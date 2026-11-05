# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the multicard smoke test and print a human-readable summary.

For real tensor-parallel multi-card runs, launch with torchrun (one process
per card).  For single-card, plain python works.

Requirements:
  - transformers==5.15.x  (uv pip install "transformers==5.15.0")

SPYRE_DEVICES tells the Flex runtime which card indices to use per rank.
Index-to-physical-card mapping is handled internally by Flex and is not
independently verifiable from this script.

1-card example::

    export SPYRE_DEVICES=0
    torchrun --nproc-per-node=1 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16

2-card example (valid indices are node-specific — check yours first)::

    export SPYRE_DEVICES=0,1
    torchrun --nproc-per-node=2 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16

4-card example::

    export SPYRE_DEVICES=0,1,2,3
    torchrun --nproc-per-node=4 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16

Use ``--max-new-tokens`` to control output length.  By default the model
stops at EOS (typically just a few tokens).  Pass ``--min-new-tokens``
equal to ``--max-new-tokens`` to suppress EOS and force exactly N tokens
every run — useful for consistent latency benchmarking::

    export SPYRE_DEVICES=0
    torchrun --nproc-per-node=1 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16 \\
        --max-new-tokens 256 --min-new-tokens 256

Pass ``--with-profiling`` to run the timed generate calls under one
``torch.profiler.profile`` (CPU + PrivateUse1 activities, shapes, memory)
driven by ``torch.profiler.schedule``: the first ``--profile-warmup`` timed
runs (default 1) warm the profiler up and are discarded, and the next
``--profile-active`` (default 5) are recorded into one trace, so
``--iterations`` must be at least their sum.  Model load, torch.compile and
the untimed warmup generate run before the profiler starts.  Each rank writes
a TensorBoard trace to ``--trace-dir`` (one ``<hostname>_<pid>.pt.trace.json``
per rank); view with ``tensorboard --logdir`` or https://ui.perfetto.dev.
Profiling adds overhead, so TTFT / ITL from a profiled run are not comparable
to an unprofiled one::

    export SPYRE_DEVICES=0
    torchrun --nproc-per-node=1 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16 --iterations 6 \\
        --with-profiling --trace-dir ./spyre_traces

To measure profiling overhead, record each mode with
``scripts/run_profiling_modes.py`` and compare with
``scripts/compare_multicard_runs.py``.

The --model argument accepts any HuggingFace repo ID or local path
(default: ibm-granite/granite-4.2-8b).

The script exits with code 0 on PASS and code 1 on FAIL or ERROR. A FAIL is
reported by that exit code alone, not by an exception, so under torchrun the
only thing printed after the summary is the elastic agent's teardown:

    Sending process <pid> closing signal SIGTERM
    failed (exitcode: 1) local_rank: 0 ...
    torch.distributed.elastic.multiprocessing.errors.ChildFailedError

That SIGTERM goes to the ranks that were still running; torchrun stops the
whole gang once any rank exits nonzero. It is the result of the failure, never
the cause -- read the RESULTS SUMMARY above it for the real verdict.

Note: a ``corrupted double-linked list`` / SIGABRT crash may appear after the
RESULTS SUMMARY prints.  This is a known shutdown bug in the Spyre runtime
(libsenlib-dd2.so destructors) and does not affect correctness — if the
summary shows ``Status: PASS`` the test passed.
"""

import argparse
import contextlib
import json
import os
import sys
import time

import torch
import torch.distributed as dist

# Ensure the project root (parent of scripts/) is on sys.path so that
# tests.spyre.test_multicard_spyre can be imported when running directly
# from anywhere inside the repo.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.spyre.test_multicard_spyre import run_multicard_smoke_test  # noqa: E402

DEFAULT_PROMPT = "The capital of France is"
DEFAULT_MODEL = "ibm-granite/granite-4.2-8b"
DEFAULT_MAX_NEW_TOKENS = 8
DEFAULT_TRACE_DIR = "./spyre_traces"
DEFAULT_PROFILE_ROWS = 30
DEFAULT_PROFILE_WARMUP = 1
DEFAULT_PROFILE_ACTIVE = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model path or local dir (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Token generation budget (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help=(
            "Torch dtype to pass to AutoSpyreModelForCausalLM.from_pretrained "
            "(e.g. float16, bfloat16, float32).  Omit to let the model decide."
        ),
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=None,
        help=(
            "Suppress EOS until this many tokens have been generated.  "
            "Set equal to --max-new-tokens to force exactly N tokens every run "
            "(useful for latency benchmarking).  Omit to let the model stop naturally."
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        metavar="N",
        help="Number of identical prompts to batch together (default: 1).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        metavar="N",
        help="Number of timed generate runs to perform (default: 1).",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help=f"Read prompt from the file (default: {DEFAULT_PROMPT})",
    )
    parser.add_argument(
        "--with-profiling",
        action="store_true",
        help=(
            "Profile the timed generate runs with torch.profiler on a schedule "
            "(load, compile and the untimed warmup excluded); needs --iterations "
            ">= --profile-warmup + --profile-active."
        ),
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=DEFAULT_PROFILE_WARMUP,
        metavar="N",
        help=(
            "Timed runs that warm the profiler up and are discarded "
            f"(default: {DEFAULT_PROFILE_WARMUP})."
        ),
    )
    parser.add_argument(
        "--profile-active",
        type=int,
        default=DEFAULT_PROFILE_ACTIVE,
        metavar="N",
        help=(
            "Timed runs recorded into the trace after the warmup "
            f"(default: {DEFAULT_PROFILE_ACTIVE})."
        ),
    )
    parser.add_argument(
        "--trace-dir",
        default=DEFAULT_TRACE_DIR,
        help=(
            "Directory for TensorBoard profiler traces, used with "
            f"--with-profiling (default: {DEFAULT_TRACE_DIR})."
        ),
    )
    parser.add_argument(
        "--profile-rows",
        type=int,
        default=DEFAULT_PROFILE_ROWS,
        metavar="N",
        help=(
            "Rows in the key_averages() table printed after a profiled run "
            f"(default: {DEFAULT_PROFILE_ROWS}; 0 disables the table)."
        ),
    )
    parser.add_argument(
        "--results-json",
        default=None,
        metavar="PATH",
        help=(
            "Append one JSON line per rank per timed iteration to PATH, "
            "for comparing runs with scripts/compare_multicard_runs.py."
        ),
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Name for this run in --results-json (default: a config summary).",
    )
    return parser


def _fmt(value: float | None, unit: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}{unit}"


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.with_profiling:
        if args.profile_warmup < 0 or args.profile_active < 1:
            parser.error("--profile-warmup must be >= 0 and --profile-active >= 1")
        if args.iterations < args.profile_warmup + args.profile_active:
            parser.error(
                f"--with-profiling records timed runs {args.profile_warmup + 1}-"
                f"{args.profile_warmup + args.profile_active}, so it needs "
                f"--iterations >= {args.profile_warmup + args.profile_active} "
                f"(got {args.iterations})"
            )
    if args.dtype is not None:
        if args.dtype not in _DTYPE_MAP:
            raise ValueError(
                f"Unknown --dtype {args.dtype!r}. "
                f"Choose from: {', '.join(_DTYPE_MAP)}"
            )
    dtype = _DTYPE_MAP.get(args.dtype) if args.dtype is not None else None

    aiu_ids = os.environ.get("AIU_IDS")
    spyre_devices = os.environ.get("SPYRE_DEVICES")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_rank0 = local_rank == 0
    # Read by the runtime, not by this script; recorded so --results-json can
    # tell "--with-profiling" runs apart from "PROFILE_LEVEL=light" ones.
    profile_level = os.environ.get("PROFILE_LEVEL")

    # Trim AIU_IDS to WORLD_SIZE entries so a 4-address env var doesn't get
    # printed when only 1 or 2 ranks are actually launched.
    if aiu_ids is not None:
        _ids = [c.strip() for c in aiu_ids.split(",") if c.strip()]
        if len(_ids) > world_size:
            aiu_ids = ",".join(_ids[:world_size])

    if is_rank0:
        print()
        print("=" * 70)
        print("  run_multicard_smoke")
        print("=" * 70)
        print(f"  SPYRE_DEVICES    : {spyre_devices!r}")
        print(f"  AIU_IDS          : {aiu_ids!r}  (fallback; SPYRE_DEVICES is primary)")
        print(f"  Model            : {args.model}")
        print(f"  max_new_tokens   : {args.max_new_tokens}")
        print(f"  batch            : {args.batch}")
        print(f"  iterations       : {args.iterations}")
        print(f"  dtype            : {args.dtype or '(not set — model default)'}")
        print(f"  Prompt File      : {args.prompt_file}")
        print(
            f"  Profiling        : {args.trace_dir if args.with_profiling else 'off'}"
            + (
                f"  (warmup={args.profile_warmup}, active={args.profile_active})"
                if args.with_profiling
                else ""
            )
        )
        print(f"  PROFILE_LEVEL    : {profile_level!r}")
        print("=" * 70)

    if args.prompt_file is not None:
        try:
            with open(args.prompt_file, "r") as _f:
                file_prompts = [line.strip() for line in _f if line.strip()]
        except OSError as e:
            raise RuntimeError(
                f"Cannot read prompt file {args.prompt_file!r}: {e}"
            ) from e
        if not file_prompts:
            raise RuntimeError(f"Prompt file {args.prompt_file!r} has no prompts")
        prompt = file_prompts[0]
    else:
        file_prompts = None
        prompt = DEFAULT_PROMPT

    # run_multicard_smoke_test encodes the inputs right before the warmup
    # generate and enters generate_ctx right after it, so the time between the
    # two is the warmup.  Swap the encoder it imports to record the start and,
    # with --prompt-file, to give each batch row its own line (cycling through
    # the file when --batch exceeds the number of lines) instead of repeating
    # one prompt --batch times.
    import tests.conftest as _conftest

    _encode = _conftest.encode_generation_inputs
    phase_t: dict[str, float] = {}

    def _encode_and_mark(tokenizer, prompts):
        if file_prompts is not None:
            prompts = [file_prompts[i % len(file_prompts)] for i in range(len(prompts))]
        encoded = _encode(tokenizer, prompts)
        phase_t["warmup_start"] = time.time()
        return encoded

    _conftest.encode_generation_inputs = _encode_and_mark

    # One profiler spans every timed run: generate_ctx() starts it on the first
    # run and steps it after each, so the schedule discards the warmup runs and
    # records the active ones into a single trace.  The trace is exported
    # (on_trace_ready) by the step after the last active run, inside
    # run_multicard_smoke_test and outside its gen_s timing.
    prof = None
    prof_started = False
    if args.with_profiling:
        from torch.profiler import ProfilerActivity, profile, schedule

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            schedule=schedule(
                wait=0,
                warmup=args.profile_warmup,
                active=args.profile_active,
                repeat=1,
            ),
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.trace_dir),
        )

    @contextlib.contextmanager
    def generate_ctx():
        nonlocal prof_started
        # Entered right before each timed run; only the first ends the warmup.
        phase_t.setdefault("warmup_end", time.time())
        if prof is not None and not prof_started:
            prof.start()
            prof_started = True
        yield
        if prof is not None:
            prof.step()

    result = run_multicard_smoke_test(
        args.model,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        dtype=dtype,
        batch_size=args.batch,
        prompt=prompt,
        num_iterations=args.iterations,
        generate_ctx=generate_ctx,
    )
    # Wall clock of the timed runs: from the end of the untimed warmup (the
    # first generate_ctx() entry) to the return above, so it excludes load and
    # warmup but includes the per-run checks and the profiler's trace export.
    # None when load or warmup failed and no timed run started.
    run_s = time.time() - phase_t["warmup_end"] if "warmup_end" in phase_t else None
    # Saves a partial trace when a failure cut the active runs short; a no-op
    # once the schedule has finished.
    if prof_started:
        prof.stop()

    warmup_s = None
    if "warmup_start" in phase_t and "warmup_end" in phase_t:
        warmup_s = phase_t["warmup_end"] - phase_t["warmup_start"]

    # ── Summary table ──────────────────────────────────────────────────────
    # Each rank builds its summary as one string and writes it atomically.
    # A brief sleep stagger keeps blocks from interleaving in the terminal.
    def _rank_summary(r: dict) -> str:
        its = r["iterations"]
        lines = [
            "",
            "=" * 70,
            f"  RESULTS SUMMARY  [rank {r['local_rank']}/{r['world_size']}]",
            "=" * 70,
            f"  SPYRE_DEVICES : {r['spyre_devices_env']!r}",
        ]
        if r["resolved_cards"]:
            lines.append(
                f"  PCI hint      : {', '.join(r['resolved_cards'])}"
                f"  (AIU_WORLD_RANK_* guess, unconfirmed)"
            )
        lines += [
            f"  AIU_IDS       : {r['aiu_ids_env']!r}",
            f"  Model         : {r['model']}",
            f"  Batch      : {r['batch_size']}",
            f"  Status     : {r['status']}",
            f"  Load       : {_fmt(r['load_s'], 's')}",
            f"  Warmup     : {_fmt(warmup_s, 's')}  (untimed generate, incl. compile)",
            f"  Generate   : {_fmt(r['gen_s'], 's')}",
            f"  TTFT       : {_fmt(r['ttft_ms'], ' ms')}",
            f"  Decode avg : {_fmt(r['decode_ms'], ' ms')}",
            f"  Steady ITL : {_fmt(r['steady_itl_ms'], ' ms')}  (outliers excluded)",
            f"  Output     : {r['output']!r}",
            f"  Run wall   : {_fmt(run_s, 's')}  (all timed runs; load and warmup excluded)",
        ]
        # Generate..Steady ITL above are the last timed run; list them all.
        if len(its) > 1:
            lines.append(f"  Timed runs : {len(its)}/{args.iterations}")
            for i, it in enumerate(its):
                tag = ""
                if args.with_profiling:
                    tag = (
                        "  (profiler warmup)"
                        if i < args.profile_warmup
                        else "  (profiled)"
                        if i < args.profile_warmup + args.profile_active
                        else ""
                    )
                lines.append(
                    f"    [{i}] {it['status']:<5} gen {_fmt(it['gen_s'], 's')}"
                    f"  TTFT {_fmt(it['ttft_ms'], ' ms')}"
                    f"  ITL {_fmt(it['steady_itl_ms'], ' ms')}{tag}"
                )
        # gen_s is None when load or warmup failed, i.e. the profiler never ran.
        if args.with_profiling and r["gen_s"] is not None:
            lines.append(f"  Trace dir  : {args.trace_dir}")
        if r["error"]:
            lines += ["", "  ERROR DETAIL:"]
            lines += [f"    {line}" for line in r["error"].splitlines()]
        lines += ["=" * 70, ""]
        return "\n".join(lines)

    def _profile_table(prof) -> str:
        try:
            averages = prof.key_averages()
        except Exception:  # nothing recorded, e.g. a failure during warmup
            return ""
        # Sort by Spyre (PrivateUse1) time; fall back to CPU time on torch
        # versions without the device_time fields.
        for sort_by in ("self_device_time_total", "self_cpu_time_total"):
            try:
                table = averages.table(sort_by=sort_by, row_limit=args.profile_rows)
                break
            except (AttributeError, KeyError, ValueError):
                continue
        else:
            return ""
        header = (
            f"  PROFILE  [rank {local_rank}/{world_size}]  "
            f"top {args.profile_rows} ops by {sort_by}, "
            f"over {args.profile_active} profiled run(s)"
        )
        return "\n".join(["", "=" * 70, header, "=" * 70, table, ""])

    profile_table = ""
    if prof_started and args.profile_rows > 0:
        profile_table = _profile_table(prof)
    time.sleep(local_rank * 0.3)
    sys.stdout.write(profile_table)
    sys.stdout.write(_rank_summary(result))
    sys.stdout.flush()

    # Every rank writes one line per timed run, after the stagger above, so on
    # multi-card runs each card's timings are kept and the lines don't
    # interleave.  Load, warmup and wall times are per process, so they repeat
    # on each of its lines.  A load or warmup failure, which leaves no timed
    # run, still gets one line so the failure shows up in the compare.
    if args.results_json:
        label = args.label or (
            f"{args.dtype or 'default'} b{args.batch} "
            f"t{args.max_new_tokens} ws{world_size}"
            + (" prof" if args.with_profiling else "")
            + (f" PROFILE_LEVEL={profile_level}" if profile_level else "")
        )
        base = {
            "label": label,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": args.model,
            "dtype": args.dtype,
            "batch": args.batch,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "prompt_file": args.prompt_file,
            "with_profiling": args.with_profiling,
            "profile_level": profile_level,
            "world_size": world_size,
            "rank": local_rank,
            "spyre_devices": spyre_devices,
            "iterations": args.iterations,
            "profile_warmup": args.profile_warmup if args.with_profiling else None,
            "profile_active": args.profile_active if args.with_profiling else None,
            "load_s": result["load_s"],
            "warmup_s": warmup_s,
            "run_s": run_s,
        }
        no_run = {
            "status": result["status"],
            "gen_s": None,
            "ttft_ms": None,
            "decode_ms": None,
            "steady_itl_ms": None,
        }
        records = [
            {**base, "iteration": i, **it} for i, it in enumerate(result["iterations"])
        ] or [{**base, "iteration": None, **no_run}]
        try:
            with open(args.results_json, "a") as _f:
                for record in records:
                    _f.write(json.dumps(record) + "\n")
        except OSError as e:
            sys.stdout.write(f"\nCould not write --results-json {args.results_json!r}: {e}\n")
        sys.stdout.flush()

    failed = result["status"] != "PASS"

    # A FAIL is reported by exiting nonzero, not by raising, so torchrun has no
    # traceback to show. Say so explicitly: otherwise the only thing in the log
    # is the agent's SIGTERM/ChildFailedError teardown, which reads like an
    # abort or a timeout rather than a verdict this script already reached.
    if failed and local_rank == 0:
        sys.stdout.write(
            "\n"
            f"multicard smoke FAILED (status={result['status']!r}); "
            "exiting rank 0 with code 1.\n"
            "The torchrun 'SIGTERM' / ChildFailedError below is the elastic "
            "agent tearing down the remaining ranks after this exit. It is the "
            "consequence of this failure, not its cause; the real reason is in "
            "the RESULTS SUMMARY above.\n"
        )
        sys.stdout.flush()

    # Ranks reach their verdict a few ms apart. Whichever exits first leaves its
    # siblings alive, and torchrun SIGTERMs them mid-print -- truncating their
    # summaries and making one rank's failure look like an unrelated signal
    # kill. Sync first so every rank prints fully and they exit together.
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception as e:  # pragma: no cover - teardown best effort
            sys.stdout.write(f"\n[rank {local_rank}] exit barrier skipped: {e}\n")
            sys.stdout.flush()

    # os._exit (not sys.exit) is deliberate: it skips interpreter teardown and
    # so avoids the libsenlib-dd2.so destructor SIGABRT described in the module
    # docstring. It also skips atexit/buffer flushing, hence the flushes above.
    os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
