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

Pass ``--with-profiling`` to wrap the timed generate call in
``torch.profiler.profile`` (CPU + PrivateUse1 activities, shapes, memory).
Model load, torch.compile and the warmup generate run before the profiler
starts, so the trace covers steady-state generation only.  Each rank writes a
TensorBoard trace to ``--trace-dir`` (one ``<hostname>_<pid>.pt.trace.json``
per rank); view with ``tensorboard --logdir`` or https://ui.perfetto.dev.
Profiling adds overhead, so TTFT / ITL from a profiled run are not comparable
to an unprofiled one::

    export SPYRE_DEVICES=0
    torchrun --nproc-per-node=1 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16 \\
        --with-profiling --trace-dir ./spyre_traces

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
            "Profile the timed generate call with torch.profiler "
            "(load, compile and warmup excluded)."
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
            "Append this run's summary as one JSON line to PATH (rank 0 only), "
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
    wall_t0 = time.time()
    args = build_parser().parse_args(argv)
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

    # Holds the profiler once generate_ctx() has been called, so its
    # key_averages() table can be printed after the run.
    profilers: list = []

    def generate_ctx():
        phase_t["warmup_end"] = time.time()
        if not args.with_profiling:
            return contextlib.nullcontext()
        from torch.profiler import ProfilerActivity, profile

        # Exits (and on_trace_ready fires) inside run_multicard_smoke_test,
        # well before the os._exit() below skips all teardown.
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.trace_dir),
        )
        profilers.append(prof)
        return prof

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

    warmup_s = None
    if "warmup_start" in phase_t and "warmup_end" in phase_t:
        warmup_s = phase_t["warmup_end"] - phase_t["warmup_start"]

    # ── Summary table ──────────────────────────────────────────────────────
    # Each rank builds its summary as one string and writes it atomically.
    # A brief sleep stagger keeps blocks from interleaving in the terminal.
    def _rank_summary(r: dict, total_s: float) -> str:
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
            f"  Total wall : {_fmt(total_s, 's')}  (main() start to this summary)",
        ]
        # gen_s is None when load or warmup failed, i.e. the profiler never ran.
        if args.with_profiling and r["gen_s"] is not None:
            lines.append(f"  Trace dir  : {args.trace_dir}")
        if r["error"]:
            lines += ["", "  ERROR DETAIL:"]
            lines += [f"    {line}" for line in r["error"].splitlines()]
        lines += ["=" * 70, ""]
        return "\n".join(lines)

    def _profile_table(prof) -> str:
        averages = prof.key_averages()
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
            f"top {args.profile_rows} ops by {sort_by}"
        )
        return "\n".join(["", "=" * 70, header, "=" * 70, table, ""])

    profile_table = ""
    if profilers and result["gen_s"] is not None and args.profile_rows > 0:
        profile_table = _profile_table(profilers[-1])
    # Measured after the profile table is built, so it covers everything this
    # rank did: load, warmup, generate, trace export and table aggregation.
    total_s = time.time() - wall_t0

    time.sleep(local_rank * 0.3)
    sys.stdout.write(profile_table)
    sys.stdout.write(_rank_summary(result, total_s))
    sys.stdout.flush()

    if args.results_json and is_rank0:
        label = args.label or (
            f"{args.dtype or 'default'} b{args.batch} "
            f"t{args.max_new_tokens} ws{world_size}"
            + (" prof" if args.with_profiling else "")
            + (f" PROFILE_LEVEL={profile_level}" if profile_level else "")
        )
        record = {
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
            "status": result["status"],
            "load_s": result["load_s"],
            "warmup_s": warmup_s,
            "gen_s": result["gen_s"],
            "ttft_ms": result["ttft_ms"],
            "decode_ms": result["decode_ms"],
            "steady_itl_ms": result["steady_itl_ms"],
            "total_s": total_s,
        }
        try:
            with open(args.results_json, "a") as _f:
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
