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

"""Record one code version's profiling overhead, one mode at a time.

Runs ``scripts/run_multicard_smoke.py`` under torchrun once per profiling
mode per round, finishing every iteration of a mode before starting the next:

  off    no profiling
  prof   --with-profiling
  light  --with-profiling with PROFILE_LEVEL=light

Each mode is one process that loads and compiles once, then does
``--iterations`` timed generate runs (default 6: with ``--with-profiling`` the
first warms the profiler up and the next 5 are recorded, via
``torch.profiler.schedule``).  Every rank appends one JSON line per timed run
to ``--out``, and ``scripts/compare_multicard_runs.py`` takes the mean per
mode::

    python3 scripts/run_profiling_modes.py --out before.jsonl   # old code
    # switch the installed packages to the version with your changes, then:
    python3 scripts/run_profiling_modes.py --out after.jsonl
    python3 scripts/compare_multicard_runs.py before.jsonl after.jsonl

Multi-card: list the cards in ``SPYRE_DEVICES``; one rank is launched per
card.  Use the same cards for every version you compare::

    SPYRE_DEVICES=0,1 python3 scripts/run_profiling_modes.py --out before.jsonl

Arguments after ``--`` go to every run_multicard_smoke.py run, and replace
the defaults below when they repeat one, e.g. ``-- --batch 4``.  Pass the
same ones for every version, and never ``--label``.

``--rounds N`` interleaves the modes: each round runs every mode once, as its
own process, and every other round reverses the order (off,prof,light then
light,prof,off), so each mode gets N independent processes and its average
position in the session is the same.  That removes the run-order bias and
the differences between processes that a single block per mode leaves in
TTFT and ITL.  Every JSON line records its ``round``, so a round with all
modes can be compared as one paired sample::

    python3 scripts/run_profiling_modes.py --out before.jsonl --rounds 4

Each run's console output goes to ``<out>.<mode>.log`` (``<out>.r<round>.
<mode>.log`` with ``--rounds``) and its traces to ``<trace-root>/<out>_<mode>/``
(``--trace-root`` defaults to ``/tmp/spyre_traces``, on the pod's local disk:
trace export time depends heavily on the filesystem, and on a network or
shared one it varies from run to run, more so with several ranks writing at
once; copy the traces off afterwards).  Before every mode the runner calls
``sync``, so trace data still being written to disk from the previous mode
cannot slow the next one.  A
run that fails is recorded with its status and the next one still runs; the
exit code is 1 if any failed.
"""

import argparse
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SMOKE = os.path.join("scripts", "run_multicard_smoke.py")
# Local disk on the pod, so trace export time isn't set by a network or
# shared filesystem.
_DEFAULT_TRACE_ROOT = "/tmp/spyre_traces"

# mode -> (extra smoke-script args, PROFILE_LEVEL or None)
MODES = {
    "off": ([], None),
    "prof": (["--with-profiling"], None),
    "light": (["--with-profiling"], "light"),
}

DEFAULT_SMOKE_ARGS = [
    "--dtype", "float16",
    "--prompt-file", "scripts/prompts.txt",
    "--batch", "64",
    "--max-new-tokens", "128",
    "--min-new-tokens", "128",
]  # fmt: skip


def _merge(defaults: list[str], extra: list[str]) -> list[str]:
    """``defaults`` with any ``--flag value`` pair that ``extra`` repeats removed."""
    given = {a.split("=", 1)[0] for a in extra if a.startswith("--")}
    out = []
    i = 0
    while i < len(defaults):
        flag = defaults[i]
        takes_value = i + 1 < len(defaults) and not defaults[i + 1].startswith("--")
        if flag not in given:
            out.append(flag)
            if takes_value:
                out.append(defaults[i + 1])
        i += 2 if takes_value else 1
    return out + extra


def _cards(spyre_devices: str) -> int:
    cards = [c for c in spyre_devices.replace(" ", "").split(",")]
    if not cards or not all(c.isdigit() for c in cards):
        sys.exit(
            f"SPYRE_DEVICES={spyre_devices!r}: expected comma-separated card "
            "indices like 0,1"
        )
    return len(cards)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    extra: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1 :]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        required=True,
        help="JSONL file to record into; must not exist yet.",
    )
    parser.add_argument(
        "--modes",
        default=",".join(MODES),
        help=f"Comma-separated modes, run in this order (default: {','.join(MODES)}).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=6,
        metavar="N",
        help=(
            "Timed generate runs per mode (default: 6; profiled modes need at "
            "least --profile-warmup + --profile-active, 1 + 5 by default)."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Run every mode once per round, reversing the order on every other "
            "round (default: 1, a single pass in --modes order)."
        ),
    )
    parser.add_argument(
        "--master-port",
        default="29500",
        help="torchrun rendezvous port (default: 29500).",
    )
    parser.add_argument(
        "--trace-root",
        default=_DEFAULT_TRACE_ROOT,
        help=(
            "Directory for per-mode trace subdirectories; keep it on local "
            f"disk (default: {_DEFAULT_TRACE_ROOT})."
        ),
    )
    args = parser.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown or not modes:
        parser.error(f"unknown mode(s) {unknown}; choose from {list(MODES)}")
    if "--label" in extra or "--iterations" in extra:
        parser.error("pass --iterations to this script, and never --label")
    if args.rounds < 1:
        parser.error("--rounds must be >= 1")

    os.chdir(_REPO_ROOT)
    # --results-json appends, so old runs would be mixed into the mean.
    if os.path.exists(args.out):
        sys.exit(f"{args.out} already exists; delete it or pick another name")

    spyre_devices = os.environ.get("SPYRE_DEVICES", "0").replace(" ", "")
    nproc = _cards(spyre_devices)
    stem = os.path.splitext(os.path.basename(args.out))[0]

    env = dict(os.environ)
    env["SPYRE_DEVICES"] = spyre_devices
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (_REPO_ROOT, env.get("PYTHONPATH")) if p
    )
    smoke_args = _merge(DEFAULT_SMOKE_ARGS, extra)

    # Each round runs every mode once; reversing every other round gives each
    # mode the same average position, so drift over the session cancels.
    schedule = [
        (r, mode)
        for r in range(1, args.rounds + 1)
        for mode in (modes if r % 2 else modes[::-1])
    ]
    print(
        f"{args.out}: modes {', '.join(modes)} x {args.iterations} iterations"
        + (f", {args.rounds} interleaved rounds" if args.rounds > 1 else "")
        + f" on SPYRE_DEVICES={spyre_devices} ({nproc} rank(s))",
        flush=True,
    )
    failed = []
    for n, (rnd, mode) in enumerate(schedule, 1):
        mode_args, profile_level = MODES[mode]
        # Only the light mode sets it, so an exported value must not leak
        # into the others.
        mode_env = {k: v for k, v in env.items() if k != "PROFILE_LEVEL"}
        if profile_level is not None:
            mode_env["PROFILE_LEVEL"] = profile_level
        # Read by run_multicard_smoke.py into each JSON line.
        mode_env["PROFILING_ROUND"] = str(rnd)
        cmd = [
            "torchrun",
            f"--nproc-per-node={nproc}",
            f"--master-port={args.master_port}",
            _SMOKE,
            *smoke_args,
            "--iterations", str(args.iterations),
            "--results-json", args.out,
            *mode_args,
        ]  # fmt: skip
        if "--with-profiling" in mode_args:
            cmd += ["--trace-dir", os.path.join(args.trace_root, f"{stem}_{mode}")]

        name = f"r{rnd}.{mode}" if args.rounds > 1 else mode
        log_path = f"{stem}.{name}.log"
        print(
            f"\n===== [{n}/{len(schedule)}] "
            + (f"round {rnd}, " if args.rounds > 1 else "")
            + f"mode {mode}: {' '.join(cmd)}",
            flush=True,
        )
        if profile_level is not None:
            print(f"      PROFILE_LEVEL={profile_level}", flush=True)
        print(f"      log: {log_path}", flush=True)
        # Flush everything still waiting to be written to disk, mostly the
        # previous mode's traces, so that writeback can't slow this mode's
        # load, warmup or timed runs.  Done before every mode, the first too,
        # so a set never starts with another set's data still being written.
        sync_t0 = time.perf_counter()
        os.sync()
        print(f"      sync: {time.perf_counter() - sync_t0:.1f}s", flush=True)
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                cmd,
                env=mode_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            code = proc.wait()
        print(f"===== {name}: exit code {code}", flush=True)
        if code != 0:
            failed.append(name)

    lines = 0
    if os.path.exists(args.out):
        with open(args.out) as f:
            lines = sum(1 for line in f if line.strip())
    print(f"\nWrote {args.out} ({lines} lines: one per rank per timed run)")
    if failed:
        print(f"Failed runs: {', '.join(failed)}; see their logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
