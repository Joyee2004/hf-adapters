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

"""Compare runs recorded by ``run_multicard_smoke.py --results-json``.

Each JSON line is one timed iteration of one rank.  Lines with the same label
(and, on multi-card runs, the same rank) are repeats of one config; each is
reduced to its median after dropping:

- iterations below ``--skip-first`` (default 1) in every process.  On a
  profiled run that is the profiler's warmup run (``--profile-warmup 1``); on
  an unprofiled one it is the first timed run, dropped too so both modes are
  measured over the same iterations;
- on a profiled run, iterations after the profiler's active window, which
  ran unprofiled.

Older files without an ``iteration`` field had one line per process; there
the first ``--skip-first`` lines of each label are dropped instead.

With one file, each label is compared against the first label (or
``--baseline LABEL``)::

    python3 scripts/compare_multicard_runs.py runs.jsonl
    python3 scripts/compare_multicard_runs.py runs.jsonl --baseline "float16 b8 t64 ws1"

With several files, each file is one code version (e.g. before and after a
profiling change) and every label is compared against the same label in the
first file.  Record each version with ``scripts/run_profiling_modes.py``,
which runs every iteration of one profiling mode before moving to the next::

    python3 scripts/run_profiling_modes.py --out before.jsonl   # old code
    python3 scripts/run_profiling_modes.py --out after.jsonl    # new code
    python3 scripts/compare_multicard_runs.py before.jsonl after.jsonl

On multi-card runs every rank writes its own lines, so each card gets its own
row (e.g. ``float16 b16 t128 ws2 prof rank0`` and ``... rank1``) and is
compared against the same card in the baseline.

Needs no torch; it can run on a laptop after copying the JSONL off the pod.
"""

import argparse
import json
import os
import statistics
import sys

# (record key, column header, format)
_METRICS = [
    ("ttft_ms", "TTFT ms", "{:.3f}"),
    ("steady_itl_ms", "ITL ms", "{:.3f}"),
    ("gen_s", "Gen s", "{:.3f}"),
    ("wall_s", "Wall s", "{:.3f}"),
    ("wall_noexport_s", "Wall-exp s", "{:.3f}"),
    ("tok_s", "Tok/s", "{:.3f}"),
]


def _load(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"skipping {path}:{n}: {e}", file=sys.stderr)
    return records


def _key(r: dict) -> str:
    """Label, plus the rank on multi-card runs so each card gets its own row.

    Records written before ranks were recorded came from rank 0 only.
    """
    if r.get("world_size", 1) > 1:
        return f"{r['label']} rank{r.get('rank', 0)}"
    return r["label"]


def _measured(r: dict, skip_first: int) -> bool:
    """Whether a per-iteration line is kept (see the module docstring)."""
    i = r["iteration"]
    if i is None:  # load or warmup failed; keep it so the failure shows
        return True
    if i < skip_first:
        return False
    if r.get("with_profiling") and r.get("profile_active") is not None:
        return i < (r.get("profile_warmup") or 0) + r["profile_active"]
    return True


def _group(records: list[dict], skip_first: int) -> dict[str, dict]:
    """Median of each metric per label and rank, after dropping skipped runs.

    Returns rows keyed by label (plus `` rankN`` on multi-card runs), in
    first-seen order.
    """
    groups: dict[str, list[dict]] = {}
    for r in records:
        itl = r.get("steady_itl_ms")
        r["tok_s"] = r["batch"] * 1000.0 / itl if itl else None
        groups.setdefault(_key(r), []).append(r)

    rows = {}
    for label, runs in groups.items():
        if all("iteration" in r for r in runs):
            kept = [r for r in runs if _measured(r, skip_first)]
        else:
            kept = runs[skip_first:]
        if not kept:
            print(
                f"label {label!r}: only {len(runs)} run(s), all dropped by "
                f"--skip-first {skip_first}",
                file=sys.stderr,
            )
            continue
        row = {
            "label": label,
            "runs": f"{len(kept)}/{len(runs)}",
            "status": "/".join(sorted({r["status"] for r in kept})),
        }
        for key, _, _ in _METRICS:
            vals = [r[key] for r in kept if r.get(key) is not None]
            row[key] = statistics.median(vals) if vals else None
        rows[label] = row
    return rows


def _cell(value, fmt: str, base) -> str:
    if value is None:
        return "N/A"
    text = fmt.format(value)
    if base:
        text += f" ({(value - base) / base * 100:+.2f}%)"
    return text


def _print_table(rows: list[tuple[str, dict, dict | None]]) -> None:
    """Print (run name, row, baseline row or None) tuples as an aligned table."""
    headers = ["Run", "N", "Status"] + [h for _, h, _ in _METRICS]
    table = [headers]
    for name, row, base in rows:
        cells = [name, row["runs"], row["status"]]
        cells += [
            _cell(row[key], fmt, base[key] if base else None) for key, _, fmt in _METRICS
        ]
        table.append(cells)

    widths = [max(len(t[i]) for t in table) for i in range(len(headers))]
    for i, cells in enumerate(table):
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))
        if i == 0:
            print("  ".join("-" * w for w in widths))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="JSONL file(s) from --results-json; several files = several code versions.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="With one file: label to compare against (default: the first label).",
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Drop iterations below N in every process, or each label's first N "
            "lines in older files without iterations (default: 1)."
        ),
    )
    args = parser.parse_args(argv)

    versions = {path: _group(_load(path), args.skip_first) for path in args.results}
    if not any(versions.values()):
        sys.exit("no runs found")

    if len(versions) == 1:
        rows = next(iter(versions.values()))
        base_label = args.baseline or next(iter(rows))
        if base_label not in rows:
            sys.exit(f"baseline {base_label!r} not found; labels: {list(rows)}")
        base = rows[base_label]
        _print_table(
            [
                (label + (" *" if row is base else ""), row, None if row is base else base)
                for label, row in rows.items()
            ]
        )
        print("\n* baseline.", end="  ")
    else:
        if args.baseline is not None:
            sys.exit("--baseline applies to a single file; with several, the first is the baseline")
        base_rows = versions[args.results[0]]
        labels = list(dict.fromkeys(label for rows in versions.values() for label in rows))
        names = {path: os.path.basename(path) for path in args.results}
        for label in labels:
            print(f"\n== {label}")
            base = base_rows.get(label)
            table = []
            for path, rows in versions.items():
                if label not in rows:
                    continue
                is_base = path == args.results[0]
                table.append(
                    (names[path] + (" *" if is_base else ""), rows[label], None if is_base else base)
                )
            _print_table(table)
        print(f"\n* baseline ({names[args.results[0]]}).", end="  ")

    print(
        f"Medians after dropping iterations below {args.skip_first} (and unprofiled "
        "ones on profiled runs); N = kept/total lines."
    )
    print(
        "Tok/s = batch / steady ITL (decode throughput).  Wall s = per-run wall "
        "clock (generate, profiler step and trace export, output checks); "
        "Wall-exp s = Wall s minus the trace export."
    )


if __name__ == "__main__":
    main()
