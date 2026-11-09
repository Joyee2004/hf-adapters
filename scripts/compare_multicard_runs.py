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
reduced to the mean of its runs after dropping:

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

With ``run_profiling_modes.py --rounds N`` every line records its round.
When a file has two or more rounds, a second table follows the main one:
each round in which a label and its baseline both ran is one paired sample,
and the table shows the mean difference across rounds with a 95% interval
(Student's t) and whether it is above, below or within noise of zero::

    python3 scripts/run_profiling_modes.py --out before.jsonl --rounds 4
    python3 scripts/compare_multicard_runs.py before.jsonl

``--per-round`` also prints one table per round, with each mode's mean
durations in that round and their % against the baseline in the same round,
to spot a round that is off from the others::

    python3 scripts/compare_multicard_runs.py before.jsonl --per-round

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
    """Mean of each metric per label and rank, after dropping skipped runs.

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
            row[key] = statistics.fmean(vals) if vals else None
        rows[label] = row
    return rows


# Two-sided 95% Student's t by degrees of freedom; 1.96 beyond the table.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}  # fmt: skip


def _t95(df: int) -> float:
    known = [d for d in _T95 if d <= df]
    return _T95[max(known)] if known else _T95[1]


def _by_round(records: list[dict], skip_first: int) -> dict[str, dict[int, dict]]:
    """Mean of each metric per label and round: {label: {round: {key: mean}}}."""
    vals: dict[str, dict[int, dict[str, list]]] = {}
    for r in records:
        rnd = r.get("round")
        if rnd is None or r.get("iteration") is None or not _measured(r, skip_first):
            continue
        if r["status"] != "PASS":
            continue
        per_round = vals.setdefault(_key(r), {}).setdefault(rnd, {})
        for key, _, _ in _METRICS:
            if r.get(key) is not None:
                per_round.setdefault(key, []).append(r[key])
    return {
        label: {rnd: {k: statistics.fmean(v) for k, v in m.items()} for rnd, m in rounds.items()}
        for label, rounds in vals.items()
    }


def _print_paired(records: list[dict], skip_first: int, base_label: str) -> None:
    """Paired per-round differences against ``base_label``, if rounds were recorded."""
    rounds = _by_round(records, skip_first)
    base_key = next((k for k in rounds if k == base_label), None)
    if base_key is None or len({r for rs in rounds.values() for r in rs}) < 2:
        return
    base = rounds[base_key]
    # Labels compared against the baseline row for the same card.
    suffix = base_key[len(base_label.split(" rank")[0]):]
    others = [k for k in rounds if k != base_key and k.endswith(suffix)]
    if not others:
        return
    keys = [k for k, _, _ in _METRICS if k != "tok_s"]
    headers = ["Run vs baseline", "Rounds"] + [h for k, h, _ in _METRICS if k in keys]
    table = [headers]
    for label in others:
        cells = [label]
        common = sorted(set(rounds[label]) & set(base))
        cells.append(str(len(common)))
        for key in keys:
            diffs = [
                (rounds[label][r][key] / base[r][key] - 1) * 100
                for r in common
                if key in rounds[label][r] and base[r].get(key)
            ]
            if len(diffs) < 2:
                cells.append("N/A")
                continue
            mean = statistics.fmean(diffs)
            ci = _t95(len(diffs) - 1) * statistics.stdev(diffs) / len(diffs) ** 0.5
            verdict = "higher" if mean - ci > 0 else "lower" if mean + ci < 0 else "noise"
            cells.append(f"{mean:+.2f}% ±{ci:.2f} {verdict}")
        table.append(cells)
    print(f"\nPer-round paired difference vs {base_key} (mean ± 95% interval):")
    widths = [max(len(t[i]) for t in table) for i in range(len(headers))]
    for i, cells in enumerate(table):
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))
        if i == 0:
            print("  ".join("-" * w for w in widths))
    print(
        "higher / lower = the whole interval is above / below the baseline; "
        "noise = the interval includes 0."
    )


def _print_per_round(records: list[dict], skip_first: int, base_label: str) -> None:
    """One table per recorded round: each label's means, vs the same round's baseline."""
    rounds = sorted({r["round"] for r in records if r.get("round") is not None})
    if not rounds:
        print("\n--per-round: no round recorded in this file (needs --rounds).")
        return
    for rnd in rounds:
        rows = _group([r for r in records if r.get("round") == rnd], skip_first)
        if not rows:
            continue
        base = rows.get(base_label)
        print(f"\n== round {rnd}" + ("" if base else f"  ({base_label!r} missing; no %)"))
        _print_table(
            [
                (label + (" *" if row is base else ""), row, None if row is base else base)
                for label, row in rows.items()
            ]
        )


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
            "lines in older files without iterations, before taking the mean "
            "(default: 1; 0 keeps every run)."
        ),
    )
    parser.add_argument(
        "--per-round",
        action="store_true",
        help=(
            "With one file recorded with --rounds: also print each round's own "
            "table of mean durations per mode."
        ),
    )
    args = parser.parse_args(argv)

    loaded = {path: _load(path) for path in args.results}
    versions = {path: _group(recs, args.skip_first) for path, recs in loaded.items()}
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
        _print_paired(next(iter(loaded.values())), args.skip_first, base_label)
        if args.per_round:
            _print_per_round(next(iter(loaded.values())), args.skip_first, base_label)
        print("\n* baseline.", end="  ")
    else:
        if args.baseline is not None:
            sys.exit("--baseline applies to a single file; with several, the first is the baseline")
        if args.per_round:
            sys.exit("--per-round applies to a single file")
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
        f"Means after dropping iterations below {args.skip_first} (and unprofiled "
        "ones on profiled runs); N = kept/total lines."
    )
    print(
        "Tok/s = batch / steady ITL (decode throughput).  Wall s = per-run wall "
        "clock (generate, profiler step and trace export, output checks); "
        "Wall-exp s = Wall s minus the trace export."
    )


if __name__ == "__main__":
    main()
