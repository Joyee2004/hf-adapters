#!/usr/bin/env bash
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
#
# Profiling overhead test: records one code version's runs of
# scripts/run_multicard_smoke.py in three modes, interleaved REPEATS times:
#   1. no profiling
#   2. --with-profiling
#   3. --with-profiling + PROFILE_LEVEL=light
#
# Usage (on the pod, from anywhere):
#   bash scripts/run_profiling_overhead.sh before          # code without your changes
#   # switch the installed packages to the version with your changes, then:
#   bash scripts/run_profiling_overhead.sh after
#   python3 scripts/compare_multicard_runs.py before.jsonl after.jsonl
#
# Multi-card: list the cards in SPYRE_DEVICES; one rank is launched per card.
#   SPYRE_DEVICES=0,1 bash scripts/run_profiling_overhead.sh before
# Use the same SPYRE_DEVICES for every version you compare: labels end in
# ws<cards>, and before/after are only matched at the same card count.
#
# Between versions, check the package switch took effect (pip show, git commit
# or the loaded .so path), and clear the compile caches if build time should be
# part of the comparison.  Keep the model and the pod the same.
#
# Each NAME writes, in the repo root:
#   NAME.jsonl           REPEATS x 3 lines, one per run, written by rank 0
#   NAME.log             full console output (every rank's summary)
#   traces/NAME_prof/    profiler traces for mode 2, one file per rank per run
#   traces/NAME_light/   profiler traces for mode 3, one file per rank per run
#
# Extra arguments are passed on to every run, e.g. to change the batch:
#   bash scripts/run_profiling_overhead.sh before --batch 4
# Pass the same extra arguments for every version you compare, and never
# --label or --iterations: before/after are matched by the default label, and
# each run is a fresh process with one timed generate.
#
# Environment:
#   SPYRE_DEVICES  comma-separated card indices (default 0); valid indices are
#                  node-specific
#   REPEATS        passes over the three modes (default 6; the compare script
#                  drops each mode's first run, so 6 gives a median of 5)
#   MASTER_PORT    torchrun rendezvous port (default 29500)
#
# The very first run doubles as a preflight: if it fails (bad card index,
# model doesn't load, ...) the whole set stops instead of burning every run.
# Later failures are recorded with their status and the loop keeps going.
# Each run loads and compiles from scratch, so a set takes a while: start it in
# tmux or under nohup.  With a median of 5, treat differences of about 1-2% as
# noise.

set -euo pipefail

if [[ $# -lt 1 || "$1" == -* ]]; then
    sed -n '/^# Usage/,/^#   SPYRE_DEVICES=0,1/p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
name=$1
shift

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SPYRE_DEVICES=${SPYRE_DEVICES:-0}
# Only mode 3 sets it, so an exported value must not leak into modes 1 and 2.
unset PROFILE_LEVEL

# One rank per card, so the rank count can't disagree with SPYRE_DEVICES.
SPYRE_DEVICES=${SPYRE_DEVICES// /}
IFS=, read -ra cards <<<"$SPYRE_DEVICES"
nproc=0
for c in "${cards[@]}"; do
    if [[ ! "$c" =~ ^[0-9]+$ ]]; then
        echo "SPYRE_DEVICES=$SPYRE_DEVICES: expected comma-separated card indices like 0,1" >&2
        exit 1
    fi
    nproc=$((nproc + 1))
done
if ((nproc == 0)); then
    echo "SPYRE_DEVICES is empty" >&2
    exit 1
fi

repeats=${REPEATS:-6}
out="$name.jsonl"
# --results-json appends, so old runs would be mixed into the median.
if [[ -e "$out" ]]; then
    echo "$out already exists; delete it or pick another name" >&2
    exit 1
fi
mkdir -p traces

run() {
    torchrun --nproc-per-node="$nproc" --master-port="${MASTER_PORT:-29500}" \
        scripts/run_multicard_smoke.py \
        --dtype float16 --prompt-file scripts/prompts.txt --batch 16 \
        --max-new-tokens 128 --min-new-tokens 128 \
        --results-json "$out" "$@"
}

echo "===== $name: $repeats passes x 3 modes on SPYRE_DEVICES=$SPYRE_DEVICES ($nproc rank(s))"

# Interleaving the modes spreads node noise across all three.
for ((i = 1; i <= repeats; i++)); do
    echo "===== pass $i/$repeats: no profiling"
    if ! run "$@" && ((i == 1)); then
        echo "===== first run failed; stopping (see the summary above)"
        exit 1
    fi
    echo "===== pass $i/$repeats: --with-profiling"
    run --with-profiling --trace-dir "traces/${name}_prof" "$@" || true
    echo "===== pass $i/$repeats: --with-profiling PROFILE_LEVEL=light"
    PROFILE_LEVEL=light run --with-profiling --trace-dir "traces/${name}_light" "$@" || true
done 2>&1 | tee "$name.log"

echo "Wrote $out ($(wc -l <"$out" | tr -d ' ') runs), $name.log and traces/${name}_{prof,light}/"
