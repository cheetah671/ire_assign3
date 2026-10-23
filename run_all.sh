#!/usr/bin/env bash
# Regenerates every number, table and figure from the data files in data/.
#   bash run_all.sh          full run (1M vectors), writes results/
#   bash run_all.sh --quick  first 100K base vectors, writes results_quick/
set -euo pipefail
cd "$(dirname "$0")"

QUICK=""
if [[ "${1:-}" == "--quick" ]]; then QUICK="--quick"; fi
PY="${PYTHON:-python}"

echo "== unit tests"
"$PY" -m pytest -q tests

echo "== environment"
"$PY" scripts/env_info.py $QUICK

echo "== Part 1: exact baseline"
"$PY" scripts/part1_exact.py $QUICK

echo "== Part 2: HNSW / IVF-Flat / IVF-PQ benchmark"
"$PY" scripts/part2_benchmark.py $QUICK
"$PY" scripts/part2_report.py $QUICK
