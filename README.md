# Assignment 3: Benchmarking Approximate Nearest Neighbour Search

CS4.406 Information Retrieval & Extraction.

## Setup

```bash
pip install -r requirements.txt
```

Put the two ann-benchmarks files in `data/` (they are not in the repo):

```
data/sift-128-euclidean.hdf5
data/glove-100-angular.hdf5
```

Set `ANN_DATA_DIR` to use a different folder. Exact ground truth is computed once
with FAISS and cached in `data/cache/`.

## Running

```bash
bash run_all.sh          # full run on all base vectors -> results/
bash run_all.sh --quick  # first 100,000 base vectors  -> results_quick/
```

`run_all.sh` runs the unit tests, records the machine and library versions
(`results/environment.json`), then runs every part and regenerates all
numbers, tables and figures from the data files.

## Layout

| Path | What |
|---|---|
| `ann/data.py` | loading HDF5, GloVe normalisation, query split, exact ground truth |
| `ann/metrics.py` | hand-written recall@k (tie-aware) |
| `ann/timing.py` | timing (single thread, one query at a time, median of 5) |
| `ann/results.py` | output locations, CSV writer |
| `tests/test_recall.py` | unit tests for recall@k |
| `scripts/part1_exact.py` | Part 1: exact baseline, check against file, timing |
| `ann/indexes.py` | index building, build settings, search-parameter sweeps |
| `scripts/part2_benchmark.py` | Part 2: build + sweep every index on both datasets |
| `scripts/part2_report.py` | Part 2: trade-off plots, QPS-at-recall and build-cost tables |

## Conventions

- GloVe vectors and queries are L2-normalised; Euclidean distance is used everywhere.
- Queries 0-4,999 are used for tuning, 5,000-9,999 for reported numbers.
- Recall@10: a returned id counts as correct if its exact distance is no larger
  than the true 10th-neighbour distance, so ties at the 10th neighbour are not mistakes.
- Speed: queries per second, one query per call, one thread, median of 5 timed passes
  (Part 2: first 2,000 test queries; exact one-at-a-time: first 500, as it takes ~30 ms/query).
  Index building uses all cores.
- Operating points (e.g. "QPS at recall 0.95") are chosen on the tune queries only
  (tune recall >= target, highest QPS on tune queries); the reported recall and QPS
  are then measured on the test queries.
- Outputs: `results/raw/*.csv` (every run), `results/tables/`, `results/figures/`.
