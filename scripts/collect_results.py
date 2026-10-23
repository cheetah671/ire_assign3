"""Merge every benchmark run (results/raw/*.csv) into results/results.csv.

One row per benchmark run (one index + build setting + search setting on one
dataset, or one exact-search timing). The `part` column says which script
produced it; columns that do not apply to a part are left empty.
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ann.results import RAW_DIR, RESULTS_DIR  # noqa: E402

ORDER = ["part1_exact", "part2_sift_glove", "part3_compression", "part4_geometry", "dataset_stats", "check_timing_bias"]


def key(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return ORDER.index(name) if name in ORDER else len(ORDER)


frames = [pd.read_csv(f) for f in sorted(glob.glob(str(RAW_DIR / "*.csv")), key=key)]
df = pd.concat(frames, ignore_index=True, sort=False)
first = ["part", "dataset", "data", "n_base", "dim", "index", "variant", "build_params",
         "search_param", "search_value", "mode", "threads", "recall10", "recall10_tune",
         "qps", "qps_min", "qps_max", "n_queries", "n_time_queries", "repeats",
         "build_s", "train_s", "add_s", "index_bytes", "bytes_per_vec"]
cols = [c for c in first if c in df.columns] + [c for c in df.columns if c not in first]
df = df[cols]
df.to_csv(RESULTS_DIR / "results.csv", index=False)
print(f"results.csv: {len(df)} rows, {len(cols)} columns")
print(df["part"].value_counts().to_string())
