"""Checks one way the timings could mislead: single-thread speed measured
straight after an all-core index build, versus after waiting.

Builds IVF-PQ (100K SIFT vectors), then times the same search repeatedly at
0, 10, 20, 30 and 60 s after the build finished (0, 5, 10 s with --quick). Writes
results/raw/check_timing_bias.csv.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ann import data  # noqa: E402
from ann.indexes import build  # noqa: E402
from ann.results import QUICK, save_rows  # noqa: E402
from ann.timing import measure_qps  # noqa: E402

ds = data.load("sift", 100_000)
xq = ds.xq[ds.test][:2000]
index, _, _ = build("IVF-PQ", ds.xb, dict(nlist=512, m=32))
index.nprobe = 1
t_built = time.perf_counter()
rows = []
for wait in ((0, 5, 10) if QUICK else (0, 10, 20, 30, 60)):
    time.sleep(max(0.0, wait - (time.perf_counter() - t_built)))
    qps, runs = measure_qps(index.search, xq, 10, repeats=5, threads=1)
    rows.append(dict(seconds_after_build=round(time.perf_counter() - t_built, 1), qps=qps,
                     qps_min=min(runs), qps_max=max(runs)))
    print(rows[-1], flush=True)
save_rows(rows, "check_timing_bias")
