"""Timing helpers: wall-clock, repeated, median."""
import time

import faiss
import numpy as np


def _run(search, xq, k, one_at_a_time):
    t0 = time.perf_counter()
    if one_at_a_time:
        for i in range(xq.shape[0]):
            search(xq[i:i + 1], k)
    else:
        search(xq, k)
    return time.perf_counter() - t0


def measure_qps(search, xq, k=10, repeats=5, threads=1, one_at_a_time=True, warmup=50):
    """Queries per second of search(x, k), median over `repeats` timed passes.

    one_at_a_time=True sends each query as its own call (latency-style use);
    False sends all queries in one call (batch use).
    Returns (median_qps, list_of_qps).
    """
    faiss.omp_set_num_threads(threads)
    if warmup:  # untimed pass so caches and lazy allocations are warm
        _run(search, xq[:warmup], k, one_at_a_time)
    qps = [xq.shape[0] / _run(search, xq, k, one_at_a_time) for _ in range(repeats)]
    return float(np.median(qps)), qps
