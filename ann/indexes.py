"""Building FAISS indexes and sweeping their search-time parameter."""
import os
import time

import faiss
import numpy as np

from .metrics import recall_vs_truth
from .timing import measure_qps

# search-time parameter grids (ascending; sweeps stop early, see sweep())
EF_GRID = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
NPROBE_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def pq_m_options(d):
    """PQ sub-vector counts: the divisors of d closest to d/8, d/4 and d/2
    (about 8, 4 and 2 dimensions per sub-vector, one byte each).
    SIFT (d=128): 16, 32, 64.  GloVe (d=100): 10, 25, 50."""
    divs = [m for m in range(1, d + 1) if d % m == 0]
    return [min(divs, key=lambda m: (abs(m - d / t), m)) for t in (8, 4, 2)]


def nlist_options(n):
    """Number of IVF lists: powers of two near sqrt(n), 2 sqrt(n), 4 sqrt(n).
    1M vectors: 1024, 2048, 4096.  100K vectors: 256, 512, 1024."""
    return [int(2 ** round(np.log2(c * np.sqrt(n)))) for c in (1, 2, 4)]


def build_settings(n, d):
    """Build settings tried for each index type."""
    nl = nlist_options(n)
    return {
        "HNSW": [dict(M=M, efC=100) for M in (8, 16, 32)],
        "IVF-Flat": [dict(nlist=x) for x in nl],
        "IVF-PQ": [dict(nlist=nl[1], m=m) for m in pq_m_options(d)],
    }


def build(kind, xb, params, threads=None):
    """Build an index. Returns (index, train_seconds, add_seconds).

    kind: "HNSW" (params M, efC), "IVF-Flat" (nlist), "IVF-PQ" (nlist, m).
    Building uses all cores; searching is timed on one thread elsewhere.
    """
    faiss.omp_set_num_threads(threads or os.cpu_count())
    d = xb.shape[1]
    if kind == "HNSW":
        index = faiss.IndexHNSWFlat(d, params["M"])
        index.hnsw.efConstruction = params["efC"]
    elif kind == "IVF-Flat":
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, params["nlist"])
    elif kind == "IVF-PQ":
        index = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, params["nlist"], params["m"], 8)
    else:
        raise ValueError(kind)

    t0 = time.perf_counter()
    if not index.is_trained:
        # k-means uses at most 256 points per centroid; sample them ourselves
        # with a fixed seed so every run trains on the same points
        n_train = min(xb.shape[0], 256 * params["nlist"])
        rs = np.random.default_rng(123)
        sample = xb[np.sort(rs.choice(xb.shape[0], n_train, replace=False))]
        index.cp.seed = 1234  # k-means seed (IVF quantiser)
        index.train(sample)
    t1 = time.perf_counter()
    index.add(xb)
    t2 = time.perf_counter()
    return index, t1 - t0, t2 - t1


def set_search_param(index, kind, value):
    if kind == "HNSW":
        index.hnsw.efSearch = value
    else:
        index.nprobe = value


def search_param_name(kind):
    return "efSearch" if kind == "HNSW" else "nprobe"


def grid_for(kind, params):
    if kind == "HNSW":
        return EF_GRID
    return [p for p in NPROBE_GRID if p <= params["nlist"]]


def index_bytes(index):
    return int(faiss.serialize_index(index).size)


def sweep(kind, params, index, ds, gt_D, gt_I, n_time, repeats=5, stop_recall=0.995, k=10, log=print):
    """Sweep the search parameter. For each value:

    - recall@10 on all tune queries (used for choosing settings)
    - recall@10 on all test queries (reported)
    - QPS on the first n_time test queries: one query per call, one thread,
      median of `repeats` passes (reported)
    - QPS on the first n_time//2 tune queries, one pass (used only for choosing)

    Stops once tune recall >= stop_recall, or when recall has stopped
    improving (PQ indexes plateau below 1).
    """
    xq_tune, xq_test = ds.xq[ds.tune], ds.xq[ds.test]
    rows, prev = [], []
    for value in grid_for(kind, params):
        set_search_param(index, kind, value)
        faiss.omp_set_num_threads(os.cpu_count())  # recall: batch, any threads
        _, I_tune = index.search(xq_tune, k)
        _, I_test = index.search(xq_test, k)
        r_tune = recall_vs_truth(I_tune, xq_tune, ds.xb, gt_D[ds.tune], gt_I[ds.tune], k)
        r_test = recall_vs_truth(I_test, xq_test, ds.xb, gt_D[ds.test], gt_I[ds.test], k)

        qps, runs = measure_qps(index.search, xq_test[:n_time], k, repeats=repeats, threads=1)
        qps_tune, _ = measure_qps(index.search, xq_tune[:n_time // 2], k, repeats=1, threads=1, warmup=0)
        log(f"    {search_param_name(kind)}={value:5d}  recall tune={r_tune:.4f} "
            f"test={r_test:.4f}  QPS={qps:9.1f}")
        rows.append(dict(search_param=search_param_name(kind), search_value=value,
                         recall10_tune=r_tune, recall10=r_test, qps=qps,
                         qps_min=min(runs), qps_max=max(runs), qps_tune=qps_tune,
                         n_time_queries=n_time, repeats=repeats, threads=1))
        prev.append(r_tune)
        if r_tune >= stop_recall:
            break
        if len(prev) >= 3 and prev[-1] > 0.5 and prev[-1] - prev[-3] < 0.002:  # plateau
            break
    return rows
