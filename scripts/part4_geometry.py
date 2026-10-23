"""Part 4 (Hard geometry): intrinsic dimension vs HNSW and IVF-Flat.

Synthetic Gaussian data with intrinsic dimension d in {8,16,32,64,128},
stored as 128-d vectors; control = the first N real SIFT vectors. Fixed build
settings; sweeps, recall and timing as in Part 2. Also records relative
contrast, distance computations per query and exact-search QPS.

Usage: python scripts/part4_geometry.py [--quick]
"""
import argparse
import os
import sys
import time

import faiss
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ann import data, synthetic  # noqa: E402
from ann.indexes import build, index_bytes, nlist_options, sweep  # noqa: E402
from ann.results import QUICK, save_rows  # noqa: E402
from ann.timing import measure_qps, on_ac_power  # noqa: E402

DIMS = [8, 16, 32, 64, 128]
N = 50_000 if QUICK else 200_000


def distance_counter(kind, index, xq):
    """Function(value) -> average distance computations per query (1 thread)."""
    def f(value):
        faiss.omp_set_num_threads(1)
        stats = faiss.cvar.hnsw_stats if kind == "HNSW" else faiss.cvar.indexIVF_stats
        stats.reset()
        index.search(xq, 10)
        return dict(ndis_per_query=stats.ndis / len(xq))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    n_time = 300 if QUICK else 2000

    settings = [(f"gauss d={d}", d, lambda d=d: synthetic.gaussian(d, N)) for d in DIMS]
    settings.append(("SIFT (control)", None, lambda: data.load("sift", N)))

    rows = []
    for label, d_int, make in settings:
        ds = make()
        gt_D, gt_I = data.ground_truth(ds, k=100)
        rc = synthetic.relative_contrast(ds, gt_D, ds.test)
        xq_test = ds.xq[ds.test]
        print(f"\n== {label}: {ds.n:,} x {ds.d}, relative contrast median {np.median(rc):.3f}", flush=True)

        common = dict(data=label, intrinsic_dim=d_int, n_base=ds.n, dim=ds.d,
                      rel_contrast_median=float(np.median(rc)), rel_contrast_mean=float(rc.mean()))
        first_row = len(rows)
        for kind, params in (("HNSW", dict(M=16, efC=100)),
                             ("IVF-Flat", dict(nlist=nlist_options(ds.n)[0]))):
            t0 = time.perf_counter()
            index, train_s, add_s = build(kind, ds.xb, params)
            size = index_bytes(index)
            print(f"  {kind} {params}: build {train_s + add_s:.1f}s", flush=True)
            for r in sweep(kind, params, index, ds, gt_D, gt_I, n_time, repeats=args.repeats,
                           extra=distance_counter(kind, index, xq_test[:1000]),
                           log=lambda s: print(s, flush=True)):
                rows.append({**common, "index": kind,
                             "build_params": " ".join(f"{k}={v}" for k, v in params.items()),
                             **params, "build_s": train_s + add_s, "index_bytes": size,
                             "bytes_per_vec": size / ds.n, **r})
            del index
            print(f"    ({time.perf_counter() - t0:.0f}s)", flush=True)

        # Exact search is timed last, after a pause: timed first (straight
        # after data generation and ground truth) it came out ~2x too slow.
        exact = faiss.IndexFlatL2(ds.d)
        exact.add(ds.xb)
        time.sleep(5 if QUICK else 20)
        exact_qps, _ = measure_qps(exact.search, xq_test[:300], 10, repeats=args.repeats, threads=1)
        print(f"  exact: {exact_qps:.1f} QPS", flush=True)
        del exact
        ac = on_ac_power()
        for r in rows[first_row:]:
            r["exact_qps"] = exact_qps
            r["exact_ac_power"] = ac
        save_rows(rows, "part4_geometry")


if __name__ == "__main__":
    main()
