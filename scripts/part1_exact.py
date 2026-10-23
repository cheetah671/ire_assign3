"""Part 1: exact search baseline.

- builds FAISS IndexFlatL2 and computes exact top-100 for all queries
- checks it against the neighbours stored in the data file
- times exact search one query at a time and as one batch

Usage: python scripts/part1_exact.py [--quick] [--repeats 5]
"""
import argparse
import os
import sys
import time

import faiss
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ann import data  # noqa: E402
from ann.metrics import exact_distances, recall_at_k, recall_vs_truth  # noqa: E402
from ann.results import n_base, save_rows  # noqa: E402
from ann.timing import measure_qps  # noqa: E402

N_TIME_SINGLE = 500  # test queries used for one-at-a-time exact timing (slow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    rows = []
    for name in data.DATASETS:
        ds = data.load(name, n_base())
        print(f"\n== {name}: {ds.n:,} base vectors, d={ds.d}")

        t0 = time.perf_counter()
        index = faiss.IndexFlatL2(ds.d)
        index.add(ds.xb)
        build = time.perf_counter() - t0

        gt_D, gt_I = data.ground_truth(ds, k=100)

        # 1) agreement with the neighbours stored in the file
        check = {}
        if ds.file_gt is not None:
            f_I = ds.file_gt.astype(np.int64)
            for k in (10, 100):
                f_D = exact_distances(ds.xq, ds.xb, f_I[:, :k])
                check[f"file_recall@{k}"] = recall_at_k(f_I, gt_I, k=k, pred_dists=f_D, gt_dists=gt_D)
                check[f"file_strict_recall@{k}"] = recall_at_k(f_I, gt_I, k=k)
            same = np.array([set(a) == set(b) for a, b in zip(f_I[:, :10], gt_I[:, :10])])
            check["file_identical_top10_frac"] = float(same.mean())
            # queries where the 10th and 11th true distances are tied
            check["ties_at_10th_frac"] = float(np.mean(np.isclose(gt_D[:, 9], gt_D[:, 10], rtol=1e-6)))
            print("  agreement with file:", {k: round(v, 5) for k, v in check.items()})
        else:
            print("  subset of base vectors: file neighbours not applicable, skipping check")

        # 2) exact search should give recall 1 on the test split (sanity)
        xq = ds.xq[ds.test]
        _, I = index.search(xq, 10)
        rec = recall_vs_truth(I, xq, ds.xb, gt_D[ds.test], gt_I[ds.test])
        print(f"  recall@10 of exact search on test queries: {rec:.4f}")

        # 3) timing
        search = index.search
        timings = [
            ("one_at_a_time", 1, xq[:N_TIME_SINGLE], True),
            ("batch", 1, xq, False),
            ("batch", os.cpu_count(), xq, False),
        ]
        for mode, threads, q, single in timings:
            qps, all_qps = measure_qps(search, q, k=10, repeats=args.repeats,
                                       threads=threads, one_at_a_time=single)
            print(f"  {mode:14s} threads={threads:2d} nq={len(q):5d}: {qps:9.1f} QPS "
                  f"({1000 / qps:.3f} ms/query)  runs={[round(x, 1) for x in all_qps]}")
            rows.append(dict(dataset=name, n_base=ds.n, dim=ds.d, index="Flat (exact)",
                             build_params="", search_params="", mode=mode, threads=threads,
                             n_queries=len(q), recall10=rec, qps=qps,
                             qps_min=min(all_qps), qps_max=max(all_qps),
                             build_s=build, index_bytes=ds.xb.nbytes,
                             bytes_per_vec=ds.xb.nbytes / ds.n, **check))
    df = save_rows(rows, "part1_exact")
    print("\n", df[["dataset", "mode", "threads", "qps", "recall10"]].to_string(index=False))


if __name__ == "__main__":
    main()
