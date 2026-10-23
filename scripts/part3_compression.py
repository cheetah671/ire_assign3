"""Part 3 (Compression): IVF-PQ vs IVF-PQ + exact re-ranking of the top 100, on SIFT.

Varies m (PQ sub-vectors, 1 byte each) with nlist fixed; for each m the same
trained index is searched plain and with re-ranking while sweeping nprobe.
Also records candidate recall: how many of the true top 10 are in the PQ top
100 at all (the most that re-ranking can recover).

Usage: python scripts/part3_compression.py [--quick]
"""
import argparse
import os
import sys
import time

import faiss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ann import data  # noqa: E402
from ann.indexes import build, index_bytes, nlist_options, sweep  # noqa: E402
from ann.metrics import candidate_recall  # noqa: E402
from ann.results import QUICK, n_base, save_rows  # noqa: E402

M_VALUES = [8, 16, 32, 64]
RERANK = 100  # candidates re-ranked exactly
K = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    n_time = 300 if QUICK else 2000

    ds = data.load("sift", n_base())
    gt_D, gt_I = data.ground_truth(ds, k=100)
    nlist = nlist_options(ds.n)[1]  # same nlist as Part 2's IVF-PQ
    xq_test = ds.xq[ds.test]
    print(f"== sift {ds.n:,} vectors, nlist={nlist}", flush=True)

    rows = []
    for m in M_VALUES:
        params = dict(nlist=nlist, m=m)
        base, train_s, add_s = build("IVF-PQ", ds.xb, params)
        refine = faiss.IndexRefineFlat(base, faiss.swig_ptr(ds.xb))
        refine.k_factor = RERANK / K
        size_plain, size_rerank = index_bytes(base), index_bytes(refine)
        print(f"  m={m}: train {train_s:.1f}s add {add_s:.1f}s  "
              f"{size_plain / ds.n:.1f} B/vec plain, {size_rerank / ds.n:.1f} B/vec with re-ranking",
              flush=True)

        def cand(value):  # nprobe is already set on `base` by sweep()
            faiss.omp_set_num_threads(os.cpu_count())
            _, I = base.search(xq_test, RERANK)
            return dict(candidate_recall_test=candidate_recall(I, xq_test, ds.xb, gt_D[ds.test], K))

        for variant, search, size in (("plain", base.search, size_plain),
                                      (f"rerank{RERANK}", refine.search, size_rerank)):
            t0 = time.perf_counter()
            print(f"   {variant}", flush=True)
            for r in sweep("IVF-PQ", params, base, ds, gt_D, gt_I, n_time, repeats=args.repeats,
                           search=search, extra=cand if variant == "plain" else None,
                           log=lambda s: print(s, flush=True)):
                rows.append(dict(dataset="sift", n_base=ds.n, dim=ds.d, index="IVF-PQ",
                                 variant=variant, nlist=nlist, m=m, code_bytes=m,
                                 build_params=f"nlist={nlist} m={m}", train_s=train_s,
                                 add_s=add_s, build_s=train_s + add_s, index_bytes=size,
                                 bytes_per_vec=size / ds.n, **r))
            print(f"   ({time.perf_counter() - t0:.0f}s)", flush=True)
        save_rows(rows, "part3_compression")
        del refine, base


if __name__ == "__main__":
    main()
