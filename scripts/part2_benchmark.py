"""Part 2: benchmark HNSW, IVF-Flat and IVF-PQ on SIFT and GloVe.

For each index type and build setting, build once (all cores), then sweep the
search parameter, recording recall@10 (tune and test queries) and
single-thread one-query-at-a-time QPS (test queries, median of 5).

Usage: python scripts/part2_benchmark.py [--quick] [--datasets sift glove]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ann import data  # noqa: E402
from ann.indexes import build, build_settings, index_bytes, sweep  # noqa: E402
from ann.results import QUICK, n_base, save_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=list(data.DATASETS))
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    n_time = 300 if QUICK else 2000  # test queries used for timing

    rows = []
    for name in args.datasets:
        ds = data.load(name, n_base())
        gt_D, gt_I = data.ground_truth(ds, k=100)
        print(f"\n== {name}: {ds.n:,} x {ds.d}", flush=True)
        for kind, settings in build_settings(ds.n, ds.d).items():
            for params in settings:
                t0 = time.perf_counter()
                index, train_s, add_s = build(kind, ds.xb, params)
                size = index_bytes(index)
                print(f"  {kind} {params}: train {train_s:.1f}s add {add_s:.1f}s "
                      f"size {size / 2**20:.1f} MiB ({size / ds.n:.1f} B/vec)", flush=True)
                common = dict(dataset=name, n_base=ds.n, dim=ds.d, index=kind,
                              build_params=" ".join(f"{k}={v}" for k, v in params.items()),
                              **params, train_s=train_s, add_s=add_s, build_s=train_s + add_s,
                              build_threads=os.cpu_count(), index_bytes=size,
                              bytes_per_vec=size / ds.n)
                for r in sweep(kind, params, index, ds, gt_D, gt_I, n_time,
                               repeats=args.repeats, log=lambda s: print(s, flush=True)):
                    rows.append({**common, **r})
                del index
                save_rows(rows, "part2_" + "_".join(args.datasets))  # save as we go
                print(f"    ({time.perf_counter() - t0:.0f}s for this setting)", flush=True)


if __name__ == "__main__":
    main()
