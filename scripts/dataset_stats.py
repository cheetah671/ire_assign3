"""Relative contrast of the real datasets (test queries), for comparison with
Part 4. Writes results/raw/dataset_stats.csv."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ann import data, synthetic  # noqa: E402
from ann.results import n_base, save_rows  # noqa: E402

rows = []
for name in data.DATASETS:
    ds = data.load(name, n_base())
    gt_D, _ = data.ground_truth(ds, k=100)
    rc = synthetic.relative_contrast(ds, gt_D, ds.test)
    rows.append(dict(dataset=name, n_base=ds.n, dim=ds.d, rel_contrast_median=float(np.median(rc)),
                     rel_contrast_mean=float(rc.mean())))
    print(rows[-1], flush=True)
save_rows(rows, "dataset_stats")
