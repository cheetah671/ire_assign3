"""Seeded synthetic data for Part 4.

gaussian(d): standard Gaussian in d dimensions, mapped into 128 dimensions by a
fixed random orthonormal matrix. Distances are unchanged by the mapping, so the
data has intrinsic dimension d but every vector costs the same to compare.
"""
import numpy as np

from .data import Dataset

AMBIENT = 128
N_QUERIES = 10_000


def gaussian(d, n, seed=2026, ambient=AMBIENT, nq=N_QUERIES):
    rng = np.random.default_rng(seed + d)
    # random orthonormal columns: (ambient x d), Q^T Q = I
    q, _ = np.linalg.qr(rng.standard_normal((ambient, d)))
    z = rng.standard_normal((n + nq, d))
    x = np.ascontiguousarray((z @ q.T).astype(np.float32))
    return Dataset(f"gauss_d{d}_s{seed}", x[:n], x[n:], None)


def relative_contrast(ds, gt_D, sl, n_sample=1000, seed=0):
    """Per-query relative contrast: mean distance to random base vectors
    divided by the distance to the nearest neighbour (Euclidean, not squared).
    Values near 1 mean the nearest neighbour is barely closer than anything."""
    rng = np.random.default_rng(seed)
    xq = ds.xq[sl]
    sample = ds.xb[rng.choice(ds.n, n_sample, replace=False)]
    d2 = (np.sum(xq ** 2, 1)[:, None] + np.sum(sample ** 2, 1)[None, :] - 2 * xq @ sample.T)
    mean_d = np.sqrt(np.maximum(d2, 0)).mean(axis=1)
    nn_d = np.sqrt(np.maximum(gt_D[sl, 0], 1e-12))
    return mean_d / nn_d
