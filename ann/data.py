"""Loading the ann-benchmarks HDF5 files, query split and exact ground truth."""
import os
from dataclasses import dataclass
from pathlib import Path

import faiss
import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("ANN_DATA_DIR", REPO / "data"))
CACHE_DIR = DATA_DIR / "cache"

# name -> (file, normalise?)  GloVe is angular, so we L2-normalise it and use
# Euclidean distance; on unit vectors this gives the same neighbour order.
DATASETS = {
    "sift": ("sift-128-euclidean.hdf5", False),
    "glove": ("glove-100-angular.hdf5", True),
}

N_TUNE = 5000  # first 5,000 queries: tuning. last 5,000: reporting.


def normalise(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # leave all-zero vectors as they are
    return np.ascontiguousarray(x / norms, dtype=np.float32)


@dataclass
class Dataset:
    name: str
    xb: np.ndarray            # base vectors (n, d)
    xq: np.ndarray            # all 10,000 queries
    file_gt: np.ndarray | None  # neighbours stored in the file (only valid for the full base set)

    @property
    def d(self):
        return self.xb.shape[1]

    @property
    def n(self):
        return self.xb.shape[0]

    @property
    def tune(self):
        return slice(0, N_TUNE)

    @property
    def test(self):
        return slice(N_TUNE, self.xq.shape[0])


def load(name, n_base=None):
    """Load a dataset. n_base keeps only the first n_base base vectors."""
    fname, norm = DATASETS[name]
    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; put the ann-benchmarks files in {DATA_DIR}")
    with h5py.File(path, "r") as f:
        total = f["train"].shape[0]
        xb = f["train"][:n_base] if n_base else f["train"][:]
        xq = f["test"][:]
        file_gt = f["neighbors"][:] if (n_base is None or n_base >= total) else None
    xb = np.ascontiguousarray(xb, dtype=np.float32)
    xq = np.ascontiguousarray(xq, dtype=np.float32)
    if norm:
        xb, xq = normalise(xb), normalise(xq)
    return Dataset(name, xb, xq, file_gt)


def ground_truth(ds, k=100):
    """Exact top-k (squared L2 distances, ids) for all queries, via FAISS IndexFlatL2.

    Cached on disk because it is reused by every part.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"gt_{ds.name}_{ds.n}_k{k}.npz"
    if path.exists():
        z = np.load(path)
        return z["D"], z["I"]
    index = faiss.IndexFlatL2(ds.d)
    index.add(ds.xb)
    faiss.omp_set_num_threads(os.cpu_count())
    D, I = index.search(ds.xq, k)
    np.savez(path, D=D, I=I)
    return D, I
