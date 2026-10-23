"""Where results go, and a small CSV writer."""
import os
import sys
from pathlib import Path

import pandas as pd

from .data import REPO

QUICK = "--quick" in sys.argv
RESULTS_DIR = Path(os.environ.get("ANN_RESULTS_DIR", REPO / ("results_quick" if QUICK else "results")))
RAW_DIR = RESULTS_DIR / "raw"
FIG_DIR = RESULTS_DIR / "figures"
QUICK_N = 100_000  # base vectors used in --quick mode


def n_base():
    """Number of base vectors to use: None (all) normally, 100K with --quick."""
    return QUICK_N if QUICK else None


def save_rows(rows, name):
    """Write one part's benchmark rows to results/raw/<name>.csv (overwrites)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.insert(0, "part", name)
    df.to_csv(RAW_DIR / f"{name}.csv", index=False)
    return df
