"""Part 4 main figure and table from results/raw/part4_geometry.csv.

Operating points are chosen on tune queries (as in Part 2).
"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ann.results import FIG_DIR, RAW_DIR, RESULTS_DIR  # noqa: E402
from part2_report import STYLE, operating_point  # noqa: E402

CTRL = "SIFT (control)"
GRAY = "#52514e"
TARGET = 0.95


def summarise(df):
    rows = []
    for (data, kind), g in df.groupby(["data", "index"], sort=False):
        base = dict(data=data, intrinsic_dim=g["intrinsic_dim"].iloc[0], index=kind,
                    rel_contrast=g["rel_contrast_median"].iloc[0], exact_qps=g["exact_qps"].iloc[0],
                    n_base=g["n_base"].iloc[0], max_recall10=g["recall10"].max())
        for t in (0.90, 0.95):
            op = operating_point(g, t)
            if op is None:
                base.update({f"param@{t:.2f}": None, f"recall@{t:.2f}": None, f"qps@{t:.2f}": None,
                             f"ndis@{t:.2f}": None, f"speedup@{t:.2f}": None})
                continue
            base.update({f"param@{t:.2f}": f"{op['search_param']}={int(op['search_value'])}",
                         f"recall@{t:.2f}": op["recall10"], f"qps@{t:.2f}": op["qps"],
                         f"ndis@{t:.2f}": op["ndis_per_query"],
                         f"speedup@{t:.2f}": op["qps"] / op["exact_qps"]})
        rows.append(base)
    return pd.DataFrame(rows)


def plot(s):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    syn = s[s["data"] != CTRL].copy()
    syn["intrinsic_dim"] = syn["intrinsic_dim"].astype(int)
    ctrl = s[s["data"] == CTRL]
    dims = sorted(syn["intrinsic_dim"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), dpi=150)
    t = f"{TARGET:.2f}"

    # (a) contrast
    a = axes[0]
    rc = syn.groupby("intrinsic_dim")["rel_contrast"].first()
    a.plot(rc.index, rc.values, color=GRAY, marker="o", lw=2, label="synthetic Gaussian")
    if not ctrl.empty:
        a.axhline(ctrl["rel_contrast"].iloc[0], color=GRAY, ls="--", lw=1,
                  label=f"SIFT control ({ctrl['rel_contrast'].iloc[0]:.2f})")
    a.axhline(1, color="#b5b4ae", lw=0.8)
    a.set_ylabel("relative contrast (median)\nmean distance / nearest distance")
    a.set_title("(a) Nearest neighbour stands out less", loc="left", fontsize=10)

    # (b) distance computations, (c) QPS
    for ax, col, title in ((axes[1], f"ndis@{t}", f"(b) Distance computations / query at recall {t}"),
                           (axes[2], f"qps@{t}", f"(c) Queries per second at recall {t}")):
        for kind in ("HNSW", "IVF-Flat"):
            color, marker = STYLE[kind]
            g = syn[syn["index"] == kind].sort_values("intrinsic_dim")
            ok = g[g[col].notna()]
            ax.plot(ok["intrinsic_dim"], ok[col], color=color, marker=marker, lw=2, label=kind)
            miss = g[g[col].isna()]
            if not miss.empty:  # x on the bottom edge where the target was never reached
                best = ", ".join(f"{m:.2f} at d={d}" for d, m in zip(miss["intrinsic_dim"], miss["max_recall10"]))
                ax.scatter(miss["intrinsic_dim"], [0.02] * len(miss), marker="x", s=60, color=color,
                           transform=ax.get_xaxis_transform(), clip_on=False, zorder=5,
                           label=f"{kind}: {t} not reached (best {best})")
            c = ctrl[ctrl["index"] == kind]
            if not c.empty and pd.notna(c[col].iloc[0]):
                ax.axhline(c[col].iloc[0], color=color, ls="--", lw=1, label=f"{kind}, SIFT control")
        if col.startswith("ndis"):
            n = syn["n_base"].iloc[0]
            ax.axhline(n, color=GRAY, lw=1, ls=":", label=f"exact search ({n:,})")
        else:
            ex = syn.groupby("intrinsic_dim")["exact_qps"].first()
            ax.plot(ex.index, ex.values, color=GRAY, lw=1.2, ls=":", marker="*", label="exact search")
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=10)
    axes[1].set_ylabel("distance computations per query (log)")
    axes[2].set_ylabel("QPS, 1 thread (log)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims, [str(d) for d in dims])
        ax.set_xlabel("intrinsic dimension d (stored as 128-d)")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    handles, labels = axes[2].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    for h, l in zip(h1, l1):
        if l not in labels:
            handles.append(h)
            labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    n = syn["n_base"].iloc[0]
    fig.suptitle(f"Part 4: intrinsic dimension vs HNSW (M=16) and IVF-Flat, {n:,} vectors",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(FIG_DIR / "part4_geometry.png")
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(RAW_DIR / "part4_geometry.csv")
    s = summarise(df)
    out = RESULTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    s.to_csv(out / "part4_geometry.csv", index=False)
    with open(out / "part4_tables.md", "w") as f:
        f.write("## Part 4: intrinsic dimension (operating points chosen on tune queries)\n\n")
        f.write(s.to_markdown(index=False, floatfmt=".3f") + "\n")
    print(open(out / "part4_tables.md").read())
    plot(s)
