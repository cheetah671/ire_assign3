"""Part 3 figure and table from results/raw/part3_compression.csv.

Control: IVF-Flat with the same nlist on the same data, taken from Part 2.
Operating points are chosen on tune queries (as in Part 2).
"""
import glob
import os
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ann.results import FIG_DIR, RAW_DIR, RESULTS_DIR  # noqa: E402
from part2_report import operating_point  # noqa: E402

M_COLORS = {8: "#9ec5f0", 16: "#5a9de6", 32: "#2a78d6", 64: "#174f94"}  # light -> dark = more bytes
CONTROL_COLOR = "#52514e"
SERIES = {"plain": "#eb6834", "rerank": "#2a78d6", "cand": "#1baf7a"}


def load():
    df = pd.read_csv(RAW_DIR / "part3_compression.csv")
    p2 = pd.concat([pd.read_csv(f) for f in glob.glob(str(RAW_DIR / "part2_*.csv"))])
    nlist = df["nlist"].iloc[0]
    ctrl = p2[(p2["dataset"] == "sift") & (p2["index"] == "IVF-Flat") & (p2["nlist"] == nlist)]
    return df, ctrl


def plot(df, ctrl):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150,
                               gridspec_kw=dict(width_ratios=[1.6, 1]))
    rr = [v for v in df["variant"].unique() if v != "plain"][0]
    for m, g in df.groupby("m"):
        c = M_COLORS.get(m, "#2a78d6")
        p = g[g["variant"] == "plain"].sort_values("search_value")
        r = g[g["variant"] == rr].sort_values("search_value")
        a.plot(p["recall10"], p["qps"], color=c, lw=1.6, ls="--", marker="^", ms=4)
        a.plot(r["recall10"], r["qps"], color=c, lw=2, marker="o", ms=4, label=f"m={m} ({m} B/vec codes)")
    if not ctrl.empty:
        c = ctrl.sort_values("search_value")
        a.plot(c["recall10"], c["qps"], color=CONTROL_COLOR, lw=1.6, marker="s", ms=4,
               label=f"control: IVF-Flat, no compression")
    a.plot([], [], color="#8a8984", ls="--", marker="^", ms=4, label="dashed: plain IVF-PQ")
    a.plot([], [], color="#8a8984", marker="o", ms=4, label="solid: + re-rank top 100")
    a.axvline(0.95, color="#b5b4ae", lw=0.8, ls="--", zorder=0)
    a.set_yscale("log")
    a.set_xlabel("recall@10 (test queries)")
    a.set_ylabel("queries per second (1 thread, log)")
    nlist, n = df["nlist"].iloc[0], df["n_base"].iloc[0]
    a.set_title(f"IVF-PQ on SIFT ({n:,} vectors, nlist={nlist}), sweeping nprobe", loc="left", fontsize=10)
    a.legend(frameon=False, fontsize=7.5, loc="lower left")

    # right: best reachable recall vs code size
    best = df.groupby(["m", "variant"])["recall10"].max().unstack()
    cand = df[df["variant"] == "plain"].groupby("m")["candidate_recall_test"].max()
    ms = best.index.values
    b.plot(ms, best["plain"], color=SERIES["plain"], marker="^", lw=2, label="plain IVF-PQ")
    b.plot(ms, best[rr], color=SERIES["rerank"], marker="o", lw=2, label="+ re-rank top 100")
    b.plot(ms, cand, color=SERIES["cand"], marker="D", lw=1.4, ls=":",
           label="true top-10 inside PQ top-100")
    if not ctrl.empty:
        b.axhline(ctrl["recall10"].max(), color=CONTROL_COLOR, lw=1, ls="--",
                  label="IVF-Flat (control), best")
    b.set_xscale("log", base=2)
    b.set_xticks(ms, [str(m) for m in ms])
    b.set_xlabel("m = bytes per vector in PQ codes")
    b.set_ylabel("highest recall@10 reached (test)")
    b.set_title("Recall ceiling vs code size", loc="left", fontsize=10)
    b.legend(frameon=False, fontsize=7.5, loc="lower right")
    for ax in (a, b):
        ax.grid(True, color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "part3_compression.png")
    plt.close(fig)


def table(df, ctrl):
    rows = []
    groups = [(f"IVF-PQ m={m} {v}", g) for (m, v), g in df.groupby(["m", "variant"])]
    if not ctrl.empty:
        groups.append(("IVF-Flat (control)", ctrl))
    for name, g in groups:
        row = dict(config=name, bytes_per_vec=g["bytes_per_vec"].iloc[0],
                   build_s=g["build_s"].iloc[0], max_recall10=g["recall10"].max())
        if "candidate_recall_test" in g and g["candidate_recall_test"].notna().any():
            row["top10_in_top100"] = g["candidate_recall_test"].max()
        for t in (0.90, 0.95):
            op = operating_point(g, t)
            row[f"qps@{t:.2f}"] = None if op is None else op["qps"]
            row[f"nprobe@{t:.2f}"] = None if op is None else int(op["search_value"])
        rows.append(row)
    t = pd.DataFrame(rows)
    out = RESULTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "part3_compression.csv", index=False)
    with open(out / "part3_tables.md", "w") as f:
        f.write("## Part 3: compression (SIFT)\n\n" + t.to_markdown(index=False, floatfmt=".3f") + "\n")
    print(open(out / "part3_tables.md").read())


if __name__ == "__main__":
    df, ctrl = load()
    plot(df, ctrl)
    table(df, ctrl)
