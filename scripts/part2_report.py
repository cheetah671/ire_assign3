"""Part 2 figures and tables from results/raw/part2_*.csv and part1_exact.csv.

Operating points are chosen on the TUNE queries only: among all settings of an
index type whose tune recall@10 >= target, take the one with the highest QPS
measured on tune queries. The table then reports that setting's recall and
QPS on the TEST queries.
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

STYLE = {  # fixed colour + marker per index type (marker = non-colour cue)
    "HNSW": ("#2a78d6", "o"),
    "IVF-Flat": ("#eb6834", "s"),
    "IVF-PQ": ("#1baf7a", "^"),
}
EXACT_COLOR = "#52514e"
TARGETS = (0.90, 0.95)


def load():
    files = sorted(glob.glob(str(RAW_DIR / "part2_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["dataset", "index", "build_params", "search_value"], keep="last")
    exact = pd.read_csv(RAW_DIR / "part1_exact.csv")
    exact = exact[(exact["mode"] == "one_at_a_time") & (exact["threads"] == 1)]
    return df, exact


def pareto(g):
    """Points not beaten on both recall and QPS by another point."""
    g = g.sort_values(["recall10", "qps"], ascending=[False, False])
    keep, best = [], -1.0
    for _, r in g.iterrows():
        if r["qps"] > best:
            keep.append(r)
            best = r["qps"]
    return pd.DataFrame(keep).sort_values("recall10")


def operating_point(g, target):
    ok = g[g["recall10_tune"] >= target]
    if ok.empty:
        return None
    return ok.loc[ok["qps_tune"].idxmax()]


def plot(df, exact):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ds, d in df.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
        for kind, (color, marker) in STYLE.items():
            g = d[d["index"] == kind]
            if g.empty:
                continue
            ax.scatter(g["recall10"], g["qps"], s=10, color=color, alpha=0.25, linewidths=0)
            p = pareto(g)
            ax.plot(p["recall10"], p["qps"], color=color, lw=2, marker=marker, ms=5,
                    label=f"{kind} (best of {g['build_params'].nunique()} builds)")
        e = exact[exact["dataset"] == ds]
        if not e.empty:
            ax.scatter([1.0], e["qps"], marker="*", s=140, color=EXACT_COLOR, zorder=5,
                       label=f"Exact (Flat): {e['qps'].iloc[0]:.0f} QPS")
        for t in TARGETS:
            ax.axvline(t, color="#b5b4ae", lw=0.8, ls="--", zorder=0)
            ax.annotate(f"{t:.2f}", (t, 1), xycoords=("data", "axes fraction"), xytext=(2, -10),
                        textcoords="offset points", fontsize=7, color="#8a8984")
        ax.set_yscale("log")
        ax.set_xlim(left=max(0.0, d["recall10"].min() - 0.02), right=1.012)
        ax.set_xlabel("recall@10 (test queries)")
        ax.set_ylabel("queries per second (1 thread, log scale)")
        n = d["n_base"].iloc[0]
        ax.set_title(f"{ds.upper()} ({n:,} vectors): recall vs speed", loc="left", fontsize=11)
        ax.grid(True, which="major", color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"part2_tradeoff_{ds}.png")
        plt.close(fig)


def tables(df, exact):
    qps_rows, cost_rows = [], []
    for (ds, kind), g in df.groupby(["dataset", "index"], sort=False):
        for t in TARGETS:
            op = operating_point(g, t)
            row = dict(dataset=ds, index=kind, target=t)
            if op is None:
                best = g.loc[g["recall10_tune"].idxmax()]
                row.update(setting="not reached", recall10=None, qps=None,
                           max_recall10=best["recall10"],
                           note=f"max {best['recall10']:.3f} at {best['build_params']} nprobe={best['search_value']}")
            else:
                row.update(setting=f"{op['build_params']} {op['search_param']}={op['search_value']}",
                           recall10=op["recall10"], qps=op["qps"],
                           speedup_vs_exact=op["qps"] / exact.loc[exact["dataset"] == ds, "qps"].iloc[0])
            qps_rows.append(row)
            if t == 0.95:
                src = op if op is not None else g.loc[g["recall10_tune"].idxmax()]
                cost_rows.append(dict(dataset=ds, index=kind,
                                      setting=src["build_params"] + ("" if op is not None else " (max recall; 0.95 not reached)"),
                                      build_s=src["build_s"], train_s=src["train_s"], add_s=src["add_s"],
                                      index_mib=src["index_bytes"] / 2**20,
                                      bytes_per_vec=src["bytes_per_vec"]))
    for _, e in exact.iterrows():
        qps_rows.append(dict(dataset=e["dataset"], index="Exact (Flat)", target="any",
                             setting="", recall10=1.0, qps=e["qps"], speedup_vs_exact=1.0))
        cost_rows.append(dict(dataset=e["dataset"], index="Exact (Flat)", setting="",
                              build_s=e["build_s"], index_mib=e["index_bytes"] / 2**20,
                              bytes_per_vec=e["bytes_per_vec"]))
    out = RESULTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    q, c = pd.DataFrame(qps_rows), pd.DataFrame(cost_rows)
    q.to_csv(out / "part2_qps_at_recall.csv", index=False)
    c.to_csv(out / "part2_build_cost.csv", index=False)
    with open(out / "part2_tables.md", "w") as f:
        f.write("## QPS at recall@10 targets (chosen on tune queries, measured on test)\n\n")
        f.write(q.to_markdown(index=False, floatfmt=".3f") + "\n\n")
        f.write("## Build time and size at recall 0.95\n\n")
        f.write(c.to_markdown(index=False, floatfmt=".2f") + "\n")
    print(open(out / "part2_tables.md").read())


if __name__ == "__main__":
    df, exact = load()
    plot(df, exact)
    tables(df, exact)
