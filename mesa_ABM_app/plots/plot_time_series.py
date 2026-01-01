
# plots/plot_time_series.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser(description="Plot S0/S1/S2/Ncells time-series for a single tag.")
    ap.add_argument("--condition", required=True, choices=["proliferative", "invasive"])
    ap.add_argument("--sweep", required=True, choices=["speed_adhesion", "proliferation", "fragmentation", "density"])
    ap.add_argument("--tag", required=True, help="e.g., proliferative_v2_adh0.7 or proliferative_p0.005")
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="figures/ch4")
    args = ap.parse_args()

    base = Path(args.results) / args.condition / args.sweep / args.tag
    runs = sorted(base.glob("run_*/summary_S012.csv"))
    if not runs:
        raise FileNotFoundError(f"No runs found under {base}")

    dfs = [pd.read_csv(fp).assign(run=fp.parent.name) for fp in runs]
    df = pd.concat(dfs, ignore_index=True)

    g = df.groupby("hours")
    mean = g[["S0","S1","S2","Ncells"]].mean()
    sem  = g[["S0","S1","S2","Ncells"]].sem()
    ci95 = 1.96*sem

    fig, axs = plt.subplots(2,2, figsize=(9,6), sharex=True)
    axs = axs.ravel()
    titles = ["S0: number of clusters","S1: mean cluster size","S2: mean cluster size²","Total cells"]
    for ax, col, title in zip(axs, ["S0","S1","S2","Ncells"], titles):
        ax.plot(mean.index, mean[col], color="tab:blue", lw=2)
        ax.fill_between(mean.index, mean[col]-ci95[col], mean[col]+ci95[col],
                        color="tab:blue", alpha=0.2, lw=0)
        ax.set_title(title)
        ax.set_xlabel("Time (hours)")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{args.condition.capitalize()} — {args.sweep} — {args.tag}", y=1.02)
    fig.tight_layout()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"time_series__{args.condition}__{args.sweep}__{args.tag}.png"
    fig.savefig(outpath, dpi=300)
    print(f"[saved] {outpath}")

if __name__ == "__main__":
    main()
