
# sweeps/plot_sweeps.py
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot ABM sweep results")
    parser.add_argument("--in", dest="in_csv", type=str, default="results/sweeps.csv")
    parser.add_argument("--outdir", type=str, default="figs_sweeps")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Heatmap: mean of mean_size_final across reps, vs (merge_prob, invasive_speed)
    pivot = df.groupby(["merge_prob", "invasive_speed"])["mean_size_final"].mean().reset_index()
    piv = pivot.pivot(index="invasive_speed", columns="merge_prob", values="mean_size_final")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{y:.2f}" for y in piv.index])
    ax.set_xlabel("merge_prob")
    ax.set_ylabel("invasive_speed")
    ax.set_title("Final mean cluster size (averaged over reps)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean_size_final")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "heatmap_mean_size_final.png"), dpi=200)
    plt.close(fig)

    # 2) Lines: n_final vs merge_prob (one line per invasive_speed)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for inv_speed, sub in df.groupby("invasive_speed"):
        means = sub.groupby("merge_prob")["n_final"].mean()
        ax2.plot(means.index, means.values, marker="o", label=f"invasive_speed={inv_speed:.2f}")
    ax2.set_xlabel("merge_prob")
    ax2.set_ylabel("N clusters (final, mean over reps)")
    ax2.set_title("Final number of clusters vs merge_prob")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="best", frameon=False)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.outdir, "lines_n_final_vs_merge.png"), dpi=200)
    plt.close(fig2)

    # 3) Lines: total_cells_final vs invasive_speed (one line per merge_prob)
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    for merge_prob, sub in df.groupby("merge_prob"):
        means = sub.groupby("invasive_speed")["total_cells_final"].mean()
        ax3.plot(means.index, means.values, marker="o", label=f"merge_prob={merge_prob:.2f}")
    ax3.set_xlabel("invasive_speed")
    ax3.set_ylabel("Total cells (final, mean over reps)")
    ax3.set_title("Total cells vs invasive_speed")
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc="best", frameon=False)
    fig3.tight_layout()
    fig3.savefig(os.path.join(args.outdir, "lines_total_cells_vs_invasive_speed.png"), dpi=200)
    plt.close(fig3)

    print(f"Wrote figures -> {args.outdir}")


if __name__ == "__main__":
    main()
