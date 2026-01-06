
# new_sweep/plot_sweeps.py
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from new_sweep_old.plot_thesis_style import use_thesis_style

def main():
    parser = argparse.ArgumentParser(description='Plot ABM sweep results (basic)')
    parser.add_argument('--in', dest='in_csv', type=str, default='results/sweeps.csv')
    parser.add_argument('--outdir', type=str, default='figs_new_sweeps')
    args = parser.parse_args()

    use_thesis_style()
    df = pd.read_csv(args.in_csv)
    os.makedirs(args.outdir, exist_ok=True)

    pivot = df.groupby(['merge_prob', 'invasive_speed'])['mean_size_final'].mean().reset_index()
    piv = pivot.pivot(index='invasive_speed', columns='merge_prob', values='mean_size_final')
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(piv.values, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{y:.2f}" for y in piv.index])
    ax.set_xlabel('Merge probability')
    ax.set_ylabel('Invasive speed')
    ax.set_title('Final mean cluster size (averaged over reps)')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('mean_size_final')
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'heatmap_mean_size_final_basic.png'))

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for inv_speed, sub in df.groupby('invasive_speed'):
        means = sub.groupby('merge_prob')['n_final'].mean()
        ax2.plot(means.index, means.values, marker='o', label=f"invasive_speed={inv_speed:.2f}")
    ax2.set_xlabel('Merge probability')
    ax2.set_ylabel('N clusters (final, mean over reps)')
    ax2.set_title('Final number of clusters vs merge probability')
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='best', frameon=False)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.outdir, 'lines_n_final_vs_merge_basic.png'))

    print(f"Wrote figures -> {args.outdir}")

if __name__ == '__main__':
    main()
