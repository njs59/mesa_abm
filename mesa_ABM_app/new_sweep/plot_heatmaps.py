
# new_sweep/plot_heatmaps.py
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from new_sweep_old.plot_thesis_style import use_thesis_style

def heatmap(ax, Z, xlabel, ylabel, title, xticklabels, yticklabels, cmap='viridis', cbar_label=None):
    im = ax.imshow(Z, aspect='auto', origin='lower', cmap=cmap)
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels([f"{x:.2f}" for x in xticklabels])
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels([f"{y:.2f}" for y in yticklabels])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = ax.figure.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)

def main():
    parser = argparse.ArgumentParser(description='Thesis-ready heatmaps for extended sweeps')
    parser.add_argument('--in_csv', type=str, default='new_results/sweeps_extended.csv')
    parser.add_argument('--outdir', type=str, default='new_results/figs')
    args = parser.parse_args()

    use_thesis_style()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.in_csv)
    g = df.groupby(['merge_prob', 'inv_speed']).agg({'mean_size_final': 'mean', 'n_final': 'mean', 'total_cells_final': 'mean'}).reset_index()

    piv = g.pivot(index='inv_speed', columns='merge_prob', values='mean_size_final')
    fig, ax = plt.subplots(figsize=(6.4, 5))
    heatmap(ax, piv.values, xlabel='Merge probability', ylabel='Invasive speed', title='Final mean cluster size', xticklabels=piv.columns, yticklabels=piv.index, cbar_label='Mean size (final)')
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'ext_heatmap_mean_size_final.png'))

    piv2 = g.pivot(index='inv_speed', columns='merge_prob', values='n_final')
    fig2, ax2 = plt.subplots(figsize=(6.4, 5))
    heatmap(ax2, piv2.values, xlabel='Merge probability', ylabel='Invasive speed', title='Final number of clusters', xticklabels=piv2.columns, yticklabels=piv2.index, cbar_label='N clusters (final)')
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.outdir, 'ext_heatmap_n_final.png'))

    print(f"Wrote heatmaps -> {args.outdir}")

if __name__ == '__main__':
    main()
