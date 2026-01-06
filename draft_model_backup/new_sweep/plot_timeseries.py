
# new_sweep/plot_timeseries.py
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from new_sweep_old.plot_thesis_style import use_thesis_style

def main():
    parser = argparse.ArgumentParser(description='Plot time-series summary statistics for ABM runs')
    parser.add_argument('--in_csv', type=str, default='new_results/timeseries/timeseries_demo.csv')
    parser.add_argument('--outdir', type=str, default='new_results/figs')
    parser.add_argument('--title', type=str, default='Clustering dynamics — time-series summaries')
    args = parser.parse_args()

    use_thesis_style()
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.in_csv)

    fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    ax[0].plot(df['time_min'], df['S0'], color='#1f77b4')
    ax[0].set_ylabel('S₀(t): number of clusters')
    ax[1].plot(df['time_min'], df['S1'], color='#d62728')
    ax[1].set_ylabel('S₁(t): mean cluster size')
    ax[2].plot(df['time_min'], df['S2'], color='#2ca02c')
    ax[2].set_ylabel('S₂(t): mean squared size')
    ax[2].set_xlabel('Time (minutes)')
    fig.suptitle(args.title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(args.outdir, 'timeseries_S012.png'))

    fig2, ax2 = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax2[0].plot(df['time_min'], df['total_cells'], color='#9467bd')
    ax2[0].set_ylabel('Total cells in system')
    ax2[1].plot(df['time_min'], df['mean_speed'], color='#8c564b')
    ax2[1].set_ylabel('Mean speed of clusters')
    ax2[1].set_xlabel('Time (minutes)')
    fig2.suptitle('Global quantities over time')
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(os.path.join(args.outdir, 'timeseries_global.png'))

    print(f"Wrote time-series plots -> {args.outdir}")

if __name__ == '__main__':
    main()
