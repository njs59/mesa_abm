#!/usr/bin/env python3
"""
Plot time-series for the 3 movement models (invasive only).
Reads:  Multi_phase_analysis/results/three_conditions/summary_timeseries.csv
Writes: Multi_phase_analysis/results/three_conditions/plots/*

Adds a compact 2×2 “core metrics” figure:
  [n_clusters, mean_size, var_size, median_nnd]
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

METRICS = ["n_clusters","mean_size","var_size","median_nnd","R","z"]
CORE = ["n_clusters","mean_size","var_size","median_nnd"]
MODEL_STYLES = {"phase2_only":"-","two_phase_interact":"--","two_phase_no_interact":":"}
MODEL_COLOURS = {"phase2_only":"C0","two_phase_interact":"C1","two_phase_no_interact":"C2"}

def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def plot_core_grid(agg: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes = axes.ravel()
    for i, m in enumerate(CORE):
        ax = axes[i]
        for cond, sub in agg.groupby('condition'):
            ax.plot(sub['time_min'], sub[m],
                    color=MODEL_COLOURS.get(cond,'k'),
                    linestyle=MODEL_STYLES.get(cond,'-'),
                    linewidth=2, label=cond)
        ax.set_title(m)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel(m)
        ax.grid(alpha=0.25)
    # single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.suptitle("Core metrics over time — 3 movement models", y=0.995)
    fig.savefig(out_dir / "CORE_four_metrics.png", dpi=300)
    plt.close(fig)

def main():
    res_dir = Path(__file__).resolve().parent / 'results' / 'three_conditions'
    csv_path = res_dir / 'summary_timeseries.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}. Run run_three_conditions.py first.")
    df = pd.read_csv(csv_path)
    agg = df.groupby(['condition','step'], as_index=False)[METRICS + ['time_min']].mean()

    out_dir = ensure(res_dir / 'plots')

    # Per-metric panels (unchanged)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10,5))
        for cond, sub in agg.groupby('condition'):
            ax.plot(sub['time_min'], sub[metric],
                    color=MODEL_COLOURS.get(cond,'k'), linestyle=MODEL_STYLES.get(cond,'-'),
                    linewidth=2, label=cond)
        ax.set_title(f"{metric} over time — 3 movement models")
        ax.set_xlabel('Time (min)')
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.png", dpi=300)
        plt.close(fig)

    # NEW: 2×2 core metrics figure
    plot_core_grid(agg, out_dir)

    print('Plots written to:', out_dir)

if __name__ == '__main__':
    main()