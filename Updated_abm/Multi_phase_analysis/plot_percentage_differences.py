#!/usr/bin/env python3
"""
% difference vs phase2_only baseline for the 15-condition experiment.
For each cell_type and each metric:
  %diff = 100 * (two_phase - phase2_only)/phase2_only
Reads:  Multi_phase_analysis/results/fifteen_conditions/summary_timeseries.csv
Writes: Multi_phase_analysis/results/fifteen_conditions/plots_pct/*

Adds a compact 2×2 “core metrics” %Δ figure:
  [n_clusters, mean_size, var_size, median_nnd]
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS = ["n_clusters","mean_size","var_size","median_nnd","R","z"]
CORE = ["n_clusters","mean_size","var_size","median_nnd"]
CELL_COLOURS = {
    "baseline":"C0","decP1":"C1","incP1":"C2","decP2":"C3","incP2":"C4"
}
PHASES_CMP = ["two_phase_interact","two_phase_no_interact"]

def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def plot_core_pct(agg: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes = axes.ravel()
    for i, m in enumerate(CORE):
        ax = axes[i]
        for cell, subc in agg.groupby('cell_type'):
            base = subc[subc['phase_model']=='phase2_only'][['step','time_min',m]].set_index('step')
            if base.empty:
                continue
            for phase in PHASES_CMP:
                comp = subc[subc['phase_model']==phase][['step',m]].set_index('step')
                joined = base.join(comp, how='inner', rsuffix='_comp')
                if joined.empty:
                    continue
                base_vals = joined[m].to_numpy(dtype=float)
                comp_vals = joined[f'{m}_comp'].to_numpy(dtype=float)
                delta = comp_vals - base_vals
                pct = np.full_like(delta, np.nan, dtype=float)
                np.divide(delta, base_vals, out=pct, where=(base_vals != 0))
                pct *= 100.0
                ax.plot(joined['time_min'], pct,
                        color=CELL_COLOURS.get(cell,'k'),
                        linestyle='--' if phase.endswith('no_interact') else '-',
                        linewidth=1.5, label=f"{cell} — {phase}")
        ax.set_title(f"%Δ {m} vs phase2_only")
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('% difference')
        ax.grid(alpha=0.25)
    # global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.suptitle("Core metrics — % difference vs phase2_only (15 conditions)", y=0.995)
    fig.savefig(outdir / "CORE_four_metrics_pct.png", dpi=300)
    plt.close(fig)

def main():
    res_dir = Path(__file__).resolve().parent / 'results' / 'fifteen_conditions'
    csv_path = res_dir / 'summary_timeseries.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}. Run run_fifteen_conditions.py first.")
    df = pd.read_csv(csv_path)
    # Aggregate over repeats
    agg = df.groupby(['condition','step'], as_index=False)[METRICS + ['time_min']].mean()
    # Split condition
    parts = agg['condition'].str.split('__', n=1, expand=True)
    agg['cell_type']   = parts[0]
    agg['phase_model'] = parts[1]

    outdir = ensure(res_dir / 'plots_pct')

    # Per-metric % plots (safe division)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(12,6))
        for cell, subc in agg.groupby('cell_type'):
            base = subc[subc['phase_model']=='phase2_only'].set_index('step')
            if base.empty:
                continue
            for phase in PHASES_CMP:
                comp = subc[subc['phase_model']==phase].set_index('step')
                joined = base[['time_min',metric]].join(comp[[metric]], how='inner', rsuffix='_comp')
                if joined.empty:
                    continue
                base_vals = joined[metric].to_numpy(dtype=float)
                comp_vals = joined[f'{metric}_comp'].to_numpy(dtype=float)
                delta = comp_vals - base_vals
                pct = np.full_like(delta, np.nan, dtype=float)
                np.divide(delta, base_vals, out=pct, where=(base_vals != 0))
                pct *= 100.0
                ax.plot(joined['time_min'], pct,
                        color=CELL_COLOURS.get(cell,'k'),
                        linestyle='--' if phase.endswith('no_interact') else '-',
                        linewidth=1.8, label=f"{cell} — {phase}")
        ax.set_title(f"% difference vs phase2_only — {metric}")
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('% difference')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(outdir / f"pctdiff__{metric}.png", dpi=300)
        plt.close(fig)

    # NEW: compact 2×2 core metrics %Δ figure
    plot_core_pct(agg, outdir)

    print("Percentage-difference plots written to:", outdir)

if __name__ == '__main__':
    main()