#!/usr/bin/env python3
"""
% difference vs phase2_only for the 3 movement models (invasive only).
Reads:  Multi_phase_analysis/results/three_conditions/summary_timeseries.csv
Writes: Multi_phase_analysis/results/three_conditions/plots_pct/*

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
MODEL_STYLES = {"two_phase_interact":"-","two_phase_no_interact":"--"}
MODEL_COLOURS = {"two_phase_interact":"C1","two_phase_no_interact":"C2"}

def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def safe_pct(base_vals: np.ndarray, comp_vals: np.ndarray) -> np.ndarray:
    delta = comp_vals - base_vals
    pct = np.full_like(delta, np.nan, dtype=float)
    np.divide(delta, base_vals, out=pct, where=(base_vals != 0))
    return pct * 100.0

def plot_core_grid_pct(agg: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes = axes.ravel()

    base = agg[agg['condition']=='phase2_only'][['step','time_min'] + CORE].set_index('step')
    if base.empty:
        return

    for i, m in enumerate(CORE):
        ax = axes[i]
        for model in ['two_phase_interact','two_phase_no_interact']:
            comp = agg[agg['condition']==model][['step', m]].set_index('step')
            joined = base.join(comp, how='inner', rsuffix='_comp')
            if joined.empty:
                continue
            base_vals = joined[m].to_numpy(dtype=float)
            comp_vals = joined[f'{m}_comp'].to_numpy(dtype=float)
            pct = safe_pct(base_vals, comp_vals)
            ax.plot(joined['time_min'], pct,
                    color=MODEL_COLOURS.get(model,'k'), linestyle=MODEL_STYLES.get(model,'-'),
                    linewidth=2, label=model)
        ax.set_title(f"%Δ {m} vs phase2_only")
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('% difference')
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.suptitle("Core metrics — % difference vs phase2_only (3 models)", y=0.995)
    fig.savefig(out_dir / "CORE_four_metrics_pct.png", dpi=300)
    plt.close(fig)

def main():
    res_dir = Path(__file__).resolve().parent / 'results' / 'three_conditions'
    csv_path = res_dir / 'summary_timeseries.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}. Run run_three_conditions.py first.")
    df = pd.read_csv(csv_path)
    agg = df.groupby(['condition','step'], as_index=False)[METRICS + ['time_min']].mean()

    out_dir = ensure(res_dir / 'plots_pct')

    # Per-metric % plots (safe division)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10,5))
        base = agg[agg['condition']=='phase2_only'][['step','time_min',metric]].set_index('step')
        if base.empty:
            plt.close(fig); continue
        for model in ['two_phase_interact','two_phase_no_interact']:
            comp = agg[agg['condition']==model][['step',metric]].set_index('step')
            joined = base.join(comp, how='inner', rsuffix='_comp')
            if joined.empty:
                continue
            base_vals = joined[metric].to_numpy(dtype=float)
            comp_vals = joined[f'{metric}_comp'].to_numpy(dtype=float)
            delta = comp_vals - base_vals
            pct = np.full_like(delta, np.nan, dtype=float)
            np.divide(delta, base_vals, out=pct, where=(base_vals != 0))
            pct *= 100.0
            ax.plot(joined['time_min'], pct,
                    color=MODEL_COLOURS.get(model,'k'), linestyle=MODEL_STYLES.get(model,'-'),
                    linewidth=2, label=model)
        ax.set_title(f"% difference vs phase2_only — {metric}")
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('% difference')
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"pctdiff__{metric}.png", dpi=300)
        plt.close(fig)

    # NEW: compact 2×2 core metrics %Δ figure
    plot_core_grid_pct(agg, out_dir)

    print('Three-conditions %diff plots written to:', out_dir)

if __name__ == '__main__':
    main()