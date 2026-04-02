#!/usr/bin/env python3
"""
Plot time-series for kappa turning conditions × 3 phase models.
Reads:  Multi_phase_analysis/results/kappa_conditions/summary_timeseries.csv
Writes: Multi_phase_analysis/results/kappa_conditions/plots_raw/* and plots_pct/*

Adds compact 2×2 “core metrics” figures:
  RAW  → plots_raw/CORE_four_metrics_raw.png
  %Δ   → plots_pct/CORE_four_metrics_pct.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS = ["n_clusters","mean_size","var_size","median_nnd","R","z"]
CORE = ["n_clusters","mean_size","var_size","median_nnd"]
MODEL_STYLES = {
    "phase2_only":"-", "two_phase_interact":"--", "two_phase_no_interact":":"
}

def kappa_colours(klist):
    cmap = plt.get_cmap('tab10')
    return {k:cmap(i % 10) for i,k in enumerate(klist)}

def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def plot_core_raw(agg: pd.DataFrame, kappas, KCOL, out_raw: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes = axes.ravel()
    for i, m in enumerate(CORE):
        ax = axes[i]
        for k in kappas:
            for model in MODEL_STYLES:
                sub = agg[(agg['kappa']==k) & (agg['model']==model)][['time_min', m]]
                if sub.empty:
                    continue
                ax.plot(sub['time_min'], sub[m], color=KCOL[k], linestyle=MODEL_STYLES[model],
                        linewidth=1.8, label=f"{k} — {model}")
        ax.set_title(m)
        ax.set_xlabel('Time (min)'); ax.set_ylabel(m); ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.suptitle("Core metrics over time — κ conditions × models", y=0.995)
    fig.savefig(out_raw / "CORE_four_metrics_raw.png", dpi=300)
    plt.close(fig)

def plot_core_pct(agg: pd.DataFrame, kappas, KCOL, out_pct: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes = axes.ravel()
    for i, m in enumerate(CORE):
        ax = axes[i]
        for k in kappas:
            base = agg[(agg['kappa']==k) & (agg['model']=='phase2_only')][['step','time_min',m]].set_index('step')
            if base.empty:
                continue
            for model in ['two_phase_interact','two_phase_no_interact']:
                comp = agg[(agg['kappa']==k) & (agg['model']==model)][['step',m]].set_index('step')
                joined = base.join(comp, how='inner', rsuffix='_comp')
                if joined.empty:
                    continue
                base_vals = joined[m].to_numpy(dtype=float)
                comp_vals = joined[f'{m}_comp'].to_numpy(dtype=float)
                delta = comp_vals - base_vals
                pct = np.full_like(delta, np.nan, dtype=float)
                np.divide(delta, base_vals, out=pct, where=(base_vals != 0))
                pct *= 100.0
                ax.plot(joined['time_min'], pct, color=KCOL[k], linestyle=MODEL_STYLES[model],
                        linewidth=1.8, label=f"{k} — {model}")
        ax.set_title(f"%Δ {m} vs phase2_only")
        ax.set_xlabel('Time (min)'); ax.set_ylabel('% difference'); ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.suptitle("Core metrics — % difference vs phase2_only (κ conditions)", y=0.995)
    fig.savefig(out_pct / "CORE_four_metrics_pct.png", dpi=300)
    plt.close(fig)

def main():
    res_dir = Path(__file__).resolve().parent / 'results' / 'kappa_conditions'
    csv_path = res_dir / 'summary_timeseries.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}. Run run_kappa_conditions.py first.")
    df = pd.read_csv(csv_path)

    # Aggregate to mean over repeats
    agg = df.groupby(['condition','step'], as_index=False)[METRICS + ['time_min']].mean()
    # Split condition into kappa label and model
    parts = agg['condition'].str.split('__', n=1, expand=True)
    agg['kappa'] = parts[0]
    agg['model'] = parts[1]

    kappas = sorted(agg['kappa'].unique(), key=lambda x: (x!='Default', x))
    KCOL = kappa_colours(kappas)

    out_raw = ensure(res_dir / 'plots_raw')
    out_pct = ensure(res_dir / 'plots_pct')

    # RAW (per metric)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(12,6))
        for k in kappas:
            for model in MODEL_STYLES:
                sub = agg[(agg['kappa']==k) & (agg['model']==model)][['time_min', metric]]
                if sub.empty:
                    continue
                ax.plot(sub['time_min'], sub[metric], color=KCOL[k], linestyle=MODEL_STYLES[model],
                        linewidth=1.8, label=f"{k} — {model}")
        ax.set_title(f"Raw time-series: {metric}")
        ax.set_xlabel('Time (min)')
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_raw / f"{metric}__raw.png", dpi=300)
        plt.close(fig)

    # NEW: 2×2 core metrics RAW
    plot_core_raw(agg, kappas, KCOL, out_raw)

    # % DIFF vs phase2_only (per metric; safe division)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(12,6))
        for k in kappas:
            base = agg[(agg['kappa']==k) & (agg['model']=='phase2_only')][['step','time_min',metric]].set_index('step')
            if base.empty:
                continue
            for model in ['two_phase_interact','two_phase_no_interact']:
                comp = agg[(agg['kappa']==k) & (agg['model']==model)][['step',metric]].set_index('step')
                joined = base.join(comp, how='inner', rsuffix='_comp')
                if joined.empty:
                    continue
                base_vals = joined[metric].to_numpy(dtype=float)
                comp_vals = joined[f'{metric}_comp'].to_numpy(dtype=float)
                delta = comp_vals - base_vals
                pct = np.full_like(delta, np.nan, dtype=float)
                np.divide(delta, base_vals, out=pct, where=(base_vals != 0))
                pct *= 100.0
                ax.plot(joined['time_min'], pct, color=KCOL[k], linestyle=MODEL_STYLES[model],
                        linewidth=1.8, label=f"{k} — {model}")
        ax.set_title(f"% difference vs phase2_only: {metric}")
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('% difference')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_pct / f"{metric}__pct.png", dpi=300)
        plt.close(fig)

    # NEW: 2×2 core metrics %Δ
    plot_core_pct(agg, kappas, KCOL, out_pct)

    print('Plots written to:')
    print(' -', out_raw)
    print(' -', out_pct)

if __name__ == '__main__':
    main()