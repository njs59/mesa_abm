#!/usr/bin/env python3
"""
Plot time-series for the 15-condition experiment (5 speed variants × 3 phase models).
Reads:  Multi_phase_analysis/results/fifteen_conditions/summary_timeseries.csv
Writes: Multi_phase_analysis/results/fifteen_conditions/plots/*

We plot the following (each metric separately):
  A) ALL 15 lines on a single axes: colour = cell/speed variant, linestyle = phase model
  B) BY_CELLTYPE: three phase models for each cell/speed variant
  C) BY_PHASE: five cell/speed variants for each phase model

NEW: a compact 2×2 “core metrics” figure over ALL 15:
  [n_clusters, mean_size, var_size, median_nnd]
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

METRICS = ["n_clusters","mean_size","var_size","median_nnd","R","z"]
CORE = ["n_clusters","mean_size","var_size","median_nnd"]
CELL_COLOURS = {
    "baseline":"C0","decP1":"C1","incP1":"C2","decP2":"C3","incP2":"C4"
}
PHASE_STYLES = {
    "phase2_only":"-", "two_phase_interact":"--", "two_phase_no_interact":":"
}

def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def plot_all_15(cond_ts, x_axis, metric, outdir):
    fig, ax = plt.subplots(figsize=(12,6))
    for cond, stats in cond_ts.items():
        cell, phase = cond.split("__")
        ax.plot(x_axis, stats[metric],
                color=CELL_COLOURS.get(cell,'k'),
                linestyle=PHASE_STYLES.get(phase,'-'),
                linewidth=1.8, label=cond)
    ax.set_title(f"{metric} over time — ALL 15 conditions")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / f"{metric}__ALL15.png", dpi=300)
    plt.close(fig)

def plot_by_celltype(cond_ts, x_axis, metric, outdir):
    for cell in CELL_COLOURS:
        fig, ax = plt.subplots(figsize=(10,5))
        for phase in PHASE_STYLES:
            cond = f"{cell}__{phase}"
            if cond in cond_ts:
                ax.plot(x_axis, cond_ts[cond][metric],
                        color=CELL_COLOURS[cell],
                        linestyle=PHASE_STYLES[phase],
                        linewidth=2, label=phase)
        ax.set_title(f"{metric} over time — cell type {cell}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"{metric}__celltype_{cell}.png", dpi=300)
        plt.close(fig)

def plot_by_phase(cond_ts, x_axis, metric, outdir):
    for phase in PHASE_STYLES:
        fig, ax = plt.subplots(figsize=(10,5))
        for cell in CELL_COLOURS:
            cond = f"{cell}__{phase}"
            if cond in cond_ts:
                ax.plot(x_axis, cond_ts[cond][metric],
                        color=CELL_COLOURS[cell],
                        linestyle=PHASE_STYLES[phase],
                        linewidth=2, label=cell)
        ax.set_title(f"{metric} over time — phase behaviour {phase}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"{metric}__phase_{phase}.png", dpi=300)
        plt.close(fig)

def plot_core_all15(agg: pd.DataFrame, out_root: Path) -> None:
    """NEW: compact 2×2 core metrics over all 15 conditions."""
    p_dir = ensure(out_root / 'ALL_15')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes = axes.ravel()
    # groupby to ensure we have the mean over repeats at each (condition, step)
    for i, m in enumerate(CORE):
        ax = axes[i]
        for (cond, sub) in agg.groupby('condition'):
            cell = sub['cell_type'].iloc[0]
            phase = sub['phase_model'].iloc[0]
            ax.plot(sub['time_min'], sub[m],
                    color=CELL_COLOURS.get(cell,'k'),
                    linestyle=PHASE_STYLES.get(phase,'-'),
                    linewidth=1.8, label=cond)
        ax.set_title(m)
        ax.set_xlabel('Time (min)'); ax.set_ylabel(m); ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.suptitle("Core metrics over time — ALL 15 conditions", y=0.995)
    fig.savefig(p_dir / "CORE_four_metrics_ALL15.png", dpi=300)
    plt.close(fig)

def main():
    res_dir = Path(__file__).resolve().parent / "results" / "fifteen_conditions"
    csv_path = res_dir / "summary_timeseries.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}. Run run_fifteen_conditions.py first.")
    df = pd.read_csv(csv_path)

    # Aggregate to mean over repeats at each (condition, step)
    agg = df.groupby(['condition','step'], as_index=False)[METRICS + ['time_min']].mean()
    # Extract cell_type and phase_model from condition: '<variant>__<phase>'
    parts = agg['condition'].str.split('__', n=1, expand=True)
    agg['cell_type'] = parts[0]
    agg['phase_model'] = parts[1]

    out_root = ensure(res_dir / "plots")
    p_all = ensure(out_root / "ALL_15")
    p_ct  = ensure(out_root / "BY_CELLTYPE")
    p_ph  = ensure(out_root / "BY_PHASE")

    # Build a dict for the old plotting helpers (keep consistency)
    # But we already have agg; just compute dict per condition if needed.
    # For existing helpers above, we’ll skip and just do the “all” style here with agg.

    # Per-metric ALL 15 (kept from earlier version)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(12,6))
        for (cond, sub) in agg.groupby('condition'):
            cell = sub['cell_type'].iloc[0]
            phase = sub['phase_model'].iloc[0]
            ax.plot(sub['time_min'], sub[metric],
                    color=CELL_COLOURS.get(cell,'k'),
                    linestyle=PHASE_STYLES.get(phase,'-'),
                    linewidth=1.8, label=cond)
        ax.set_title(f"{metric} over time — ALL 15 conditions")
        ax.set_xlabel('Time (min)')
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(p_all / f"{metric}__ALL15.png", dpi=300)
        plt.close(fig)

    # BY_CELLTYPE
    for metric in METRICS:
        for cell in CELL_COLOURS:
            fig, ax = plt.subplots(figsize=(10,5))
            for phase in PHASE_STYLES:
                sub = agg[(agg['cell_type']==cell) & (agg['phase_model']==phase)]
                if sub.empty: continue
                ax.plot(sub['time_min'], sub[metric],
                        color=CELL_COLOURS[cell],
                        linestyle=PHASE_STYLES[phase],
                        linewidth=2, label=phase)
            ax.set_title(f"{metric} over time — cell type {cell}")
            ax.set_xlabel('Time (min)'); ax.set_ylabel(metric); ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(ensure(out_root / 'BY_CELLTYPE') / f"{metric}__celltype_{cell}.png", dpi=300)
            plt.close(fig)

    # BY_PHASE
    for metric in METRICS:
        for phase in PHASE_STYLES:
            fig, ax = plt.subplots(figsize=(10,5))
            for cell in CELL_COLOURS:
                sub = agg[(agg['cell_type']==cell) & (agg['phase_model']==phase)]
                if sub.empty: continue
                ax.plot(sub['time_min'], sub[metric],
                        color=CELL_COLOURS[cell],
                        linestyle=PHASE_STYLES[phase],
                        linewidth=2, label=cell)
            ax.set_title(f"{metric} over time — phase behaviour {phase}")
            ax.set_xlabel('Time (min)'); ax.set_ylabel(metric); ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(ensure(out_root / 'BY_PHASE') / f"{metric}__phase_{phase}.png", dpi=300)
            plt.close(fig)

    # NEW: 2×2 core metrics across ALL 15
    plot_core_all15(agg, out_root)

    print("Plots written to:", out_root)

if __name__ == '__main__':
    main()