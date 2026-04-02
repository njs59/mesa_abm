#!/usr/bin/env python3
"""
Log‑scale timeseries plotting for 15-condition ABM experiment.

Reads:
    results/results_5_cell_types_3_conditions/summary_timeseries.csv

Produces:
    results/results_5_cell_types_3_conditions/plots_timeseries_logy/

Three plot sets:
  (A) ALL 15 conditions — 15 curves per plot (colour = cell type, linestyle = phase model)
  (B) BY CELL TYPE — 3 curves per plot (phase models)
  (C) BY PHASE MODEL — 5 curves per plot (cell types)
All y-axes are LOG SCALE.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------
# Settings
# -------------------------------------

CELL_TYPES = ["baseline", "decP1", "incP1", "decP2", "incP2"]
PHASE_MODELS = ["phase2_only", "two_phase_interact", "two_phase_no_interact"]

METRICS = [
    "n_clusters",
    "mean_size",
    "var_size",
    "median_nnd",
    "R",
    "z",
]

CELL_COLOURS = {
    "baseline": "C0",
    "decP1":    "C1",
    "incP1":    "C2",
    "decP2":    "C3",
    "incP2":    "C4",
}

PHASE_STYLES = {
    "phase2_only":          "-",
    "two_phase_interact":   "--",
    "two_phase_no_interact":"dotted",
}


# -------------------------------------
# Utilities
# -------------------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# -------------------------------------
# Plotting functions
# -------------------------------------

def plot_all_15(df, metric, outdir):
    fig, ax = plt.subplots(figsize=(12,6))

    for cond, g in df.groupby("condition"):
        cell = g["cell_type"].iloc[0]
        phase = g["phase_model"].iloc[0]
        ax.plot(
            g["time_min"], g[metric],
            color=CELL_COLOURS[cell],
            linestyle=PHASE_STYLES[phase],
            linewidth=1.8,
            label=cond
        )

    ax.set_yscale("log")
    ax.set_title(f"{metric} over time — ALL 15 conditions (log scale)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    fig.savefig(outdir / f"{metric}__ALL15_logy.png", dpi=300)
    plt.close(fig)


def plot_by_celltype(df, metric, outdir):
    for cell in CELL_TYPES:
        fig, ax = plt.subplots(figsize=(10,5))

        df_cell = df[df["cell_type"] == cell]

        for phase in PHASE_MODELS:
            d = df_cell[df_cell["phase_model"] == phase]
            if len(d) > 0:
                ax.plot(
                    d["time_min"], d[metric],
                    color=CELL_COLOURS[cell],
                    linestyle=PHASE_STYLES[phase],
                    linewidth=2,
                    label=phase
                )

        ax.set_yscale("log")
        ax.set_title(f"{metric} over time — cell type {cell} (log scale)")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend()

        fig.tight_layout()
        fig.savefig(outdir / f"{metric}__celltype_{cell}_logy.png", dpi=300)
        plt.close(fig)


def plot_by_phase(df, metric, outdir):
    for phase in PHASE_MODELS:
        fig, ax = plt.subplots(figsize=(10,5))

        df_phase = df[df["phase_model"] == phase]

        for cell in CELL_TYPES:
            d = df_phase[df_phase["cell_type"] == cell]
            if len(d) > 0:
                ax.plot(
                    d["time_min"], d[metric],
                    color=CELL_COLOURS[cell],
                    linestyle=PHASE_STYLES[phase],
                    linewidth=2,
                    label=cell
                )

        ax.set_yscale("log")
        ax.set_title(f"{metric} over time — phase behaviour {phase} (log scale)")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend()

        fig.tight_layout()
        fig.savefig(outdir / f"{metric}__phase_{phase}_logy.png", dpi=300)
        plt.close(fig)


# -------------------------------------
# MAIN
# -------------------------------------

def main():
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results" / "results_5_cell_types_3_conditions"

    # Input
    csv_path = results_dir / "summary_timeseries.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # Output
    plots_dir = ensure_dir(results_dir / "plots_timeseries_logy")
    out_all15 = ensure_dir(plots_dir / "ALL_15")
    out_ct    = ensure_dir(plots_dir / "BY_CELLTYPE")
    out_ph    = ensure_dir(plots_dir / "BY_PHASE")

    # Make all the plots
    for metric in METRICS:
        plot_all_15(df, metric, out_all15)
        plot_by_celltype(df, metric, out_ct)
        plot_by_phase(df, metric, out_ph)

    print("Finished writing LOG-SCALE plots to:", plots_dir)


if __name__ == "__main__":
    main()