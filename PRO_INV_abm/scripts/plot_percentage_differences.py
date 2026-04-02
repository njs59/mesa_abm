#!/usr/bin/env python3

"""
Plots percentage differences between each two‑phase model and the
phase2_only model, for each phenotype and each summary metric.

% diff = (two_phase - phase2_only) / phase2_only

Output directory:
 results/results_5_cell_types_3_conditions/plots_percentage_diff/
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === constants matching the existing script ===
CELL_TYPES = ["baseline", "decP1", "incP1", "decP2", "incP2"]
PHASE_MODELS = ["phase2_only", "two_phase_interact", "two_phase_no_interact"]
METRICS = ["n_clusters", "mean_size", "var_size", "median_nnd", "R", "z"]

CELL_COLOURS = {
    "baseline": "C0",
    "decP1": "C1",
    "incP1": "C2",
    "decP2": "C3",
    "incP2": "C4",
}

PHASE_STYLES = {
    "two_phase_interact": "-",
    "two_phase_no_interact": "--",
}

# ==========================================================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# ==========================================================
def main():

    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results" / "results_5_cell_types_3_conditions"

    # Load the summary file produced by the main script
    summary_file = results_dir / "summary_timeseries.csv"
    if not summary_file.exists():
        raise FileNotFoundError(
            "summary_timeseries.csv not found. Run plot_movement_15_conditions.py first."
        )

    df = pd.read_csv(summary_file)

    outdir = ensure_dir(results_dir / "plots_percentage_diff")

    # ================================================
    # For each metric, produce one plot
    # ================================================
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(12, 6))

        for cell in CELL_TYPES:
            # subset baseline models
            df_cell = df[df["cell_type"] == cell]

            # get phase2_only time series
            base = df_cell[df_cell["phase_model"] == "phase2_only"]

            if base.empty:
                continue

            t = base["time_min"].to_numpy()
            base_vals = base[metric].to_numpy()

            # compare each two-phase model
            for phase in ["two_phase_interact", "two_phase_no_interact"]:
                tp = df_cell[df_cell["phase_model"] == phase]

                if tp.empty:
                    continue

                tp_vals = tp[metric].to_numpy()

                # compute % diff: (two_phase - base) / base
                pct = (tp_vals - base_vals) / base_vals

                ax.plot(
                    t,
                    pct * 100,  # express as percent
                    color=CELL_COLOURS[cell],
                    linestyle=PHASE_STYLES[phase],
                    linewidth=1.8,
                    label=f"{cell} — {phase}",
                )

        ax.set_title(f"Percentage difference relative to phase2_only — {metric}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("% difference")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()

        fig.savefig(outdir / f"pctdiff__{metric}.png", dpi=300)
        plt.close(fig)

    print(f"Percentage-difference plots written to: {outdir}")


if __name__ == "__main__":
    main()