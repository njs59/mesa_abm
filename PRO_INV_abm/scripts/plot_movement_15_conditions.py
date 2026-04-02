#!/usr/bin/env python3
"""
Timeseries plotting for 15-condition ABM experiment.

Adds:
1. CSV export of all summary stats over time for all 15 conditions.
2. Colour = cell type (5 colours), linestyle = movement model (3 styles)
   for the combined 15-line plots.

Conditions use:
    baseline, decP1, incP1, decP2, incP2   (cell types)
    phase2_only, two_phase_interact, two_phase_no_interact (movement models)

Reads from:
    results/results_5_cell_types_3_conditions/trajectories/<condition>/repeat_*.csv

Outputs to:
    results/results_5_cell_types_3_conditions/plots_timeseries/
    results/results_5_cell_types_3_conditions/summary_timeseries.csv
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# ==========================================================
# DEFINITIONS
# ==========================================================

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

# 5 cell-type colours
CELL_COLOURS = {
    "baseline": "C0",
    "decP1":    "C1",
    "incP1":    "C2",
    "decP2":    "C3",
    "incP2":    "C4",
}

# 3 movement model linestyles
PHASE_STYLES = {
    "phase2_only":          "-",
    "two_phase_interact":   "--",
    "two_phase_no_interact":"dotted",
}


# ==========================================================
# UTILITIES
# ==========================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_spatial(df_t, W, H, torus=True):
    """Compute median NND and Clark–Evans R, z for one timestep."""
    if df_t.empty or len(df_t) < 2:
        return dict(median_nnd=np.nan, R=np.nan, z=np.nan)

    x = df_t["x"].to_numpy(float)
    y = df_t["y"].to_numpy(float)
    n = len(x)
    area = W * H

    dx = x[:,None] - x[None,:]
    dy = y[:,None] - y[None,:]

    if torus:
        dx = (dx + W/2) % W - W/2
        dy = (dy + H/2) % H - H/2

    d2 = dx*dx + dy*dy
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))

    r_obs = float(nn.mean())
    lam = n / area
    if lam <= 0:
        return dict(median_nnd=np.nan, R=np.nan, z=np.nan)

    r_exp = 1/(2*np.sqrt(lam))
    se = 0.26136/np.sqrt(n*lam)
    if se <= 0:
        return dict(median_nnd=np.nan, R=np.nan, z=np.nan)

    z = (r_obs - r_exp)/se
    R = r_obs/r_exp

    return dict(
        median_nnd=float(np.median(nn)),
        R=R,
        z=z
    )


def compute_timeseries(condition, files, W, H, torus=True):
    """Return dict(metric→array(T)), and T."""
    # determine longest trajectory
    max_step = 0
    for f in files:
        df = pd.read_csv(f, usecols=["step"])
        if len(df):
            max_step = max(max_step, int(df["step"].max()))

    T = max_step + 1

    rep_arrays = {m: [] for m in METRICS}

    for f in files:
        df = pd.read_csv(f)
        arr = {m: np.full(T, np.nan) for m in METRICS}

        for t in range(T):
            df_t = df[df["step"] == t]
            if df_t.empty:
                continue

            arr["n_clusters"][t] = len(df_t)
            sizes = df_t["size"].to_numpy(float)
            arr["mean_size"][t] = sizes.mean()
            arr["var_size"][t] = sizes.var(ddof=1) if len(sizes)>1 else 0.0

            spatial = compute_spatial(df_t, W, H, torus)
            arr["median_nnd"][t] = spatial["median_nnd"]
            arr["R"][t]          = spatial["R"]
            arr["z"][t]          = spatial["z"]

        for m in METRICS:
            rep_arrays[m].append(arr[m])

    # average over repeats
    out = {}
    for m in METRICS:
        M = np.vstack(rep_arrays[m])
        out[m] = np.nanmean(M, axis=0)

    return out, T


# ==========================================================
# PLOTTING
# ==========================================================

def plot_all_15(cond_ts, x_axis, metric, outdir):
    fig, ax = plt.subplots(figsize=(12,6))
    for cond, stats in cond_ts.items():
        cell, phase = cond.split("__")
        ax.plot(
            x_axis,
            stats[metric],
            color=CELL_COLOURS[cell],
            linestyle=PHASE_STYLES[phase],
            linewidth=1.8,
            label=cond
        )
    ax.set_title(f"{metric} over time — ALL 15 conditions")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / f"{metric}__ALL15.png", dpi=300)
    plt.close(fig)


def plot_by_celltype(cond_ts, x_axis, metric, outdir):
    for cell in CELL_TYPES:
        fig, ax = plt.subplots(figsize=(10,5))
        for phase in PHASE_MODELS:
            cond = f"{cell}__{phase}"
            if cond in cond_ts:
                ax.plot(
                    x_axis,
                    cond_ts[cond][metric],
                    color=CELL_COLOURS[cell],
                    linestyle=PHASE_STYLES[phase],
                    linewidth=2,
                    label=phase
                )
        ax.set_title(f"{metric} over time — cell type {cell}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"{metric}__celltype_{cell}.png", dpi=300)
        plt.close(fig)


def plot_by_phase(cond_ts, x_axis, metric, outdir):
    for phase in PHASE_MODELS:
        fig, ax = plt.subplots(figsize=(10,5))
        for cell in CELL_TYPES:
            cond = f"{cell}__{phase}"
            if cond in cond_ts:
                ax.plot(
                    x_axis,
                    cond_ts[cond][metric],
                    color=CELL_COLOURS[cell],
                    linestyle=PHASE_STYLES[phase],
                    linewidth=2,
                    label=cell
                )
        ax.set_title(f"{metric} over time — phase behaviour {phase}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"{metric}__phase_{phase}.png", dpi=300)
        plt.close(fig)


# ==========================================================
# MAIN
# ==========================================================

def main():
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results" / "results_5_cell_types_3_conditions"
    traj_dir = results_dir / "trajectories"

    plots_dir = ensure_dir(results_dir / "plots_timeseries")

    # domain
    W, H = 1344.0, 1025.0
    torus = True

    # find condition folders
    cond_files = {}
    for d in sorted(p for p in traj_dir.iterdir() if p.is_dir()):
        files = sorted(d.glob("repeat_*.csv"))
        if files:
            cond_files[d.name] = files

    # infer dt
    first_file = next(iter(next(iter(cond_files.values()))))
    df_dt = pd.read_csv(first_file, usecols=["step","time_min"])
    dt = float(np.median(df_dt.loc[df_dt.step>0, "time_min"] /
                         df_dt.loc[df_dt.step>0, "step"]))

    # compute full timeseries
    cond_ts = {}
    Tlist = []
    for cond, files in cond_files.items():
        print("Processing", cond)
        ts, T = compute_timeseries(cond, files, W, H, torus)
        cond_ts[cond] = ts
        Tlist.append(T)

    Tmax = max(Tlist)
    x_axis = np.arange(Tmax) * dt

    # ======================================================
    # WRITE MASTER CSV FOR ALL SUMMARY STATS
    # ======================================================
    rows = []
    for cond, ts in cond_ts.items():
        cell, phase = cond.split("__")
        for t in range(Tmax):
            d = dict(
                condition=cond,
                cell_type=cell,
                phase_model=phase,
                step=t,
                time_min=x_axis[t],
            )
            for m in METRICS:
                d[m] = ts[m][t]
            rows.append(d)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(results_dir / "summary_timeseries.csv", index=False)
    print("Wrote summary_timeseries.csv")

    # ======================================================
    # MAKE PLOTS
    # ======================================================

    # A) All 15
    p_all = ensure_dir(plots_dir / "ALL_15")
    for m in METRICS:
        plot_all_15(cond_ts, x_axis, m, p_all)

    # B) cell types
    p_ct = ensure_dir(plots_dir / "BY_CELLTYPE")
    for m in METRICS:
        plot_by_celltype(cond_ts, x_axis, m, p_ct)

    # C) phases
    p_ph = ensure_dir(plots_dir / "BY_PHASE")
    for m in METRICS:
        plot_by_phase(cond_ts, x_axis, m, p_ph)

    print("\nAll plots written to:", plots_dir)


if __name__ == "__main__":
    main()