#!/usr/bin/env python3
"""
Timeseries plotting for 15‑condition ABM experiment.

Produces:

(A) ALL 15 conditions (15 lines per plot)
(B) 5 plots: each cell‑type with 3 lines (phase behaviour)
(C) 3 plots: each phase behaviour with 5 lines (cell types)

Reads from:
    results/results_5_cell_types_3_conditions/trajectories/<condition>/repeat_###.csv
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ----------------------------------------------------
#  Condition structure
# ----------------------------------------------------

CELL_TYPES = ["baseline", "decP1", "incP1", "decP2", "incP2"]
PHASE_TYPES = ["phase2_only", "two_phase_interact", "two_phase_no_interact"]

METRICS = [
    "n_clusters",
    "mean_size",
    "var_size",
    "median_nnd",
    "R",
    "z",
]


# ----------------------------------------------------
# Utilities
# ----------------------------------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_spatial(df_t, W, H, torus=True):
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

    r_exp = 1.0 / (2.0 * math.sqrt(lam))
    se = 0.26136 / math.sqrt(n * lam)
    if se <= 0:
        return dict(median_nnd=np.nan, R=np.nan, z=np.nan)

    z = (r_obs - r_exp) / se
    R = r_obs / r_exp

    return dict(
        median_nnd=float(np.median(nn)),
        R=R,
        z=z
    )


def compute_timeseries(condition_name, files, W, H, torus=True):
    # find longest run
    max_step = 0
    for f in files:
        df = pd.read_csv(f, usecols=["step"])
        if df.size > 0:
            max_step = max(max_step, int(df["step"].max()))

    T = max_step + 1
    per_rep = {m: [] for m in METRICS}

    for f in files:
        df = pd.read_csv(f)
        arrays = {m: np.full(T, np.nan) for m in METRICS}

        for t in range(T):
            df_t = df[df["step"] == t]
            if df_t.empty:
                continue

            arrays["n_clusters"][t] = len(df_t)
            sizes = df_t["size"].to_numpy(float)
            arrays["mean_size"][t] = sizes.mean()
            arrays["var_size"][t] = sizes.var(ddof=1) if len(sizes) > 1 else 0.0

            spatial = compute_spatial(df_t, W, H, torus)
            arrays["median_nnd"][t] = spatial["median_nnd"]
            arrays["R"][t] = spatial["R"]
            arrays["z"][t] = spatial["z"]

        for m in METRICS:
            per_rep[m].append(arrays[m])

    out = {}
    for m in METRICS:
        M = np.vstack(per_rep[m])
        out[m] = np.nanmean(M, axis=0)

    return out, T


# ----------------------------------------------------
# Plot types
# ----------------------------------------------------

def plot_all_15(cond_ts, x_axis, metric, outdir):
    fig, ax = plt.subplots(figsize=(12, 6))

    for cond, data in cond_ts.items():
        ax.plot(x_axis, data[metric], lw=1.5, label=cond)

    ax.set_title(f"{metric} over time — ALL 15 conditions")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / f"{metric}__ALL15.png", dpi=300)
    plt.close(fig)


def plot_by_celltype(cond_ts, x_axis, metric, outdir):
    for ct in CELL_TYPES:
        fig, ax = plt.subplots(figsize=(10,5))

        for ph in PHASE_TYPES:
            cname = f"{ct}__{ph}"
            if cname in cond_ts:
                ax.plot(x_axis, cond_ts[cname][metric], lw=2,
                        label=ph)

        ax.set_title(f"{metric} over time — cell type {ct}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"{metric}__celltype_{ct}.png", dpi=300)
        plt.close(fig)


def plot_by_phase(cond_ts, x_axis, metric, outdir):
    for ph in PHASE_TYPES:
        fig, ax = plt.subplots(figsize=(10,5))

        for ct in CELL_TYPES:
            cname = f"{ct}__{ph}"
            if cname in cond_ts:
                ax.plot(x_axis, cond_ts[cname][metric], lw=2,
                        label=ct)

        ax.set_title(f"{metric} over time — phase behaviour {ph}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"{metric}__phase_{ph}.png", dpi=300)
        plt.close(fig)


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

def main():
    root = Path(__file__).resolve().parents[1]

    results_dir = root / "results" / "results_5_cell_types_3_conditions"
    traj_dir = results_dir / "trajectories"

    plots_dir = ensure_dir(results_dir / "plots_timeseries")

    # domain consistent with your ABM
    W = 1344.0
    H = 1025.0
    torus = True

    # load all condition directories
    cond_files = {}
    for cdir in sorted([p for p in traj_dir.iterdir() if p.is_dir()]):
        files = sorted(cdir.glob("repeat_*.csv"))
        if files:
            cond_files[cdir.name] = files

    # infer dt
    example_file = next(iter(next(iter(cond_files.values()))))
    df_example = pd.read_csv(example_file, usecols=["step","time_min"])
    dt = np.median(df_example.loc[df_example.step>0, "time_min"]
                   / df_example.loc[df_example.step>0, "step"])

    # compute timeseries
    cond_ts = {}
    Tlist = []

    for cond, files in cond_files.items():
        print("Processing", cond)
        ts, T = compute_timeseries(cond, files, W, H, torus)
        cond_ts[cond] = ts
        Tlist.append(T)

    Tmax = max(Tlist)
    x_axis = np.arange(Tmax) * dt

    # (A) ALL CONDITIONS
    out_all15 = ensure_dir(plots_dir / "ALL_15")
    for m in METRICS:
        plot_all_15(cond_ts, x_axis, m, out_all15)

    # (B) BY CELL TYPE
    out_ct = ensure_dir(plots_dir / "BY_CELLTYPE")
    for m in METRICS:
        plot_by_celltype(cond_ts, x_axis, m, out_ct)

    # (C) BY PHASE BEHAVIOUR
    out_ph = ensure_dir(plots_dir / "BY_PHASE")
    for m in METRICS:
        plot_by_phase(cond_ts, x_axis, m, out_ph)

    print("\nAll plots saved to:", plots_dir)


if __name__ == "__main__":
    main()