#!/usr/bin/env python3
"""
Timeseries summary + PRO vs INV + PERCENT DIFFERENCE plots.

Reads trajectories from:

    results/PRO_vs_INV_1_vs_2_stages/trajectories/invasive_phase2_only/
    results/PRO_vs_INV_1_vs_2_stages/trajectories/prolif_phase2_only/

Computes:
    n_clusters, mean_size, var_size, median_nnd, R, z

Outputs:
    summary_timeseries_PRO_vs_INV.csv
    plots_timeseries_PRO_vs_INV/<metric>__PRO_vs_INV.png
    plots_timeseries_PRO_vs_INV/<metric>__pctdiff.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# Colours (as specified)
PRO_COLOR = (186/256, 29/256, 186/256)   # purple-ish
INV_COLOR = (70/256, 158/256, 44/256)    # green-ish
PCT_COLOR = "C3"                         # red for percent diff

METRICS = [
    "n_clusters",
    "mean_size",
    "var_size",
    "median_nnd",
    "R",
    "z",
]


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_spatial(df_t, W, H, torus=True):
    """Compute median NND and Clark–Evans R, z for a timestep."""
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
    se = 0.26136 / np.sqrt(n*lam)
    if se <= 0:
        return dict(median_nnd=np.nan, R=np.nan, z=np.nan)

    z = (r_obs - r_exp) / se
    R = r_obs / r_exp

    return dict(
        median_nnd=float(np.median(nn)),
        R=R,
        z=z
    )


def compute_timeseries(files, W, H, torus=True):
    """Return dict(metric → array(T)), and T."""
    max_step = 0
    for f in files:
        df = pd.read_csv(f, usecols=["step"])
        if len(df):
            max_step = max(max_step, int(df["step"].max()))

    T = max_step + 1
    per_rep = {m: [] for m in METRICS}

    for f in files:
        df = pd.read_csv(f)
        arr = {m: np.full(T, np.nan) for m in METRICS}

        for t in range(T):
            df_t = df[df["step"] == t]
            if df_t.empty: continue

            arr["n_clusters"][t] = len(df_t)

            sizes = df_t["size"].to_numpy(float)
            arr["mean_size"][t] = sizes.mean()
            arr["var_size"][t]  = sizes.var(ddof=1) if len(sizes)>1 else 0.0

            spatial = compute_spatial(df_t, W, H, torus)
            arr["median_nnd"][t] = spatial["median_nnd"]
            arr["R"][t]          = spatial["R"]
            arr["z"][t]          = spatial["z"]

        for m in METRICS:
            per_rep[m].append(arr[m])

    out = {}
    for m in METRICS:
        M = np.vstack(per_rep[m])
        out[m] = np.nanmean(M, axis=0)

    return out, T


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "results" / "PRO_vs_INV_1_vs_2_stages"
    traj_dir = base / "trajectories"

    inv_dir = traj_dir / "invasive_phase2_only"
    pro_dir = traj_dir / "prolif_phase2_only"

    inv_files = sorted(inv_dir.glob("repeat_*.csv"))
    pro_files = sorted(pro_dir.glob("repeat_*.csv"))

    if not inv_files:
        raise RuntimeError(f"No INV files in {inv_dir}")
    if not pro_files:
        raise RuntimeError(f"No PRO files in {pro_dir}")

    # infer dt from INV
    df_dt = pd.read_csv(inv_files[0], usecols=["step","time_min"])
    df_dt = df_dt[df_dt.step > 0]
    dt = float(np.median(df_dt["time_min"] / df_dt["step"]))

    W, H = 1344.0, 1025.0
    torus = True

    print("Computing INV timeseries...")
    inv_ts, T_inv = compute_timeseries(inv_files, W, H, torus)

    print("Computing PRO timeseries...")
    pro_ts, T_pro = compute_timeseries(pro_files, W, H, torus)

    T = max(T_inv, T_pro)
    x_axis = np.arange(T) * dt

    # -----------------------------------------------------
    # Save combined summary CSV
    # -----------------------------------------------------
    rows = []

    for t in range(T):
        d_inv = dict(
            condition="INV",
            phenotype="INV",
            step=t, time_min=x_axis[t],
        )
        for m in METRICS:
            d_inv[m] = inv_ts[m][t]
        rows.append(d_inv)

        d_pro = dict(
            condition="PRO",
            phenotype="PRO",
            step=t, time_min=x_axis[t],
        )
        for m in METRICS:
            d_pro[m] = pro_ts[m][t]
        rows.append(d_pro)

    df_out = pd.DataFrame(rows)
    csv_path = base / "summary_timeseries_PRO_vs_INV.csv"
    df_out.to_csv(csv_path, index=False)
    print("Wrote:", csv_path)

    # -----------------------------------------------------
    # Plotting
    # -----------------------------------------------------
    plots_dir = ensure_dir(base / "plots_timeseries_PRO_vs_INV")

    for m in METRICS:
        fig, ax = plt.subplots(figsize=(10,6))

        ax.plot(x_axis, inv_ts[m], color=INV_COLOR, lw=2.2, label="INV")
        ax.plot(x_axis, pro_ts[m], color=PRO_COLOR, lw=2.2, label="PRO")

        ax.set_title(f"{m} over time — PRO vs INV")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(m)
        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(plots_dir / f"{m}__PRO_vs_INV.png", dpi=300)
        plt.close(fig)

    # -----------------------------------------------------
    # NEW — Percentage difference plots
    # -----------------------------------------------------
    pct_dir = ensure_dir(plots_dir / "pctdiff")

    for m in METRICS:
        inv_vals = inv_ts[m]
        pro_vals = pro_ts[m]

        # Percent difference: positive -> INV larger, negative -> PRO larger
        pct = 100 * (inv_vals - pro_vals) / pro_vals

        fig, ax = plt.subplots(figsize=(10,6))

        ax.plot(x_axis, pct, color=PCT_COLOR, lw=2.2)

        ax.axhline(0, color="black", lw=1, linestyle="--")

        ax.set_title(f"Percent difference in {m} (INV vs PRO)")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("% difference (INV > PRO positive)")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(pct_dir / f"{m}__pctdiff.png", dpi=300)
        plt.close(fig)

    print("Created percent‑difference plots at:", pct_dir)


if __name__ == "__main__":
    main()
