#!/usr/bin/env python3
from __future__ import annotations

from multiprocessing import Pool
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn

from sweeps.sweep_config import load_sweep_config
from sweeps.sweep_utils import (
    merge_baseline_and_override,
    load_observed_with_scaler,
    distance_l2_scaled,
)

from abm.clusters_model import ClustersModel
from abcp.compute_summary import simulate_timeseries


# Ordering of columns returned from simulate_timeseries
ALL_STATS = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]

# Parameters to sweep
PARAM_A = "softness"
PARAM_B = "fragment_minsep_factor"

values_A = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
values_B = [1.05, 1.2, 1.4, 1.6, 1.8, 2.0]


# # Define A-range using min, max, step
# A_min, A_max, A_step = 0.4, 0.7, 0.025
# values_A = list(np.arange(A_min, A_max + 1e-12, A_step))

# # Define B-range using min, max, step
# B_min, B_max, B_step = 0.0, 0.002, 0.0005
# values_B = list(np.arange(B_min, B_max + 1e-12, B_step))



def final_stats(df: pd.DataFrame):
    """Return final-time biological stats."""
    return {
        "final_S0": df["S0"].iloc[-1],
        "final_S1": df["S1"].iloc[-1],
        "final_S2": df["S2"].iloc[-1],
        "final_NND": df["NND_med"].iloc[-1],
    }


def _run_single(args):
    """Worker: run one replicate and return a DataFrame of the 4 stats."""
    a, b, rep, cfg, obs_scaled_vec, scaler = args

    # Build full nested params
    flat = {PARAM_A: a, PARAM_B: b}
    nested = merge_baseline_and_override(cfg["params"], flat)

    # Run simulation
    sim = simulate_timeseries(
        lambda p: ClustersModel(params=p, seed=cfg["seed_base"] + rep),
        params=nested,
        total_steps=cfg["total_steps"],
        sample_steps=tuple(cfg["timesteps"]),  # ensures correct timesteps
    )

    # Name columns → slice to the 4 stats we are fitting
    df_full = pd.DataFrame(sim, columns=ALL_STATS)
    return df_full[cfg["summary_stats"]]


def _heatmap(df, field, pa, pb):
    """Plot and save a heatmap for a given metric."""
    pv = df.pivot(index=pa, columns=pb, values=field)
    sns.heatmap(pv, cmap="viridis", annot=True, fmt=".3g")
    plt.title(field)
    plt.tight_layout()
    plt.savefig(f"results/{field}_heatmap.png", dpi=180)
    plt.close()


def main():
    cfg = load_sweep_config("sweeps/sweep_defaults.yaml")

    # Fit to exactly 4 stats (S0, S1, S2, NND_med)
    if len(cfg["summary_stats"]) != 4:
        raise ValueError("[sweeps] Expected exactly 4 summary_stats.")

    # Load observed data, get MaxAbsScaler + valid timesteps
    obs_scaled_vec, valid_ts, scaler = load_observed_with_scaler(
        cfg["observed_csv"], cfg["summary_stats"], cfg["timesteps"]
    )
    cfg["timesteps"] = valid_ts  # ensure simulation uses matched timesteps

    records = []
    total_jobs = len(values_A) * len(values_B) * cfg["replicates"]

    with Progress(
        "[bold blue]2-Parameter Sweep...",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:

        task = progress.add_task("Sweeping...", total=total_jobs)

        for a in values_A:
            for b in values_B:

                tasks = [(a, b, r, cfg, obs_scaled_vec, scaler)
                         for r in range(cfg["replicates"])]

                with Pool(cfg["workers"]) as pool:
                    sims = []
                    for df in pool.imap_unordered(_run_single, tasks):
                        sims.append(df)
                        progress.update(task, advance=1)

                # Compute distances (MaxAbs‑scaled)
                dists = [
                    distance_l2_scaled(df, scaler, obs_scaled_vec, cfg["summary_stats"])
                    for df in sims
                ]

                # Compute final biological stats
                fin = pd.DataFrame([final_stats(df) for df in sims]).mean()

                # Save aggregate result for this (a,b)
                records.append({
                    PARAM_A: a,
                    PARAM_B: b,
                    "avg_fit": float(np.mean(dists)),
                    "best_fit": float(np.min(dists)),
                    **fin.to_dict(),
                })

    out = pd.DataFrame(records)
    out.to_csv("results/sweep_2params_mp_results.csv", index=False)

    # Heatmaps for performance
    for fld in ["avg_fit", "best_fit"]:
        _heatmap(out, fld, PARAM_A, PARAM_B)

    # Heatmaps for biological final states
    for fld in ["final_S0", "final_S1", "final_S2", "final_NND"]:
        _heatmap(out, fld, PARAM_A, PARAM_B)

    print("[sweeps] Saved 2-parameter sweep + heatmaps.")


if __name__ == "__main__":
    main()