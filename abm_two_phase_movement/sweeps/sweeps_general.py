#!/usr/bin/env python3
from __future__ import annotations

import itertools
from multiprocessing import Pool
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn

from sweeps.sweep_config import load_sweep_config
from sweeps.sweep_utils import (
    merge_baseline_and_override,
    load_observed_with_scaler,
    distance_l2_scaled,
)

from abm.clusters_model import ClustersModel
from abcp.compute_summary import simulate_timeseries

# Full output ordering from simulate_timeseries
ALL_STATS = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]


def _run_single(args: Tuple[Dict[str, Any], int, Dict[str, Any], np.ndarray, Any]):
    flat, rep, cfg, obs_scaled_vec, scaler = args
    nested = merge_baseline_and_override(cfg["params"], flat)

    sim = simulate_timeseries(
        lambda p: ClustersModel(params=p, seed=cfg["seed_base"] + rep),
        params=nested,
        total_steps=cfg["total_steps"],
        sample_steps=tuple(cfg["timesteps"]),  # uses valid_ts
    )

    df_full = pd.DataFrame(sim, columns=ALL_STATS)
    df = df_full[cfg["summary_stats"]]

    return distance_l2_scaled(df, scaler, obs_scaled_vec, cfg["summary_stats"])


def main():
    cfg = load_sweep_config("sweeps/sweep_defaults.yaml")

    # Load observed and the MaxAbsScaler + valid timesteps
    obs_scaled_vec, valid_ts, scaler = load_observed_with_scaler(
        cfg["observed_csv"], cfg["summary_stats"], cfg["timesteps"]
    )
    cfg["timesteps"] = valid_ts

    # --- Define the parameter grid to sweep (override here as needed) ---
    # Provide flat names; any name omitted keeps its baseline value from YAML.
    # param_grid = {
    #     "fragment_rate": [0.0,0.001,0.002,0.003,0.004],
    #     "p_merge": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #     "n_init": [600,700,800,900,1000,1100]
    # }

    param_grid = {
        "fragment_rate": [0.0, 0.001, 0.002, 0.004],
        "p_merge": [0.4, 0.6, 0.8, 1.0],
        "n_init": [600, 800, 1000]
    }

    combos = list(itertools.product(*param_grid.values()))
    keys = list(param_grid.keys())

    rows = []

    with Progress(
        "[bold green]General Sweep...",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:

        task = progress.add_task("Sweeping...", total=len(combos) * cfg["replicates"])

        for combo in combos:
            flat = dict(zip(keys, combo))
            tasks = [(flat, r, cfg, obs_scaled_vec, scaler)
                     for r in range(cfg["replicates"])]

            with Pool(cfg["workers"]) as pool:
                for dist in pool.imap_unordered(_run_single, tasks):
                    rows.append({**flat, "fit": dist})
                    progress.update(task, advance=1)

    # Aggregate results
    df = pd.DataFrame(rows)
    summary = df.groupby(keys).agg(
        avg_fit=("fit", "mean"),
        best_fit=("fit", "min"),
    ).reset_index()

    # -----------------------------------------
    # NEW: Print best-performing parameter set
    # -----------------------------------------
    best_row = summary.loc[summary["best_fit"].idxmin()]

    print("\n" + "="*60)
    print("[ sweeps ] Best-performing parameter set:")
    for k in keys:
        print(f"  {k:20s} = {best_row[k]}")
    print(f"  {'best_fit':20s} = {best_row['best_fit']:.6g}")
    print("="*60 + "\n")

    # -----------------------------------------
    # NEW: Print ranked top 5 parameter sets
    # -----------------------------------------
    print("[ sweeps ] Top 5 parameter combinations by best_fit:")
    ranked = summary.sort_values("best_fit").head(5)
    for idx, row in ranked.iterrows():
        print("  -", {k: row[k] for k in keys}, f"best_fit = {row['best_fit']:.6g}")
    print()

    # Save CSV
    out_path = "results/sweep_general_mp_results.csv"
    summary.to_csv(out_path, index=False)
    print(f"[sweeps] Saved {out_path}")


if __name__ == "__main__":
    main()