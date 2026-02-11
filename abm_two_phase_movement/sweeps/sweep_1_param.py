#!/usr/bin/env python3
from __future__ import annotations

from multiprocessing import Pool
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from sweeps.sweep_config import load_sweep_config
from sweeps.sweep_utils import (
    merge_baseline_and_override,
    load_observed_with_scaler,
    distance_l2_scaled,
)

from abm.clusters_model import ClustersModel
from abcp.compute_summary import simulate_timeseries

ALL_STATS = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]

PARAM = "n_init"

# initial_scale = 1e-2
initial_scale = 50
VALUES_run = np.arange(0, 11)  # 0..16 inclusive
VALUES = 500 + initial_scale * VALUES_run

# print("Values are", VALUES.tolist())  # convert to list for pretty printing if you like


def _run_single(args):
    val, rep, cfg, obs_scaled_vec, scaler = args
    flat = {PARAM: val}
    nested = merge_baseline_and_override(cfg["params"], flat)

    sim = simulate_timeseries(
        lambda p: ClustersModel(params=p, seed=cfg["seed_base"] + rep),
        params=nested,
        total_steps=cfg["total_steps"],
        sample_steps=tuple(cfg["timesteps"]),
    )

    df_full = pd.DataFrame(sim, columns=ALL_STATS)
    return df_full[cfg["summary_stats"]]


def main():
    cfg = load_sweep_config("sweeps/sweep_defaults.yaml")

    obs_scaled_vec, valid_ts, scaler = load_observed_with_scaler(
        cfg["observed_csv"], cfg["summary_stats"], cfg["timesteps"]
    )
    cfg["timesteps"] = valid_ts

    summary_rows, dist_rows = [], []
    total_jobs = len(VALUES) * cfg["replicates"]

    with Progress(
        "[bold magenta]1D Sweep...",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:

        task = progress.add_task("Sweeping...", total=total_jobs)

        for val in VALUES:
            tasks = [(val, r, cfg, obs_scaled_vec, scaler)
                     for r in range(cfg["replicates"])]

            with Pool(cfg["workers"]) as pool:
                sims = []
                for df in pool.imap_unordered(_run_single, tasks):
                    sims.append(df)
                    progress.update(task, advance=1)

            dists = [distance_l2_scaled(df, scaler, obs_scaled_vec, cfg["summary_stats"])
                     for df in sims]

            S0 = [df["S0"].iloc[-1] for df in sims]
            NND = [df["NND_med"].iloc[-1] for df in sims]

            summary_rows.append({
                "param": val,
                "avg_distance": np.mean(dists),
                "best_distance": np.min(dists),
                "S0_mean": np.mean(S0),
                "S0_95_low": np.percentile(S0, 2.5),
                "S0_95_high": np.percentile(S0, 97.5),
                "NND_mean": np.mean(NND),
                "NND_95_low": np.percentile(NND, 2.5),
                "NND_95_high": np.percentile(NND, 97.5),
            })

            for d in dists:
                dist_rows.append({"param": val, "distance": d})

    pd.DataFrame(summary_rows).to_csv("results/sweep_1param_mp_summary.csv", index=False)
    pd.DataFrame(dist_rows).to_csv("results/sweep_1param_mp_distances.csv", index=False)

    # ---- CI Plot ----
    df = pd.DataFrame(summary_rows)
    plt.figure()
    plt.errorbar(
        df["param"], df["S0_mean"],
        yerr=[df["S0_mean"] - df["S0_95_low"], df["S0_95_high"] - df["S0_mean"]],
        fmt="o-", capsize=4,
    )
    try:
        obs = pd.read_csv(cfg["observed_csv"]).sort_values("timestep")
        plt.axhline(float(obs["S0"].iloc[-1]), color="red", linestyle="--")
    except Exception:
        pass
    plt.xlabel(PARAM)
    plt.ylabel("Final S0")
    plt.tight_layout()
    plt.savefig("results/ci_S0.png", dpi=180)
    plt.close()

    # ---- Boxplot ----
    dist_df = pd.DataFrame(dist_rows)
    plt.figure()
    sns.boxplot(data=dist_df, x="param", y="distance")
    plt.tight_layout()
    plt.savefig("results/boxplot_distance.png", dpi=180)
    plt.close()

    print("[sweeps] Saved 1-parameter CI sweep results.")


if __name__ == "__main__":
    main()