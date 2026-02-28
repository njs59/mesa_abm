#!/usr/bin/env python3
"""
Generate figures for Thesis Section 4.1 (ABM Behaviour Exhibits)

Reads defaults from scripts/scripts_defaults.yaml, runs ABM, and produces:
 - trajectories of cluster statistics
 - phase-switch time distribution (finite times only)
 - dispersal patterns under perturbed motility

Outputs -> results/abm_behaviour_exhibits/
"""

import os
import argparse
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# --- Import ABM modules ---
from abm.clusters_model import ClustersModel
from abm.utils import export_timeseries_state, DEFAULTS as ABM_DEFAULTS

sns.set(style="whitegrid")

# ============================================================
# I/O paths
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULTS_YAML = os.path.join(SCRIPT_DIR, "scripts_defaults.yaml")
OUTDIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results", "abm_behaviour_exhibits")
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# Config helpers
# ============================================================

def load_yaml_defaults(path=DEFAULTS_YAML):
    """
    Load scripts_defaults.yaml and return as dict.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg  # contains your values for steps, prolif_rate, p_merge, etc.


def merge_into_abm_params(yaml_cfg):
    """
    Create a full ABM params dict by starting from ABM_DEFAULTS and
    overlaying values found in scripts_defaults.yaml where applicable.
    """
    params = copy.deepcopy(ABM_DEFAULTS)

    # time/steps if present
    if "time" in yaml_cfg and "steps" in yaml_cfg["time"]:
        params["time"]["steps"] = float(yaml_cfg["time"]["steps"])
    elif "total_steps" in yaml_cfg:
        params["time"]["steps"] = int(yaml_cfg["total_steps"])

    if "time" in yaml_cfg and "dt" in yaml_cfg["time"]:
        params["time"]["dt"] = float(yaml_cfg["time"]["dt"])

    # physics block overrides
    if "physics" in yaml_cfg:
        params["physics"].update(yaml_cfg["physics"])

    # merge probability
    if "merge" in yaml_cfg:
        params["merge"].update(yaml_cfg["merge"])
    elif "params" in yaml_cfg and "p_merge" in yaml_cfg["params"]:
        params["merge"]["p_merge"] = float(yaml_cfg["params"]["p_merge"])

    # phenotypes.proliferative rates from your params block
    if "params" in yaml_cfg:
        p = yaml_cfg["params"]
        if "prolif_rate" in p:
            params["phenotypes"]["proliferative"]["prolif_rate"] = float(p["prolif_rate"])
        if "fragment_rate" in p:
            params["phenotypes"]["proliferative"]["fragment_rate"] = float(p["fragment_rate"])
        if "softness" in p:
            params["physics"]["softness"] = float(p["softness"])
        if "fragment_minsep_factor" in p:
            params["physics"]["fragment_minsep_factor"] = float(p["fragment_minsep_factor"])

    # initial conditions
    if "init" in yaml_cfg:
        params["init"].update(yaml_cfg["init"])
    else:
        # Some files expose n_init as params.n_init
        if "params" in yaml_cfg and "n_init" in yaml_cfg["params"]:
            params["init"]["n_clusters"] = int(yaml_cfg["params"]["n_init"])

    # movement_v2 and interactions are already structurally correct in ABM_DEFAULTS,
    # but scripts_defaults.yaml also includes a complete movement_v2 block; overlay if present
    if "movement_v2" in yaml_cfg:
        # deep-merge each sub-block if provided
        for k, v in yaml_cfg["movement_v2"].items():
            if isinstance(v, dict) and k in params["movement_v2"]:
                params["movement_v2"][k].update(v)
            else:
                params["movement_v2"][k] = v

    if "interactions" in yaml_cfg:
        params["interactions"].update(yaml_cfg["interactions"])

    # space block (width/height/torus)
    if "space" in yaml_cfg:
        params["space"].update(yaml_cfg["space"])

    return params


# ============================================================
# ABM runners and summarisation
# ============================================================

def run_abm(params=None, steps=None, seed=1):
    model = ClustersModel(params=params, seed=seed)
    # If steps explicitly provided, override the params["time"]["steps"] count
    total_steps = int(steps if steps is not None else params.get("time", {}).get("steps", 300))
    for _ in range(total_steps):
        model.step()
    return model


def summarise_abm(model):
    df = export_timeseries_state(
        model,
        out_csv=os.path.join(OUTDIR, "timeseries_state.csv")
    )
    summary = (
        df.groupby("step")
          .agg(
              num_clusters=("agent_id", "nunique"),
              mean_cluster_size=("size", "mean"),
              mean_squared_cluster_size=("size", lambda x: np.mean(x**2)),
          )
          .reset_index()
    )
    summary.to_csv(os.path.join(OUTDIR, "summary_statistics.csv"), index=False)
    return summary


# ============================================================
# Plotting
# ============================================================

def plot_cluster_trajectories(summary_df):
    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax[0].plot(summary_df["step"], summary_df["num_clusters"], lw=2)
    ax[0].set_ylabel("Number of clusters")

    ax[1].plot(summary_df["step"], summary_df["mean_cluster_size"], lw=2, color="green")
    ax[1].set_ylabel("Mean cluster size")

    ax[2].plot(
        summary_df["step"],
        summary_df["mean_squared_cluster_size"],
        lw=2,
        color="orange"
    )
    ax[2].set_ylabel("Mean squared cluster size")
    ax[2].set_xlabel("Simulation step")

    fig.suptitle("ABM Trajectories of Cluster Statistics (Baseline)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "trajectories.png"), dpi=150)
    plt.close()


def plot_phase_switch_distribution(model):
    # Extract raw switch times (some will be inf for children born in Phase 2)
    raw_times = np.array([a.phase_switch_time for a in model.agent_set], dtype=float)

    finite_mask = np.isfinite(raw_times)
    switch_times = raw_times[finite_mask]
    num_inf = int((~finite_mask).sum())
    print(f"Excluded {num_inf} agents with phase_switch_time = inf (born in Phase 2).")

    if len(switch_times) == 0:
        print("Warning: No finite phase-switch times available to plot.")
        return

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax[0].hist(switch_times, bins=40, color="purple", alpha=0.7)
    ax[0].set_title("Histogram of Phase-Switch Times (Finite Only)")
    ax[0].set_xlabel("Switch time")
    ax[0].set_ylabel("Count")

    # Empirical CDF
    sorted_times = np.sort(switch_times)
    yvals = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
    ax[1].plot(sorted_times, yvals, lw=2)
    ax[1].set_title("Empirical CDF of Phase-Switch Times (Finite Only)")
    ax[1].set_xlabel("Switch time")
    ax[1].set_ylabel("Cumulative probability")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "phase_switch_distribution.png"), dpi=150)
    plt.close()


def plot_dispersal_patterns(param_scenarios, labels, steps=200, seed=1):
    fig, axes = plt.subplots(1, len(param_scenarios), figsize=(6 * len(param_scenarios), 6))
    if len(param_scenarios) == 1:
        axes = [axes]  # handle edge case

    for i, (params, label) in enumerate(zip(param_scenarios, labels)):
        model = ClustersModel(params=params, seed=seed)
        for _ in range(steps):
            model.step()

        ids, pos, radii, sizes, speeds = model._snapshot_alive()

        axes[i].scatter(pos[:, 0], pos[:, 1], s=5, alpha=0.6)
        axes[i].set_title(label)
        axes[i].set_xlabel("x position")
        axes[i].set_ylabel("y position")
        axes[i].set_aspect("equal")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "dispersal_patterns.png"), dpi=150)
    plt.close()


# ============================================================
# CLI
# ============================================================

def build_argparser():
    p = argparse.ArgumentParser(description="ABM Behaviour Exhibits (Section 4.1)")
    p.add_argument("--defaults", type=str, default=DEFAULTS_YAML,
                   help="Path to scripts_defaults.yaml")
    p.add_argument("--seed", type=int, default=1, help="Random seed for baseline run")
    p.add_argument("--steps", type=int, default=None,
                   help="Override number of steps for baseline run")
    p.add_argument("--disp-steps", type=int, default=200,
                   help="Steps for dispersal pattern snapshots")
    p.add_argument("--fast-scale1", type=float, default=7.0,
                   help="Phase1 speed 'scale' for fast scenario")
    p.add_argument("--fast-scale2", type=float, default=6.0,
                   help="Phase2 speed 'scale' for fast scenario")
    p.add_argument("--slow-scale1", type=float, default=2.0,
                   help="Phase1 speed 'scale' for slow scenario")
    p.add_argument("--slow-scale2", type=float, default=1.5,
                   help="Phase2 speed 'scale' for slow scenario")
    return p


# ============================================================
# MAIN
# ============================================================

def main():
    args = build_argparser().parse_args()

    # 1) Load YAML defaults (your file) and merge into full ABM params
    user_cfg = load_yaml_defaults(args.defaults)          # from scripts_defaults.yaml
    params = merge_into_abm_params(user_cfg)              # full ABM params, ready for ClustersModel

    # 2) Baseline ABM
    print("Running baseline ABM...")
    model = run_abm(params=params, steps=args.steps, seed=args.seed)

    # 3) Summaries + core plots
    print("Summarising output...")
    summary = summarise_abm(model)

    print("Plotting trajectories...")
    plot_cluster_trajectories(summary)

    print("Plotting phase-switch distribution...")
    plot_phase_switch_distribution(model)

    # 4) Dispersal patterns: build three scenarios (baseline + fast + slow)
    print("Plotting dispersal patterns...")
    # baseline uses the same 'params' as above
    params_fast = copy.deepcopy(params)
    params_fast.setdefault("movement_v2", {}).setdefault("phase1", {}).setdefault("speed_dist", {}).setdefault("params", {})["scale"] = float(args.fast_scale1)
    params_fast["movement_v2"].setdefault("phase2", {}).setdefault("speed_dist", {}).setdefault("params", {})["scale"] = float(args.fast_scale2)

    params_slow = copy.deepcopy(params)
    params_slow.setdefault("movement_v2", {}).setdefault("phase1", {}).setdefault("speed_dist", {}).setdefault("params", {})["scale"] = float(args.slow_scale1)
    params_slow["movement_v2"].setdefault("phase2", {}).setdefault("speed_dist", {}).setdefault("params", {})["scale"] = float(args.slow_scale2)

    plot_dispersal_patterns(
        [params, params_fast, params_slow],
        ["Baseline", "Fast motility", "Slow motility"],
        steps=args.disp_steps,
        seed=args.seed
    )

    print(f"All plots saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
