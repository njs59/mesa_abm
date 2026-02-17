#!/usr/bin/env python3
"""
Parallel forward simulation of ABM from a specified timepoint.

Summary statistics per step (across runs we report mean and 95% CIs):
 1) Number of clusters
 2) Mean cluster size
 3) Mean squared cluster size

Reads:
- ODE_fitting/background/mean_initial_clusters.json
- ODE_fitting/background/best_model.json
- scripts_defaults.yaml (ABM defaults)

Outputs:
- forward_means_stats.csv
- forward_ci_stats.csv
- plot_num_clusters.png
- plot_mean_cluster_size.png
- plot_mean_squared_cluster_size.png
- abm_input.yaml  <-- NEW: snapshot of inputs used

New flags (minimal changes):
- --movement-phase {1,2}  : 2 = force Phase-2 only (existing behaviour), 1 = natural Phase1→2 transitions
- --init-singletons <int> : if set, ignore best_model sampling and initialise all clusters as singletons
"""
from __future__ import annotations
import os
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# Import ABM
from abm.clusters_model import ClustersModel

# ------------------------------------------------------------
# Directory resolution
# ------------------------------------------------------------
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))  # ODE_fitting/scripts
PKG_ROOT   = os.path.dirname(MODULE_DIR)                 # ODE_fitting/
PROJECT_ROOT = os.path.dirname(PKG_ROOT)                 # project root

BACKGROUND_DIR   = os.path.join(PROJECT_ROOT, "ODE_fitting/background")
RESULTS_DIR      = os.path.join(PROJECT_ROOT, "ODE_fitting/results")
os.makedirs(RESULTS_DIR, exist_ok=True)
DEFAULTS_PATH    = os.path.join(MODULE_DIR, "scripts_defaults.yaml")
BEST_MODEL_PATH  = os.path.join(BACKGROUND_DIR, "best_model.json")
MEAN_CLUSTER_PATH= os.path.join(BACKGROUND_DIR, "mean_initial_clusters.json")

# ------------------------------------------------------------
# Distribution sampler (unchanged)
# ------------------------------------------------------------
def sample_from_best_dist(best_info: dict, n_samples: int, seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    name  = str(best_info["best_model"]).strip()
    params = best_info["params"]

    def f(x):
        if isinstance(x, list):
            return [float(v) for v in x]
        return float(x)

    lname = name.lower()
    if "log" in lname and "norm" in lname:
        s, loc, scale = map(f, params)
        x = loc + scale * np.exp(rng.normal(0, s, n_samples))
    elif "gamma" in lname:
        a, loc, scale = map(f, params)
        x = loc + rng.gamma(a, scale, n_samples)
    elif "weibull" in lname:
        c, loc, scale = map(f, params)
        x = loc + scale * rng.weibull(c, n_samples)
    elif "exponential" in lname:
        loc, scale = map(f, params)
        x = loc + rng.exponential(scale, n_samples)
    elif "normal" in lname and "log" not in lname:
        mu, sd = map(f, params)
        x = rng.normal(mu, sd, n_samples)
    elif "logistic" in lname:
        loc, scale = map(f, params)
        x = loc + scale * rng.logistic(0.0, 1.0, n_samples)
    elif "cauchy" in lname:
        loc, scale = map(f, params)
        x = loc + scale * rng.standard_cauchy(n_samples)
    elif "gev" in lname:
        c, loc, scale = map(f, params)
        u = rng.uniform(0, 1, n_samples)
        u = np.clip(u, 1e-12, 1 - 1e-12)
        x = loc + scale * ((-np.log(u)) ** (-c) - 1.0) / c
    elif "gpd" in lname:
        c, loc, scale = map(f, params)
        u = rng.uniform(0, 1, n_samples)
        u = np.clip(u, 1e-12, 1 - 1e-12)
        x = loc + scale * (u ** (-c) - 1.0) / c
    elif "beta" in lname:
        (a, b, _loc2, _scale2) = params[0]
        vmin, vmax = params[1], params[2]
        y = rng.beta(float(a), float(b), n_samples)
        x = vmin + (vmax - vmin) * y
    else:
        raise ValueError(f"Unsupported distribution name: {name}")

    x = np.maximum(1, np.round(x).astype(int))
    return x

# ------------------------------------------------------------
# Worker function (MINIMAL CHANGE: movement_phase argument)
# ------------------------------------------------------------
def run_simulation_worker(args):
    default_params, init_sizes, seed, start_step, movement_phase = args

    # Shallow copy to isolate workers
    params = json.loads(json.dumps(default_params))

    dt          = float(params["time"]["dt"])
    total_steps = int(params["time"]["steps"])

    # How many *future* steps to simulate?
    steps_to_run = max(0, total_steps - start_step)
    time_offset  = float(start_step) * dt

    # Build initial clusters: all proliferative singletons or provided sizes
    init_clusters = [{"size": int(s), "phenotype": "proliferative"} for s in init_sizes]

    model = ClustersModel(params=params, seed=seed, init_clusters=init_clusters)

    # **NEW**: movement-phase control
    #   - if 2: force Phase-2 only (existing behaviour)
    #   - if 1: allow natural Phase-1 -> Phase-2 transitions (do nothing)
    if int(movement_phase) == 2:
        for a in model.agent_set:
            a.movement_phase     = 2
            a.phase_switch_time  = float("inf")

    # Set model time offset
    model.time = time_offset

    n_list, m1_list, m2_list = [], [], []

    for _ in range(steps_to_run):
        model.step()
        sizes = np.asarray(model.size_log[-1], dtype=float)
        n = sizes.size
        if n == 0:
            n_list.append(0); m1_list.append(0.0); m2_list.append(0.0)
        else:
            n_list.append(int(n))
            m1_list.append(float(sizes.mean()))
            m2_list.append(float((sizes ** 2).mean()))

    return np.column_stack([n_list, m1_list, m2_list])

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Parallel forward ABM simulation (Phase control + singleton IC)")

    p.add_argument("--start-step",     type=int, default=71)
    p.add_argument("--n-runs",         type=int, default=100)
    p.add_argument("--n-workers",      type=int, default=max(1, cpu_count() - 1))
    p.add_argument("--seed",           type=int, default=12345)
    p.add_argument("--defaults",       type=str, default=DEFAULTS_PATH)
    p.add_argument("--best-model",     type=str, default=BEST_MODEL_PATH)
    p.add_argument("--mean-clusters",  type=str, default=MEAN_CLUSTER_PATH)
    p.add_argument("--results-dir",    type=str, default=RESULTS_DIR)

    # -------- NEW FLAGS (minimal additions) --------
    p.add_argument("--movement-phase", type=int, choices=[1, 2], default=2,
                   help="2 = force Phase-2 only (existing behaviour); 1 = natural Phase-1→2 transitions.")
    p.add_argument("--init-singletons", type=int, default=None,
                   help="If provided, initialise ALL clusters as singletons and ignore best_model sampling.")
    # -----------------------------------------------

    return p.parse_args()

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # Load defaults
    with open(args.defaults, "r") as f:
        default_params = yaml.safe_load(f)

    # NEW: initial sizes — either singleton IC or sampled from best model
    if args.init_singletons is not None:
        init_sizes = np.ones(int(args.init_singletons), dtype=int)
    else:
        with open(args.mean_clusters, "r") as f:
            mean_info = json.load(f)
        n_clusters = int(mean_info["mean_initial_clusters"])

        with open(args.best_model, "r") as f:
            best_info = json.load(f)

        init_sizes = sample_from_best_dist(best_info, n_clusters, seed=args.seed)

    # Seeds per run
    seeds = [args.seed + i for i in range(args.n_runs)]
    worker_args = [
        (default_params, init_sizes, s, args.start_step, args.movement_phase)
        for s in seeds
    ]

    print(f"Running {args.n_runs} forward simulations with {args.n_workers} workers...")
    with Pool(processes=args.n_workers) as pool:
        runs = pool.map(run_simulation_worker, worker_args)

    runs = np.stack(runs, axis=0)  # (n_runs, T, 3)
    T = runs.shape[1]

    # Means & CIs
    mean_stats  = runs.mean(axis=0)
    lower_stats = np.percentile(runs, 2.5, axis=0)
    upper_stats = np.percentile(runs, 97.5, axis=0)

    steps = np.arange(args.start_step, args.start_step + T)
    stat_names = ["num_clusters", "mean_cluster_size", "mean_squared_cluster_size"]

    # Means CSV
    df_mean = pd.DataFrame({
        "step": steps,
        stat_names[0]: mean_stats[:, 0],
        stat_names[1]: mean_stats[:, 1],
        stat_names[2]: mean_stats[:, 2],
    })

    # CI CSV
    df_ci = pd.DataFrame({
        "step": steps,
        f"{stat_names[0]}_mean":       mean_stats[:, 0],
        f"{stat_names[0]}_lower_95":   lower_stats[:, 0],
        f"{stat_names[0]}_upper_95":   upper_stats[:, 0],
        f"{stat_names[1]}_mean":       mean_stats[:, 1],
        f"{stat_names[1]}_lower_95":   lower_stats[:, 1],
        f"{stat_names[1]}_upper_95":   upper_stats[:, 1],
        f"{stat_names[2]}_mean":       mean_stats[:, 2],
        f"{stat_names[2]}_lower_95":   lower_stats[:, 2],
        f"{stat_names[2]}_upper_95":   upper_stats[:, 2],
    })

    mean_path = os.path.join(args.results_dir, "forward_means_stats.csv")
    ci_path   = os.path.join(args.results_dir, "forward_ci_stats.csv")
    df_mean.to_csv(mean_path, index=False)
    df_ci.to_csv(ci_path, index=False)

    # Helper for plotting
    def plot_stat(idx, title, ylabel, fname):
        plt.figure(figsize=(10, 6))
        plt.plot(steps, mean_stats[:, idx], color="blue", label="Mean")
        plt.fill_between(steps, lower_stats[:, idx], upper_stats[:, idx],
                         alpha=0.3, color="lightblue", label="95% CI")
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        out = os.path.join(args.results_dir, fname)
        plt.savefig(out, dpi=150)
        plt.close()
        return out

    p1 = plot_stat(0, "Number of Clusters (mean ± 95% CI)", "Number of clusters", "plot_num_clusters.png")
    p2 = plot_stat(1, "Mean Cluster Size (mean ± 95% CI)", "Mean cluster size", "plot_mean_cluster_size.png")
    p3 = plot_stat(2, "Mean Squared Cluster Size (mean ± 95% CI)", "Mean squared size", "plot_mean_squared_cluster_size.png")

    # --- NEW: Save ABM input snapshot for reproducibility ---
    abm_snapshot = {
        "start_step":        args.start_step,
        "n_runs":            args.n_runs,
        "n_workers":         args.n_workers,
        "seed":              args.seed,
        "defaults_yaml":     args.defaults,
        "best_model":        args.best_model,
        "mean_clusters":     args.mean_clusters,
        "movement_phase":    int(args.movement_phase),
        "init_singletons":   int(args.init_singletons) if args.init_singletons is not None else None,
    }
    with open(os.path.join(args.results_dir, "abm_input.yaml"), "w") as f:
        yaml.safe_dump(abm_snapshot, f, sort_keys=False)

    print("\nSaved:")
    print(f"  {mean_path}")
    print(f"  {ci_path}")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")

if __name__ == "__main__":
    main()