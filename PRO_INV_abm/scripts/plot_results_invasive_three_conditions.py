#!/usr/bin/env python3
"""
Plot ABM results for the 3‑condition invasive‑only experiment.

Works with directory structure:

results-Invasive_3/
    summary_repeats.csv
    trajectories/
        phase2_only/
        two_phase_interact/
        two_phase_no_interact/
    plots/
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===============================================================
# PATH HELPERS
# ===============================================================

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ===============================================================
# CONFIDENCE INTERVAL HELPERS
# ===============================================================

def t_critical_975(df: int) -> float:
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except Exception:
        return 1.96


def mean_ci_95(values: np.ndarray) -> Tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(vals))
    if n == 1:
        return (mean, np.nan, np.nan)
    se = float(np.std(vals, ddof=1) / np.sqrt(n))
    tcrit = t_critical_975(n - 1)
    return (mean, mean - tcrit * se, mean + tcrit * se)


def mean_ci_95_over_repeats(matrix: np.ndarray):
    """Compute mean ±95% CI column‑wise from repeat×time matrix."""
    M = np.asarray(matrix, float)
    T = M.shape[1]
    mean = np.full(T, np.nan)
    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)

    for t in range(T):
        vals = M[:, t]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        mean[t] = np.mean(vals)
        if len(vals) > 1:
            se = np.std(vals, ddof=1) / np.sqrt(len(vals))
            tcrit = t_critical_975(len(vals) - 1)
            lo[t] = mean[t] - tcrit * se
            hi[t] = mean[t] + tcrit * se
    return mean, lo, hi


# ===============================================================
# TRAJECTORY DISCOVERY
# ===============================================================

def find_trajectory_files(traj_root: Path) -> Dict[str, List[Path]]:
    out = {}
    if not traj_root.exists():
        return out
    for cond in sorted(p for p in traj_root.iterdir() if p.is_dir()):
        files = sorted(cond.glob("repeat_*.csv"))
        if files:
            out[cond.name] = files
    return out


# ===============================================================
# SPATIAL STATISTICS PER TIMESTEP
# ===============================================================

def compute_spatial_stats_at_step(df_t, W, H, torus=True):
    if df_t.empty:
        return dict(median_nnd=np.nan, r_obs=np.nan, r_exp=np.nan,
                    R=np.nan, z=np.nan, p=np.nan)

    x = df_t["x"].to_numpy(float)
    y = df_t["y"].to_numpy(float)
    N = len(x)
    area = W * H

    if N < 2:
        return dict(median_nnd=np.nan, r_obs=np.nan, r_exp=np.nan,
                    R=np.nan, z=np.nan, p=np.nan)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]

    if torus:
        dx = (dx + W/2) % W - W/2
        dy = (dy + H/2) % H - H/2

    d2 = dx*dx + dy*dy
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))

    r_obs = float(nn.mean())
    lam = N / area
    r_exp = 1.0 / (2.0 * np.sqrt(lam)) if lam > 0 else np.nan
    se = 0.26136 / np.sqrt(N * lam) if (N * lam) > 0 else np.nan
    z = (r_obs - r_exp) / se if (se and np.isfinite(se)) else np.nan
    p = math.erfc(abs(z) / np.sqrt(2)) if np.isfinite(z) else np.nan
    R = r_obs / r_exp if (r_exp and np.isfinite(r_exp)) else np.nan

    return dict(
        median_nnd=float(np.median(nn)),
        r_obs=r_obs,
        r_exp=r_exp,
        R=R,
        z=z,
        p=p,
    )


# ===============================================================
# TIME‑SERIES SUMMARY STATISTICS FOR EACH CONDITION
# ===============================================================

def compute_timeseries_for_condition(files, W, H, torus=True):
    """Return dict of per‑timestep metrics averaged across repeats."""
    # find max step
    max_step = 0
    for f in files:
        df = pd.read_csv(f, usecols=["step"])
        if not df.empty:
            max_step = max(max_step, int(df["step"].max()))

    metrics = [
        "n_clusters", "mean_size", "var_size",
        "median_nnd", "r_obs", "r_exp", "R", "z", "p"
    ]

    per_rep_arrays = {m: [] for m in metrics}

    for f in files:
        df = pd.read_csv(f)
        arrs = {m: np.full(max_step+1, np.nan) for m in metrics}

        for t in range(max_step+1):
            df_t = df[df["step"] == t]
            if df_t.empty:
                continue

            sizes = df_t["size"].to_numpy(float)
            arrs["n_clusters"][t] = len(sizes)
            arrs["mean_size"][t] = sizes.mean()
            arrs["var_size"][t] = sizes.var(ddof=1) if len(sizes) > 1 else 0.0

            spatial = compute_spatial_stats_at_step(df_t, W, H, torus)
            for k, v in spatial.items():
                arrs[k][t] = v

        for m in metrics:
            per_rep_arrays[m].append(arrs[m])

    timeseries = {}
    for m in metrics:
        M = np.vstack(per_rep_arrays[m])
        mean = np.nanmean(M, axis=0)
        timeseries[m] = mean

    return timeseries, max_step


# ===============================================================
# PLOTTING: TIME‑SERIES SUMMARY STATISTICS
# ===============================================================

def plot_timeseries_across_conditions(cond_to_ts: Dict[str, Dict[str, np.ndarray]],
                                      x_axis: np.ndarray,
                                      out_dir: Path):
    out_dir = ensure_dir(out_dir)

    metrics = [
        "n_clusters",
        "mean_size",
        "var_size",
        "median_nnd",
        "R",
        "z",
        "p",
    ]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 5))

        for cond, ts in cond_to_ts.items():
            if metric not in ts:
                continue
            ax.plot(x_axis, ts[metric], lw=2, label=cond)

        ax.set_title(f"{metric.replace('_',' ')} over time")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}_timeseries.png", dpi=300)
        plt.close(fig)


# ===============================================================
# ORIGINAL SUMMARY STATISTICS (ENDPOINT)
# ===============================================================

def build_condition_summary(df_rep: pd.DataFrame, metrics: List[str]):
    rows = []
    for cond, g in df_rep.groupby("condition"):
        out = {"condition": cond, "n_repeats": len(g)}
        for m in metrics:
            mean, lo, hi = mean_ci_95(g[m].to_numpy())
            out[f"{m}_mean"] = mean
            out[f"{m}_ci95_lo"] = lo
            out[f"{m}_ci95_hi"] = hi
        rows.append(out)
    return pd.DataFrame(rows)


def plot_metric_bar(df_cond, metric, outpath):
    m = df_cond[f"{metric}_mean"].to_numpy()
    lo = df_cond[f"{metric}_ci95_lo"].to_numpy()
    hi = df_cond[f"{metric}_ci95_hi"].to_numpy()
    x = np.arange(len(df_cond))
    yerr = np.vstack([m - lo, hi - m])
    yerr = np.where(np.isfinite(yerr), yerr, 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, m, yerr=yerr, capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df_cond["condition"], rotation=25, ha="right")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(metric.replace("_", " "))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# ===============================================================
# MAIN
# ===============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_summaries", action="store_true")
    args = parser.parse_args()

    root = project_root()

    # *** NEW SAVE LOCATION ***
    results_dir = root / "results/results_invasive_3"
    plots_dir = ensure_dir(results_dir / "plots")
    traj_root = results_dir / "trajectories"

    rep_path = results_dir / "summary_repeats.csv"
    if not rep_path.exists():
        raise FileNotFoundError(f"summary_repeats.csv not found at {rep_path}")

    df_rep = pd.read_csv(rep_path)

    trajectory_map = find_trajectory_files(traj_root)
    if not trajectory_map:
        raise RuntimeError("No trajectories found.")

    # Expected 3‑condition order
    conds = ["phase2_only", "two_phase_interact", "two_phase_no_interact"]
    conds = [c for c in conds if c in trajectory_map]

    # Derive dt from first CSV
    example_file = next(iter(trajectory_map[conds[0]]))
    df_ex = pd.read_csv(example_file, usecols=["step", "time_min"])
    dt = float(np.mean(df_ex["time_min"] / (df_ex["step"] + 1e-9)))

    # Spatial domain
    # Use DEFAULTS-like values directly (your model is fixed)
    W = 1344.0
    H = 1025.0
    torus = True

    # ======================================================
    # TIME‑SERIES SUMMARY STATISTICS (NEW FEATURE)
    # ======================================================
    cond_to_ts = {}
    max_steps_list = []

    for cond in conds:
        print(f"Computing time-series stats for {cond} ...")
        ts, max_step = compute_timeseries_for_condition(
            trajectory_map[cond], W, H, torus=True
        )
        cond_to_ts[cond] = ts
        max_steps_list.append(max_step)

    global_max = max(max_steps_list)
    x_axis = np.arange(global_max+1) * dt

    plot_timeseries_across_conditions(
        cond_to_ts, x_axis, plots_dir / "summary_timeseries"
    )

    print("Time‑series summary plots saved.")

    # ======================================================
    # (OPTIONAL) ENDPOINT SUMMARY PLOTS
    # ======================================================
    if not args.skip_summaries:
        metrics = [
            "n_clusters",
            "mean_size",
            "var_size",
            "median_nnd",
            "clark_evans_R",
            "clark_evans_z",
            "clark_evans_p",
        ]

        cond_summary = build_condition_summary(df_rep, metrics)
        for m in metrics:
            plot_metric_bar(cond_summary, m, plots_dir / f"summary_{m}.png")

    print(f"All plots saved under {plots_dir}")


if __name__ == "__main__":
    main()