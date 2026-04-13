#!/usr/bin/env python3
"""
forward_phase2_invasive_convergence_plus.py

Same model setup as forward_phase2_invasive_spaghetti_plots.py (phase-2 invasive singletons),
but focuses on convergence diagnostics.

Runs N repeats (default 1000) and computes summary stats time series:
- n_clusters
- mean_size
- var_size
- median_nnd

Produces:
1) Convergence plots for FINAL timepoint:
   - Running mean vs number of sims (1..N)
   - 95% CI band based on SEM (fast)
   - BOTH linear x scale and log x scale
   - Saved as 2x2 grids AND (optionally) separate plots per stat

2) Mean-over-time curves using first i sims for i in {10,20,50,100,200,500,1000}
   - Saved as 2x2 grid AND (optionally) separate plots per stat
   - Optional CI band (SEM) for largest i to show uncertainty

Also writes a recommendation report for how many sims are needed based on:
- CI half-width tolerance (absolute and/or relative)
- Stability criterion: running mean changes little over last W sims
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# Ensure we can import sibling package `abm` (same pattern as your existing script)
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from abm.clusters_model import ClustersModel
from abm.utils import DEFAULTS

# User convention: dt=1 is 30 minutes (same as your existing script)
MIN_PER_DT = 30.0


# -----------------------------------------------------------------------------
# Parameter helpers (same as existing)
# -----------------------------------------------------------------------------
def clone_params(base: dict) -> dict:
    import copy
    return copy.deepcopy(base)


def prepare_params_phase2_invasive(base: dict, *, n_clusters: int, steps: int) -> dict:
    p = clone_params(base)
    p.setdefault("time", {})["steps"] = int(steps)
    p.setdefault("init", {})["n_clusters"] = int(n_clusters)
    p["init"]["size"] = 1
    p["init"]["phenotype"] = "invasive"
    return p


def spawn_invasive_phase2_singletons(model: ClustersModel, n: int) -> None:
    for _ in range(int(n)):
        a = model.spawn_cluster(size=1, phenotype="invasive", phase_switch_time=0.0)
        a.movement_phase = 2
        a.phase_switch_time = float("-inf")


def run_single(seed: int, params: dict) -> Dict[str, List[np.ndarray]]:
    """
    Same single-run logic as your existing script:
    instantiate model, spawn invasive singletons forced into phase 2, step, return logs.
    """
    m = ClustersModel(params=params, seed=seed, init_clusters=[])
    n0 = int(params.get("init", {}).get("n_clusters", 800))
    spawn_invasive_phase2_singletons(m, n0)

    n_steps = int(params.get("time", {}).get("steps", 145))
    for _ in range(n_steps):
        m.step()

    return {
        "sizes": m.size_log,
        "positions": m.pos_log,
        "dt": float(getattr(m, "dt", 1.0)),
    }


# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------
def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else float("nan")


def safe_var(x: np.ndarray) -> float:
    return float(np.var(x)) if x.size else float("nan")


def median_nearest_neighbour_distance(positions: np.ndarray) -> float:
    if positions is None or len(positions) < 2:
        return float("nan")
    pts = np.asarray(positions, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return float("nan")
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    return float(np.median(dists[:, 1]))


def fix_initial_empty_log(
    sizes_log: List[np.ndarray],
    pos_log: List[np.ndarray],
    mode: str = "copy_next",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Fix occasional empty t=0 logs.

    mode:
      - 'none'      : no changes
      - 'copy_next' : if t=0 empty, replace it with first subsequent non-empty snapshot
      - 'drop'      : drop t=0 and pad the end with final snapshot to keep length
    """
    if mode == "none":
        return sizes_log, pos_log
    if not sizes_log or not pos_log:
        return sizes_log, pos_log

    try:
        if len(sizes_log[0]) > 0:
            return sizes_log, pos_log
    except Exception:
        return sizes_log, pos_log

    if mode == "copy_next":
        k = None
        for j in range(1, min(len(sizes_log), len(pos_log))):
            try:
                if len(sizes_log[j]) > 0:
                    k = j
                    break
            except Exception:
                continue
        if k is not None:
            sizes_log = list(sizes_log)
            pos_log = list(pos_log)
            sizes_log[0] = sizes_log[k]
            pos_log[0] = pos_log[k]
        return sizes_log, pos_log

    if mode == "drop":
        if len(sizes_log) >= 2 and len(pos_log) >= 2:
            sizes_log = list(sizes_log[1:]) + [sizes_log[-1]]
            pos_log = list(pos_log[1:]) + [pos_log[-1]]
        return sizes_log, pos_log

    raise ValueError(f"Unknown fix_t0 mode: {mode}")


# -----------------------------------------------------------------------------
# Worker: simulate and summarise
# -----------------------------------------------------------------------------
def worker_simulate_and_summarise(seed: int, params: dict, steps: int, fix_t0: str) -> Dict[str, np.ndarray]:
    out = run_single(seed, params)
    sizes_log = out["sizes"]
    pos_log = out["positions"]
    dt = float(out.get("dt", 1.0))

    sizes_log, pos_log = fix_initial_empty_log(sizes_log, pos_log, mode=fix_t0)

    T = steps + 1
    n_clusters = np.full((T,), np.nan, dtype=np.float32)
    mean_size = np.full((T,), np.nan, dtype=np.float32)
    var_size = np.full((T,), np.nan, dtype=np.float32)
    med_nnd = np.full((T,), np.nan, dtype=np.float32)

    T_run = min(T, len(sizes_log), len(pos_log))
    for t in range(T_run):
        sizes = np.asarray(sizes_log[t], dtype=float)
        pos = np.asarray(pos_log[t], dtype=float)
        n_clusters[t] = float(sizes.size)
        mean_size[t] = safe_mean(sizes)
        var_size[t] = safe_var(sizes)
        med_nnd[t] = median_nearest_neighbour_distance(pos)

    return {
        "dt": np.array([dt], dtype=np.float32),
        "n_clusters": n_clusters,
        "mean_size": mean_size,
        "var_size": var_size,
        "median_nnd": med_nnd,
    }


# -----------------------------------------------------------------------------
# Plotting / maths utilities
# -----------------------------------------------------------------------------
def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def time_axis(T: int, dt: float, units: str = "h") -> np.ndarray:
    minutes = np.arange(T) * float(dt) * MIN_PER_DT
    return minutes if units == "min" else minutes / 60.0


def time_label(units: str) -> str:
    return "Time (min)" if units == "min" else "Time (h)"


def add_panel_labels_outside(fig, axes_2x2, *, fontsize: int = 26, dx: float = 0.012, dy: float = 0.010) -> None:
    labels = ["A", "B", "C", "D"]
    k = 0
    for r in range(2):
        for c in range(2):
            ax = axes_2x2[r][c]
            bb = ax.get_position()
            x = bb.x0 - dx
            y = bb.y1 + dy
            fig.text(
                x, y, labels[k],
                ha="right", va="bottom",
                fontweight="bold", fontsize=fontsize
            )
            k += 1


def running_mean_and_sem(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute running mean and running SEM for a 1D array y (length N),
    using cumulative sums (fast). NaNs are handled by treating them as missing.

    Returns:
      mean_n: (N,)
      sem_n:  (N,)
    """
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y).astype(float)
    y0 = np.where(np.isfinite(y), y, 0.0)

    csum = np.cumsum(y0)
    csum2 = np.cumsum(y0 * y0)
    n = np.cumsum(finite)

    mean = np.full_like(csum, np.nan, dtype=float)
    sem = np.full_like(csum, np.nan, dtype=float)

    ok = n > 0
    mean[ok] = csum[ok] / n[ok]

    ok2 = n > 1
    var = np.full_like(csum, np.nan, dtype=float)
    var[ok2] = (csum2[ok2] - (csum[ok2] * csum[ok2]) / n[ok2]) / (n[ok2] - 1.0)
    var = np.maximum(var, 0.0)
    sem[ok2] = np.sqrt(var[ok2] / n[ok2])

    return mean, sem


def first_i_mean_and_sem(arr: np.ndarray, i: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a (N, T) array, compute mean and SEM across first i rows at each time.
    Handles NaNs.
    """
    x = np.asarray(arr[:i, :], dtype=float)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0, ddof=1)
    n = np.sum(np.isfinite(x), axis=0)
    sem = std / np.sqrt(np.maximum(n, 1))
    return mean, sem


def recommend_n(
    y: np.ndarray,
    ci_level: float,
    tol_abs: float,
    tol_rel: float,
    stability_window: int,
    min_n: int = 30,
) -> Dict[str, object]:
    """
    Recommendation for required number of simulations based on:
    - CI half-width <= max(tol_abs, tol_rel*|mean|)
    - running mean stability: over last W sims, mean changes <= same tolerance

    Returns dict with recommended n and diagnostic arrays.
    """
    mean, sem = running_mean_and_sem(y)
    alpha = (1.0 - float(ci_level)) / 2.0
    z = float(norm.ppf(1.0 - alpha))
    half_width = z * sem

    thr = np.maximum(float(tol_abs), float(tol_rel) * np.abs(mean))

    stab = np.full_like(mean, np.nan, dtype=float)
    W = int(stability_window)
    if W >= 1:
        stab[W:] = np.abs(mean[W:] - mean[:-W])

    stable = np.zeros_like(mean, dtype=bool)
    ok = np.isfinite(stab) & np.isfinite(thr)
    stable[ok] = stab[ok] <= thr[ok]

    good = (half_width <= thr) & stable & np.isfinite(half_width) & np.isfinite(thr)
    good[: max(0, int(min_n) - 1)] = False

    idx = np.where(good)[0]
    n_rec = int(idx[0] + 1) if idx.size else None

    return {
        "n_recommended": n_rec,
        "running_mean": mean,
        "running_sem": sem,
        "ci_half_width": half_width,
        "threshold": thr,
        "stable": stable,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run N ABM realisations and plot convergence diagnostics (linear + log) + mean-over-time by i."
    )
    parser.add_argument("--repeats", type=int, default=1000, help="Number of repeats (default: 1000)")
    parser.add_argument(
        "--steps",
        type=int,
        default=int(DEFAULTS.get("time", {}).get("steps", 145)),
        help="Number of steps per repeat",
    )
    parser.add_argument(
        "--n0",
        type=int,
        default=int(DEFAULTS.get("init", {}).get("n_clusters", 800)),
        help="Initial number of invasive singletons",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed (run i uses seed + i)")
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output folder (default: outputs/convergence_plus_<timestamp>/)",
    )
    parser.add_argument("--time_units", choices=["min", "h"], default="h", help="Time units")
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--fix_t0",
        choices=["copy_next", "drop", "none"],
        default="copy_next",
        help="Fix empty t=0 logs",
    )

    parser.add_argument(
        "--i_values",
        type=int,
        nargs="*",
        default=[10, 20, 50, 100, 200, 500, 1000],
        help="Values of i for mean-over-time plots",
    )
    parser.add_argument("--ci_level", type=float, default=0.95, help="CI level for SEM bands (default: 0.95)")

    # Convergence recommendation thresholds
    parser.add_argument(
        "--tol_abs",
        type=float,
        default=0.0,
        help="Absolute tolerance for convergence (CI half-width and stability). Default 0 (use relative).",
    )
    parser.add_argument(
        "--tol_rel",
        type=float,
        default=0.02,
        help="Relative tolerance (fraction of |mean|). Default 0.02 (2%%).",
    )
    parser.add_argument(
        "--stability_window",
        type=int,
        default=50,
        help="Window W: require running mean change over last W sims <= tolerance. Default 50.",
    )
    parser.add_argument(
        "--min_n",
        type=int,
        default=30,
        help="Do not declare convergence before min_n sims. Default 30.",
    )

    parser.add_argument(
        "--plot_separate",
        action="store_true",
        help="Also save separate per-stat plots (in addition to 2x2 grids).",
    )
    parser.add_argument(
        "--ci_over_time_for_max_i",
        action="store_true",
        help="Add CI band (SEM) over time for the largest i curve in mean-over-time plots.",
    )

    args = parser.parse_args()

    repeats = int(args.repeats)
    steps = int(args.steps)
    n0 = int(args.n0)

    i_values = sorted(set(int(x) for x in args.i_values))
    i_values = [i for i in i_values if 1 <= i <= repeats]
    if not i_values:
        raise ValueError("No valid i_values after filtering to [1..repeats].")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = (
        os.path.join(THIS_DIR, "outputs", f"convergence_plus_{timestamp}")
        if args.outdir is None
        else args.outdir
    )
    ensure_outdir(out_base)

    params = prepare_params_phase2_invasive(DEFAULTS, n_clusters=n0, steps=steps)
    T = steps + 1

    # Arrays: (repeats, T)
    n_clusters_arr = np.full((repeats, T), np.nan, dtype=np.float32)
    mean_size_arr = np.full((repeats, T), np.nan, dtype=np.float32)
    var_size_arr = np.full((repeats, T), np.nan, dtype=np.float32)
    med_nnd_arr = np.full((repeats, T), np.nan, dtype=np.float32)

    seeds = [int(args.seed) + i for i in range(repeats)]

    print(f"Running {repeats} realisations with {int(args.n_jobs)} workers…")
    dt_used = None

    with ProcessPoolExecutor(max_workers=int(args.n_jobs)) as ex:
        futures = {
            ex.submit(worker_simulate_and_summarise, seed, params, steps, args.fix_t0): i
            for i, seed in enumerate(seeds)
        }
        completed = 0
        for fut in as_completed(futures):
            i = futures[fut]
            res = fut.result()
            if dt_used is None:
                dt_used = float(res["dt"][0])

            n_clusters_arr[i, :] = res["n_clusters"]
            mean_size_arr[i, :] = res["mean_size"]
            var_size_arr[i, :] = res["var_size"]
            med_nnd_arr[i, :] = res["median_nnd"]

            completed += 1
            if completed % max(1, repeats // 10) == 0:
                print(f"  {completed}/{repeats} completed")

    if dt_used is None:
        dt_used = 1.0

    t_axis = time_axis(T, dt_used, units=args.time_units)

    # Save arrays for reuse
    np.savez_compressed(
        os.path.join(out_base, "summary_arrays.npz"),
        n_clusters=n_clusters_arr,
        mean_size=mean_size_arr,
        var_size=var_size_arr,
        median_nnd=med_nnd_arr,
        dt=np.array([dt_used], dtype=np.float32),
        min_per_dt=np.array([MIN_PER_DT], dtype=np.float32),
        steps=np.array([steps], dtype=int),
        repeats=np.array([repeats], dtype=int),
        i_values=np.array(i_values, dtype=int),
        ci_level=np.array([float(args.ci_level)], dtype=np.float32),
        tol_abs=np.array([float(args.tol_abs)], dtype=np.float32),
        tol_rel=np.array([float(args.tol_rel)], dtype=np.float32),
        stability_window=np.array([int(args.stability_window)], dtype=int),
        min_n=np.array([int(args.min_n)], dtype=int),
    )

    stat_arrays = {
        "n_clusters": n_clusters_arr.astype(float),
        "mean_size": mean_size_arr.astype(float),
        "var_size": var_size_arr.astype(float),
        "median_nnd": med_nnd_arr.astype(float),
    }
    pretty = {
        "n_clusters": "Number of clusters",
        "mean_size": "Mean cluster size",
        "var_size": "Variance of cluster size",
        "median_nnd": "Median nearest-neighbour distance",
    }
    order = ["n_clusters", "mean_size", "var_size", "median_nnd"]

    # -------------------------------
    # (1) Convergence plots (final time)
    # -------------------------------
    alpha = (1.0 - float(args.ci_level)) / 2.0
    z = float(norm.ppf(1.0 - alpha))
    x = np.arange(1, repeats + 1, dtype=float)

    rec_lines = []
    rec_lines.append(
        f"Convergence recommendation (CI level={args.ci_level}, tol_abs={args.tol_abs}, tol_rel={args.tol_rel}, "
        f"stability_window={args.stability_window}, min_n={args.min_n})"
    )

    fig_lin, axes_lin = plt.subplots(2, 2, figsize=(13.2, 9.6), squeeze=False)
    fig_log, axes_log = plt.subplots(2, 2, figsize=(13.2, 9.6), squeeze=False)

    for k, name in enumerate(order):
        r, c = divmod(k, 2)
        y = stat_arrays[name][:, -1]  # final timepoint

        mean, sem = running_mean_and_sem(y)
        half = z * sem
        low = mean - half
        high = mean + half

        rec = recommend_n(
            y,
            ci_level=float(args.ci_level),
            tol_abs=float(args.tol_abs),
            tol_rel=float(args.tol_rel),
            stability_window=int(args.stability_window),
            min_n=int(args.min_n),
        )
        nrec = rec["n_recommended"]
        if nrec is None:
            rec_lines.append(f"- {name}: NOT converged by N={repeats} under these tolerances.")
        else:
            rec_lines.append(f"- {name}: recommended N ≈ {nrec}")

        # Linear x panel
        ax = axes_lin[r][c]
        ax.plot(x, mean, lw=2.2, label="Running mean")
        ax.fill_between(x, low, high, alpha=0.25, label=f"{int(args.ci_level*100)}% CI (SEM)")
        ax.set_title(f"{pretty[name]} (final time)")
        ax.set_xlabel("Number of simulations (first x)")
        ax.set_ylabel("Running mean")
        ax.grid(True, alpha=0.25)
        if nrec is not None:
            ax.axvline(nrec, color="tab:red", ls="--", lw=1.6, alpha=0.8, label=f"Recommended N≈{nrec}")
        ax.legend(fontsize=9)

        # Log x panel
        ax2 = axes_log[r][c]
        ax2.plot(x, mean, lw=2.2, label="Running mean")
        ax2.fill_between(x, low, high, alpha=0.25, label=f"{int(args.ci_level*100)}% CI (SEM)")
        ax2.set_xscale("log")
        ax2.set_title(f"{pretty[name]} (final time)")
        ax2.set_xlabel("Number of simulations (first x, log scale)")
        ax2.set_ylabel("Running mean")
        ax2.grid(True, alpha=0.25, which="both")
        if nrec is not None:
            ax2.axvline(nrec, color="tab:red", ls="--", lw=1.6, alpha=0.8, label=f"Recommended N≈{nrec}")
        ax2.legend(fontsize=9)

        # Optional separate plots per stat
        if args.plot_separate:
            # linear
            f, a = plt.subplots(figsize=(8.8, 5.6))
            a.plot(x, mean, lw=2.3)
            a.fill_between(x, low, high, alpha=0.25)
            a.set_title(f"Convergence: {pretty[name]} (final time)")
            a.set_xlabel("Number of simulations (first x)")
            a.set_ylabel("Running mean")
            a.grid(True, alpha=0.25)
            if nrec is not None:
                a.axvline(nrec, color="tab:red", ls="--", lw=1.6, alpha=0.8)
            f.tight_layout()
            f.savefig(os.path.join(out_base, f"convergence_final_{name}_linear.png"), dpi=300, bbox_inches="tight")
            plt.close(f)

            # log
            f, a = plt.subplots(figsize=(8.8, 5.6))
            a.plot(x, mean, lw=2.3)
            a.fill_between(x, low, high, alpha=0.25)
            a.set_xscale("log")
            a.set_title(f"Convergence: {pretty[name]} (final time, log x)")
            a.set_xlabel("Number of simulations (first x, log scale)")
            a.set_ylabel("Running mean")
            a.grid(True, alpha=0.25, which="both")
            if nrec is not None:
                a.axvline(nrec, color="tab:red", ls="--", lw=1.6, alpha=0.8)
            f.tight_layout()
            f.savefig(os.path.join(out_base, f"convergence_final_{name}_logx.png"), dpi=300, bbox_inches="tight")
            plt.close(f)

    add_panel_labels_outside(fig_lin, axes_lin)
    fig_lin.tight_layout()
    fig_lin.savefig(os.path.join(out_base, "convergence_final_2x2_linear.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_lin)

    add_panel_labels_outside(fig_log, axes_log)
    fig_log.tight_layout()
    fig_log.savefig(os.path.join(out_base, "convergence_final_2x2_logx.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_log)

    report_path = os.path.join(out_base, "convergence_recommendation.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rec_lines) + "\n")

    print("\n".join(rec_lines))
    print(f"(Saved report to {report_path})")

    # -------------------------------
    # (2) Mean-over-time curves for different i
    # -------------------------------
    cmap = plt.get_cmap("viridis")
    colours = [cmap(j / max(1, len(i_values) - 1)) for j in range(len(i_values))]
    i_max = max(i_values)

    fig_mt, axes_mt = plt.subplots(2, 2, figsize=(13.2, 9.6), squeeze=False)

    for k, name in enumerate(order):
        r, c = divmod(k, 2)
        ax = axes_mt[r][c]

        for j, i in enumerate(i_values):
            mean_i, _sem_i = first_i_mean_and_sem(stat_arrays[name], i)
            ax.plot(t_axis, mean_i, lw=2.0, color=colours[j], label=f"i = {i}")

        if args.ci_over_time_for_max_i:
            mean_i, sem_i = first_i_mean_and_sem(stat_arrays[name], i_max)
            half_i = z * sem_i
            ax.fill_between(
                t_axis,
                mean_i - half_i,
                mean_i + half_i,
                color="grey",
                alpha=0.18,
                label=f"{int(args.ci_level*100)}% CI (SEM), i={i_max}",
            )

        ax.set_title(pretty[name])
        ax.set_xlabel(time_label(args.time_units))
        ax.set_ylabel("Mean")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, ncol=2)

        if args.plot_separate:
            f, a = plt.subplots(figsize=(9.0, 5.6))
            for j, i in enumerate(i_values):
                mean_i, _sem_i = first_i_mean_and_sem(stat_arrays[name], i)
                a.plot(t_axis, mean_i, lw=2.2, color=colours[j], label=f"i = {i}")

            if args.ci_over_time_for_max_i:
                mean_i, sem_i = first_i_mean_and_sem(stat_arrays[name], i_max)
                half_i = z * sem_i
                a.fill_between(
                    t_axis,
                    mean_i - half_i,
                    mean_i + half_i,
                    color="grey",
                    alpha=0.18,
                    label=f"{int(args.ci_level*100)}% CI (SEM), i={i_max}",
                )

            a.set_title(f"Mean over time: {pretty[name]}")
            a.set_xlabel(time_label(args.time_units))
            a.set_ylabel("Mean")
            a.grid(True, alpha=0.25)
            a.legend(fontsize=9, ncol=2)
            f.tight_layout()
            f.savefig(os.path.join(out_base, f"mean_over_time_by_i_{name}.png"), dpi=300, bbox_inches="tight")
            plt.close(f)

    add_panel_labels_outside(fig_mt, axes_mt)
    fig_mt.tight_layout()
    fig_mt.savefig(os.path.join(out_base, "mean_over_time_by_i_2x2.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_mt)

    print("\nDone. Outputs saved to:")
    print(f"  {out_base}")
    print("Key figures:")
    print("  - convergence_final_2x2_linear.png")
    print("  - convergence_final_2x2_logx.png")
    print("  - mean_over_time_by_i_2x2.png")
    if args.plot_separate:
        print("  - (plus separate per-stat plots)")
    print("Also saved:")
    print("  - summary_arrays.npz")
    print("  - convergence_recommendation.txt")


if __name__ == "__main__":
    main()