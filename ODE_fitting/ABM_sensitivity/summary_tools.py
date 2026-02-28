"""
summary_tools.py
================

Per-timestep summary metrics for ABM sensitivity runs, and helpers to
aggregate across repeats.

Metrics (per timestep):
- num_clusters
- merges (raw count that tick)  --> used to compute P_merge across repeats
- mean_size, var_size, cv_size, gini_size
- median_nn (median nearest-neighbour distance; all clusters)
- morisita (Morisita index; grid-based, domain-aware)

Also:
- summarize_repeat(...) -> DataFrame per repeat
- aggregate_over_repeats(...) -> DataFrame with mean and 95% CI
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Basic statistics on cluster sizes
# ---------------------------------------------------------------------------
def safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def safe_var(x: np.ndarray) -> float:
    if x.size <= 1:
        return float("nan")
    return float(np.var(x, ddof=1))


def safe_cv(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    m = np.mean(x)
    if m == 0:
        return float("nan")
    s = np.std(x, ddof=1) if x.size > 1 else 0.0
    return float(s / m)


def gini_coefficient(x: np.ndarray) -> float:
    """
    Gini coefficient of a non-negative vector x.
    Returns NaN if x is empty or all zeros.

    Formula (mean absolute difference):
        G = sum_i sum_j |x_i - x_j| / (2 n sum_i x_i)
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    if np.allclose(x, 0.0):
        return float("nan")
    # Use sorted cumulative method for O(n log n)
    x_sorted = np.sort(x)
    n = x_sorted.size
    cumx = np.cumsum(x_sorted)
    sumx = cumx[-1]
    # Gini via: (n+1 - 2 * sum((n - i) * x_i) / sumx) / n, i from 1..n
    # Equivalent compact form using cumulative sums:
    # sum((2i - n - 1) * x_i) / (n * sumx), i=1..n
    i = np.arange(1, n + 1, dtype=float)
    g = (np.sum((2 * i - n - 1) * x_sorted)) / (n * sumx)
    return float(g)


# ---------------------------------------------------------------------------
# Spatial metrics
# ---------------------------------------------------------------------------
def median_nearest_neighbour(xy: np.ndarray) -> float:
    """
    Median NN distance for 2D points xy[N,2].
    Returns NaN if fewer than 2 points.
    """
    xy = np.asarray(xy, dtype=float)
    n = xy.shape[0]
    if n < 2:
        return float("nan")
    # Compute pairwise squared distances efficiently
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    x2 = np.sum(xy * xy, axis=1, keepdims=True)  # [N,1]
    d2 = x2 + x2.T - 2.0 * (xy @ xy.T)          # [N,N]
    # Remove self-distance
    np.fill_diagonal(d2, np.inf)
    # NN per point, then median
    nn = np.sqrt(np.min(d2, axis=1))
    return float(np.median(nn))


def morisita_index(xy: np.ndarray, width: float, height: float,
                   grid_x: int, grid_y: int) -> float:
    """
    Morisita index I_M for a set of 2D points within [0,width]x[0,height].
    Uses grid_x * grid_y rectangular cells.

    I_M = M * sum_i n_i (n_i - 1) / [ N (N - 1) ],  where M = grid_x * grid_y.
    Returns NaN for N < 2.
    """
    xy = np.asarray(xy, dtype=float)
    N = xy.shape[0]
    if N < 2:
        return float("nan")
    # Histogram points into grid cells
    H, xedges, yedges = np.histogram2d(
        xy[:, 0], xy[:, 1],
        bins=[grid_x, grid_y],
        range=[[0.0, width], [0.0, height]]
    )
    n = H.ravel()
    M = float(grid_x * grid_y)
    num = float(np.sum(n * (n - 1.0)))
    den = float(N * (N - 1.0))
    if den <= 0:
        return float("nan")
    return float(M * num / den)


# ---------------------------------------------------------------------------
# Per-timestep computation
# ---------------------------------------------------------------------------
def compute_timestep_metrics(
    pos: np.ndarray,
    sizes: np.ndarray,
    *,
    merges_this_step: int,
    width: float,
    height: float,
    grid_x: int,
    grid_y: int
) -> Dict[str, float]:
    """
    Compute all metrics for one timestep given positions and sizes.
    """
    pos = np.asarray(pos, dtype=float)
    sizes = np.asarray(sizes, dtype=float)

    num_clusters = int(pos.shape[0])

    # Size-based metrics
    mean_size = safe_mean(sizes)
    var_size = safe_var(sizes)
    cv_size = safe_cv(sizes)
    gini_size = gini_coefficient(sizes)

    # Spatial metrics (skip if 0 or 1 clusters)
    if num_clusters >= 2:
        median_nn = median_nearest_neighbour(pos)
        morisita = morisita_index(pos, width, height, grid_x, grid_y)
    else:
        median_nn = float("nan")
        morisita = float("nan")

    return {
        "num_clusters": num_clusters,
        "merges": int(merges_this_step),
        "mean_size": float(mean_size),
        "var_size": float(var_size),
        "cv_size": float(cv_size),
        "gini_size": float(gini_size),
        "median_nn": float(median_nn),
        "morisita": float(morisita),
    }


# ---------------------------------------------------------------------------
# Repeat-level summary
# ---------------------------------------------------------------------------
def summarize_repeat(
    *,
    id_log: List[np.ndarray],
    pos_log: List[np.ndarray],
    size_log: List[np.ndarray],
    dt: float,
    width: float,
    height: float,
    grid_x: int,
    grid_y: int,
    merges_by_step: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Build a per-timestep DataFrame for one repeat.

    Inputs:
      - id_log, pos_log, size_log: lists aligned by timestep (from model)
      - dt, width, height: domain/time settings
      - grid_x, grid_y: Morisita grid
      - merges_by_step: list of ints per step (same length), optional; if None,
        merges treated as zero.

    Output columns:
      ['step', 'time', 'num_clusters', 'merges', 'mean_size', 'var_size',
       'cv_size', 'gini_size', 'median_nn', 'morisita']
    """
    T = min(len(id_log), len(pos_log), len(size_log))
    data_rows = []
    merges = merges_by_step if merges_by_step is not None else [0] * T

    for t in range(T):
        pos = np.asarray(pos_log[t], dtype=float)
        sizes = np.asarray(size_log[t], dtype=float)

        metrics = compute_timestep_metrics(
            pos, sizes,
            merges_this_step=int(merges[t]) if t < len(merges) else 0,
            width=width, height=height,
            grid_x=grid_x, grid_y=grid_y
        )

        data_rows.append({
            "step": int(t),
            "time": float(t * dt),
            **metrics
        })

    return pd.DataFrame(data_rows)


# ---------------------------------------------------------------------------
# Aggregation across repeats (per condition)
# ---------------------------------------------------------------------------
def _mean_and_ci(x: np.ndarray, axis: int = 0, ci: float = 0.95):
    """
    Return mean, lower, upper (normal approx) along axis.
    """
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x, axis=axis)
    s = np.nanstd(x, axis=axis, ddof=1)
    n = np.sum(~np.isnan(x), axis=axis)
    se = np.where(n > 0, s / np.sqrt(np.maximum(n, 1)), np.nan)
    z = 1.96 if ci == 0.95 else 1.96  # simple; extend if needed
    lo = m - z * se
    hi = m + z * se
    return m, lo, hi, n


def aggregate_over_repeats(
    repeat_frames: List[pd.DataFrame],
    *,
    compute_coag_prob: bool = True
) -> pd.DataFrame:
    """
    Combine per-repeat summary frames (aligned by 'step') into a single
    condition-level DataFrame with mean and 95% CI for each metric.

    For coagulation probability P_merge(t):
        P(t) = sum_r merges_r(t) / sum_r num_clusters_r(t)

    Returns a DataFrame with columns:
        step, time,
        mean_num_clusters, lo_num_clusters, hi_num_clusters,
        mean_mean_size, lo_mean_size, hi_mean_size,
        mean_var_size, lo_var_size, hi_var_size,
        mean_cv_size, lo_cv_size, hi_cv_size,
        mean_gini_size, lo_gini_size, hi_gini_size,
        mean_median_nn, lo_median_nn, hi_median_nn,
        mean_morisita, lo_morisita, hi_morisita,
        (optionally) coag_prob
    """
    if not repeat_frames:
        return pd.DataFrame()

    # Align by 'step'
    # Assume all repeats share the same steps; if not, outer-join and reindex
    steps = sorted(set().union(*[df["step"].unique().tolist() for df in repeat_frames]))
    T = len(steps)

    # Stack each metric across repeats into arrays [R, T]
    metrics = ["num_clusters", "mean_size", "var_size", "cv_size",
               "gini_size", "median_nn", "morisita", "merges"]

    stacks: Dict[str, np.ndarray] = {}
    times = np.zeros(T, dtype=float)

    for m in metrics:
        mat = np.full((len(repeat_frames), T), np.nan, dtype=float)
        for r, df in enumerate(repeat_frames):
            dfr = df.set_index("step").reindex(steps)
            if m == "num_clusters":
                times = dfr["time"].to_numpy(dtype=float)
            mat[r, :] = dfr[m].to_numpy(dtype=float)
        stacks[m] = mat

    out_rows = []
    for j, step in enumerate(steps):
        row = {"step": int(step), "time": float(times[j])}

        # mean & CI for metrics (except coag prob)
        for key in ["num_clusters", "mean_size", "var_size", "cv_size",
                    "gini_size", "median_nn", "morisita"]:
            mean, lo, hi, n = _mean_and_ci(stacks[key][:, j], axis=0, ci=0.95)
            row[f"mean_{key}"] = float(mean)
            row[f"lo_{key}"] = float(lo)
            row[f"hi_{key}"] = float(hi)
            row[f"n_{key}"] = int(n) if not np.isnan(n) else 0

        # coagulation probability at step j
        if compute_coag_prob:
            merges_sum = float(np.nansum(stacks["merges"][:, j]))
            clusters_sum = float(np.nansum(stacks["num_clusters"][:, j]))
            if clusters_sum > 0:
                row["coag_prob"] = merges_sum / clusters_sum
            else:
                row["coag_prob"] = float("nan")

        out_rows.append(row)

    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Convenience: pick early/mid/final timesteps
# ---------------------------------------------------------------------------
def pick_early_mid_final_indices(T: int) -> Tuple[int, int, int]:
    """
    Choose (early, mid, final) indices for a series length T.
    Early ~ 10% into the simulation (at least 0),
    Mid   ~ 50%,
    Final = T-1.
    """
    if T <= 0:
        return (0, 0, 0)
    early = max(0, int(round(0.10 * (T - 1))))
    mid = max(0, int(round(0.50 * (T - 1))))
    final = max(0, T - 1)
    return (early, mid, final)