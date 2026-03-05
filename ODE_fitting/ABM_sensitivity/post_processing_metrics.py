#!/usr/bin/env python3
"""post_process_metrics.py

Post-processing for ABM sensitivity outputs.

What this script does
---------------------
For each run folder inside ABM_sensitivity_results/<run_name>/ it:

1) Reads raw simulation timepoint CSVs:
     simulations/<condition_id>/repeat_XX/t_*.csv

   Each t_*.csv is expected to contain, per cluster/agent at that timestep:
     - an identifier column (ignored)
     - 2D position columns (x and y)
     - a size column

   Column naming varies across projects, so we use robust heuristics to detect
   x/y/size, with optional CLI overrides.

2) Computes five per-timestep metrics per repeat:
     (a) num_clusters            = N
     (b) mean_cluster_size       = mean(size)
     (c) var_cluster_size        = sample variance(size, ddof=1)
     (d) median_nnd              = median nearest-neighbour distance
     (e) clark_evans_R           = Clark–Evans R index (mean NN / expected NN)

   Additionally, it computes clark_evans_z (z-score) and stores it in tables,
   but plots focus on the five metrics above.

   Clark–Evans uses an assumed rectangular domain area A = width*height.
   If width/height are not provided, we estimate them from the first available
   timepoint file (usually t_0000.csv) by taking (max-min) per axis.

3) Saves new tables under:
     <run_dir>/post-processing/tables/<condition_id>/
       summary_repeat_XX.csv
       summary_aggregated.csv

   Aggregation mirrors your existing approach:
     mean_*, lo_*, hi_*, n_* using normal approximation 95% CI.

4) Generates plots ONLY for these five metrics (in this order):
     1. num_clusters
     2. mean_cluster_size
     3. var_cluster_size
     4. median_nnd
     5. clark_evans_R

   Using early/mid/final snapshots of the aggregated time series.

   Plots are saved under:
     <run_dir>/post-processing/plots/<snapshot>/{grid|metrics}/<variant>/

   The plotting style matches the previous extra_plots script:
     - total grid (rows=metrics, cols=params)
     - individual metric figures
     - if exactly 2 varying params: colour by the other + sister plots

Usage
-----
From repo root:
  python -m ABM_sensitivity.post_process_metrics --run <RUN_NAME>

Optional:
  --results-root PATH
  --dt 1.0
  --width W --height H
  --x-col X --y-col Y --size-col SIZE

"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# -----------------------------
# Configuration
# -----------------------------

IGNORE_PARAMS = {"scenario"}
JITTER_SEED = 0
CONTINUOUS_CMAP_NAME = "viridis"

# Metric order requested by user
METRIC_ORDER = [
    "num_clusters",
    "mean_cluster_size",
    "var_cluster_size",
    "median_nnd",
    "clark_evans_R",
]

# -----------------------------
# Helpers: robust parsing of condition id -> parameters
# -----------------------------

_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?([eE]-?\d+)?$")
_P_DECIMAL_RE = re.compile(r"^-?\d+p\d+([eE]-?\d+)?$")


def parse_value(token: str):
    t = token.strip()
    if _P_DECIMAL_RE.fullmatch(t):
        t = t.replace("p", ".")
    if _NUMERIC_RE.fullmatch(t):
        try:
            return float(t)
        except ValueError:
            return token
    return token


def parse_condition_id(condition_id: str) -> Dict[str, object]:
    """Parse condition folder name into ABM parameter dict, ignoring IGNORE_PARAMS."""
    out: Dict[str, object] = {}
    parts = condition_id.split("__")

    for part in parts:
        if "_" not in part:
            if part not in IGNORE_PARAMS:
                out[part] = part
            continue
        key, rest = part.split("_", 1)
        if key in IGNORE_PARAMS:
            continue
        out[key] = parse_value(rest)
    return out


def maybe_to_numeric(s: pd.Series, threshold: float = 0.9) -> pd.Series:
    """Convert to numeric only if most non-null entries convert successfully."""
    if s is None or len(s) == 0:
        return s
    converted = pd.to_numeric(s, errors="coerce")
    non_null = s.notna().sum()
    if non_null == 0:
        return s
    success = converted.notna().sum()
    return converted if (success / non_null) >= threshold else s


# -----------------------------
# Helpers: domain and file parsing
# -----------------------------


def sanitise_for_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_condition_dirs(simulations_dir: Path) -> List[Path]:
    return sorted([p for p in simulations_dir.iterdir() if p.is_dir()])


def list_repeat_dirs(condition_dir: Path) -> List[Path]:
    return sorted([p for p in condition_dir.iterdir() if p.is_dir() and p.name.startswith("repeat_")])


def list_timepoint_files(repeat_dir: Path) -> List[Path]:
    return sorted(repeat_dir.glob("t_*.csv"))


def timestep_from_filename(path: Path) -> Optional[int]:
    m = re.search(r"t_(\d+)\.csv$", path.name)
    return int(m.group(1)) if m else None


def infer_xy_size_columns(df: pd.DataFrame,
                          x_col: Optional[str] = None,
                          y_col: Optional[str] = None,
                          size_col: Optional[str] = None) -> Tuple[str, str, str]:
    """Infer x/y/size column names using heuristics, unless provided."""
    cols = list(df.columns)

    def pick_first(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in cols:
                return c
        return None

    if x_col is None:
        x_col = pick_first(["x", "X", "pos_x", "px", "x_pos", "xcoord", "x_coord"])
    if y_col is None:
        y_col = pick_first(["y", "Y", "pos_y", "py", "y_pos", "ycoord", "y_coord"])
    if size_col is None:
        size_col = pick_first(["size", "Size", "cluster_size", "volume", "area", "radius", "r", "mass"])

    # Fallbacks: choose numeric columns
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    if x_col is None or y_col is None:
        # Try the first two numeric columns
        if len(numeric_cols) >= 2:
            x_col = x_col or numeric_cols[0]
            y_col = y_col or numeric_cols[1]

    if size_col is None:
        # Prefer a remaining numeric column different from x/y; else last numeric
        remaining = [c for c in numeric_cols if c not in {x_col, y_col}]
        size_col = remaining[0] if remaining else (numeric_cols[-1] if numeric_cols else None)

    if x_col is None or y_col is None or size_col is None:
        raise ValueError(f"Could not infer x/y/size columns from: {cols}")

    return x_col, y_col, size_col


def estimate_domain_from_file(tp_file: Path, x_col: str, y_col: str) -> Tuple[float, float, float, float]:
    """Return (minx, maxx, miny, maxy) from a single timepoint file."""
    df = pd.read_csv(tp_file)
    minx, maxx = float(df[x_col].min()), float(df[x_col].max())
    miny, maxy = float(df[y_col].min()), float(df[y_col].max())
    return minx, maxx, miny, maxy


# -----------------------------
# Metrics
# -----------------------------


def sample_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return float("nan")
    return float(np.var(x, ddof=1))


def median_nearest_neighbour(xy: np.ndarray) -> float:
    """Median nearest neighbour distance (uses KDTree if available)."""
    xy = np.asarray(xy, dtype=float)
    n = xy.shape[0]
    if n < 2:
        return float("nan")

    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(xy)
        dists, _ = tree.query(xy, k=2)  # first is 0 (self), second is NN
        nn = dists[:, 1]
        return float(np.median(nn))
    except Exception:
        # Fallback: full pairwise distances (O(n^2))
        x2 = np.sum(xy * xy, axis=1, keepdims=True)
        d2 = x2 + x2.T - 2.0 * (xy @ xy.T)
        np.fill_diagonal(d2, np.inf)
        nn = np.sqrt(np.min(d2, axis=1))
        return float(np.median(nn))


def mean_nearest_neighbour(xy: np.ndarray) -> float:
    """Mean nearest neighbour distance (needed for Clark–Evans)."""
    xy = np.asarray(xy, dtype=float)
    n = xy.shape[0]
    if n < 2:
        return float("nan")

    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(xy)
        dists, _ = tree.query(xy, k=2)
        nn = dists[:, 1]
        return float(np.mean(nn))
    except Exception:
        x2 = np.sum(xy * xy, axis=1, keepdims=True)
        d2 = x2 + x2.T - 2.0 * (xy @ xy.T)
        np.fill_diagonal(d2, np.inf)
        nn = np.sqrt(np.min(d2, axis=1))
        return float(np.mean(nn))


def clark_evans(xy: np.ndarray, width: float, height: float) -> Tuple[float, float]:
    """Return (R, z) for Clark–Evans NN test, using a rectangular domain.

    R = r_o / r_e
      r_o: observed mean NN distance
      r_e: expected mean NN distance under CSR (Poisson) = 0.5 / sqrt(lambda)
    where lambda = N / A, A = width*height

    z = (r_o - r_e) / SE
      SE approximated as 0.26136 / sqrt(N * lambda)

    Returns (nan, nan) for N < 2 or invalid geometry.
    """
    xy = np.asarray(xy, dtype=float)
    N = xy.shape[0]
    if N < 2:
        return float("nan"), float("nan")

    A = float(width) * float(height)
    if A <= 0:
        return float("nan"), float("nan")

    lam = N / A
    if lam <= 0:
        return float("nan"), float("nan")

    ro = mean_nearest_neighbour(xy)
    re = 0.5 / np.sqrt(lam)
    R = ro / re if re > 0 else float("nan")

    # Standard error (classic approximation)
    se = 0.26136 / np.sqrt(N * lam)
    z = (ro - re) / se if se > 0 else float("nan")

    return float(R), float(z)


# -----------------------------
# Per-repeat computation
# -----------------------------


def compute_repeat_timeseries(
    repeat_dir: Path,
    *,
    dt: float,
    width: Optional[float],
    height: Optional[float],
    x_col: Optional[str],
    y_col: Optional[str],
    size_col: Optional[str],
) -> pd.DataFrame:
    """Compute time series of the 5 metrics for one repeat from raw t_*.csv."""
    files = list_timepoint_files(repeat_dir)
    if not files:
        return pd.DataFrame()

    # Read first file to infer columns
    df0 = pd.read_csv(files[0])
    xname, yname, sname = infer_xy_size_columns(df0, x_col=x_col, y_col=y_col, size_col=size_col)

    # Estimate domain if not provided
    if width is None or height is None:
        minx, maxx, miny, maxy = estimate_domain_from_file(files[0], xname, yname)
        est_w = max(1e-12, maxx - minx)
        est_h = max(1e-12, maxy - miny)
        width = float(width) if width is not None else float(est_w)
        height = float(height) if height is not None else float(est_h)

    rows = []
    for f in files:
        step = timestep_from_filename(f)
        if step is None:
            continue
        df = pd.read_csv(f)
        # allow for slight column differences; re-infer if missing
        if xname not in df.columns or yname not in df.columns or sname not in df.columns:
            xname, yname, sname = infer_xy_size_columns(df, x_col=x_col, y_col=y_col, size_col=size_col)

        xy = df[[xname, yname]].to_numpy(dtype=float)
        sizes = df[sname].to_numpy(dtype=float)

        N = int(df.shape[0])
        mean_size = float(np.mean(sizes)) if N > 0 else float("nan")
        var_size = sample_var(sizes)
        med_nnd = median_nearest_neighbour(xy)
        R, z = clark_evans(xy, width=width, height=height)

        rows.append({
            "step": int(step),
            "time": float(step * dt),
            "num_clusters": float(N),
            "mean_cluster_size": mean_size,
            "var_cluster_size": var_size,
            "median_nnd": med_nnd,
            "clark_evans_R": R,
            "clark_evans_z": z,
        })

    out = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    return out


# -----------------------------
# Aggregation across repeats
# -----------------------------


def mean_and_ci(x: np.ndarray, ci: float = 0.95):
    """Return mean, lower, upper (normal approx) and n (non-NaN)."""
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x, ddof=1)
    n = int(np.sum(~np.isnan(x)))
    if n <= 1:
        return float(m), float("nan"), float("nan"), n
    se = s / np.sqrt(n)
    z = 1.96 if ci == 0.95 else 1.96
    lo = m - z * se
    hi = m + z * se
    return float(m), float(lo), float(hi), n


def aggregate_over_repeats(repeat_frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not repeat_frames:
        return pd.DataFrame()

    steps = sorted(set().union(*[set(df["step"].tolist()) for df in repeat_frames if not df.empty]))
    if not steps:
        return pd.DataFrame()

    # Align by step
    metrics = ["num_clusters", "mean_cluster_size", "var_cluster_size", "median_nnd", "clark_evans_R", "clark_evans_z"]
    T = len(steps)
    Rn = len(repeat_frames)

    stacks = {m: np.full((Rn, T), np.nan, dtype=float) for m in metrics}
    times = np.full(T, np.nan, dtype=float)

    for r, df in enumerate(repeat_frames):
        dfr = df.set_index("step").reindex(steps)
        times[:] = dfr["time"].to_numpy(dtype=float)
        for m in metrics:
            stacks[m][r, :] = dfr[m].to_numpy(dtype=float)

    out_rows = []
    for j, step in enumerate(steps):
        row = {"step": int(step), "time": float(times[j])}
        for m in metrics:
            mean, lo, hi, n = mean_and_ci(stacks[m][:, j], ci=0.95)
            row[f"mean_{m}"] = mean
            row[f"lo_{m}"] = lo
            row[f"hi_{m}"] = hi
            row[f"n_{m}"] = n
        out_rows.append(row)

    return pd.DataFrame(out_rows)


# -----------------------------
# Snapshot reduction (early/mid/final)
# -----------------------------


def pick_early_mid_final_indices(T: int) -> Tuple[int, int, int]:
    if T <= 0:
        return (0, 0, 0)
    early = max(0, int(round(0.10 * (T - 1))))
    mid = max(0, int(round(0.50 * (T - 1))))
    final = max(0, T - 1)
    return early, mid, final


def aggregated_snapshots(agg_df: pd.DataFrame, metrics: List[str]) -> Dict[str, Dict[str, float]]:
    """Return early/mid/final values from aggregated dataframe for mean_ columns."""
    T = len(agg_df)
    ie, im, ifin = pick_early_mid_final_indices(T)

    snaps = {}
    for name, idx in [("early", ie), ("mid", im), ("final", ifin)]:
        idx = int(np.clip(idx, 0, T - 1))
        row = agg_df.iloc[idx]
        d = {"snap_step": float(row["step"]), "snap_time": float(row["time"]) }
        for m in metrics:
            d[m] = float(row[m])
        snaps[name] = d
    return snaps


# -----------------------------
# Plotting (re-using style from previous script)
# -----------------------------


def colour_mapping(series: pd.Series, numeric_threshold: float = 0.9):
    name = series.name
    cmap_cont = mpl.colormaps[CONTINUOUS_CMAP_NAME]

    if pd.api.types.is_numeric_dtype(series):
        v = series.to_numpy(dtype=float)
        norm = mpl.colors.Normalize(vmin=np.nanmin(v), vmax=np.nanmax(v))
        colours = cmap_cont(norm(v))
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_cont)
        return colours, mappable, None, name

    numeric = pd.to_numeric(series, errors="coerce")
    non_null = series.notna().sum()
    success = numeric.notna().sum()
    if non_null > 0 and (success / non_null) >= numeric_threshold:
        v = numeric.to_numpy(dtype=float)
        norm = mpl.colors.Normalize(vmin=np.nanmin(v), vmax=np.nanmax(v))
        colours = cmap_cont(norm(v))
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_cont)
        return colours, mappable, None, name

    cats = pd.Categorical(series.astype(str))
    categories = list(cats.categories)

    def _sort_key(x: str):
        try:
            return (0, float(x))
        except Exception:
            return (1, x)

    categories_sorted = sorted(categories, key=_sort_key)
    cat_to_idx = {cat: i for i, cat in enumerate(categories_sorted)}
    n = len(categories_sorted)
    denom = max(n - 1, 1)
    vals = np.array([cat_to_idx[c] / denom for c in cats], dtype=float)
    colours = cmap_cont(vals)

    handles = []
    for cat in categories_sorted:
        colour = cmap_cont(cat_to_idx[cat] / denom)
        handles.append(
            mpl.lines.Line2D([0], [0], marker="o", linestyle="", color=colour,
                             label=str(cat), markersize=6)
        )
    return colours, None, handles, name


def cat_positions(vals: pd.Series) -> Tuple[np.ndarray, List[str]]:
    cats = pd.Categorical(vals.astype(str))
    return cats.codes.astype(float), [str(x) for x in cats.categories]


def add_jitter(x: np.ndarray, scale: float = 0.08) -> np.ndarray:
    rng = np.random.default_rng(JITTER_SEED)
    return x + rng.normal(0.0, scale, size=x.shape)


def make_gridspec_figure(nrows: int, ncols: int, *, add_cbar: bool, figsize: Tuple[float, float]):
    fig = plt.figure(figsize=figsize)
    if add_cbar:
        gs = fig.add_gridspec(
            nrows=nrows, ncols=ncols + 1,
            width_ratios=[1.0] * ncols + [0.06],
            wspace=0.35, hspace=0.50
        )
        axes = np.empty((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j] = fig.add_subplot(gs[i, j])
        cax = fig.add_subplot(gs[:, -1])
        return fig, axes, cax
    else:
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.35, hspace=0.50)
        axes = np.empty((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j] = fig.add_subplot(gs[i, j])
        return fig, axes, None


def save_png(fig: plt.Figure, outpath: Path, dpi: int = 220) -> None:
    fig.savefig(outpath.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]


def variant_dir(base: Path, plot_type: str, variant: str) -> Path:
    d = base / plot_type / variant
    ensure_dir(d)
    return d


def plot_grid_pages(
    df: pd.DataFrame,
    metrics: List[str],
    x_params: List[str],
    outdir: Path,
    prefix: str,
    title_prefix: str,
    *,
    colour_by: Optional[str] = None,
    metrics_per_page: int = 10,
    params_per_page: int = 4
) -> None:
    if df.empty or not metrics or not x_params:
        return

    metric_pages = chunk_list(metrics, metrics_per_page)
    param_pages = chunk_list(x_params, params_per_page)

    colours = mappable = legend_handles = colour_label = None
    if colour_by is not None and colour_by in df.columns:
        colours, mappable, legend_handles, colour_label = colour_mapping(df[colour_by])

    for mp_i, metrics_subset in enumerate(metric_pages):
        for pp_i, params_subset in enumerate(param_pages):
            nrows = len(metrics_subset)
            ncols = len(params_subset)

            fig_w = max(7, 4.8 * ncols + (0.8 if mappable is not None else 0.0))
            fig_h = max(4, 2.7 * nrows)
            fig, axes, cax = make_gridspec_figure(
                nrows, ncols,
                add_cbar=(mappable is not None),
                figsize=(fig_w, fig_h)
            )

            for i, metric in enumerate(metrics_subset):
                y = df[metric].to_numpy(dtype=float)
                for j, xparam in enumerate(params_subset):
                    ax = axes[i, j]
                    xser = df[xparam]
                    if pd.api.types.is_numeric_dtype(xser):
                        x = xser.to_numpy(dtype=float)
                        ax.scatter(x, y, s=18, alpha=0.78, c=(colours if colours is not None else None))
                    else:
                        x, labels = cat_positions(xser)
                        x = add_jitter(x, 0.08)
                        ax.scatter(x, y, s=18, alpha=0.78, c=(colours if colours is not None else None))
                        ax.set_xticks(np.arange(len(labels)))
                        ax.set_xticklabels(labels, rotation=35, ha="right")

                    if j == 0:
                        ax.set_ylabel(metric)
                    if i == nrows - 1:
                        ax.set_xlabel(xparam)
                    ax.grid(True, alpha=0.2)

            fig.suptitle(
                f"{title_prefix}\n(metrics page {mp_i+1}/{len(metric_pages)}, params page {pp_i+1}/{len(param_pages)})",
                y=0.995, fontsize=13
            )

            if mappable is not None and cax is not None:
                cb = fig.colorbar(mappable, cax=cax)
                cb.set_label(colour_label)
            elif legend_handles is not None:
                fig.subplots_adjust(right=0.82)
                fig.legend(
                    handles=legend_handles,
                    title=colour_label,
                    loc="center left",
                    bbox_to_anchor=(0.86, 0.5),
                    frameon=False
                )

            outpath = outdir / f"{prefix}__m{mp_i+1:02d}_p{pp_i+1:02d}"
            save_png(fig, outpath)


def plot_each_metric_pages(
    df: pd.DataFrame,
    metric: str,
    x_params: List[str],
    outdir: Path,
    prefix: str,
    title_prefix: str,
    *,
    colour_by: Optional[str] = None,
    params_per_page: int = 6
) -> None:
    if df.empty or metric not in df.columns or not x_params:
        return

    param_pages = chunk_list(x_params, params_per_page)

    colours = mappable = legend_handles = colour_label = None
    if colour_by is not None and colour_by in df.columns:
        colours, mappable, legend_handles, colour_label = colour_mapping(df[colour_by])

    y = df[metric].to_numpy(dtype=float)

    for pp_i, params_subset in enumerate(param_pages):
        n = len(params_subset)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))

        fig_w = max(7, 4.8 * ncols + (0.8 if mappable is not None else 0.0))
        fig_h = max(4, 3.2 * nrows)

        fig, axes, cax = make_gridspec_figure(
            nrows, ncols,
            add_cbar=(mappable is not None),
            figsize=(fig_w, fig_h)
        )
        axes_flat = axes.ravel()

        for j, xparam in enumerate(params_subset):
            ax = axes_flat[j]
            xser = df[xparam]

            if pd.api.types.is_numeric_dtype(xser):
                x = xser.to_numpy(dtype=float)
                ax.scatter(x, y, s=22, alpha=0.82, c=(colours if colours is not None else None))
            else:
                x, labels = cat_positions(xser)
                x = add_jitter(x, 0.08)
                ax.scatter(x, y, s=22, alpha=0.82, c=(colours if colours is not None else None))
                ax.set_xticks(np.arange(len(labels)))
                ax.set_xticklabels(labels, rotation=35, ha="right")

            ax.set_xlabel(xparam)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} vs {xparam}")
            ax.grid(True, alpha=0.2)

        for k in range(n, len(axes_flat)):
            axes_flat[k].axis("off")

        fig.suptitle(
            f"{title_prefix}\n(params page {pp_i+1}/{len(param_pages)})",
            y=0.995, fontsize=13
        )

        if mappable is not None and cax is not None:
            cb = fig.colorbar(mappable, cax=cax)
            cb.set_label(colour_label)
        elif legend_handles is not None:
            fig.subplots_adjust(right=0.82)
            fig.legend(
                handles=legend_handles,
                title=colour_label,
                loc="center left",
                bbox_to_anchor=(0.86, 0.5),
                frameon=False
            )

        outpath = outdir / f"{prefix}__{sanitise_for_filename(metric)}__p{pp_i+1:02d}"
        save_png(fig, outpath)


# -----------------------------
# Main workflow
# -----------------------------


def process_run(
    run_dir: Path,
    *,
    dt: float,
    width: Optional[float],
    height: Optional[float],
    x_col: Optional[str],
    y_col: Optional[str],
    size_col: Optional[str],
    metrics_per_page: int,
    params_per_page: int,
    per_metric_params_per_page: int,
) -> None:
    sim_dir = run_dir / "simulations"
    if not sim_dir.exists():
        print(f"[WARN] {run_dir.name}: no simulations/ folder. Skipping.")
        return

    post_dir = run_dir / "post-processing"
    tables_root = post_dir / "tables"
    plots_root = post_dir / "plots"
    ensure_dir(tables_root)
    ensure_dir(plots_root)

    # Build snapshot tables (one row per condition) from aggregated processed summaries
    snapshot_rows = {"early": [], "mid": [], "final": []}
    param_cols_union: Set[str] = set()

    condition_dirs = list_condition_dirs(sim_dir)
    if not condition_dirs:
        print(f"[WARN] {run_dir.name}: no condition folders in simulations/.")
        return

    for cond_dir in condition_dirs:
        cond_id = cond_dir.name
        cond_tables_dir = tables_root / cond_id
        ensure_dir(cond_tables_dir)

        # Parse parameters from condition id
        params = parse_condition_id(cond_id)
        param_cols_union.update(params.keys())

        # Compute repeat summaries
        repeat_dirs = list_repeat_dirs(cond_dir)
        repeat_frames: List[pd.DataFrame] = []

        for rd in repeat_dirs:
            rep_name = rd.name
            rep_df = compute_repeat_timeseries(
                rd,
                dt=dt,
                width=width,
                height=height,
                x_col=x_col,
                y_col=y_col,
                size_col=size_col,
            )
            if rep_df.empty:
                continue
            repeat_frames.append(rep_df)
            rep_df.to_csv(cond_tables_dir / f"summary_{rep_name}.csv", index=False)

        if not repeat_frames:
            continue

        agg = aggregate_over_repeats(repeat_frames)
        agg.to_csv(cond_tables_dir / "summary_aggregated.csv", index=False)

        # Reduce to early/mid/final snapshot rows for plotting
        mean_cols = [f"mean_{m}" for m in METRIC_ORDER]
        snaps = aggregated_snapshots(agg, mean_cols)

        for snap_name in ("early", "mid", "final"):
            row = {"condition_id": cond_id}
            row.update(params)
            row.update(snaps[snap_name])
            snapshot_rows[snap_name].append(row)

    # Build snapshot dataframes
    df_snap = {k: pd.DataFrame(v) for k, v in snapshot_rows.items()}
    for k, df in df_snap.items():
        if df.empty:
            print(f"[WARN] {run_dir.name}: snapshot '{k}' has no rows.")
            return
        for c in list(param_cols_union):
            if c in df.columns:
                df[c] = maybe_to_numeric(df[c])

    # Determine varying parameters (excluding scenario)
    param_cols = [c for c in sorted(param_cols_union) if c not in IGNORE_PARAMS]
    varying_params = []
    if param_cols:
        varying_params = [c for c in param_cols if df_snap["final"][c].nunique(dropna=True) > 1]

    print(f"[INFO] {run_dir.name}: varying ABM parameters = {varying_params}")

    # Prepare plotting metric list in the requested order (using mean_ columns)
    metrics_to_plot = [f"mean_{m}" for m in METRIC_ORDER]

    # Output plots under post-processing/plots/<snapshot>/...
    for snap_name in ("early", "mid", "final"):
        snap_base = plots_root / snap_name
        ensure_dir(snap_base)
        df = df_snap[snap_name]

        # Total grid (no colour)
        gdir = variant_dir(snap_base, "grid", "no_colour")
        plot_grid_pages(
            df=df,
            metrics=metrics_to_plot,
            x_params=varying_params if varying_params else param_cols,
            outdir=gdir,
            prefix=f"{snap_name}__grid__no_colour",
            title_prefix=f"{run_dir.name}: {snap_name} snapshot — selected metrics vs varying parameters",
            colour_by=None,
            metrics_per_page=min(metrics_per_page, len(metrics_to_plot)),
            params_per_page=params_per_page,
        )

        # Per-metric figures (no colour)
        mdir = variant_dir(snap_base, "metrics", "no_colour")
        for metric in metrics_to_plot:
            plot_each_metric_pages(
                df=df,
                metric=metric,
                x_params=varying_params if varying_params else param_cols,
                outdir=mdir,
                prefix=f"{snap_name}__metric__no_colour",
                title_prefix=f"{run_dir.name}: {snap_name} — {metric} vs parameters",
                colour_by=None,
                params_per_page=per_metric_params_per_page,
            )

        # If exactly two varying params: colour + sister plots
        if len(varying_params) == 2:
            p1, p2 = varying_params
            v12 = f"x_{sanitise_for_filename(p1)}__colour_{sanitise_for_filename(p2)}"
            v21 = f"x_{sanitise_for_filename(p2)}__colour_{sanitise_for_filename(p1)}"

            gdir12 = variant_dir(snap_base, "grid", v12)
            plot_grid_pages(
                df=df,
                metrics=metrics_to_plot,
                x_params=[p1],
                outdir=gdir12,
                prefix=f"{snap_name}__grid__{v12}",
                title_prefix=f"{run_dir.name}: {snap_name} — selected metrics vs {p1} (colour={p2})",
                colour_by=p2,
                metrics_per_page=min(metrics_per_page, len(metrics_to_plot)),
                params_per_page=1,
            )

            gdir21 = variant_dir(snap_base, "grid", v21)
            plot_grid_pages(
                df=df,
                metrics=metrics_to_plot,
                x_params=[p2],
                outdir=gdir21,
                prefix=f"{snap_name}__grid__{v21}",
                title_prefix=f"{run_dir.name}: {snap_name} — selected metrics vs {p2} (colour={p1})",
                colour_by=p1,
                metrics_per_page=min(metrics_per_page, len(metrics_to_plot)),
                params_per_page=1,
            )

            mdir12 = variant_dir(snap_base, "metrics", v12)
            mdir21 = variant_dir(snap_base, "metrics", v21)
            for metric in metrics_to_plot:
                plot_each_metric_pages(
                    df=df,
                    metric=metric,
                    x_params=[p1],
                    outdir=mdir12,
                    prefix=f"{snap_name}__metric__{v12}",
                    title_prefix=f"{run_dir.name}: {snap_name} — {metric} vs {p1} (colour={p2})",
                    colour_by=p2,
                    params_per_page=1,
                )
                plot_each_metric_pages(
                    df=df,
                    metric=metric,
                    x_params=[p2],
                    outdir=mdir21,
                    prefix=f"{snap_name}__metric__{v21}",
                    title_prefix=f"{run_dir.name}: {snap_name} — {metric} vs {p2} (colour={p1})",
                    colour_by=p1,
                    params_per_page=1,
                )

    # Save the snapshot tables for convenience
    snap_tables_dir = post_dir / "snapshot_tables"
    ensure_dir(snap_tables_dir)
    for snap_name, df in df_snap.items():
        df.to_csv(snap_tables_dir / f"snapshots_{snap_name}.csv", index=False)

    print(f"[DONE] {run_dir.name}: wrote post-processing outputs to {post_dir}")


def main():
    parser = argparse.ArgumentParser(description="Post-process ABM sensitivity runs to compute selected metrics.")
    parser.add_argument("--results-root", type=str, default=None,
                        help="Path to ABM_sensitivity_results. Default: sibling of ABM_sensitivity/.")
    parser.add_argument("--run", type=str, default="all",
                        help="Run folder name to process, or 'all'.")
    parser.add_argument("--dt", type=float, default=1.0,
                        help="Time step size for time = step*dt in outputs.")
    parser.add_argument("--width", type=float, default=None,
                        help="Domain width for Clark–Evans. If omitted, estimated from first timepoint.")
    parser.add_argument("--height", type=float, default=None,
                        help="Domain height for Clark–Evans. If omitted, estimated from first timepoint.")
    parser.add_argument("--x-col", type=str, default=None, help="Override x column name in t_*.csv")
    parser.add_argument("--y-col", type=str, default=None, help="Override y column name in t_*.csv")
    parser.add_argument("--size-col", type=str, default=None, help="Override size column name in t_*.csv")

    parser.add_argument("--metrics-per-page", type=int, default=10,
                        help="How many metric rows per grid page (will clamp to 5).")
    parser.add_argument("--params-per-page", type=int, default=4,
                        help="How many parameter columns per grid page.")
    parser.add_argument("--per-metric-params-per-page", type=int, default=6,
                        help="How many parameter subplots per per-metric page.")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_results_root = script_dir.parent / "ABM_sensitivity_results"
    results_root = Path(args.results_root).resolve() if args.results_root else default_results_root

    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    run_dirs = sorted([p for p in results_root.iterdir() if p.is_dir()])
    if args.run != "all":
        run_dirs = [p for p in run_dirs if p.name == args.run]
        if not run_dirs:
            raise FileNotFoundError(f"No run folder named '{args.run}' in {results_root}")

    for run_dir in run_dirs:
        process_run(
            run_dir,
            dt=float(args.dt),
            width=args.width,
            height=args.height,
            x_col=args.x_col,
            y_col=args.y_col,
            size_col=args.size_col,
            metrics_per_page=min(int(args.metrics_per_page), 5),
            params_per_page=int(args.params_per_page),
            per_metric_params_per_page=int(args.per_metric_params_per_page),
        )


if __name__ == "__main__":
    main()
