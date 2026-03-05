#!/usr/bin/env python3
"""
Extra plots for ABM sensitivity runs (PNG only), using early/mid/final snapshots.

Key features:
  - Parses ABM parameters ONLY from condition folder names
  - Ignores 'scenario' as a parameter
  - Uses early/mid/final snapshot reduction from summary_aggregated.csv time series
  - Saves BOTH:
      * total grid (rows=metrics, cols=params)
      * individual per-metric PNGs (subplots over params)
  - If exactly 2 varying params:
      * colour points by the other param, and produce the sister plot (swap x/colour)
  - Colourbars are placed in a dedicated column (never overlap plots)
  - Outputs organised into subfolders by snapshot / plot-type / variant
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
CONTINUOUS_CMAP_NAME = "viridis"  # good perceptual ordering


# -----------------------------
# Robust numeric coercion (pandas-safe)
# -----------------------------

def maybe_to_numeric(s: pd.Series, threshold: float = 0.9) -> pd.Series:
    """
    Try converting a Series to numeric. Keep numeric version only if most
    non-null values convert successfully. Prevents turning categorical params into NaN.
    """
    if s is None or len(s) == 0:
        return s
    converted = pd.to_numeric(s, errors="coerce")
    non_null = s.notna().sum()
    if non_null == 0:
        return s
    success = converted.notna().sum()
    return converted if (success / non_null) >= threshold else s


# -----------------------------
# Parse condition folder names -> ABM parameters
# -----------------------------

_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?([eE]-?\d+)?$")
_P_DECIMAL_RE = re.compile(r"^-?\d+p\d+([eE]-?\d+)?$")


def parse_value(token: str):
    """Convert folder-encoded values to Python types (e.g. 0p5 -> 0.5)."""
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
    """
    Parse condition folder name into ABM parameter dict, ignoring IGNORE_PARAMS.

    Example:
      scenario_00__mode_singletons_phase2_fit_all__a_0p5__prolif_rate_0p004
    """
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


# -----------------------------
# Snapshot selection (matches summary_tools.pick_early_mid_final_indices)
# -----------------------------

def pick_early_mid_final_indices(T: int) -> Tuple[int, int, int]:
    """Early ~10%, mid ~50%, final = T-1."""
    if T <= 0:
        return (0, 0, 0)
    early = max(0, int(round(0.10 * (T - 1))))
    mid = max(0, int(round(0.50 * (T - 1))))
    final = max(0, T - 1)
    return early, mid, final


# -----------------------------
# Load and reduce per-condition time-series summaries into snapshots
# -----------------------------

def load_condition_aggregated(cond_dir: Path) -> Optional[pd.DataFrame]:
    f = cond_dir / "summary_aggregated.csv"
    if not f.exists():
        return None
    try:
        df = pd.read_csv(f)
    except Exception:
        return None
    if df.empty:
        return None
    return df


def identify_metrics_in_aggregated(df: pd.DataFrame) -> List[str]:
    """Use numeric mean_* columns plus coag_prob if present."""
    mean_cols = [
        c for c in df.columns
        if c.startswith("mean_") and pd.api.types.is_numeric_dtype(df[c])
    ]
    metrics = sorted(mean_cols)
    if "coag_prob" in df.columns and pd.api.types.is_numeric_dtype(df["coag_prob"]):
        metrics.append("coag_prob")
    return metrics


def condition_snapshots(cond_df: pd.DataFrame, metrics: List[str]) -> Dict[str, Dict[str, float]]:
    """Extract early/mid/final snapshot rows."""
    T = len(cond_df)
    ie, im, ifin = pick_early_mid_final_indices(T)

    snapshots: Dict[str, Dict[str, float]] = {}
    for label, idx in [("early", ie), ("mid", im), ("final", ifin)]:
        idx = int(np.clip(idx, 0, T - 1))
        row = cond_df.iloc[idx]

        snap = {
            "step": float(row["step"]) if "step" in cond_df.columns else float(idx),
            "time": float(row["time"]) if "time" in cond_df.columns else float("nan"),
        }
        for m in metrics:
            snap[m] = float(row[m]) if m in cond_df.columns else float("nan")
        snapshots[label] = snap

    return snapshots


# -----------------------------
# Identify varying ABM parameters (ONLY parsed params)
# -----------------------------

def identify_varying_parameters(df: pd.DataFrame, param_cols: List[str]) -> List[str]:
    varying = []
    for c in param_cols:
        if c in df.columns and df[c].nunique(dropna=True) > 1:
            varying.append(c)
    return varying


# -----------------------------
# Plot helpers
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitise_for_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def save_png(fig: plt.Figure, outpath: Path, dpi: int = 220) -> None:
    # Avoid tight_layout warnings; bbox_inches='tight' captures legends safely
    fig.savefig(outpath.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def cat_positions(vals: pd.Series) -> Tuple[np.ndarray, List[str]]:
    cats = pd.Categorical(vals.astype(str))
    return cats.codes.astype(float), [str(x) for x in cats.categories]


def add_jitter(x: np.ndarray, scale: float = 0.08) -> np.ndarray:
    rng = np.random.default_rng(JITTER_SEED)
    return x + rng.normal(0.0, scale, size=x.shape)


def colour_mapping(series: pd.Series, numeric_threshold: float = 0.9):
    """
    Provide sensible colours:
      - numeric (or mostly numeric) => continuous colormap + colourbar
      - categorical => sequential gradient over sorted categories + legend
    """
    name = series.name

    cmap_cont = mpl.colormaps[CONTINUOUS_CMAP_NAME]

    # 1) Numeric
    if pd.api.types.is_numeric_dtype(series):
        v = series.to_numpy(dtype=float)
        norm = mpl.colors.Normalize(vmin=np.nanmin(v), vmax=np.nanmax(v))
        colours = cmap_cont(norm(v))
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_cont)
        return colours, mappable, None, name

    # 2) Mostly numeric
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = series.notna().sum()
    success = numeric.notna().sum()
    if non_null > 0 and (success / non_null) >= numeric_threshold:
        v = numeric.to_numpy(dtype=float)
        norm = mpl.colors.Normalize(vmin=np.nanmin(v), vmax=np.nanmax(v))
        colours = cmap_cont(norm(v))
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_cont)
        return colours, mappable, None, name

    # 3) Categorical: smooth gradient in sorted order (not random)
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
            mpl.lines.Line2D([0], [0], marker="o", linestyle="",
                             color=colour, label=str(cat), markersize=6)
        )
    return colours, None, handles, name


def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]


def variant_dir(base: Path, plot_type: str, variant: str) -> Path:
    """
    base = extra_plots/<snap>/
    plot_type = 'grid' or 'metrics'
    variant = 'no_colour' or 'x_<p1>__colour_<p2>' etc.
    """
    d = base / plot_type / variant
    ensure_dir(d)
    return d


# -----------------------------
# Figure builders with dedicated colourbar column
# -----------------------------

def make_gridspec_figure(nrows: int, ncols: int, *, add_cbar: bool, figsize: Tuple[float, float]):
    """
    Create a figure with an optional extra column reserved for a colourbar.
    Returns (fig, axes[nrows,ncols], cax_or_None).
    """
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
        gs = fig.add_gridspec(
            nrows=nrows, ncols=ncols,
            wspace=0.35, hspace=0.50
        )
        axes = np.empty((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j] = fig.add_subplot(gs[i, j])
        return fig, axes, None


# -----------------------------
# Plotting: total grid (rows=metrics, cols=params)
# -----------------------------

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

                    if is_numeric(xser):
                        x = xser.to_numpy(dtype=float)
                        ax.scatter(x, y, s=18, alpha=0.78,
                                   c=(colours if colours is not None else None))
                    else:
                        x, labels = cat_positions(xser)
                        x = add_jitter(x, 0.08)
                        ax.scatter(x, y, s=18, alpha=0.78,
                                   c=(colours if colours is not None else None))
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
                # Put legend outside, centred vertically; enlarge right margin slightly
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


# -----------------------------
# Plotting: per-metric figures (subplots over params)
# -----------------------------

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
    if df.empty or not metric or metric not in df.columns or not x_params:
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

            if is_numeric(xser):
                x = xser.to_numpy(dtype=float)
                ax.scatter(x, y, s=22, alpha=0.82,
                           c=(colours if colours is not None else None))
            else:
                x, labels = cat_positions(xser)
                x = add_jitter(x, 0.08)
                ax.scatter(x, y, s=22, alpha=0.82,
                           c=(colours if colours is not None else None))
                ax.set_xticks(np.arange(len(labels)))
                ax.set_xticklabels(labels, rotation=35, ha="right")

            ax.set_xlabel(xparam)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} vs {xparam}")
            ax.grid(True, alpha=0.2)

        # hide unused axes
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
# Main per-run processing
# -----------------------------

def make_plots_for_run(
    run_dir: Path,
    metrics_per_page: int,
    params_per_page: int,
    per_metric_params_per_page: int
) -> None:
    summaries_dir = run_dir / "summaries"
    if not summaries_dir.exists():
        print(f"[WARN] {run_dir.name}: no summaries/ folder, skipping.")
        return

    rows_by_snap = {"early": [], "mid": [], "final": []}
    param_cols_union: Set[str] = set()
    metrics_master: Optional[List[str]] = None

    cond_dirs = sorted([p for p in summaries_dir.iterdir() if p.is_dir()])
    if not cond_dirs:
        print(f"[WARN] {run_dir.name}: no condition folders in summaries/.")
        return

    for cond_dir in cond_dirs:
        cond_id = cond_dir.name
        cond_df = load_condition_aggregated(cond_dir)
        if cond_df is None:
            continue

        params = parse_condition_id(cond_id)  # scenario is excluded here
        param_cols_union.update(params.keys())

        if metrics_master is None:
            metrics_master = identify_metrics_in_aggregated(cond_df)
        metrics = metrics_master or []

        snaps = condition_snapshots(cond_df, metrics)

        for snap_name in ("early", "mid", "final"):
            row = {"condition_id": cond_id}
            row.update(params)
            row["snap_step"] = snaps[snap_name]["step"]
            row["snap_time"] = snaps[snap_name]["time"]
            for m in metrics:
                row[m] = snaps[snap_name][m]
            rows_by_snap[snap_name].append(row)

    if metrics_master is None:
        print(f"[WARN] {run_dir.name}: no summary_aggregated.csv found anywhere.")
        return

    df_snap = {k: pd.DataFrame(v) for k, v in rows_by_snap.items()}
    for k, df in df_snap.items():
        if df.empty:
            print(f"[WARN] {run_dir.name}: snapshot '{k}' has no rows.")
            return
        for c in list(param_cols_union):
            if c in df.columns:
                df[c] = maybe_to_numeric(df[c])

    # Ensure scenario excluded even if it sneaks in
    param_cols = [c for c in sorted(param_cols_union) if c not in IGNORE_PARAMS]
    varying_params = identify_varying_parameters(df_snap["final"], param_cols)

    print(f"[INFO] {run_dir.name}: varying ABM parameters = {varying_params}")
    if not varying_params:
        print(f"[INFO] {run_dir.name}: no varying ABM parameters detected.")
        return

    metrics_to_plot = metrics_master

    extra_root = run_dir / "extra_plots"
    ensure_dir(extra_root)

    for snap_name in ("early", "mid", "final"):
        df = df_snap[snap_name]
        snap_base = extra_root / snap_name
        ensure_dir(snap_base)

        # -------- Total grid (no colour) --------
        gdir = variant_dir(snap_base, "grid", "no_colour")
        plot_grid_pages(
            df=df,
            metrics=metrics_to_plot,
            x_params=varying_params,
            outdir=gdir,
            prefix=f"{snap_name}__grid__no_colour",
            title_prefix=f"{run_dir.name}: {snap_name} snapshot — metrics vs varying parameters (scenario ignored)",
            colour_by=None,
            metrics_per_page=metrics_per_page,
            params_per_page=params_per_page
        )

        # -------- Per-metric figures (no colour) --------
        mdir = variant_dir(snap_base, "metrics", "no_colour")
        for metric in metrics_to_plot:
            plot_each_metric_pages(
                df=df,
                metric=metric,
                x_params=varying_params,
                outdir=mdir,
                prefix=f"{snap_name}__metric__no_colour",
                title_prefix=f"{run_dir.name}: {snap_name} — {metric} vs parameters (no colour)",
                colour_by=None,
                params_per_page=per_metric_params_per_page
            )

        # -------- If exactly 2 varying params: colour + sister plots --------
        if len(varying_params) == 2:
            p1, p2 = varying_params
            v12 = f"x_{sanitise_for_filename(p1)}__colour_{sanitise_for_filename(p2)}"
            v21 = f"x_{sanitise_for_filename(p2)}__colour_{sanitise_for_filename(p1)}"

            # Total-grid: x=p1 coloured by p2
            gdir12 = variant_dir(snap_base, "grid", v12)
            plot_grid_pages(
                df=df,
                metrics=metrics_to_plot,
                x_params=[p1],
                outdir=gdir12,
                prefix=f"{snap_name}__grid__{v12}",
                title_prefix=f"{run_dir.name}: {snap_name} — metrics vs {p1} (colour={p2})",
                colour_by=p2,
                metrics_per_page=metrics_per_page,
                params_per_page=1
            )

            # Total-grid: x=p2 coloured by p1 (sister)
            gdir21 = variant_dir(snap_base, "grid", v21)
            plot_grid_pages(
                df=df,
                metrics=metrics_to_plot,
                x_params=[p2],
                outdir=gdir21,
                prefix=f"{snap_name}__grid__{v21}",
                title_prefix=f"{run_dir.name}: {snap_name} — metrics vs {p2} (colour={p1})",
                colour_by=p1,
                metrics_per_page=metrics_per_page,
                params_per_page=1
            )

            # Per-metric: x=p1 colour=p2 and sister
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
                    params_per_page=1
                )
                plot_each_metric_pages(
                    df=df,
                    metric=metric,
                    x_params=[p2],
                    outdir=mdir21,
                    prefix=f"{snap_name}__metric__{v21}",
                    title_prefix=f"{run_dir.name}: {snap_name} — {metric} vs {p2} (colour={p1})",
                    colour_by=p1,
                    params_per_page=1
                )

    print(f"[DONE] {run_dir.name}: saved PNGs to {extra_root}")


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate extra PNG plots (early/mid/final snapshots).")
    parser.add_argument("--results-root", type=str, default=None,
                        help="Path to ABM_sensitivity_results. Default: sibling of ABM_sensitivity/.")
    parser.add_argument("--run", type=str, default="all",
                        help="Run folder name to process, or 'all'.")
    parser.add_argument("--metrics-per-page", type=int, default=10,
                        help="How many metric rows per grid figure page.")
    parser.add_argument("--params-per-page", type=int, default=4,
                        help="How many parameter columns per grid figure page.")
    parser.add_argument("--per-metric-params-per-page", type=int, default=6,
                        help="How many parameter subplots per per-metric figure page.")
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
        make_plots_for_run(
            run_dir,
            metrics_per_page=args.metrics_per_page,
            params_per_page=args.params_per_page,
            per_metric_params_per_page=args.per_metric_params_per_page
        )


if __name__ == "__main__":
    main()