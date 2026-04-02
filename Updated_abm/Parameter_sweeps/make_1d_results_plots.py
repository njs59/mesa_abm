#!/usr/bin/env python3
"""
make_1d_results_plots.py
------------------------
Final-time CI plots for **1-parameter** sweeps with neat x-axes and a
2×2 "core metrics" panel:
  [Number of clusters, Mean cluster size, Variance of cluster size, Median NND]

Reads one sweep *run* written by your parameter_sweep pipeline:
  Parameter_sweeps_results/<run_timestamp>/
    summaries/<condition>/summary_repeat_00.csv, ...
    summaries/<condition>/summary_aggregated.csv (optional)

Per condition (scenario):
  • load all summary_repeat_*.csv
  • take the **final row** (largest 'step') from each repeat
  • compute mean and **95% CI** (mean ± 1.96·sd/√n)

Outputs:
  - One line+CI plot per metric vs the swept parameter
  - A collective grid (all metrics)
  - A compact 2×2 "core metrics" grid:
      [num_clusters, mean_size, var_size, median_nn]

Saved to:
  Parameter_sweeps_results/<run>/plots_1d_final/
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Parse condition ids (fix: split at **last** underscore, and parse 0p1 -> 0.1)
# -----------------------------
IGNORE_PARAMS = {"scenario"}

import re
_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?([eE]-?\d+)?$")        # e.g., 1, 0.1, 1e-3
_P_DECIMAL_RE = re.compile(r"^-?\d+p\d+([eE]-?\d+)?$")          # e.g., 0p1 -> 0.1


def _parse_value(token: str):
    """Convert folder-encoded values to Python types (e.g. 0p1 -> 0.1)."""
    t = token.strip()
    if _P_DECIMAL_RE.fullmatch(t):            # 0p1, 1p0, etc.
        t = t.replace("p", ".")
    if _NUMERIC_RE.fullmatch(t):              # numeric after optional 'p' -> '.'
        try:
            return float(t)
        except ValueError:
            return token
    return token


def parse_condition_id(condition_id: str) -> Dict[str, object]:
    """
    Extract parameters from a condition id like:
      scenario_00__mode_singletons_phase2_fit_all__p_merge_0p1
    We now split each part at the **last** underscore, so:
      key='p_merge', value='0p1'  ->  0.1
    """
    out: Dict[str, object] = {}
    parts = condition_id.split("__")
    for part in parts:
        if "_" not in part:
            if part not in IGNORE_PARAMS:
                out[part] = part
            continue
        # split at the **last** underscore to separate key vs numeric token
        key, rest = part.rsplit("_", 1)
        if key in IGNORE_PARAMS:
            continue
        out[key] = _parse_value(rest)
    return out


# -----------------------------
# IO helpers
# -----------------------------
def find_run_dir(results_root: Path, run: Optional[str]) -> Path:
    if run in (None, "latest", "LAST", "Latest"):
        runs = sorted([p for p in results_root.iterdir() if p.is_dir()])
        if not runs:
            raise FileNotFoundError(f"No runs found under {results_root}")
        return runs[-1]
    rd = results_root / run
    if not rd.exists():
        raise FileNotFoundError(f"Run folder not found: {rd}")
    return rd


# -----------------------------
# Stats helpers
# -----------------------------
def ci95(mean: float, sd: float, n: int) -> Tuple[float, float]:
    if n <= 1 or not np.isfinite(sd):
        return (np.nan, np.nan)
    half = 1.96 * (sd / math.sqrt(n))
    return (mean - half, mean + half)


# -----------------------------
# Core extraction
# -----------------------------
def load_final_rows_for_condition(cond_dir: Path) -> pd.DataFrame:
    """
    Load final row from each repeat summary in cond_dir/summaries/<cond>/.
    Returns a DataFrame with one row per repeat (final-time values).
    """
    files = sorted(cond_dir.glob("summary_repeat_*.csv"))
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty:
            continue
        if "step" in df.columns:
            df = df.sort_values("step")
        row = df.iloc[-1].to_dict()
        row["__repeat_file__"] = f.name
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------
# Pretty axis & label helpers
# -----------------------------
# EXACTLY the four metrics you asked for:
CORE4 = [
    ("num_clusters", "Number of clusters"),
    ("mean_size",    "Mean cluster size"),
    ("var_size",     "Variance of cluster size"),
    ("median_nn",    "Median NND"),
]


def _format_float_compact(x: float) -> str:
    """
    A neat float formatter:
      - up to ~6 decimals with trailing zeros removed
      - scientific notation only for very small/large magnitudes
    """
    ax = abs(x)
    if (ax > 0 and ax < 1e-3) or ax >= 1e4:
        return f"{x:.3e}"
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s not in {"", "-"} else "0"


def prepare_x_axis(xvals: np.ndarray, max_ticks: int = 12):
    """
    Return:
      x_plot      : numeric positions used for plotting
      tick_pos    : tick positions
      tick_labels : tick labels
      is_numeric  : whether original xvals were numeric
    """
    is_numeric = np.issubdtype(np.asarray(xvals).dtype, np.number)
    if is_numeric:
        xv = np.asarray(xvals, dtype=float)
        x_unique = np.unique(xv)
        n = len(x_unique)
        if n <= max_ticks:
            tick_pos = x_unique
        else:
            step = int(math.ceil(n / max_ticks))
            tick_pos = x_unique[::step]
            if tick_pos[-1] != x_unique[-1]:
                tick_pos = np.append(tick_pos, x_unique[-1])
        tick_labels = [_format_float_compact(v) for v in tick_pos]
        x_plot = xv
    else:
        vals = pd.Index(pd.Categorical(xvals.astype(str)).categories).tolist()
        val_to_idx = {v: i for i, v in enumerate(vals)}
        x_plot = np.array([val_to_idx[str(v)] for v in xvals], dtype=float)
        tick_pos = np.arange(len(vals), dtype=float)
        tick_labels = vals
    return x_plot, tick_pos, tick_labels, is_numeric


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def pretty_metric_name(metric: str) -> str:
    for key, disp in CORE4:
        if metric == key:
            return disp
    return metric.replace("_", " ").capitalize()


# -----------------------------
# Plotting primitives
# -----------------------------
def line_with_ci(ax, x, y, ylo, yhi, *, color=None, label=None):
    ax.plot(x, y, color=color, lw=2.0, label=label)
    ax.fill_between(x, ylo, yhi, color=color, alpha=0.25, linewidth=0)
    ax.grid(alpha=0.25)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Final-time CI plots for 1-parameter sweeps (neat x-axes + core 2x2 metrics)"
    )
    ap.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Path to Parameter_sweeps_results (default: sibling of this file).",
    )
    ap.add_argument("--run", type=str, default="latest", help="Run folder name or 'latest' (default)")
    ap.add_argument(
        "--param-key",
        type=str,
        default=None,
        help="Explicit parameter key for x-axis (auto-detect if omitted)",
    )
    ap.add_argument(
        "--outdir-name",
        type=str,
        default="plots_1d_final",
        help="Name of output subfolder under run dir",
    )
    ap.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=None,
        help="Restrict to these metrics (defaults to all found).",
    )
    ap.add_argument("--dpi", type=int, default=260)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_results_root = script_dir.parent / "Parameter_sweeps_results"
    results_root = Path(args.results_root) if args.results_root else default_results_root

    run_dir = find_run_dir(results_root, args.run)
    summaries_dir = run_dir / "summaries"
    if not summaries_dir.exists():
        raise FileNotFoundError(f"No summaries/ folder under {run_dir}")

    # Collect final rows across all conditions
    records = []
    for cond_dir in sorted([p for p in summaries_dir.iterdir() if p.is_dir()]):
        cond_id = cond_dir.name
        params = parse_condition_id(cond_id)
        df_fin = load_final_rows_for_condition(cond_dir)
        if df_fin.empty:
            continue
        metrics_all = [c for c in df_fin.columns if c not in {"step", "time", "__repeat_file__"}]
        num_cols = [c for c in metrics_all if pd.api.types.is_numeric_dtype(df_fin[c])]
        metrics_all = num_cols
        n = len(df_fin)
        rec = {"condition_id": cond_id, **params, "__n__": n}
        for m in metrics_all:
            vals = df_fin[m].to_numpy(dtype=float)
            mu = float(np.nanmean(vals))
            sd = float(np.nanstd(vals, ddof=1)) if n > 1 else float("nan")
            lo, hi = ci95(mu, sd, n)
            rec[f"mean::{m}"] = mu
            rec[f"ci_lo::{m}"] = lo
            rec[f"ci_hi::{m}"] = hi
            rec[f"sd::{m}"] = sd
        records.append(rec)

    if not records:
        raise RuntimeError("No final-time rows found in any condition.")

    df = pd.DataFrame(records)

    # X-axis parameter
    param_cols = [c for c in df.columns if c not in df.filter(like="::").columns and c not in {"condition_id", "__n__"}]
    varying = [c for c in param_cols if df[c].nunique(dropna=True) > 1]
    if args.param_key:
        xkey = args.param_key
        if xkey not in df.columns:
            raise ValueError(f"param-key '{xkey}' not found among parsed params: {param_cols}")
    else:
        if len(varying) != 1:
            raise ValueError(
                f"Expected exactly 1 varying parameter, found {len(varying)}: {varying}.\n"
                f"Pass --param-key to choose one."
            )
        xkey = varying[0]

    # Metrics to plot (for the “individual” and “all metrics” pages)
    all_metrics = sorted({c.split("::", 1)[1] for c in df.columns if c.startswith("mean::")})
    if args.metrics:
        metrics = [m for m in args.metrics if f"mean::{m}" in df.columns]
        if not metrics:
            raise ValueError(f"None of the requested metrics are available. Available: {all_metrics}")
    else:
        metrics = all_metrics

    # Sort by the x parameter for consistent ordering
    df = df.sort_values(xkey)
    x_raw = df[xkey].to_numpy()

    # Prepare x-axis once (positions + ticks + labels)
    x_plot, x_ticks, x_labels, is_numeric = prepare_x_axis(x_raw, max_ticks=12)

    out_root = run_dir / args.outdir_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Export tidy CSV
    tidy_rows = []
    for m in metrics:
        for _, r in df.iterrows():
            tidy_rows.append(
                {
                    "param": xkey,
                    "param_value": r[xkey],
                    "metric": m,
                    "mean": r[f"mean::{m}"],
                    "ci_lo": r[f"ci_lo::{m}"],
                    "ci_hi": r[f"ci_hi::{m}"],
                    "sd": r[f"sd::{m}"],
                    "n": r["__n__"],
                    "condition_id": r["condition_id"],
                }
            )
    pd.DataFrame(tidy_rows).to_csv(out_root / "final_stats_by_param.csv", index=False)

    # 1) Individual line+CI per metric
    for m in metrics:
        y  = df[f"mean::{m}"].to_numpy(dtype=float)
        ylo = df[f"ci_lo::{m}"].to_numpy(dtype=float)
        yhi = df[f"ci_hi::{m}"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        line_with_ci(ax, x_plot, y, ylo, yhi, color="C0")

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=0 if is_numeric else 30, ha="right" if not is_numeric else "center")
        ax.set_xlabel(xkey)
        ax.set_ylabel(pretty_metric_name(m))
        ax.set_title(f"Final-time {pretty_metric_name(m)} vs {xkey}")
        fig.tight_layout()
        fig.savefig(out_root / f"final_{m}_vs_{sanitize_filename(xkey)}.png", dpi=args.dpi)
        plt.close(fig)

    # 2) Collective grid for **all** metrics
    n = len(metrics)
    ncols = 2 if n >= 2 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.6 * nrows), squeeze=False)
    for i, m in enumerate(metrics):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        y  = df[f"mean::{m}"].to_numpy(dtype=float)
        ylo = df[f"ci_lo::{m}"].to_numpy(dtype=float)
        yhi = df[f"ci_hi::{m}"].to_numpy(dtype=float)
        line_with_ci(ax, x_plot, y, ylo, yhi, color="C0")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=0 if is_numeric else 30, ha="right" if not is_numeric else "center")
        ax.set_xlabel(xkey)
        ax.set_ylabel(pretty_metric_name(m))
        ax.set_title(pretty_metric_name(m))
    # Hide unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.suptitle(f"Final-time metrics vs {xkey}", y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_root / f"final_all_metrics_vs_{sanitize_filename(xkey)}.png", dpi=args.dpi)
    plt.close(fig)

    # 3) Fixed 2x2 **core metrics** (always these four, regardless of --metrics)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    axes = axes.ravel()
    for i, (m, disp) in enumerate(CORE4):
        ax = axes[i]
        if f"mean::{m}" in df.columns:
            y  = df[f"mean::{m}"].to_numpy(dtype=float)
            ylo = df[f"ci_lo::{m}"].to_numpy(dtype=float)
            yhi = df[f"ci_hi::{m}"].to_numpy(dtype=float)
            line_with_ci(ax, x_plot, y, ylo, yhi, color="C0")
        else:
            ax.text(0.5, 0.5, f"'{m}' not found", ha="center", va="center", alpha=0.6)
            ax.grid(alpha=0.25)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=0 if is_numeric else 30, ha="right" if not is_numeric else "center")
        ax.set_xlabel(xkey)
        ax.set_ylabel(disp)
        ax.set_title(disp)
    fig.suptitle(f"Final-time core metrics vs {xkey}", y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_root / f"final_core_metrics_vs_{sanitize_filename(xkey)}.png", dpi=args.dpi)
    plt.close(fig)

    print(f"Saved plots and CSV to: {out_root}")


if __name__ == "__main__":
    main()