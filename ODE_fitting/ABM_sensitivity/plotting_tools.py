"""
plotting_tools.py
=================

Improvements:
- Legends show ONLY parameters that vary across conditions (plus mode if modes differ).
- Adds `trim_first` option for time-series (and writes to a separate folder).
- Rounds sweep values used for axes to 12 dp to avoid FP artefacts on tick labels.
"""

from __future__ import annotations
import os
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Metric registry & labelling
# ---------------------------------------------------------------------------
_METRIC_SPECS = {
    "num_clusters": {
        "mean": "mean_num_clusters", "lo": "lo_num_clusters", "hi": "hi_num_clusters",
        "label": "Number of clusters"
    },
    "mean_size": {
        "mean": "mean_mean_size", "lo": "lo_mean_size", "hi": "hi_mean_size",
        "label": "Mean cluster size"
    },
    "var_size": {
        "mean": "mean_var_size", "lo": "lo_var_size", "hi": "hi_var_size",
        "label": "Variance of cluster size"
    },
    "cv_size": {
        "mean": "mean_cv_size", "lo": "lo_cv_size", "hi": "hi_cv_size",
        "label": "Coefficient of variation (size)"
    },
    "gini_size": {
        "mean": "mean_gini_size", "lo": "lo_gini_size", "hi": "hi_gini_size",
        "label": "Gini coefficient (size)"
    },
    "median_nn": {
        "mean": "mean_median_nn", "lo": "lo_median_nn", "hi": "hi_median_nn",
        "label": "Median nearest-neighbour distance"
    },
    "morisita": {
        "mean": "mean_morisita", "lo": "lo_morisita", "hi": "hi_morisita",
        "label": "Morisita index"
    },
    # coag_prob is a point estimate with no CI band
    "coag_prob": {
        "point": "coag_prob",
        "label": "Coagulation probability"
    },
}

_DEFAULT_METRICS = [
    "num_clusters", "mean_size", "var_size", "cv_size",
    "gini_size", "median_nn", "morisita", "coag_prob"
]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _format_number_pretty(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))):
        return ""
    if x == 0:
        return "0"
    ax = abs(x)
    if 1e-3 <= ax < 1e4:
        s = f"{x:.6g}"
    else:
        s = f"{x:.3e}"
    if "e" not in s and "." in s:
        s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s


def _format_number_for_axis(x: float) -> str:
    return _format_number_pretty(x)


def _round12(v: float) -> float:
    return float(np.round(v, 12))


# --------------------------- legend helpers ---------------------------------
def _collect_param_keys(condition_records: List[Dict]) -> List[str]:
    keys = set()
    for rec in condition_records:
        for k in rec.get("params", {}).keys():
            keys.add(k)
    return sorted(keys)


def _distinct_values_for_key(condition_records: List[Dict], key: str):
    vals = []
    for rec in condition_records:
        if key in rec.get("params", {}):
            vals.append(rec["params"][key])
    # Normalise floats
    norm = []
    for v in vals:
        try:
            vv = float(v)
            vv = _round12(vv)
            norm.append(vv)
        except Exception:
            norm.append(v)
    return set(norm)


def _varying_keys(condition_records: List[Dict]) -> List[str]:
    """
    Returns only those parameter keys whose values differ across conditions.
    Special handling for mode: include '__mode__' only if multiple modes present.
    """
    all_keys = _collect_param_keys(condition_records)
    varying = []
    # Check if modes differ
    mode_vals = _distinct_values_for_key(condition_records, "__mode__")
    modes_differ = len(mode_vals) > 1
    for k in all_keys:
        if k == "__mode__":
            if modes_differ:
                varying.append(k)
            continue
        vals = _distinct_values_for_key(condition_records, k)
        if len(vals) > 1:
            varying.append(k)
    return sorted(varying)


def _label_for_record(rec: Dict, var_keys: List[str]) -> str:
    """
    Build compact label: "a = 0.7, p_merge = 0.6" (and optionally "mode = X").
    """
    parts = []
    params = rec.get("params", {})
    for k in var_keys:
        if k == "__mode__":
            parts.append(f"mode = {params.get(k, '')}")
            continue
        if k in params:
            v = params[k]
            if isinstance(v, (int, float)):
                parts.append(f"{k.split('.')[-1]} = {_format_number_pretty(float(v))}")
            else:
                parts.append(f"{k.split('.')[-1]} = {v}")
    return ", ".join(parts) if parts else rec.get("name", "")


# ---------------------------------------------------------------------------
# 1D sweep: mean ± CI at early/mid/final
# ---------------------------------------------------------------------------
def plot_param_1d_ci(
    condition_records: List[Dict],
    param_key: str,
    out_dir: str,
    step_indices: Tuple[int, int, int],
    metrics: Optional[Sequence[str]] = None,
    fig_dpi: int = 160,
):
    if not condition_records:
        return

    metrics = list(metrics or _DEFAULT_METRICS)
    early_idx, mid_idx, final_idx = step_indices

    # Build (x, df, label) tuples
    var_keys = _varying_keys(condition_records)
    series = []
    for rec in condition_records:
        params = rec.get("params", {})
        if param_key not in params:
            continue
        x = _round12(float(params[param_key]))
        label = _label_for_record(rec, var_keys)
        series.append((x, rec["agg_df"], label))
    if not series:
        return

    series.sort(key=lambda kv: kv[0])
    xs = np.array([kv[0] for kv in series], dtype=float)
    frames = [kv[1] for kv in series]
    labels = [kv[2] for kv in series]

    tag_map = {"early": early_idx, "mid": mid_idx, "final": final_idx}
    plot_dir = _ensure_dir(os.path.join(out_dir, "1D"))

    for met in metrics:
        spec = _METRIC_SPECS.get(met)
        if spec is None:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, (tag, tidx) in zip(axes, tag_map.items()):
            ys = np.full_like(xs, np.nan, dtype=float)
            lo = np.full_like(xs, np.nan, dtype=float)
            hi = np.full_like(xs, np.nan, dtype=float)

            for i, df in enumerate(frames):
                if tidx >= len(df):
                    continue
                row = df.iloc[tidx]
                if "point" in spec:
                    ys[i] = float(row.get(spec["point"], np.nan))
                else:
                    ys[i] = float(row.get(spec["mean"], np.nan))
                    lo[i] = float(row.get(spec["lo"], np.nan))
                    hi[i] = float(row.get(spec["hi"], np.nan))

            if "point" in spec:
                ax.plot(xs, ys, marker="o", lw=2, color="#1f77b4")
            else:
                ax.plot(xs, ys, marker="o", lw=2, color="#1f77b4")
                mask = np.isfinite(lo) & np.isfinite(hi)
                if np.any(mask):
                    ax.fill_between(xs[mask], lo[mask], hi[mask],
                                    color="#1f77b4", alpha=0.2, linewidth=0)

            ax.set_title(f"{tag.capitalize()} (step {tidx})")
            ax.set_xlabel(param_key.split(".")[-1])
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(spec["label"])
        # Build a global figure legend that shows what varies across conditions
        fig.legend(labels, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        fig.suptitle(f"1D sweep — {spec['label']}")
        fig.tight_layout(rect=[0, 0.0, 1, 0.93])

        fname = f"1D_{met}.png"
        fig.savefig(os.path.join(plot_dir, fname), dpi=fig_dpi)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 2D sweep: heatmaps at early/mid/final
# ---------------------------------------------------------------------------
def plot_param_2d_heatmaps(
    condition_records: List[Dict],
    param_keys: Tuple[str, str],
    out_dir: str,
    step_indices: Tuple[int, int, int],
    metrics: Optional[Sequence[str]] = None,
    cmap: str = "viridis",
    fig_dpi: int = 160,
):
    if not condition_records:
        return

    metrics = list(metrics or _DEFAULT_METRICS)
    pkey1, pkey2 = param_keys
    early_idx, mid_idx, final_idx = step_indices
    tag_map = {"early": early_idx, "mid": mid_idx, "final": final_idx}

    xs = sorted({_round12(float(rec["params"][pkey1])) for rec in condition_records if pkey1 in rec["params"]})
    ys = sorted({_round12(float(rec["params"][pkey2])) for rec in condition_records if pkey2 in rec["params"]})
    if not xs or not ys:
        return

    x_to_i = {v: i for i, v in enumerate(xs)}
    y_to_j = {v: j for j, v in enumerate(ys)}

    lut = {}
    for rec in condition_records:
        params = rec.get("params", {})
        if pkey1 in params and pkey2 in params:
            xv = _round12(float(params[pkey1])); yv = _round12(float(params[pkey2]))
            lut[(xv, yv)] = rec["agg_df"]

    plot_dir = _ensure_dir(os.path.join(out_dir, "2D_heatmaps"))

    for met in metrics:
        spec = _METRIC_SPECS.get(met)
        if spec is None:
            continue

        for tag, tidx in tag_map.items():
            Z = np.full((len(ys), len(xs)), np.nan, dtype=float)

            for (xv, yv), df in lut.items():
                if tidx >= len(df):
                    continue
                row = df.iloc[tidx]
                if "point" in spec:
                    val = float(row.get(spec["point"], np.nan))
                else:
                    val = float(row.get(spec["mean"], np.nan))
                Z[y_to_j[yv], x_to_i[xv]] = val

            plt.figure(figsize=(8 + 0.25 * len(xs), 6 + 0.25 * len(ys)))
            ax = sns.heatmap(
                Z, cmap=cmap, annot=False, cbar=True, square=False,
                xticklabels=[_format_number_for_axis(v) for v in xs],
                yticklabels=[_format_number_for_axis(v) for v in ys]
            )
            ax.set_xlabel(pkey1.split(".")[-1])
            ax.set_ylabel(pkey2.split(".")[-1])
            ax.set_title(f"{_METRIC_SPECS[met]['label']} — {tag.capitalize()} (step {tidx})")
            plt.tight_layout()
            fname = f"2D_{met}_{tag}.png"
            plt.savefig(os.path.join(plot_dir, fname), dpi=fig_dpi)
            plt.clf(); plt.close("all")


# ---------------------------------------------------------------------------
# Time-series overlays across conditions (small number of conditions)
# ---------------------------------------------------------------------------
def plot_time_series_over_conditions(
    condition_records: List[Dict],
    out_dir: str,
    metrics: Optional[Sequence[str]] = None,
    max_legend: int = 10,
    fig_dpi: int = 160,
    trim_first: bool = False,     # NEW
):
    """
    Overlay time series across conditions. For CI-enabled metrics, plot mean with
    shaded 95% CI; for coag_prob, plot point estimate only.

    If trim_first=True, the first row (earliest time) is removed before plotting.
    """
    if not condition_records:
        return

    metrics = list(metrics or _DEFAULT_METRICS)
    folder = "time_series_no_t0" if trim_first else "time_series"
    plot_dir = _ensure_dir(os.path.join(out_dir, folder))

    # consistent palette
    palette = sns.color_palette("tab10", n_colors=max(10, len(condition_records)))

    var_keys = _varying_keys(condition_records)

    for met in metrics:
        spec = _METRIC_SPECS.get(met)
        if spec is None:
            continue

        plt.figure(figsize=(10, 6))
        for idx, rec in enumerate(condition_records):
            label = _label_for_record(rec, var_keys)
            df = rec["agg_df"]
            if trim_first and len(df) > 1:
                df = df.iloc[1:].reset_index(drop=True)

            colour = palette[idx % len(palette)]

            if "point" in spec:
                if "coag_prob" not in df.columns:
                    continue
                plt.plot(df["time"], df[spec["point"]], lw=2, label=label, color=colour)
            else:
                mean_col, lo_col, hi_col = spec["mean"], spec["lo"], spec["hi"]
                if not set([mean_col, lo_col, hi_col]).issubset(df.columns):
                    continue
                plt.plot(df["time"], df[mean_col], lw=2, label=label, color=colour)
                lo = df[lo_col].to_numpy(dtype=float)
                hi = df[hi_col].to_numpy(dtype=float)
                mask = np.isfinite(lo) & np.isfinite(hi)
                if np.any(mask):
                    plt.fill_between(df["time"].to_numpy(dtype=float)[mask],
                                     lo[mask], hi[mask],
                                     color=colour, alpha=0.18, linewidth=0)

        plt.xlabel("Time (arbitrary units)")
        plt.ylabel(spec["label"])
        plt.title(f"Time series — {spec['label']}" + (" (no t0)" if trim_first else ""))
        if len(condition_records) <= max_legend:
            plt.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"timeseries_{met}.png" if not trim_first else f"timeseries_{met}_no_t0.png"
        plt.savefig(os.path.join(plot_dir, fname), dpi=fig_dpi)
        plt.clf(); plt.close("all")