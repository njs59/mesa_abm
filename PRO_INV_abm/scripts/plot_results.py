#!/usr/bin/env python3
"""
Plot ABM results created by scripts/run_movement_conditions.py.

Expected:
  results/
    summary_repeats.csv
    summary_conditions.csv           (optional; can be re-derived)
    trajectories/
      <condition>/
        repeat_000.csv
        repeat_001.csv
        ...

Outputs:
  results/plots/
    summary_<metric>.png
    summary_all_metrics.png
    phase_percentages_<condition>.png
    phase_percentages_all_conditions.png
    size_dists_by_timepoint/
      step_000.png
      step_001.png
      ...
    (optional) size_dists_by_timepoint.gif
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Paths
# -----------------------------
def project_root() -> Path:
    # scripts/ is alongside abm/; root is parent of scripts/
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------
# CI utilities
# -----------------------------
def t_critical_975(df: int) -> float:
    """Return t_{0.975, df}. Uses SciPy if available, else normal approx 1.96."""
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except Exception:
        return 1.96


def mean_ci_95(values: np.ndarray) -> Tuple[float, float, float]:
    """Mean and 95% CI (t-based when possible). Ignores NaNs."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    m = float(np.mean(v))
    if n == 1:
        return (m, np.nan, np.nan)
    se = float(np.std(v, ddof=1) / math.sqrt(n))
    tcrit = t_critical_975(n - 1)
    return (m, m - tcrit * se, m + tcrit * se)


def mean_ci_95_over_repeats(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given mat shape (n_repeats, T), compute mean and 95% CI per timepoint.
    Ignores NaNs per timepoint.
    """
    mat = np.asarray(mat, dtype=float)
    T = mat.shape[1]
    mean = np.full(T, np.nan, dtype=float)
    lo = np.full(T, np.nan, dtype=float)
    hi = np.full(T, np.nan, dtype=float)

    for t in range(T):
        v = mat[:, t]
        v = v[np.isfinite(v)]
        n = len(v)
        if n == 0:
            continue
        m = float(np.mean(v))
        mean[t] = m
        if n == 1:
            continue
        se = float(np.std(v, ddof=1) / np.sqrt(n))
        tcrit = t_critical_975(n - 1)
        lo[t] = m - tcrit * se
        hi[t] = m + tcrit * se

    return mean, lo, hi


# -----------------------------
# Summary stats aggregation + plotting
# -----------------------------
def build_condition_summary(df_rep: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for cond, g in df_rep.groupby("condition"):
        out = {
            "condition": cond,
            "phenotype": str(g["phenotype"].iloc[0]) if "phenotype" in g else "",
            "phase2_only": bool(g["phase2_only"].iloc[0]) if "phase2_only" in g else False,
            "n_repeats": int(len(g)),
        }
        for m in metrics:
            mean, lo, hi = mean_ci_95(g[m].to_numpy())
            out[f"{m}_mean"] = mean
            out[f"{m}_ci95_lo"] = lo
            out[f"{m}_ci95_hi"] = hi
        rows.append(out)
    return pd.DataFrame(rows)


def plot_metric_bar(df_cond: pd.DataFrame, metric: str, outpath: Path):
    m = df_cond[f"{metric}_mean"].to_numpy()
    lo = df_cond[f"{metric}_ci95_lo"].to_numpy()
    hi = df_cond[f"{metric}_ci95_hi"].to_numpy()
    x = np.arange(len(df_cond))

    yerr = np.vstack([m - lo, hi - m])
    yerr = np.where(np.isfinite(yerr), yerr, 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, m, yerr=yerr, capsize=6, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df_cond["condition"].tolist(), rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric.replace('_', ' ')} (mean ± 95% CI)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_all_metrics_grid(df_cond: pd.DataFrame, metrics: List[str], outpath: Path, ncols: int = 3):
    n = len(metrics)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)

    x = np.arange(len(df_cond))
    labels = df_cond["condition"].tolist()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        m = df_cond[f"{metric}_mean"].to_numpy()
        lo = df_cond[f"{metric}_ci95_lo"].to_numpy()
        hi = df_cond[f"{metric}_ci95_hi"].to_numpy()
        yerr = np.vstack([m - lo, hi - m])
        yerr = np.where(np.isfinite(yerr), yerr, 0.0)

        ax.bar(x, m, yerr=yerr, capsize=5, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Condition summary statistics (mean ± 95% CI)", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Trajectories: discovery + histogram accumulation
# -----------------------------
def find_trajectory_files(traj_root: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    if not traj_root.exists():
        return out
    for cond_dir in sorted([p for p in traj_root.iterdir() if p.is_dir()]):
        files = sorted(cond_dir.glob("repeat_*.csv"))
        if files:
            out[cond_dir.name] = files
    return out


def scan_max_step_and_size(files: List[Path], max_files: int | None = None) -> Tuple[int, int]:
    max_step = 0
    max_size = 1
    files_to_scan = files if (max_files is None) else files[:max_files]
    for f in files_to_scan:
        df = pd.read_csv(f, usecols=["step", "size"])
        if not df.empty:
            max_step = max(max_step, int(df["step"].max()))
            max_size = max(max_size, int(df["size"].max()))
    return max_step, max_size


def infer_dt_minutes_from_trajectory(csv_file: Path) -> float | None:
    """
    Infer dt (minutes) from columns step and time_min.
    Returns None if cannot infer robustly.
    """
    try:
        df = pd.read_csv(csv_file, usecols=["step", "time_min"])
        df = df[df["step"] > 0]
        if df.empty:
            return None
        dt = np.median(df["time_min"].to_numpy(dtype=float) / df["step"].to_numpy(dtype=float))
        return float(dt) if np.isfinite(dt) and dt > 0 else None
    except Exception:
        return None


def accumulate_size_histograms(
    files: List[Path],
    max_step: int,
    max_size: int,
    size_cap: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accumulate counts[step, size] across files.
    counts shape: (max_step+1, n_bins), where bins represent integer sizes 1..n_bins.
    If size_cap is set, sizes above cap are clamped to the last bin (cap+).
    """
    if size_cap is not None:
        max_size = min(max_size, int(size_cap))

    n_bins = max_size
    counts = np.zeros((max_step + 1, n_bins), dtype=np.int64)

    for f in files:
        df = pd.read_csv(f, usecols=["step", "size"])
        if df.empty:
            continue

        steps = df["step"].to_numpy(dtype=int)
        sizes = df["size"].to_numpy(dtype=int)

        if size_cap is not None:
            sizes = np.clip(sizes, 1, max_size)

        sidx = sizes - 1
        mask = (steps >= 0) & (steps <= max_step) & (sidx >= 0) & (sidx < n_bins)
        steps = steps[mask]
        sidx = sidx[mask]
        np.add.at(counts, (steps, sidx), 1)

    bin_centres = np.arange(1, n_bins + 1, dtype=int)
    return counts, bin_centres


def normalise_row(row: np.ndarray) -> np.ndarray:
    row = row.astype(float)
    s = row.sum()
    return (row / s) if s > 0 else row


# -----------------------------
# Size distributions BY TIMEPOINT (all conditions on each plot)
# -----------------------------
def plot_size_dists_by_timepoint(
    counts_by_cond: Dict[str, np.ndarray],
    bin_centres: np.ndarray,
    out_dir: Path,
    x_is_time: bool,
    x_values: np.ndarray,
    log_y: bool = False,
    selected_steps: List[int] | None = None,
    colours: Dict[str, str] | None = None,
):
    out_dir = ensure_dir(out_dir)

    conds = list(counts_by_cond.keys())
    T = next(iter(counts_by_cond.values())).shape[0]

    # Validate shapes
    for c in conds:
        if counts_by_cond[c].shape[0] != T or counts_by_cond[c].shape[1] != len(bin_centres):
            raise ValueError("All conditions must share identical (steps, bins) shapes.")

    if selected_steps is None:
        steps = list(range(T))
    else:
        steps = [s for s in selected_steps if 0 <= s < T]

    if colours is None:
        colours = {
            "prolif_phase2_only": "#6a3d9a",
            "invasive_phase2_only": "#33a02c",
            "prolif_two_phase": "#1f78b4",
            "invasive_two_phase": "#e31a1c",
        }
        # fallback for any unexpected condition names
        palette = ["#6a3d9a", "#33a02c", "#1f78b4", "#e31a1c", "#ff7f00", "#b15928"]
        for i, c in enumerate(conds):
            colours.setdefault(c, palette[i % len(palette)])

    for step in steps:
        fig, ax = plt.subplots(figsize=(10, 5))

        for c in conds:
            row = normalise_row(counts_by_cond[c][step])
            ax.plot(bin_centres, row, lw=2.2, label=c, color=colours.get(c))

        ax.set_xlabel("Cluster size (cells)")
        ax.set_ylabel("Probability")
        if x_is_time:
            ax.set_title(f"Cluster size distribution — t = {x_values[step]:.2f} min (step {step})")
        else:
            ax.set_title(f"Cluster size distribution — step {step}")

        if log_y:
            ax.set_yscale("log")
            ax.set_ylabel("Probability (log scale)")

        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"step_{step:03d}.png", dpi=250)
        plt.close(fig)


def make_gif_from_pngs(png_dir: Path, out_gif: Path, fps: int = 8):
    """Optional helper to create a GIF from the step_###.png files."""
    try:
        from PIL import Image
    except Exception:
        print("Pillow not installed; skipping GIF creation.")
        return

    files = sorted(png_dir.glob("step_*.png"))
    if not files:
        print("No PNGs found for GIF creation.")
        return

    frames = [Image.open(f).convert("RGB") for f in files]
    duration_ms = int(1000 / max(1, fps))
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved GIF: {out_gif}")


# -----------------------------
# Movement phase occupancy (% in phase 1/2 over time)
# -----------------------------
def phase_fraction_from_file(path: Path, max_step: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    For one repeat file, return arrays (p1, p2) of length max_step+1.
    p1[t] = fraction of clusters in phase 1 at step t, p2 similarly.
    """
    df = pd.read_csv(path, usecols=["step", "movement_phase"])
    if df.empty:
        p1 = np.full(max_step + 1, np.nan, dtype=float)
        p2 = np.full(max_step + 1, np.nan, dtype=float)
        return p1, p2

    totals = df.groupby("step").size()
    phase1 = df[df["movement_phase"] == 1].groupby("step").size()
    phase2 = df[df["movement_phase"] == 2].groupby("step").size()

    p1 = np.full(max_step + 1, np.nan, dtype=float)
    p2 = np.full(max_step + 1, np.nan, dtype=float)

    for step, tot in totals.items():
        s = int(step)
        if 0 <= s <= max_step and tot > 0:
            n1 = int(phase1.get(step, 0))
            n2 = int(phase2.get(step, 0))
            p1[s] = n1 / tot
            p2[s] = n2 / tot

    return p1, p2


def plot_phase_percentages_for_condition(
    condition: str,
    x_axis: np.ndarray,
    x_label: str,
    p1_mean: np.ndarray, p1_lo: np.ndarray, p1_hi: np.ndarray,
    p2_mean: np.ndarray, p2_lo: np.ndarray, p2_hi: np.ndarray,
    outpath: Path,
):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x_axis, 100 * p1_mean, lw=2.2, label="Phase 1")
    ax.fill_between(x_axis, 100 * p1_lo, 100 * p1_hi, alpha=0.2)

    ax.plot(x_axis, 100 * p2_mean, lw=2.2, label="Phase 2")
    ax.fill_between(x_axis, 100 * p2_lo, 100 * p2_hi, alpha=0.2)

    ax.set_ylim(0, 100)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Clusters (%)")
    ax.set_title(f"Percentage of clusters in each movement phase — {condition}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_phase_percentages_all_conditions(
    conditions: List[str],
    x_axis: np.ndarray,
    x_label: str,
    phase_stats: Dict[str, Dict[str, np.ndarray]],
    outpath: Path,
):
    n = len(conditions)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 7))
    axes = np.array(axes).reshape(-1)

    for i, cond in enumerate(conditions):
        ax = axes[i]
        st = phase_stats[cond]

        ax.plot(x_axis, 100 * st["p1_mean"], lw=2.0, label="Phase 1")
        ax.fill_between(x_axis, 100 * st["p1_lo"], 100 * st["p1_hi"], alpha=0.2)

        ax.plot(x_axis, 100 * st["p2_mean"], lw=2.0, label="Phase 2")
        ax.fill_between(x_axis, 100 * st["p2_lo"], 100 * st["p2_hi"], alpha=0.2)

        ax.set_ylim(0, 100)
        ax.set_title(cond)
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=9)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Movement phase occupancy over time (mean ± 95% CI)", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot ABM results into results/plots/.")

    # Summary plots
    parser.add_argument("--skip_summaries", action="store_true", help="Skip summary-statistic plots.")

    # Size distribution plots (timepoint -> all conditions)
    parser.add_argument("--size_cap", type=int, default=None,
                        help="Optional cap for size bins; sizes above cap go into last bin.")
    parser.add_argument("--scan_limit", type=int, default=None,
                        help="Only scan first N trajectory files per condition to estimate max step/size faster.")
    parser.add_argument("--selected_steps", type=str, default=None,
                        help="Optional comma-separated list of steps to plot (e.g. '0,10,50,145'). Default: all steps.")
    parser.add_argument("--log_y", action="store_true", help="Use log scale for y-axis in size distributions.")
    parser.add_argument("--make_gif", action="store_true", help="Also create an animated GIF from timepoint PNGs.")
    parser.add_argument("--gif_fps", type=int, default=8, help="Frames per second for GIF (if enabled).")

    # Phase percentage plots
    parser.add_argument("--skip_phases", action="store_true", help="Skip movement-phase percentage plots.")
    parser.add_argument("--dt", type=float, default=None,
                        help="Optional dt (minutes) to use for time axis in phase plots and titles. "
                             "If omitted, dt is inferred from trajectories if possible.")

    args = parser.parse_args()

    root = project_root()
    results_dir = root / "results"
    plots_dir = ensure_dir(results_dir / "plots")
    traj_root = results_dir / "trajectories"

    rep_path = results_dir / "summary_repeats.csv"
    if not rep_path.exists():
        raise FileNotFoundError(f"Cannot find {rep_path}. Run the simulation script first.")

    df_rep = pd.read_csv(rep_path)

    metrics = [
        "n_clusters",
        "mean_size",
        "var_size",
        "median_nnd",
        "clark_evans_R",
        "clark_evans_z",
        "clark_evans_p",
    ]

    # Condition summaries (load or derive)
    cond_path = results_dir / "summary_conditions.csv"
    if cond_path.exists():
        df_cond = pd.read_csv(cond_path)
    else:
        df_cond = build_condition_summary(df_rep, metrics)
        df_cond.to_csv(cond_path, index=False)

    if "condition" in df_cond.columns:
        df_cond = df_cond.sort_values("condition").reset_index(drop=True)

    # --- Summary plots ---
    if not args.skip_summaries:
        for m in metrics:
            plot_metric_bar(df_cond, m, plots_dir / f"summary_{m}.png")
        plot_all_metrics_grid(df_cond, metrics, plots_dir / "summary_all_metrics.png", ncols=3)

    # --- Trajectories ---
    traj_map = find_trajectory_files(traj_root)
    if not traj_map:
        print(f"No trajectories found under {traj_root}; skipping time-series plots.")
        return

    # Prefer a stable, meaningful order if present; otherwise alphabetical
    preferred = [
        "prolif_phase2_only",
        "invasive_phase2_only",
        "prolif_two_phase",
        "invasive_two_phase",
    ]
    conds = [c for c in preferred if c in traj_map] + [c for c in sorted(traj_map) if c not in preferred]

    # Global max_step and max_size for consistent binning
    global_max_step = 0
    global_max_size = 1
    for cond in conds:
        ms, mz = scan_max_step_and_size(traj_map[cond], max_files=args.scan_limit)
        global_max_step = max(global_max_step, ms)
        global_max_size = max(global_max_size, mz)

    if args.size_cap is not None:
        global_max_size = min(global_max_size, int(args.size_cap))

    # Infer dt if needed
    inferred_dt = None
    for cond in conds:
        if traj_map[cond]:
            inferred_dt = infer_dt_minutes_from_trajectory(traj_map[cond][0])
            if inferred_dt is not None:
                break
    dt_to_use = args.dt if args.dt is not None else inferred_dt

    steps = np.arange(global_max_step + 1, dtype=int)
    if dt_to_use is not None:
        x_axis = steps.astype(float) * float(dt_to_use)
        x_label = "Time (min)"
        x_is_time = True
    else:
        x_axis = steps
        x_label = "Step"
        x_is_time = False

    # Parse selected steps for size distributions
    selected_steps = None
    if args.selected_steps:
        selected_steps = [int(s.strip()) for s in args.selected_steps.split(",") if s.strip()]

    # --- Size distributions at each timepoint (all 4 conditions on each plot) ---
    counts_by_cond: Dict[str, np.ndarray] = {}
    bin_centres = np.arange(1, global_max_size + 1, dtype=int)

    for cond in conds:
        counts, bc = accumulate_size_histograms(
            traj_map[cond],
            max_step=global_max_step,
            max_size=global_max_size,
            size_cap=args.size_cap,
        )
        counts_by_cond[cond] = counts
        bin_centres = bc

    out_dir_sizes = plots_dir / "size_dists_by_timepoint"
    plot_size_dists_by_timepoint(
        counts_by_cond=counts_by_cond,
        bin_centres=bin_centres,
        out_dir=out_dir_sizes,
        x_is_time=x_is_time,
        x_values=x_axis,
        log_y=args.log_y,
        selected_steps=selected_steps,
    )

    if args.make_gif:
        make_gif_from_pngs(out_dir_sizes, plots_dir / "size_dists_by_timepoint.gif", fps=args.gif_fps)

    # --- Movement phase percentages over time ---
    if not args.skip_phases:
        phase_stats: Dict[str, Dict[str, np.ndarray]] = {}

        for cond in conds:
            p1_list, p2_list = [], []
            for f in traj_map[cond]:
                # requires movement_phase column in trajectories
                p1, p2 = phase_fraction_from_file(f, max_step=global_max_step)
                p1_list.append(p1)
                p2_list.append(p2)

            p1_mat = np.vstack(p1_list) if p1_list else np.empty((0, global_max_step + 1))
            p2_mat = np.vstack(p2_list) if p2_list else np.empty((0, global_max_step + 1))

            p1_mean, p1_lo, p1_hi = mean_ci_95_over_repeats(p1_mat)
            p2_mean, p2_lo, p2_hi = mean_ci_95_over_repeats(p2_mat)

            phase_stats[cond] = {
                "p1_mean": p1_mean, "p1_lo": p1_lo, "p1_hi": p1_hi,
                "p2_mean": p2_mean, "p2_lo": p2_lo, "p2_hi": p2_hi,
            }

            plot_phase_percentages_for_condition(
                condition=cond,
                x_axis=x_axis,
                x_label=x_label,
                p1_mean=p1_mean, p1_lo=p1_lo, p1_hi=p1_hi,
                p2_mean=p2_mean, p2_lo=p2_lo, p2_hi=p2_hi,
                outpath=plots_dir / f"phase_percentages_{cond}.png",
            )

        plot_phase_percentages_all_conditions(
            conditions=conds,
            x_axis=x_axis,
            x_label=x_label,
            phase_stats=phase_stats,
            outpath=plots_dir / "phase_percentages_all_conditions.png",
        )

    print(f"Plots saved in: {plots_dir}")


if __name__ == "__main__":
    main()