#!/usr/bin/env python3
"""
plot_kappa_movement_model_ext PNGs:plot_kappa_movement_model_extremes.py
1) % difference time-series per metric
   - two lines: max-increase event pair and max-decrease event pair
   - vertical lines at the extreme-event times
2) 2×2 snapshot grid:
   - row 1: max-increase event (phase2 baseline | two-phase model)
   - row 2: max-decrease event (phase2 baseline | two-phase model)
   - true radii via abm.utils.radius_from_size_3d [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/run_kappa_conditions.py)
3) side-by-side size-distribution comparison:
   - left: max-increase time (baseline vs model)
   - right: max-decrease time (baseline vs model)

Reads:
  results/kappa_conditions/movement_effects/extremes.csv
  results/kappa_conditions/movement_effects/rerun/summary_timeseries_mean.csv
  results/kappa_conditions/movement_effects/rerun/trajectories/<condition>/repeat_XXX.csv
"""

# --- robust path setup so `import abm` works like your drivers/sweeps [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/run_kappa_conditions.py)[1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/run_kappa_conditions.py) ---
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = None
for p in [THIS_DIR] + list(THIS_DIR.parents):
    if (p / "abm").is_dir():
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    raise RuntimeError(f"Could not locate project root containing 'abm/' from {THIS_DIR}")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ----------------------------------------------------------------------

import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless rendering (batch-safe) [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/run_kappa_conditions.py)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from abm.utils import radius_from_size_3d  # true radii mapping [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/run_kappa_conditions.py)

# ONLY the 4 metrics requested [2](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/plot_kappa_conditions.py)
METRICS = ["n_clusters", "mean_size", "var_size", "median_nnd"]
EPS = 1e-12

RES_DIR = THIS_DIR / "results" / "kappa_conditions"
EFF_DIR = RES_DIR / "movement_effects"
RERUN = EFF_DIR / "rerun"
TRAJ = RERUN / "trajectories"
PLOTS = EFF_DIR / "plots_png"
PLOTS.mkdir(parents=True, exist_ok=True)

COLOUR_INC = "C0"
COLOUR_DEC = "C1"


def pct_diff(x, ref):
    ref2 = np.where(np.abs(ref) < EPS, np.nan, ref)
    return 100.0 * (x - ref2) / ref2


def detect_traj_cols(df: pd.DataFrame):
    """
    Robust detection of trajectory columns. Your trajectories come from run_single_sim,
    which can vary column naming; we handle common patterns.
    """
    step_candidates = ["step", "t", "time_step"]
    x_candidates = ["x", "pos_x", "X"]
    y_candidates = ["y", "pos_y", "Y"]
    size_candidates = ["size", "cluster_size", "n_cells"]

    step_col = next((c for c in step_candidates if c in df.columns), None)
    x_col = next((c for c in x_candidates if c in df.columns), None)
    y_col = next((c for c in y_candidates if c in df.columns), None)
    size_col = next((c for c in size_candidates if c in df.columns), None)

    if None in (step_col, x_col, y_col, size_col):
        raise RuntimeError(
            "Could not detect required trajectory columns.\n"
            f"Columns present: {list(df.columns)}\n"
            "Need at least: step + x + y + size."
        )
    return step_col, x_col, y_col, size_col


def count_repeats(condition: str) -> int:
    files = sorted((TRAJ / condition).glob("repeat_*.csv"))
    if not files:
        raise FileNotFoundError(f"No trajectories found under {TRAJ/condition}")
    return len(files)


def read_snapshot(condition: str, requested_step: int, rep: int):
    """
    Read snapshot from trajectories at requested_step.
    If that step doesn't exist in the file, use the nearest available step.
    Returns: (used_step, pos Nx2, sizes N)
    """
    traj_csv = TRAJ / condition / f"repeat_{rep:03d}.csv"
    df = pd.read_csv(traj_csv)
    step_col, x_col, y_col, size_col = detect_traj_cols(df)

    # find nearest step if needed
    steps = np.unique(df[step_col].to_numpy())
    used_step = requested_step if requested_step in set(steps) else int(steps[np.argmin(np.abs(steps - requested_step))])

    sub = df[df[step_col] == used_step][[x_col, y_col, size_col]].copy()

    # numeric cleanup
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub[size_col] = pd.to_numeric(sub[size_col], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

    pos = sub[[x_col, y_col]].to_numpy(dtype=float)
    sizes = np.maximum(1, np.round(sub[size_col].to_numpy(dtype=float))).astype(int)
    return used_step, pos, sizes


def infer_bounds(pos_list):
    """
    Shared bounds for comparing baseline vs model fairly.
    """
    if not any(p.size > 0 for p in pos_list):
        return (0, 1), (0, 1)

    all_pos = np.vstack([p for p in pos_list if p.size > 0])
    xmin, ymin = np.min(all_pos[:, 0]), np.min(all_pos[:, 1])
    xmax, ymax = np.max(all_pos[:, 0]), np.max(all_pos[:, 1])
    dx, dy = max(1e-9, xmax - xmin), max(1e-9, ymax - ymin)
    return (xmin - 0.05 * dx, xmax + 0.05 * dx), (ymin - 0.05 * dy, ymax + 0.05 * dy)


def mean_size_distribution_at_step(condition: str, step: int, repeats: int):
    """
    Mean cluster-size distribution at a step (mean over repeats).
    Returns (size_axis, mean_counts).
    """
    dists = []
    maxS = 1

    for rep in range(repeats):
        traj_csv = TRAJ / condition / f"repeat_{rep:03d}.csv"
        df = pd.read_csv(traj_csv)
        step_col, x_col, y_col, size_col = detect_traj_cols(df)

        # pick nearest step if exact missing
        steps = np.unique(df[step_col].to_numpy())
        used_step = step if step in set(steps) else int(steps[np.argmin(np.abs(steps - step))])

        sub = df[df[step_col] == used_step]
        sizes = sub[size_col].to_numpy(dtype=int) if size_col in sub.columns else np.array([], dtype=int)

        if sizes.size == 0:
            bc = np.zeros(1, dtype=float)
        else:
            bc = np.bincount(sizes, minlength=int(sizes.max()) + 1).astype(float)
            bc[0] = 0.0

        maxS = max(maxS, bc.shape[0])
        dists.append(bc)

    M = np.zeros((len(dists), maxS), dtype=float)
    for i, bc in enumerate(dists):
        M[i, : bc.shape[0]] = bc

    return np.arange(M.shape[1]), M.mean(axis=0)


def merged_pair(ts_mean: pd.DataFrame, baseline_cond: str, model_cond: str):
    """
    Join model and baseline mean time-series on (step, time_min).
    """
    base = ts_mean[ts_mean["condition"] == baseline_cond].sort_values("step")
    mod = ts_mean[ts_mean["condition"] == model_cond].sort_values("step")
    return mod.merge(base[["step", "time_min"] + METRICS], on=["step", "time_min"], suffixes=("", "_base"))


def plot_pct_timeseries(extremes: pd.DataFrame, ts_mean: pd.DataFrame):
    """
    For each metric: two lines (%Δ for max-increase event pair and max-decrease event pair),
    with vertical lines at the event times.
    """
    out = PLOTS / "pct_timeseries"
    out.mkdir(parents=True, exist_ok=True)

    inc = extremes[extremes["effect"] == "max_increase"].iloc[0]
    dec = extremes[extremes["effect"] == "max_decrease"].iloc[0]

    inc_m = merged_pair(ts_mean, inc["baseline"], inc["condition"])
    dec_m = merged_pair(ts_mean, dec["baseline"], dec["condition"])

    t_inc = float(inc["time_min"])
    t_dec = float(dec["time_min"])

    for m in METRICS:
        pct_inc = pct_diff(inc_m[m].to_numpy(), inc_m[f"{m}_base"].to_numpy())
        pct_dec = pct_diff(dec_m[m].to_numpy(), dec_m[f"{m}_base"].to_numpy())

        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        ax.plot(inc_m["time_min"], pct_inc, lw=2.2, color=COLOUR_INC,
                label=f"Max increase: {inc['kappa']} | {inc['model']}")
        ax.plot(dec_m["time_min"], pct_dec, lw=2.2, color=COLOUR_DEC,
                label=f"Max decrease: {dec['kappa']} | {dec['model']}")

        ax.axvline(t_inc, color=COLOUR_INC, lw=1.6, alpha=0.75)
        ax.axvline(t_dec, color=COLOUR_DEC, lw=1.6, alpha=0.75)

        ax.axhline(0, color="k", lw=1.0, alpha=0.5)
        ax.set_title(f"% difference vs κ phase2 baseline — {m}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("% difference")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"pctdiff__{m}.png", dpi=300)
        plt.close(fig)


def plot_snapshot_grid(extremes: pd.DataFrame, rep: int, radius_scale: float):
    """
    2×2 snapshot grid:
      row 1: max increase (phase2 baseline | two-phase)
      row 2: max decrease (phase2 baseline | two-phase)
    Uses true radii via radius_from_size_3d(size). [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/run_kappa_conditions.py)
    """
    out = PLOTS / "snapshots"
    out.mkdir(parents=True, exist_ok=True)

    inc = extremes[extremes["effect"] == "max_increase"].iloc[0]
    dec = extremes[extremes["effect"] == "max_decrease"].iloc[0]

    pairs = [("Max increase", inc, "0.5", "C0"), ("Max decrease", dec, "0.5", "C0")]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.5))

    for r, (label, row, base_col, mod_col) in enumerate(pairs):
        base = row["baseline"]
        mod = row["condition"]
        step = int(row["step"])
        tmin = float(row["time_min"])

        used_b, pos_b, sizes_b = read_snapshot(base, step, rep)
        used_m, pos_m, sizes_m = read_snapshot(mod, step, rep)

        xlim, ylim = infer_bounds([pos_b, pos_m])

        # baseline panel
        ax = axes[r, 0]
        for (x, y), s in zip(pos_b, sizes_b):
            ax.add_patch(Circle((x, y), float(radius_from_size_3d(int(s))) * radius_scale,
                                fc=base_col, ec="none", alpha=0.45))
        ax.set_title(f"{label}\nPhase2 baseline\n{base}\n t={tmin:.1f} min (step {used_b})", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([])

        # model panel
        ax = axes[r, 1]
        for (x, y), s in zip(pos_m, sizes_m):
            ax.add_patch(Circle((x, y), float(radius_from_size_3d(int(s))) * radius_scale,
                                fc=mod_col, ec="none", alpha=0.45))
        ax.set_title(f"{label}\nTwo-phase model\n{mod}\n t={tmin:.1f} min (step {used_m})", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"κ snapshots (rep {rep}): phase2 vs two-phase at extreme times", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out / f"snapshots_2x2__rep{rep:03d}.png", dpi=300)
    plt.close(fig)


def plot_distribution_side_by_side(extremes: pd.DataFrame):
    """
    Side-by-side size-distribution plots:
      left: max increase time (baseline vs model)
      right: max decrease time (baseline vs model)
    """
    out = PLOTS / "distributions"
    out.mkdir(parents=True, exist_ok=True)

    inc = extremes[extremes["effect"] == "max_increase"].iloc[0]
    dec = extremes[extremes["effect"] == "max_decrease"].iloc[0]

    reps_inc = min(count_repeats(inc["baseline"]), count_repeats(inc["condition"]))
    reps_dec = min(count_repeats(dec["baseline"]), count_repeats(dec["condition"]))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for ax, label, row, reps in [
        (axes[0], "Max increase", inc, reps_inc),
        (axes[1], "Max decrease", dec, reps_dec),
    ]:
        step = int(row["step"])
        tmin = float(row["time_min"])
        base = row["baseline"]
        mod = row["condition"]

        x_b, y_b = mean_size_distribution_at_step(base, step, reps)
        x_m, y_m = mean_size_distribution_at_step(mod, step, reps)

        valid_b = x_b >= 1
        valid_m = x_m >= 1

        ax.plot(x_b[valid_b], y_b[valid_b], lw=2.2, color="0.4", label="Phase2 baseline")
        ax.plot(x_m[valid_m], y_m[valid_m], lw=2.2, color="C0", label="Two-phase model")

        ax.set_title(f"{label}\n{row['kappa']} | {row['model']} | t={tmin:.1f} min", fontsize=9)
        ax.set_xlabel("Cluster size (cells)")
        ax.set_ylabel("Mean count (over repeats)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out / "size_distributions_side_by_side.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=0, help="which repeat to use for snapshots (default 0)")
    ap.add_argument("--radius_scale", type=float, default=1.0, help="multiply radii for visibility (default 1.0)")
    args = ap.parse_args()

    extremes_csv = EFF_DIR / "extremes.csv"
    ts_csv = RERUN / "summary_timeseries_mean.csv"

    if not extremes_csv.exists():
        raise FileNotFoundError(f"Missing {extremes_csv}. Run find_kappa_movement_model_extremes.py first.")
    if not ts_csv.exists():
        raise FileNotFoundError(f"Missing {ts_csv}. Run rerun_kappa_movement_model_extremes.py first.")

    extremes = pd.read_csv(extremes_csv)
    ts_mean = pd.read_csv(ts_csv)

    plot_pct_timeseries(extremes, ts_mean)
    plot_snapshot_grid(extremes, rep=int(args.rep), radius_scale=float(args.radius_scale))
    plot_distribution_side_by_side(extremes)

    print("Plots written to:", PLOTS)


if __name__ == "__main__":
    main()

