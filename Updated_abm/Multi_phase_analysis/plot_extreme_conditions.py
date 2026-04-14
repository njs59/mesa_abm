#!/usr/bin/env python3
"""
plot_extreme_conditions.py )

Metrics used ONLY:
- n_clusters
- mean_size
- var_size
- median_nnd

Snapshot radii:
- Uses abm.utils.radius_from_size_3d if importable (correct model mapping). [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/parameter_sweep.py)
- Optional --radius_scale multiplier (default 1.0) for visibility tweaks.

Reads:
  results/fifteen_conditions/extreme_phase2only/selected_conditions.csv
  results/fifteen_conditions/extreme_phase2only/extreme_events.csv
  results/fifteen_conditions/extreme_phase2only/rerun/summary_timeseries_mean.csv
  results/fifteen_conditions/extreme_phase2only/rerun/trajectories/<condition>/repeat_XXX.csv

Writes PNGs to:
  results/fifteen_conditions/extreme_phase2only/plots_png/
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # batch-safe headless rendering [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/parameter_sweep.py)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# -------- CONFIG --------
METRICS = ["n_clusters", "mean_size", "var_size", "median_nnd"]
EPS = 1e-12

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR / "results" / "fifteen_conditions" / "extreme_phase2only"
RERUN = ROOT / "rerun"
TRAJ = RERUN / "trajectories"
PLOTS = ROOT / "plots_png"
PLOTS.mkdir(parents=True, exist_ok=True)

COLOURS = {
    "baseline": "#4D4D4D",    # grey
    "pos_extreme": "#1f77b4", # blue
    "neg_extreme": "#ff7f0e", # orange
}
# ------------------------


def ensure_project_root_on_syspath():
    """
    Make `abm` importable if it exists locally (like your forward/sweep scripts). [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/parameter_sweep.py)[1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/parameter_sweep.py)
    """
    import sys
    here = THIS_DIR
    for p in [here] + list(here.parents):
        if (p / "abm").is_dir():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return True
    return False


# Try to import the *true* radius mapping from your ABM utils.
HAVE_ABM_RADIUS = False
radius_from_size_3d = None
if ensure_project_root_on_syspath():
    try:
        from abm.utils import radius_from_size_3d as _r  # true mapping [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/parameter_sweep.py)
        radius_from_size_3d = _r
        HAVE_ABM_RADIUS = True
    except Exception:
        HAVE_ABM_RADIUS = False


def radius_fallback(size: int) -> float:
    """
    Fallback: volume ∝ size => radius ∝ size^(1/3).
    Only used if abm import fails.
    """
    size = max(1, int(size))
    return float(size ** (1.0 / 3.0))


def radius_from_size(size: int) -> float:
    if HAVE_ABM_RADIUS and radius_from_size_3d is not None:
        return float(radius_from_size_3d(int(size)))
    return radius_fallback(size)


def pct_diff(x, ref):
    ref2 = np.where(np.abs(ref) < EPS, np.nan, ref)
    return 100.0 * (x - ref2) / ref2


def detect_traj_cols(df: pd.DataFrame):
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


def load_selected_and_events():
    sel = pd.read_csv(ROOT / "selected_conditions.csv")
    conds = {r["role"]: r["condition"] for _, r in sel.iterrows()}

    ev = pd.read_csv(ROOT / "extreme_events.csv")
    pos = ev[ev["role"] == "pos_extreme"].iloc[0].to_dict()
    neg = ev[ev["role"] == "neg_extreme"].iloc[0].to_dict()
    return conds, pos, neg


def load_ts_mean():
    p = RERUN / "summary_timeseries_mean.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run run_extreme_conditions_forward.py first.")
    return pd.read_csv(p)


def count_repeats(condition: str) -> int:
    files = sorted((TRAJ / condition).glob("repeat_*.csv"))
    if not files:
        raise FileNotFoundError(f"No trajectories found under {TRAJ/condition}")
    return len(files)


def read_snapshot(condition: str, requested_step: int, rep: int):
    """
    Return (used_step, pos Nx2, sizes N) using nearest available step if needed.
    """
    traj_csv = TRAJ / condition / f"repeat_{rep:03d}.csv"
    df = pd.read_csv(traj_csv)
    step_col, x_col, y_col, size_col = detect_traj_cols(df)

    steps = np.unique(df[step_col].to_numpy())
    if requested_step in set(steps):
        used_step = requested_step
    else:
        used_step = int(steps[np.argmin(np.abs(steps - requested_step))])

    sub = df[df[step_col] == used_step][[x_col, y_col, size_col]].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub[size_col] = pd.to_numeric(sub[size_col], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

    pos = sub[[x_col, y_col]].to_numpy(dtype=float)
    sizes = np.maximum(1, np.round(sub[size_col].to_numpy(dtype=float))).astype(int)
    return used_step, pos, sizes


def infer_shared_bounds(pos_list):
    if not any(p.size > 0 for p in pos_list):
        return (0, 1), (0, 1)
    all_pos = np.vstack([p for p in pos_list if p.size > 0])
    xmin, ymin = np.min(all_pos[:, 0]), np.min(all_pos[:, 1])
    xmax, ymax = np.max(all_pos[:, 0]), np.max(all_pos[:, 1])
    dx = max(1e-9, xmax - xmin)
    dy = max(1e-9, ymax - ymin)
    pad_x, pad_y = 0.05 * dx, 0.05 * dy
    return (xmin - pad_x, xmax + pad_x), (ymin - pad_y, ymax + pad_y)


def plot_pct_timeseries(ts_mean: pd.DataFrame, conds: dict):
    """
    For each metric: plot %diff(pos vs baseline) and %diff(neg vs baseline)
    and vertical lines at each curve's max |%|.
    """
    out = PLOTS / "pct_timeseries"
    out.mkdir(parents=True, exist_ok=True)

    base = ts_mean[ts_mean["condition"] == conds["baseline"]].sort_values("step")
    pos = ts_mean[ts_mean["condition"] == conds["pos_extreme"]].sort_values("step")
    neg = ts_mean[ts_mean["condition"] == conds["neg_extreme"]].sort_values("step")

    merged_pos = pos.merge(base[["step", "time_min"] + METRICS], on=["step", "time_min"], suffixes=("", "_base"))
    merged_neg = neg.merge(base[["step", "time_min"] + METRICS], on=["step", "time_min"], suffixes=("", "_base"))

    for m in METRICS:
        pct_pos = pct_diff(merged_pos[m].to_numpy(), merged_pos[f"{m}_base"].to_numpy())
        pct_neg = pct_diff(merged_neg[m].to_numpy(), merged_neg[f"{m}_base"].to_numpy())

        i_pos = int(np.nanargmax(np.abs(pct_pos))) if np.any(np.isfinite(pct_pos)) else None
        i_neg = int(np.nanargmax(np.abs(pct_neg))) if np.any(np.isfinite(pct_neg)) else None
        t_pos = float(merged_pos["time_min"].iloc[i_pos]) if i_pos is not None else np.nan
        t_neg = float(merged_neg["time_min"].iloc[i_neg]) if i_neg is not None else np.nan

        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        ax.plot(merged_pos["time_min"], pct_pos, lw=2.2, color=COLOURS["pos_extreme"], label=f"{conds['pos_extreme']} (%Δ)")
        ax.plot(merged_neg["time_min"], pct_neg, lw=2.2, color=COLOURS["neg_extreme"], label=f"{conds['neg_extreme']} (%Δ)")

        if np.isfinite(t_pos):
            ax.axvline(t_pos, color=COLOURS["pos_extreme"], lw=1.6, alpha=0.75)
        if np.isfinite(t_neg):
            ax.axvline(t_neg, color=COLOURS["neg_extreme"], lw=1.6, alpha=0.75)

        ax.axhline(0, color="k", lw=1.0, alpha=0.5)
        ax.set_title(f"% difference vs {conds['baseline']} — {m}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("% difference")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"pctdiff__{m}.png", dpi=300)
        plt.close(fig)


def mean_size_distribution_at_step(condition: str, step: int, repeats: int):
    dists = []
    maxS = 1
    for rep in range(repeats):
        traj_csv = TRAJ / condition / f"repeat_{rep:03d}.csv"
        df = pd.read_csv(traj_csv)
        step_col, x_col, y_col, size_col = detect_traj_cols(df)
        sub = df[df[step_col] == step]
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
    mean_counts = M.mean(axis=0)
    size_axis = np.arange(mean_counts.shape[0])
    return size_axis, mean_counts


def plot_size_distributions(conds: dict, pos_event: dict, neg_event: dict, repeats: int):
    out = PLOTS / "size_distributions"
    out.mkdir(parents=True, exist_ok=True)

    for label, ev in [("pos_time", pos_event), ("neg_time", neg_event)]:
        step = int(ev["step"])
        tmin = float(ev["time_min"])
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        for role, cname in [("baseline", conds["baseline"]), ("pos_extreme", conds["pos_extreme"]), ("neg_extreme", conds["neg_extreme"])]:
            sizes, mean_counts = mean_size_distribution_at_step(cname, step, repeats)
            valid = sizes >= 1
            ax.plot(sizes[valid], mean_counts[valid], lw=2.2, label=cname, color=COLOURS[role])
        ax.set_title(f"Mean cluster size distribution at t={tmin:.1f} min (step={step})")
        ax.set_xlabel("Cluster size (cells)")
        ax.set_ylabel("Mean count (over repeats)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"size_dist__{label}__t{int(tmin):04d}min.png", dpi=300)
        plt.close(fig)


def plot_snapshot_triptych(label: str, time_min: float, requested_step: int, conds: dict, rep: int, radius_scale: float):
    out = PLOTS / "snapshots"
    out.mkdir(parents=True, exist_ok=True)

    roles = ["baseline", "pos_extreme", "neg_extreme"]
    names = [conds[r] for r in roles]

    snaps = []
    used_steps = []
    for cname in names:
        used_step, pos, sizes = read_snapshot(cname, requested_step, rep=rep)
        snaps.append((pos, sizes))
        used_steps.append(used_step)

    pos_list = [p for (p, s) in snaps]
    xlim, ylim = infer_shared_bounds(pos_list)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True, sharey=True)

    for ax, role, cname, (pos, sizes), used_step in zip(axes, roles, names, snaps, used_steps):
        col = COLOURS[role]
        if pos.size > 0:
            # TRUE radii from model mapping (or fallback) — no autoscale
            radii = np.array([radius_from_size(int(s)) for s in sizes], dtype=float) * float(radius_scale)
            for (x, y), r in zip(pos, radii):
                ax.add_patch(Circle((x, y), r, fc=col, ec="none", alpha=0.40))

        ax.set_title(f"{cname}\n(req {requested_step}, used {used_step})", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])

        if pos.shape[0] == 0:
            ax.text(0.5, 0.5, "EMPTY (0 clusters)", transform=ax.transAxes,
                    ha="center", va="center", color="red", fontsize=12)

    subtitle = "radius_from_size_3d" if HAVE_ABM_RADIUS else "fallback r∝size^(1/3)"
    fig.suptitle(f"Snapshots at t={time_min:.1f} min ({label}), rep={rep} | radii={subtitle}, scale={radius_scale}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / f"snapshots__{label}__t{int(time_min):04d}min__rep{rep:03d}.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=0, help="which repeat to snapshot (default 0)")
    ap.add_argument("--radius_scale", type=float, default=1.0, help="multiply radii for visibility (default 1.0)")
    args = ap.parse_args()

    conds, pos_event, neg_event = load_selected_and_events()
    ts_mean = load_ts_mean()
    repeats = count_repeats(conds["baseline"])

    # 1) % difference plots
    plot_pct_timeseries(ts_mean, conds)

    # 2) size distribution overlays at pos/neg extreme times
    plot_size_distributions(conds, pos_event, neg_event, repeats)

    # 3) snapshots at pos/neg extreme times
    plot_snapshot_triptych("pos_time", float(pos_event["time_min"]), int(pos_event["step"]), conds, rep=int(args.rep), radius_scale=float(args.radius_scale))
    plot_snapshot_triptych("neg_time", float(neg_event["time_min"]), int(neg_event["step"]), conds, rep=int(args.rep), radius_scale=float(args.radius_scale))

    print("PNG plots written to:", PLOTS)
    if not HAVE_ABM_RADIUS:
        print("[WARN] Could not import abm.utils.radius_from_size_3d; used fallback radii (r ∝ size^(1/3)).")


if __name__ == "__main__":
    main()