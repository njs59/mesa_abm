#!/usr/bin/env python3
"""
Plot epsilon trajectories from a set of ABC .db files with:
 - **colour** encoding the *movement speed distribution* (constant/lognorm/gamma/weibull)
 - **line style** encoding the *motion type* (isotropic vs persistent)

This script only makes the epsilon plots (trajectories + final-epoch scatter),
so it is lightweight compared to full comparisons.

Expected DB filename pattern (robust):
  abc_{obs}_{motion}_{speed}_seed{seed}[_ts].db
where {obs} may contain underscores; {motion} in {isotropic|persistent};
{speed} in {constant|lognorm|gamma|weibull}.

Outputs (saved under --outdir):
  00_epsilon_trajectories_colstyle.png  # colour by speed, line style by motion
  01_epsilon_final_colstyle.png         # final epsilon points (colour by speed, y by model, linetype legend)
  epsilon_trajectories.csv              # long-form export (optional, handy)
  parse_log.txt                         # which DBs were used / skipped

Example:
  python plot_epsilon_by_colour_linestyle.py \
    --db_glob 'results/abc_15_pop_PRO_ABM_ready_summary_*_{lognorm,gamma}_seed42_*.db' \
    --outdir results/eps_plots_PRO
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyabc

# ---- parsing of motion/speed from filenames ---------------------------------
MOTION_SET = ("isotropic", "persistent")
SPEED_SET = ("constant", "lognorm", "gamma", "weibull")

# Pattern: abc_{obs}_{motion}_{speed}_seed{seed}[_ts].db
# {obs} can include underscores, so we capture motion/speed by the explicit tokens
PATTERN = re.compile(
    r"abc_(.+?)_(isotropic|persistent)_(constant|lognorm|gamma|weibull)_seed",
    re.IGNORECASE,
)

def parse_model_from_name(p: Path) -> Tuple[str, str]:
    m = PATTERN.search(p.name)
    if m:
        motion = m.group(2).lower()
        speed = m.group(3).lower()
        return motion, speed
    return "unknown", "unknown"

# ---- load epsilon from DB ----------------------------------------------------

def load_epsilon_trajectory(db_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (t_values, epsilon_values). If not present, return ([], [])."""
    h = pyabc.History(f"sqlite:///{db_path}")
    pops = h.get_all_populations()
    if ("t" not in pops.columns) or ("epsilon" not in pops.columns):
        return np.array([]), np.array([])
    return pops["t"].to_numpy(), pops["epsilon"].to_numpy()

# ---- plotting ---------------------------------------------------------------

def make_legends_linestyle(markerless: bool, linestyle_map: Dict[str, str], colour_map: Dict[str, tuple]):
    """Create separate legend handles for speeds (colours) and motions (linestyles)."""
    from matplotlib.lines import Line2D
    speed_handles = [
        Line2D([0], [0], color=colour_map[s], lw=2.2, linestyle='-', label=s)
        for s in ["constant","lognorm","gamma","weibull"]
    ]
    motion_handles = [
        Line2D([0], [0], color='k', lw=2.2, linestyle=linestyle_map[m], label=m)
        for m in ["isotropic","persistent"]
    ]
    return speed_handles, motion_handles


def main():
    ap = argparse.ArgumentParser(description="Plot epsilon with colour by speed and line style by motion.")
    ap.add_argument("--db_glob", type=str, required=True, help="Glob for DBs, e.g. results/abc_*_seed*.db")
    ap.add_argument("--outdir", type=str, default="results/eps_plots", help="Output directory")
    ap.add_argument("--style", type=str, default="seaborn-v0_8", help="Matplotlib style to use")
    ap.add_argument("--assume_motion", type=str, choices=list(MOTION_SET), default=None,
                    help="If filenames lack motion, assume this")
    ap.add_argument("--assume_speed", type=str, choices=list(SPEED_SET), default=None,
                    help="If filenames lack speed, assume this")
    ap.add_argument("--show_markers", action="store_true", help="Also place small markers at points (optional)")
    args = ap.parse_args()

    plt.style.use(args.style)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    db_files = sorted([p for p in Path('.').glob(args.db_glob)])
    if len(db_files) == 0:
        raise FileNotFoundError(f"No DB files matched {args.db_glob}")

    runs = []
    used, skipped = [], []

    for db in db_files:
        motion, speed = parse_model_from_name(db)
        if motion == "unknown" and args.assume_motion:
            motion = args.assume_motion
        if speed == "unknown" and args.assume_speed:
            speed = args.assume_speed
        if motion not in MOTION_SET or speed not in SPEED_SET:
            skipped.append(db.name)
            continue
        t, eps = load_epsilon_trajectory(db)
        if len(t) == 0:
            skipped.append(f"{db.name} (no epsilon trajectory)")
            continue
        runs.append({
            "db": db,
            "label": f"{motion}/{speed}",
            "motion": motion,
            "speed": speed,
            "t": t,
            "eps": eps,
            "eps_final": float(eps[-1]) if len(eps) > 0 else np.nan,
        })
        used.append(f"{db.name} → {motion}/{speed}")

    with open(outdir / "parse_log.txt", "w") as f:
        f.write("USED FILES:\n")
        for u in used:
            f.write(f" {u}\n")
        if skipped:
            f.write("\nSKIPPED FILES:\n")
            for s in skipped:
                f.write(f" {s}\n")
    if not runs:
        print("No valid runs to plot. See parse_log.txt for details.")
        return

    # Colour map by speed (consistent across plots)
    palette = sns.color_palette("tab10", n_colors=4)
    colour_map = {s: palette[i] for i, s in enumerate(["constant","lognorm","gamma","weibull"])}

    # Linestyle by motion
    linestyle_map = {
        "isotropic": "-",     # solid
        "persistent": "--",   # dashed
    }

    # -------- Plot 1: trajectories (colour by speed, style by motion)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for r in runs:
        ax.plot(
            r["t"], r["eps"],
            color=colour_map[r["speed"]],
            linestyle=linestyle_map[r["motion"]],
            linewidth=1.9,
            marker='o' if args.show_markers else None,
            markersize=3.5 if args.show_markers else 0,
            label=r["label"],
        )
    ax.set_xlabel("Population t")
    ax.set_ylabel("Epsilon")
    ax.set_title("Epsilon trajectories (colour: speed distribution, line style: motion type)")
    ax.grid(True, alpha=0.25)

    # Two separate legends (speed colours, motion linestyles)
    speed_handles, motion_handles = make_legends_linestyle(
        markerless=not args.show_markers,
        linestyle_map=linestyle_map,
        colour_map=colour_map,
    )
    leg1 = ax.legend(handles=speed_handles, title="Speed distribution", loc="upper right")
    ax.add_artist(leg1)
    ax.legend(handles=motion_handles, title="Motion type", loc="lower left")

    fig.tight_layout()
    fig.savefig(outdir / "00_epsilon_trajectories_colstyle.png", dpi=200)
    plt.close(fig)

    # -------- Plot 2: final-epoch points
    df_eps = pd.DataFrame([{ "label": r["label"], "motion": r["motion"], "speed": r["speed"], "eps_final": r["eps_final"]} for r in runs])
    df_eps = df_eps.sort_values("eps_final", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    y = np.arange(len(df_eps))
    for i, row in df_eps.iterrows():
        ax.hlines(y=i, xmin=min(0, row["eps_final"]*0), xmax=row["eps_final"], colors=colour_map[row["speed"]], linestyles=linestyle_map[row["motion"]], lw=2.0)
        ax.plot(row["eps_final"], i, color=colour_map[row["speed"]], marker='o', ms=5)
        ax.text(row["eps_final"], i, f"  {row['label']}", va="center", ha="left", fontsize=8)

    ax.set_xlabel("Final epsilon")
    ax.set_yticks([])
    ax.set_title("Final epsilon per model (colour: speed, line style: motion)")
    ax.grid(True, axis='x', alpha=0.25)

    # Legends again
    speed_handles, motion_handles = make_legends_linestyle(
        markerless=True,
        linestyle_map=linestyle_map,
        colour_map=colour_map,
    )
    leg1 = ax.legend(handles=speed_handles, title="Speed distribution", loc="upper right")
    ax.add_artist(leg1)
    ax.legend(handles=motion_handles, title="Motion type", loc="lower right")

    fig.tight_layout()
    fig.savefig(outdir / "01_epsilon_final_colstyle.png", dpi=200)
    plt.close(fig)

    # CSV export for convenience
    rows = []
    for r in runs:
        for tt, ev in zip(r["t"], r["eps"]):
            rows.append({"db": str(r["db"]), "motion": r["motion"], "speed": r["speed"], "t": int(tt), "epsilon": float(ev)})
    pd.DataFrame(rows).to_csv(outdir / "epsilon_trajectories.csv", index=False)

    print(f"Done. Saved plots to {outdir.resolve()}\n - 00_epsilon_trajectories_colstyle.png\n - 01_epsilon_final_colstyle.png")


if __name__ == "__main__":
    main()
