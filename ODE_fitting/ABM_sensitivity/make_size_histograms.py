#!/usr/bin/env python3
"""
make_size_histograms.py

Overlayed size-distribution bar charts (averaged across repeats) for each condition
in an ABM sensitivity run.

Outputs FOUR sets:
  1) percent/full      : mean of per-repeat normalised size distributions
  2) counts/full       : mean of per-repeat raw counts per size
  3) percent/chopped   : as (1), but sizes > max_size are collapsed into max_size bin (label max+)
  4) counts/chopped    : as (2), but sizes > max_size are collapsed into max_size bin (label max+)

Bars are drawn for each *individual cluster size* (discrete integer sizes), overlaid with alpha.

Folder output:
  <run_dir>/extra_plots/size_histograms/
      percent/full/frames/*.png
      percent/full/gifs/size_hist_percent.gif
      counts/full/frames/*.png
      counts/full/gifs/size_hist_counts.gif
      percent/chopped_max_<K>/frames/*.png
      percent/chopped_max_<K>/gifs/size_hist_percent_chopped_max_<K>.gif
      counts/chopped_max_<K>/frames/*.png
      counts/chopped_max_<K>/gifs/size_hist_counts_chopped_max_<K>.gif

Run:
  python -m ABM_sensitivity.make_size_histograms --run Prolif_speed_sweep

Chop tail:
  python -m ABM_sensitivity.make_size_histograms --run Prolif_speed_sweep --chop-max-size 50
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image


# -----------------------------
# Config / parsing
# -----------------------------

IGNORE_PARAMS = {"scenario"}  # treat as label, not a sweep parameter

TAB_CMAP_10 = "tab10"
TAB_CMAP_20 = "tab20"

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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def timestep_from_filename(p: Path) -> Optional[int]:
    m = re.search(r"t_(\d+)\.csv$", p.name)
    return int(m.group(1)) if m else None


def list_condition_dirs(sim_dir: Path) -> List[Path]:
    return sorted([p for p in sim_dir.iterdir() if p.is_dir()])


def list_repeat_dirs(cond_dir: Path) -> List[Path]:
    return sorted([p for p in cond_dir.iterdir() if p.is_dir() and p.name.startswith("repeat_")])


def list_timepoint_files(rep_dir: Path) -> List[Path]:
    return sorted(rep_dir.glob("t_*.csv"))


def infer_size_col(df: pd.DataFrame, size_col: Optional[str] = None) -> str:
    """Infer a size column name if not provided."""
    cols = list(df.columns)
    if size_col and size_col in cols:
        return size_col

    candidates = ["size", "Size", "cluster_size", "volume", "area", "radius", "r", "mass"]
    for c in candidates:
        if c in cols:
            return c

    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError(f"Could not infer size column from columns: {cols}")

    avoid = {"x", "y", "X", "Y", "pos_x", "pos_y", "px", "py", "x_pos", "y_pos"}
    remaining = [c for c in numeric_cols if c not in avoid]
    return remaining[-1] if remaining else numeric_cols[-1]


def build_condition_label(condition_id: str, *, only_keys: Optional[List[str]] = None) -> str:
    """Short legend label from condition_id (only varying keys, if provided)."""
    params = parse_condition_id(condition_id)
    if only_keys is not None:
        params = {k: params[k] for k in only_keys if k in params}
    parts = []
    for k, v in params.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else condition_id


# -----------------------------
# Timesteps and data extraction
# -----------------------------

def available_timesteps_for_condition(cond_dir: Path) -> Set[int]:
    steps: Set[int] = set()
    for rd in list_repeat_dirs(cond_dir):
        for f in list_timepoint_files(rd):
            t = timestep_from_filename(f)
            if t is not None:
                steps.add(t)
    return steps


def common_timesteps_across_conditions(condition_dirs: List[Path]) -> List[int]:
    """Intersection of timesteps across conditions (best for overlays)."""
    if not condition_dirs:
        return []
    common = None
    for cd in condition_dirs:
        s = available_timesteps_for_condition(cd)
        common = s if common is None else (common & s)
        if not common:
            return []
    return sorted(common)


def collect_sizes_for_condition_step(
    cond_dir: Path,
    step: int,
    size_col: Optional[str],
    *,
    chop_max_size: Optional[int] = None
) -> List[np.ndarray]:
    """
    Return list of integer size arrays (one per repeat) for condition+step.
    If chop_max_size is provided, sizes > chop_max_size are collapsed to chop_max_size.
    """
    arrays: List[np.ndarray] = []
    for rd in list_repeat_dirs(cond_dir):
        f = rd / f"t_{step:04d}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        scol = infer_size_col(df, size_col=size_col)
        s = np.rint(df[scol].to_numpy(dtype=float)).astype(int)
        if chop_max_size is not None:
            s = np.minimum(s, int(chop_max_size))
        arrays.append(s)
    return arrays


# -----------------------------
# Discrete (integer) binning + averaging
# -----------------------------

def discrete_bin_edges(min_size: int, max_size: int) -> np.ndarray:
    """
    Bin edges so each integer size has its own bar:
      size=k corresponds to bin [k-0.5, k+0.5)
    """
    return np.arange(min_size - 0.5, max_size + 1.5, 1.0)


def averaged_discrete_histograms(
    size_arrays: List[np.ndarray],
    bin_edges: np.ndarray,
    *,
    mode: str
) -> np.ndarray:
    """
    mode:
      - 'percent' => per-repeat normalise to sum 1, then average probabilities
      - 'counts'  => per-repeat raw counts, then average counts
    """
    if not size_arrays:
        return np.full(len(bin_edges) - 1, np.nan, dtype=float)

    hists = []
    for s in size_arrays:
        if s.size == 0:
            continue
        counts, _ = np.histogram(s, bins=bin_edges)

        if mode == "percent":
            total = counts.sum()
            if total == 0:
                continue
            hists.append(counts / total)
        elif mode == "counts":
            hists.append(counts.astype(float))
        else:
            raise ValueError("mode must be 'percent' or 'counts'")

    if not hists:
        return np.full(len(bin_edges) - 1, np.nan, dtype=float)

    return np.nanmean(np.vstack(hists), axis=0)


def scan_global_integer_range(
    condition_dirs: List[Path],
    size_col: Optional[str],
    *,
    scan_all: bool,
    sample_steps: List[int],
    limit_repeats: int = 3
) -> Tuple[int, int]:
    """Determine global integer min/max for sizes so bins stay fixed across time."""
    global_min = np.inf
    global_max = -np.inf

    for cd in condition_dirs:
        reps = list_repeat_dirs(cd)[:limit_repeats]
        steps_to_scan = sorted(available_timesteps_for_condition(cd)) if scan_all else sample_steps

        for rd in reps:
            for step in steps_to_scan:
                f = rd / f"t_{step:04d}.csv"
                if not f.exists():
                    continue
                df = pd.read_csv(f)
                scol = infer_size_col(df, size_col=size_col)
                s = np.rint(df[scol].to_numpy(dtype=float)).astype(int)
                if s.size == 0:
                    continue
                global_min = min(global_min, int(np.min(s)))
                global_max = max(global_max, int(np.max(s)))

    if not np.isfinite(global_min) or not np.isfinite(global_max) or global_min == global_max:
        return 0, 1
    return int(global_min), int(global_max)


# -----------------------------
# Plotting + GIF
# -----------------------------

def choose_tab_cmap(n_conditions: int) -> mpl.colors.Colormap:
    """Pick tab10 or tab20."""
    return mpl.colormaps[TAB_CMAP_10] if n_conditions <= 10 else mpl.colormaps[TAB_CMAP_20]


def plot_overlay_bars(
    step: int,
    condition_dirs: List[Path],
    bin_edges: np.ndarray,
    out_png: Path,
    *,
    size_col: Optional[str],
    labels: Dict[str, str],
    colours: Dict[str, Tuple[float, float, float, float]],
    mode: str,
    y_max: Optional[float],
    alpha: float = 0.35,
    chop_max_size: Optional[int] = None
) -> None:
    """
    Overlaid bars (same x) for each condition using alpha.
    If chop_max_size is set, label the final tick as '<max>+'.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    sizes = np.arange(
        int(np.floor(bin_edges[0] + 0.5)),
        int(np.ceil(bin_edges[-1] - 0.5)) + 1
    )
    n_bins = len(sizes)

    any_data = False
    bar_w = 0.90

    for cd in condition_dirs:
        cond_id = cd.name
        arrays = collect_sizes_for_condition_step(cd, step, size_col, chop_max_size=chop_max_size)
        hist = averaged_discrete_histograms(arrays, bin_edges, mode=mode)
        if np.all(np.isnan(hist)):
            continue
        any_data = True

        ax.bar(
            sizes,
            hist,
            width=bar_w,
            color=colours[cond_id],
            alpha=alpha,
            edgecolor=colours[cond_id],
            linewidth=0.8,
            label=labels[cond_id],
        )

    title_mode = "Percent (mean of per-repeat normalised)" if mode == "percent" else "Mean raw counts per size"
    title_extra = f" (chopped at {chop_max_size}+)" if chop_max_size is not None else ""
    ax.set_title(f"Size distribution — {title_mode}{title_extra} — t={step:04d}")
    ax.set_xlabel("Cluster size (integer)")
    ax.set_ylabel("Probability" if mode == "percent" else "Mean number of clusters")
    ax.grid(True, axis="y", alpha=0.25)

    if y_max is not None and np.isfinite(y_max):
        ax.set_ylim(0, y_max)

    # ticks: avoid huge label sets
    if n_bins <= 30:
        tick_idx = np.arange(n_bins)
    else:
        step_tick = max(1, int(np.ceil(n_bins / 25)))
        tick_idx = np.arange(0, n_bins, step_tick)

    ax.set_xticks(sizes[tick_idx])

    # If chopped, label final size as max+
    if chop_max_size is not None:
        tick_labels = [str(int(s)) for s in sizes[tick_idx]]
        if len(tick_labels) > 0 and int(sizes[tick_idx][-1]) == int(chop_max_size):
            tick_labels[-1] = f"{int(chop_max_size)}+"
        ax.set_xticklabels(tick_labels)

    if any_data:
        ax.legend(loc="upper right", fontsize=9, frameon=False)
    else:
        ax.text(0.5, 0.5, "No data for this timestep", transform=ax.transAxes,
                ha="center", va="center")

    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_gif(frame_files: List[Path], out_gif: Path, fps: int = 8) -> None:
    """Build animated GIF from PNG frames."""
    if not frame_files:
        return
    ensure_dir(out_gif.parent)
    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_files]
    duration_ms = int(1000 / max(1, fps))
    images[0].save(out_gif, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=True)


def estimate_ymax(
    condition_dirs: List[Path],
    sample_steps: List[int],
    bin_edges: np.ndarray,
    *,
    size_col: Optional[str],
    mode: str,
    chop_max_size: Optional[int]
) -> Optional[float]:
    ymax = 0.0
    for step in sample_steps:
        for cd in condition_dirs:
            arrays = collect_sizes_for_condition_step(cd, step, size_col, chop_max_size=chop_max_size)
            hist = averaged_discrete_histograms(arrays, bin_edges, mode=mode)
            if np.all(np.isnan(hist)):
                continue
            ymax = max(ymax, float(np.nanmax(hist)))
    return ymax * 1.10 if ymax > 0 else None


# -----------------------------
# Main
# -----------------------------

def process_run(
    run_dir: Path,
    *,
    max_conditions: int,
    size_col: Optional[str],
    fps: int,
    scan_all: bool,
    limit_timesteps: Optional[int],
    chop_max_size: Optional[int],
    alpha: float
) -> None:
    sim_dir = run_dir / "simulations"
    if not sim_dir.exists():
        print(f"[WARN] {run_dir.name}: no simulations/ folder. Skipping.")
        return

    condition_dirs = list_condition_dirs(sim_dir)
    if not condition_dirs:
        print(f"[WARN] {run_dir.name}: no conditions found in simulations/.")
        return

    if len(condition_dirs) > max_conditions:
        print(f"[INFO] {run_dir.name}: {len(condition_dirs)} conditions > max_conditions={max_conditions}. Skipping.")
        return

    timesteps = common_timesteps_across_conditions(condition_dirs)
    if not timesteps:
        timesteps = sorted(available_timesteps_for_condition(condition_dirs[0]))
        print(f"[WARN] {run_dir.name}: no common timesteps; using timesteps from first condition ({len(timesteps)}).")

    if limit_timesteps is not None:
        timesteps = timesteps[:limit_timesteps]

    if not timesteps:
        print(f"[WARN] {run_dir.name}: no timesteps found.")
        return

    # sample steps for y-limit estimation + scanning
    early = timesteps[int(round(0.10 * (len(timesteps) - 1)))] if len(timesteps) > 1 else timesteps[0]
    mid = timesteps[int(round(0.50 * (len(timesteps) - 1)))] if len(timesteps) > 1 else timesteps[0]
    final = timesteps[-1]
    sample_steps = sorted(set([early, mid, final]))

    # global full-range bins
    smin, smax = scan_global_integer_range(
        condition_dirs, size_col=size_col,
        scan_all=scan_all, sample_steps=sample_steps, limit_repeats=3
    )
    full_edges = discrete_bin_edges(smin, smax)

    # chopped bins (if requested)
    chopped_edges = None
    chopped_max = None
    if chop_max_size is not None:
        chopped_max = int(min(chop_max_size, smax))
        if chopped_max < smin:
            chopped_max = smin
        chopped_edges = discrete_bin_edges(smin, chopped_max)

    # determine varying keys for legend labels
    param_sets = [parse_condition_id(cd.name) for cd in condition_dirs]
    all_keys = sorted(set().union(*[set(p.keys()) for p in param_sets]))
    varying_keys = []
    for k in all_keys:
        uniq = set(str(p.get(k, None)) for p in param_sets)
        if len(uniq) > 1:
            varying_keys.append(k)
    labels = {cd.name: build_condition_label(cd.name, only_keys=varying_keys) for cd in condition_dirs}

    # colours
    cmap = choose_tab_cmap(len(condition_dirs))
    colours = {cd.name: cmap(i % cmap.N) for i, cd in enumerate(condition_dirs)}

    out_root = run_dir / "extra_plots" / "size_histograms"
    ensure_dir(out_root)

    # Helper to render a mode+variant
    def render_variant(mode: str, variant: str, edges: np.ndarray, *, chop: Optional[int]):
        frames_dir = out_root / mode / variant / "frames"
        gifs_dir = out_root / mode / variant / "gifs"
        ensure_dir(frames_dir)
        ensure_dir(gifs_dir)

        ymax = estimate_ymax(
            condition_dirs, sample_steps, edges,
            size_col=size_col, mode=mode, chop_max_size=chop
        )

        frame_files: List[Path] = []
        for step in timesteps:
            out_png = frames_dir / f"hist_{mode}_{variant}_t_{step:04d}.png"
            plot_overlay_bars(
                step=step,
                condition_dirs=condition_dirs,
                bin_edges=edges,
                out_png=out_png,
                size_col=size_col,
                labels=labels,
                colours=colours,
                mode=mode,
                y_max=ymax,
                alpha=alpha,
                chop_max_size=chop
            )
            frame_files.append(out_png)

        out_gif = gifs_dir / f"size_hist_{mode}_{variant}.gif"
        make_gif(frame_files, out_gif, fps=fps)
        print(f"[DONE] {run_dir.name}: {mode}/{variant} -> {out_gif}")

    print(f"[INFO] {run_dir.name}: generating overlay size bars ({len(timesteps)} timesteps) into {out_root}")

    # Full versions
    render_variant("percent", "full", full_edges, chop=None)
    render_variant("counts", "full", full_edges, chop=None)

    # Chopped versions (if requested)
    if chopped_edges is not None and chopped_max is not None:
        variant_name = f"chopped_max_{chopped_max}"
        render_variant("percent", variant_name, chopped_edges, chop=chopped_max)
        render_variant("counts", variant_name, chopped_edges, chop=chopped_max)


def main():
    parser = argparse.ArgumentParser(description="Overlay averaged size distribution bars (PNG per timestep + GIF).")
    parser.add_argument("--results-root", type=str, default=None,
                        help="Path to ABM_sensitivity_results. Default: sibling of ABM_sensitivity/.")
    parser.add_argument("--run", type=str, default="all",
                        help="Run folder name to process, or 'all'.")
    parser.add_argument("--max-conditions", type=int, default=10,
                        help="Only generate overlay plots if #conditions <= this.")
    parser.add_argument("--size-col", type=str, default=None,
                        help="Override size column name in t_*.csv (otherwise inferred).")
    parser.add_argument("--fps", type=int, default=8, help="FPS for GIF.")
    parser.add_argument("--scan-all", action="store_true",
                        help="Scan all timesteps to determine global integer size range (slower).")
    parser.add_argument("--limit-timesteps", type=int, default=None,
                        help="Only process the first N timesteps (quick tests).")
    parser.add_argument("--alpha", type=float, default=0.35,
                        help="Alpha for overlaid bars (smaller = more transparent).")
    parser.add_argument("--chop-max-size", type=int, default=None,
                        help="Create chopped plots where sizes > this are collapsed into this bin (labelled max+).")
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
            max_conditions=int(args.max_conditions),
            size_col=args.size_col,
            fps=int(args.fps),
            scan_all=bool(args.scan_all),
            limit_timesteps=args.limit_timesteps,
            chop_max_size=args.chop_max_size,
            alpha=float(args.alpha),
        )


if __name__ == "__main__":
    main()