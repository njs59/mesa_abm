#!/usr/bin/env python3
"""
Forward-simulation driver for ABM under **Phase 2 only** movement with
**initial invasive phenotype singletons**.

Folder layout (expected):
    project_root/
      abm/                    # your package with __init__.py, cluster_agent.py, clusters_model.py, utils.py
      Forward_simulations/
        forward_phase2_invasive.py   # <-- this script (place here)

This script runs N=100 stochastic repeats by default, producing:
  • First-run diagnostics:
      - size_distribution_frames/ (PNG per time step)
      - size_distribution.gif (histogram over time)
      - size_distribution_every25.png (overlay of every 25th step)
      - snapshots_frames/ (PNG per time step; spatial visualisation)
      - snapshots.gif (optional overview GIF of spatial views)
  • Aggregated over 100 repeats:
      - avg_size_distribution_frames/ (PNG per time step of mean counts)
      - avg_size_distribution.gif
      - avg_size_distribution_every25.png
      - CSV exports for first-run and averages
  • Selected times (t=30,40,50,60,70,80):
      - selected_first_every10_t30_to_t80/ (individual PNGs + grid + lines)
      - selected_average_every10_t30_to_t80/ (individual PNGs + grid + lines)
  • Selected times (t=30,50,70,90):
      - selected_first_every20_t30_to_t90/ (individual PNGs + grid + lines)
      - selected_average_every20_t30_to_t90/ (individual PNGs + grid + lines)

Notes
-----
• "Phase 2 only" is enforced by spawning agents with switch time 0.0 and
  then immediately setting `movement_phase=2` and `phase_switch_time = -inf`.
  All progeny produced by fragmentation are forced into phase 2 by the
  existing model logic (child inherits and is set to phase 2).
• The initial population consists of size-1 clusters of phenotype "invasive".
• Run-time parameters can be changed via command-line flags. See `--help`.
"""

import os
import sys
import math
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless for batch rendering
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image

# -----------------------------------------------------------------------------
# Ensure we can import the sibling package `abm`
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from abm.clusters_model import ClustersModel
from abm.utils import DEFAULTS, radius_from_size_3d

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clone_params(base: dict) -> dict:
    import copy
    return copy.deepcopy(base)

def prepare_params_phase2_invasive(base: dict, *, n_clusters: int, steps: int) -> dict:
    """Take DEFAULTS and return a parameters dict configured for this task."""
    p = clone_params(base)
    # Time horizon
    p.setdefault('time', {})['steps'] = int(steps)
    # Initial condition: invasive singletons
    p.setdefault('init', {})['n_clusters'] = int(n_clusters)
    p['init']['size'] = 1
    p['init']['phenotype'] = 'invasive'
    return p

def spawn_invasive_phase2_singletons(model: ClustersModel, n: int) -> None:
    """Spawn *n* invasive size-1 clusters that start immediately in Phase 2."""
    for _ in range(int(n)):
        a = model.spawn_cluster(size=1, phenotype='invasive', phase_switch_time=0.0)
        # Force Phase 2 from t=0
        a.movement_phase = 2
        a.phase_switch_time = float('-inf')

def run_single(seed: int, params: dict) -> Dict[str, List[np.ndarray]]:
    """Run one simulation and return the logged series we need."""
    # Prevent auto-spawn in the model constructor by giving an empty init list
    m = ClustersModel(params=params, seed=seed, init_clusters=[])

    # Spawn custom initial condition (number from params['init'])
    n0 = int(params.get('init', {}).get('n_clusters', 800))
    spawn_invasive_phase2_singletons(m, n0)

    # Run for the requested number of steps
    n_steps = int(params.get('time', {}).get('steps', 145))
    for _ in range(n_steps):
        m.step()

    # Package outputs: these correspond to m.*_log lists filled at t=0..n_steps
    out = {
        'sizes': m.size_log,      # list of 1D arrays (sizes per alive agent)
        'positions': m.pos_log,   # list of (N,2)
        'radii': m.radius_log,    # list of 1D arrays (not used for plotting)
        'ids': m.id_log,          # list of 1D arrays
        'speeds': m.speed_log,    # list of 1D arrays
        'dt': [m.dt],
        'space': [(m.space.x_max, m.space.y_max)],
    }
    return out

def bincount_sizes(sizes: np.ndarray, min_size: int = 1) -> np.ndarray:
    """Return counts for cluster sizes, indexed by size (index 0 unused)."""
    if sizes.size == 0:
        return np.zeros(1, dtype=int)  # only index 0 present
    max_sz = int(sizes.max())
    bc = np.bincount(sizes.astype(int), minlength=max_sz + 1)
    bc[0] = 0  # no size-0 clusters
    return bc

def aggregate_counts_over_runs(counts_per_time: Dict[int, np.ndarray], T: int) -> np.ndarray:
    """Convert {t_idx: counts_by_size} into 2D array (T x Smax+1)."""
    max_size = 1
    for t in range(T):
        arr = counts_per_time.get(t)
        if arr is not None:
            max_size = max(max_size, arr.shape[0]-1)
    M = np.zeros((T, max_size + 1), dtype=float)
    for t in range(T):
        arr = counts_per_time.get(t)
        if arr is None:
            continue
        L = min(M.shape[1], arr.shape[0])
        M[t, :L] = arr[:L]
    return M

def ensure_outdir(base_out: str) -> str:
    os.makedirs(base_out, exist_ok=True)
    return base_out

def fig_to_pil(fig, dpi=120) -> Image.Image:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)
    buf = buf[:, :, [1, 2, 3, 0]]  # ARGB -> RGBA
    im = Image.fromarray(buf, 'RGBA')
    return im

# Interpret dt=1.0 as 30 minutes (your convention)
MIN_PER_DT = 30.0

def plot_size_distribution_frame(counts: np.ndarray, t_idx: int, dt: float, out_png: str, *,
                                 color: str = '#4c78a8', xmax: int = None) -> None:
    sizes = np.arange(counts.shape[0])
    valid = sizes >= 1
    if xmax is not None:
        valid = valid & (sizes <= int(xmax))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(sizes[valid], counts[valid], color=color)
    ax.set_xlabel('Cluster size (cells)')
    ax.set_ylabel('Count')
    ax.set_title(f'Size distribution at t={t_idx} (time = {t_idx*dt*MIN_PER_DT:.1f} min)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if xmax is not None:
        ax.set_xlim(1, int(xmax))
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_overlay_every_k(M: np.ndarray, dt: float, k: int, out_png: str, title_prefix: str) -> None:
    """Overlay plot of counts-by-size for times 0, k, 2k, ..."""
    T, S = M.shape
    times = list(range(0, T, k))
    if T-1 not in times:
        times.append(T-1)
    cmap = plt.get_cmap('viridis')
    colours = [cmap(i/ max(1,(len(times)-1))) for i in range(len(times))]
    sizes = np.arange(S)
    valid = sizes >= 1

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for idx, t in enumerate(times):
        ax.plot(sizes[valid], M[t, valid],
                label=f't={t} ({t*dt*MIN_PER_DT:.0f} min)', color=colours[idx], lw=1.8)
    ax.set_xlabel('Cluster size (cells)')
    ax.set_ylabel('Count (mean over runs)' if title_prefix.startswith('Average') else 'Count')
    ax.set_title(f'{title_prefix}: every {k} steps')
    ax.legend(ncol=2, fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_snapshot(positions: np.ndarray, sizes: np.ndarray, space_wh: Tuple[float, float],
                  out_png: str, title: str) -> None:
    """Draw true circles in data coordinates, radius from size via radius_from_size_3d."""
    W, H = space_wh
    # Compute radii from sizes (volume-preserving)
    if sizes.size:
        radii = np.array([radius_from_size_3d(int(s)) for s in sizes], dtype=float)
    else:
        radii = np.zeros((0,), dtype=float)

    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    patches = []
    for i in range(positions.shape[0]):
        cx, cy = float(positions[i, 0]), float(positions[i, 1])
        r = float(radii[i]) if i < radii.size else 0.0
        patches.append(Circle((cx, cy), radius=r))
    if patches:
        pc = PatchCollection(
            patches,
            facecolor=(70/256, 158/256, 44/256, 0.6),
            edgecolor='k',
            linewidths=0.3
        )
        ax.add_collection(pc)

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def save_gif_from_folder(frames_dir: str, out_path: str, duration_ms: int = 160) -> None:
    files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith('.png')])
    if not files:
        return
    images = [Image.open(os.path.join(frames_dir, f)).convert('P', palette=Image.ADAPTIVE) for f in files]
    images[0].save(out_path, save_all=True, append_images=images[1:], optimize=True,
                   duration=duration_ms, loop=0)

def plot_size_distribution_grid(M: np.ndarray, dt: float, times: List[int], out_png: str,
                                title_prefix: str, *, palette: List[str] = None,
                                xmax: int = None) -> None:
    """Create a grid of bar charts for selected time indices."""
    if not times:
        return
    n = len(times)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.6*nrows), squeeze=False)
    if palette is None:
        palette = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
                   '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
    for i, t in enumerate(times):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sizes = np.arange(M.shape[1])
        valid = sizes >= 1
        if xmax is not None:
            valid = valid & (sizes <= int(xmax))
        color = palette[i % len(palette)]
        ax.bar(sizes[valid], M[t, valid], color=color)
        ax.set_title(f't={t} ({t*dt*MIN_PER_DT:.0f} min)')
        ax.set_xlabel('Size')
        ax.set_ylabel('Count' if not title_prefix.startswith('Average') else 'Mean count')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if xmax is not None:
            ax.set_xlim(1, int(xmax))
    # Clear any unused axes
    for j in range(n, nrows*ncols):
        r, c = divmod(j, ncols)
        fig.delaxes(axes[r][c])
    fig.suptitle(f'{title_prefix}: selected times', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def save_selected_time_plots(M: np.ndarray, dt: float, times: List[int], out_dir: str,
                             title_prefix: str, *, palette: List[str] = None,
                             xmax: int = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Individual PNGs
    for idx, t in enumerate(times):
        out_png = os.path.join(out_dir, f'bar_t{t:03d}.png')
        color = (palette[idx % len(palette)] if palette else '#4c78a8')
        plot_size_distribution_frame(M[t], t, dt, out_png, color=color, xmax=xmax)
    # Grid PNG
    grid_png = os.path.join(out_dir, 'grid_selected_times.png')
    plot_size_distribution_grid(M, dt, times, grid_png, title_prefix,
                                palette=palette, xmax=xmax)

def plot_lines_selected_times(M: np.ndarray, dt: float, times: List[int], out_png: str,
                              title_prefix: str, *, palette: List[str] = None,
                              xmax: int = None) -> None:
    """Plot size-distribution curves for multiple selected times on one axes."""
    if not times:
        return
    sizes = np.arange(M.shape[1])
    valid = sizes >= 1
    if xmax is not None:
        valid = valid & (sizes <= int(xmax))
    if palette is None:
        palette = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
                   '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for i, t in enumerate(times):
        color = palette[i % len(palette)]
        ax.plot(sizes[valid], M[t, valid], color=color, lw=2.0,
                label=f't={t} ({t*dt*MIN_PER_DT:.0f} min)')
    ax.set_xlabel('Cluster size (cells)')
    ax.set_ylabel('Count' if not title_prefix.startswith('Average') else 'Mean count')
    ax.set_title(f'{title_prefix}: selected times (line plot)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if xmax is not None:
        ax.set_xlim(1, int(xmax))
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Forward-simulate ABM: Phase 2 only, invasive singletons, with visualisations.'
    )
    parser.add_argument('--repeats', type=int, default=100,
                        help='Number of stochastic repeats (default: 100)')
    parser.add_argument('--steps', type=int, default=int(DEFAULTS['time']['steps']),
                        help='Number of steps to run per repeat')
    parser.add_argument('--n0', type=int,
                        default=int(DEFAULTS['init']['n_clusters']) if 'init' in DEFAULTS else 800,
                        help='Initial number of invasive singletons (default from DEFAULTS or 800)')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output folder (default: Forward_simulations/outputs/phase2_invasive_<timestamp>)')
    parser.add_argument('--gif_ms', type=int, default=160, help='Frame duration for GIFs in milliseconds')
    parser.add_argument('--every_k', type=int, default=25, help='Stride for overlay plot (every k steps)')

    args = parser.parse_args()

    # Prepare output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_out = os.path.join(THIS_DIR, 'outputs', f'phase2_invasive_{timestamp}')
    out_base = ensure_outdir(args.outdir or default_out)

    # Parameters template
    params = prepare_params_phase2_invasive(DEFAULTS, n_clusters=args.n0, steps=args.steps)

    # -----------------------
    # First run (with logs)
    # -----------------------
    print('[1/3] Running first simulation (detailed logs & visuals)...')
    first = run_single(seed=args.seed, params=params)
    sizes_series = first['sizes']   # list length T, arrays of sizes
    pos_series = first['positions']
    dt = float(first['dt'][0])
    W, H = first['space'][0]
    T = len(sizes_series)

    # Compute counts per time for first run
    counts_first = {}
    for t_idx, sz in enumerate(sizes_series):
        bc = bincount_sizes(np.asarray(sz))
        counts_first[t_idx] = bc

    M_first = aggregate_counts_over_runs(counts_first, T)  # shape (T, Smax+1)

    # Export per-time CSV for first run
    csv_first = os.path.join(out_base, 'first_run_size_counts.csv')
    os.makedirs(out_base, exist_ok=True)
    df_rows = []
    for t in range(T):
        for s in range(1, M_first.shape[1]):
            df_rows.append({'step': t, 'time_min': t*dt*MIN_PER_DT, 'size': s, 'count': int(M_first[t, s])})
    pd.DataFrame(df_rows).to_csv(csv_first, index=False)

    # Frames for histogram over time (first run)
    frames_dir = ensure_outdir(os.path.join(out_base, 'size_distribution_frames'))
    for t in range(T):
        out_png = os.path.join(frames_dir, f'frame_{t:04d}.png')
        plot_size_distribution_frame(M_first[t], t, dt, out_png)
    save_gif_from_folder(frames_dir, os.path.join(out_base, 'size_distribution.gif'), duration_ms=args.gif_ms)

    # Overlay every k steps (first run)
    overlay_png_first = os.path.join(out_base, 'size_distribution_every{:d}.png'.format(args.every_k))
    plot_overlay_every_k(M_first, dt, args.every_k, overlay_png_first, 'First run size distribution')

    # Spatial snapshots for first run (volume-preserving radii from sizes)
    snaps_dir = ensure_outdir(os.path.join(out_base, 'snapshots_frames'))
    for t in range(T):
        P = np.asarray(pos_series[t])
        S = np.asarray(sizes_series[t])
        if P.size == 0:
            P = np.zeros((0, 2))
            S = np.zeros((0, ))
        out_png = os.path.join(snaps_dir, f'step_{t:04d}.png')
        title = f'Spatial snapshot (t={t}, time={t*dt*MIN_PER_DT:.0f} min)'
        plot_snapshot(P, S, (W, H), out_png, title)
    save_gif_from_folder(snaps_dir, os.path.join(out_base, 'snapshots.gif'), duration_ms=args.gif_ms)

    # ---------------------------------------------------
    # Remaining runs for averages (2..N) – aggregate only
    # ---------------------------------------------------
    print('[2/3] Running remaining repeats to compute averages...')
    repeats = max(1, int(args.repeats))
    if repeats < 1:
        repeats = 1
    # Accumulators for counts
    sum_counts: Dict[int, np.ndarray] = {t: M_first[t].astype(float).copy() for t in range(T)}

    for r in range(2, repeats+1):
        seed_r = args.seed + r - 1
        run = run_single(seed=seed_r, params=params)
        sizes_series_r = run['sizes']
        Tr = min(T, len(sizes_series_r))
        for t in range(Tr):
            bc = bincount_sizes(np.asarray(sizes_series_r[t]))
            if bc.shape[0] > sum_counts[t].shape[0]:
                grow = bc.shape[0] - sum_counts[t].shape[0]
                sum_counts[t] = np.pad(sum_counts[t], (0, grow), mode='constant', constant_values=0.0)
            sum_counts[t][:bc.shape[0]] += bc

    # Compute means
    M_mean = aggregate_counts_over_runs(sum_counts, T)
    M_mean /= float(repeats)

    # Export mean CSV
    csv_avg = os.path.join(out_base, 'average_size_counts.csv')
    df_rows = []
    for t in range(T):
        for s in range(1, M_mean.shape[1]):
            df_rows.append({'step': t, 'time_min': t*dt*MIN_PER_DT, 'size': s, 'mean_count': M_mean[t, s]})
    pd.DataFrame(df_rows).to_csv(csv_avg, index=False)

    # Frames/GIF for mean distribution over time
    frames_dir_avg = ensure_outdir(os.path.join(out_base, 'avg_size_distribution_frames'))
    for t in range(T):
        out_png = os.path.join(frames_dir_avg, f'frame_{t:04d}.png')
        plot_size_distribution_frame(M_mean[t], t, dt, out_png)
    save_gif_from_folder(frames_dir_avg, os.path.join(out_base, 'avg_size_distribution.gif'),
                         duration_ms=args.gif_ms)

    overlay_png_avg = os.path.join(out_base, 'avg_size_distribution_every{:d}.png'.format(args.every_k))
    plot_overlay_every_k(M_mean, dt, args.every_k, overlay_png_avg, 'Average size distribution')

    # ---------------------------------------------------
    # Selected time plots (every 10 steps from t=30 to t=80 inclusive)
    # ---------------------------------------------------
    selected_times = list(range(30, 81, 10))
    selected_times = [t for t in selected_times if 0 <= t < T]

    # Block colour palette (non-viridis)
    block_palette = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
                     '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']

    # First run selected plots — block colours, x-axis cut at 100
    out_sel_first = os.path.join(out_base, 'selected_first_every10_t30_to_t80')
    save_selected_time_plots(M_first, dt, selected_times, out_sel_first,
                             'First run size distribution', palette=block_palette, xmax=100)
    # Combined line plot
    out_lines_first = os.path.join(out_sel_first, 'lines_selected_times.png')
    plot_lines_selected_times(M_first, dt, selected_times, out_lines_first,
                              'First run size distribution', palette=block_palette, xmax=100)

    # Average selected plots — block colours, x-axis cut at 100
    out_sel_avg = os.path.join(out_base, 'selected_average_every10_t30_to_t80')
    save_selected_time_plots(M_mean, dt, selected_times, out_sel_avg,
                             'Average size distribution', palette=block_palette, xmax=100)
    out_lines_avg = os.path.join(out_sel_avg, 'lines_selected_times.png')
    plot_lines_selected_times(M_mean, dt, selected_times, out_lines_avg,
                              'Average size distribution', palette=block_palette, xmax=100)

    # ---------------------------------------------------
    # Selected time plots (every 20 steps from t=30 to t=90 inclusive)
    # ---------------------------------------------------
    selected_times_20 = list(range(30, 91, 20))
    selected_times_20 = [t for t in selected_times_20 if 0 <= t < T]

    # First run selected plots — block colours, x-axis cut at 100
    out_sel_first_20 = os.path.join(out_base, 'selected_first_every20_t30_to_t90')
    save_selected_time_plots(M_first, dt, selected_times_20, out_sel_first_20,
                             'First run size distribution', palette=block_palette, xmax=100)
    # Combined line plot (block colours, x<=100)
    out_lines_first_20 = os.path.join(out_sel_first_20, 'lines_selected_times.png')
    plot_lines_selected_times(M_first, dt, selected_times_20, out_lines_first_20,
                              'First run size distribution', palette=block_palette, xmax=100)

    # Average selected plots — block colours, x-axis cut at 100
    out_sel_avg_20 = os.path.join(out_base, 'selected_average_every20_t30_to_t90')
    save_selected_time_plots(M_mean, dt, selected_times_20, out_sel_avg_20,
                             'Average size distribution', palette=block_palette, xmax=100)
    out_lines_avg_20 = os.path.join(out_sel_avg_20, 'lines_selected_times.png')
    plot_lines_selected_times(M_mean, dt, selected_times_20, out_lines_avg_20,
                              'Average size distribution', palette=block_palette, xmax=100)

    print('[3/3] Done. Outputs saved to:\n  ' + out_base)

if __name__ == '__main__':
    main()