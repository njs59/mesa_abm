#!/usr/bin/env python3
"""forward_phase2_invasive_spaghetti_plots.py

As before, but with improved panel labels:
• A,B,C,D labels are placed OUTSIDE the axes and larger for all 2x2 grids.

Outputs:
- Individual distribution plots (spaghetti-all, first10 coloured, mean+CI) for each selected time
- A 2x2 distribution grid for two chosen timepoints (default 30 and 70)
  • columns: timepoints
  • top row: first-10 spaghetti (coloured)
  • bottom row: mean+CI
  • panels labelled A,B,C,D outside subplots
- Individual summary-statistic plots over time
- 2x2 summary-statistic grids for each condition (spaghetti-all, first10, mean+CI)
  • panels labelled A,B,C,D outside subplots

Parallel execution:
- Uses ProcessPoolExecutor; set --n_jobs

"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# Ensure we can import sibling package `abm`
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from abm.clusters_model import ClustersModel
from abm.utils import DEFAULTS

# User convention: dt=1 is 30 minutes
MIN_PER_DT = 30.0

# -----------------------------------------------------------------------------
# Parameter helpers
# -----------------------------------------------------------------------------

def clone_params(base: dict) -> dict:
    import copy
    return copy.deepcopy(base)


def prepare_params_phase2_invasive(base: dict, *, n_clusters: int, steps: int) -> dict:
    p = clone_params(base)
    p.setdefault('time', {})['steps'] = int(steps)
    p.setdefault('init', {})['n_clusters'] = int(n_clusters)
    p['init']['size'] = 1
    p['init']['phenotype'] = 'invasive'
    return p


def spawn_invasive_phase2_singletons(model: ClustersModel, n: int) -> None:
    for _ in range(int(n)):
        a = model.spawn_cluster(size=1, phenotype='invasive', phase_switch_time=0.0)
        a.movement_phase = 2
        a.phase_switch_time = float('-inf')


def run_single(seed: int, params: dict) -> Dict[str, List[np.ndarray]]:
    m = ClustersModel(params=params, seed=seed, init_clusters=[])
    n0 = int(params.get('init', {}).get('n_clusters', 800))
    spawn_invasive_phase2_singletons(m, n0)

    n_steps = int(params.get('time', {}).get('steps', 145))
    for _ in range(n_steps):
        m.step()

    return {
        'sizes': m.size_log,
        'positions': m.pos_log,
        'dt': float(getattr(m, 'dt', 1.0)),
    }

# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------

def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else float('nan')


def safe_var(x: np.ndarray) -> float:
    return float(np.var(x)) if x.size else float('nan')


def median_nearest_neighbour_distance(positions: np.ndarray) -> float:
    if positions is None or len(positions) < 2:
        return float('nan')
    pts = np.asarray(positions, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return float('nan')
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    return float(np.median(dists[:, 1]))


def size_distribution_counts(sizes: np.ndarray, xmax: int) -> np.ndarray:
    out = np.zeros((xmax,), dtype=np.float32)
    if sizes is None or len(sizes) == 0:
        return out
    s = np.asarray(sizes, dtype=int)
    s = s[(s >= 1) & (s <= xmax)]
    if s.size == 0:
        return out
    bc = np.bincount(s, minlength=xmax + 1)
    out[:] = bc[1:xmax + 1]
    return out


def size_distribution_prob(sizes: np.ndarray, xmax: int) -> np.ndarray:
    counts = size_distribution_counts(sizes, xmax).astype(np.float32)
    total = float(counts.sum())
    return (counts / total) if total > 0 else counts


def ci_band(arr: np.ndarray, *, level: float = 0.95, method: str = 'percentile') -> Tuple[np.ndarray, np.ndarray]:
    alpha = (1.0 - float(level)) / 2.0
    if method == 'percentile':
        low = np.nanpercentile(arr, 100.0 * alpha, axis=0)
        high = np.nanpercentile(arr, 100.0 * (1.0 - alpha), axis=0)
        return low, high
    if method == 'sem':
        from scipy.stats import norm
        z = float(norm.ppf(1.0 - alpha))
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0, ddof=1)
        n = np.sum(np.isfinite(arr), axis=0)
        sem = std / np.sqrt(np.maximum(n, 1))
        return mean - z * sem, mean + z * sem
    raise ValueError(f"Unknown CI method: {method}")

# -----------------------------------------------------------------------------
# Initial timepoint fix
# -----------------------------------------------------------------------------

def fix_initial_empty_log(sizes_log: List[np.ndarray], pos_log: List[np.ndarray], mode: str = 'copy_next') -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if mode == 'none':
        return sizes_log, pos_log
    if not sizes_log or not pos_log:
        return sizes_log, pos_log

    try:
        if len(sizes_log[0]) > 0:
            return sizes_log, pos_log
    except Exception:
        return sizes_log, pos_log

    first_non_empty = None
    for i in range(1, min(len(sizes_log), len(pos_log))):
        if sizes_log[i] is not None and len(sizes_log[i]) > 0:
            first_non_empty = i
            break
    if first_non_empty is None:
        return sizes_log, pos_log
    if mode == 'drop':
        return sizes_log[1:], pos_log[1:]

    sizes_log = list(sizes_log)
    pos_log = list(pos_log)
    sizes_log[0] = np.array(sizes_log[first_non_empty], copy=True)
    pos_log[0] = np.array(pos_log[first_non_empty], copy=True)
    return sizes_log, pos_log

# -----------------------------------------------------------------------------
# Parallel worker: simulate + compute reduced outputs
# -----------------------------------------------------------------------------

def worker_simulate_and_summarise(seed: int,
                                 params: dict,
                                 steps: int,
                                 selected_times: List[int],
                                 xmax: int,
                                 dist_mode: str,
                                 fix_t0: str) -> Dict[str, np.ndarray]:
    out = run_single(seed, params)
    sizes_log = out['sizes']
    pos_log = out['positions']
    dt = float(out.get('dt', 1.0))

    sizes_log, pos_log = fix_initial_empty_log(sizes_log, pos_log, mode=fix_t0)

    T = steps + 1
    n_clusters = np.full((T,), np.nan, dtype=np.float32)
    mean_size = np.full((T,), np.nan, dtype=np.float32)
    var_size = np.full((T,), np.nan, dtype=np.float32)
    med_nnd = np.full((T,), np.nan, dtype=np.float32)

    T_run = min(T, len(sizes_log), len(pos_log))
    for t in range(T_run):
        sizes = np.asarray(sizes_log[t], dtype=float)
        pos = np.asarray(pos_log[t], dtype=float)
        n_clusters[t] = float(sizes.size)
        mean_size[t] = safe_mean(sizes)
        var_size[t] = safe_var(sizes)
        med_nnd[t] = median_nearest_neighbour_distance(pos)

    dist = np.zeros((len(selected_times), xmax), dtype=np.float32)
    for j, t_sel in enumerate(selected_times):
        if t_sel < T_run:
            s = sizes_log[t_sel]
            dist[j, :] = size_distribution_counts(s, xmax) if dist_mode == 'counts' else size_distribution_prob(s, xmax)

    return {
        'dt': np.array([dt], dtype=np.float32),
        'n_clusters': n_clusters,
        'mean_size': mean_size,
        'var_size': var_size,
        'median_nnd': med_nnd,
        'dist': dist,
    }

# -----------------------------------------------------------------------------
# Plotting utilities
# -----------------------------------------------------------------------------

def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def time_axis(T: int, dt: float, units: str = 'h') -> np.ndarray:
    minutes = np.arange(T) * float(dt) * MIN_PER_DT
    return minutes if units == 'min' else minutes / 60.0


def time_label(units: str) -> str:
    return 'Time (min)' if units == 'min' else 'Time (h)'


def add_panel_labels_outside(fig, axes_2x2, *, fontsize: int = 28, dx: float = 0.014, dy: float = 0.010) -> None:
    """Add A–D labels OUTSIDE each subplot (top-left of each axes), in figure coords.

    fontsize: larger label size
    dx,dy: padding in figure coordinates.

    Saving with bbox_inches='tight' ensures labels aren't cropped.
    """
    labels = ['A', 'B', 'C', 'D']
    k = 0
    for r in range(2):
        for c in range(2):
            ax = axes_2x2[r][c]
            bb = ax.get_position()  # figure coordinates
            x = bb.x0 - dx
            y = bb.y1 + dy
            fig.text(x, y, labels[k], ha='right', va='bottom',
                     fontweight='bold', fontsize=fontsize)
            k += 1


def plot_summary_grid_2x2(t: np.ndarray,
                          series_dict: Dict[str, np.ndarray],
                          *,
                          titles: Dict[str, str],
                          ylabels: Dict[str, str],
                          mode: str,
                          out_png: str,
                          units: str,
                          repeats: int,
                          ci_level: float,
                          ci_method: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.6), squeeze=False)
    order = ['n_clusters', 'mean_size', 'var_size', 'median_nnd']

    for k, key in enumerate(order):
        ax = axes[k // 2][k % 2]
        Y = series_dict[key]

        if mode == 'spaghetti_all':
            for r in range(Y.shape[0]):
                ax.plot(t, Y[r], color='black', alpha=0.22, lw=0.9)
            ax.set_title(titles[key] + f" — spaghetti ({repeats} realisations)")

        elif mode == 'spaghetti_first10':
            cmap = plt.get_cmap('tab10')
            n = min(10, Y.shape[0])
            for r in range(n):
                ax.plot(t, Y[r], color=cmap(r), alpha=0.95, lw=1.6, label=f'Run {r+1}')
            ax.set_title(titles[key] + " — first 10 realisations")
            ax.legend(ncol=2, fontsize=8)

        elif mode == 'mean_ci':
            mean = np.nanmean(Y, axis=0)
            low, high = ci_band(Y, level=ci_level, method=ci_method)
            ax.plot(t, mean, color='#4c78a8', lw=2.4, label='Mean')
            ax.fill_between(t, low, high, color='#4c78a8', alpha=0.25,
                            label=f'{int(ci_level*100)}% CI')
            ax.set_title(titles[key] + " — mean and CI")
            ax.legend(fontsize=9)
        else:
            raise ValueError(f'Unknown mode: {mode}')

        ax.set_xlabel(time_label(units))
        ax.set_ylabel(ylabels[key])

    fig.tight_layout()
    add_panel_labels_outside(fig, axes, fontsize=28)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_distribution_grid_timepoints(dist_arr: np.ndarray,
                                      selected_times: List[int],
                                      *,
                                      timepoints: List[int],
                                      sizes_axis: np.ndarray,
                                      dt: float,
                                      out_png: str,
                                      ylab: str,
                                      ci_level: float,
                                      ci_method: str) -> None:
    idx_map = {t: j for j, t in enumerate(selected_times)}
    if len(timepoints) != 2:
        raise ValueError('timepoints must be length 2')
    if timepoints[0] not in idx_map or timepoints[1] not in idx_map:
        raise ValueError('Requested timepoints must be included in --selected_times')

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), squeeze=False)
    cmap = plt.get_cmap('tab10')

    for col, t_sel in enumerate(timepoints):
        j = idx_map[t_sel]
        dist_by_run = dist_arr[:, j, :]
        time_minutes = t_sel * dt * MIN_PER_DT

        # Top row: first-10 spaghetti
        ax = axes[0][col]
        n = min(10, dist_by_run.shape[0])
        for r in range(n):
            ax.plot(sizes_axis, dist_by_run[r], color=cmap(r), alpha=0.95, lw=1.6)
        ax.set_title(f't={t_sel} ({time_minutes:.0f} min) — first 10 spaghetti')
        ax.set_xlabel('Cluster size (cells)')
        ax.set_ylabel(ylab)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Bottom row: mean + CI
        ax2 = axes[1][col]
        mean = np.nanmean(dist_by_run, axis=0)
        low, high = ci_band(dist_by_run, level=ci_level, method=ci_method)
        ax2.plot(sizes_axis, mean, color='#4c78a8', lw=2.5, label='Mean')
        ax2.fill_between(sizes_axis, low, high, color='#4c78a8', alpha=0.25,
                         label=f'{int(ci_level*100)}% CI')
        ax2.set_title(f't={t_sel} ({time_minutes:.0f} min) — mean and CI')
        ax2.set_xlabel('Cluster size (cells)')
        ax2.set_ylabel(ylab)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.legend(fontsize=9)

    fig.tight_layout()
    add_panel_labels_outside(fig, axes, fontsize=32)  # a bit larger for this figure
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ---- individual plot helpers (unchanged behaviour) ---------------------------

def plot_spaghetti_all(dist_by_run: np.ndarray, x: np.ndarray, title: str, out_png: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for r in range(dist_by_run.shape[0]):
        ax.plot(x, dist_by_run[r], color='black', alpha=0.18, lw=0.9)
    ax.set_xlabel('Cluster size (cells)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_spaghetti_first10(dist_by_run: np.ndarray, x: np.ndarray, title: str, out_png: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    cmap = plt.get_cmap('tab10')
    n = min(10, dist_by_run.shape[0])
    for r in range(n):
        ax.plot(x, dist_by_run[r], color=cmap(r), alpha=0.95, lw=1.6, label=f'Run {r+1}')
    ax.set_xlabel('Cluster size (cells)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_mean_ci_curve(dist_by_run: np.ndarray, x: np.ndarray, title: str, out_png: str, ylabel: str,
                       ci_level: float, ci_method: str) -> None:
    mean = np.nanmean(dist_by_run, axis=0)
    low, high = ci_band(dist_by_run, level=ci_level, method=ci_method)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(x, mean, color='#4c78a8', lw=2.5, label='Mean')
    ax.fill_between(x, low, high, color='#4c78a8', alpha=0.25, label=f'{int(ci_level*100)}% CI')
    ax.set_xlabel('Cluster size (cells)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_time_spaghetti_all(series_by_run: np.ndarray, t: np.ndarray, title: str, out_png: str, ylabel: str, units: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for r in range(series_by_run.shape[0]):
        ax.plot(t, series_by_run[r], color='black', alpha=0.22, lw=0.9)
    ax.set_xlabel(time_label(units))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_time_spaghetti_first10(series_by_run: np.ndarray, t: np.ndarray, title: str, out_png: str, ylabel: str, units: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    cmap = plt.get_cmap('tab10')
    n = min(10, series_by_run.shape[0])
    for r in range(n):
        ax.plot(t, series_by_run[r], color=cmap(r), alpha=0.95, lw=1.6, label=f'Run {r+1}')
    ax.set_xlabel(time_label(units))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_time_mean_ci(series_by_run: np.ndarray, t: np.ndarray, title: str, out_png: str, ylabel: str, units: str,
                      ci_level: float, ci_method: str) -> None:
    mean = np.nanmean(series_by_run, axis=0)
    low, high = ci_band(series_by_run, level=ci_level, method=ci_method)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(t, mean, color='#4c78a8', lw=2.5, label='Mean')
    ax.fill_between(t, low, high, color='#4c78a8', alpha=0.25, label=f'{int(ci_level*100)}% CI')
    ax.set_xlabel(time_label(units))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Run N ABM realisations and plot (parallel).')
    parser.add_argument('--repeats', type=int, default=100, help='Number of repeats (default: 100)')
    parser.add_argument('--steps', type=int, default=int(DEFAULTS.get('time', {}).get('steps', 145)),
                        help='Number of steps per repeat')
    parser.add_argument('--n0', type=int, default=int(DEFAULTS.get('init', {}).get('n_clusters', 800)),
                        help='Initial number of invasive singletons')
    parser.add_argument('--seed', type=int, default=42, help='Base seed (run i uses seed + i)')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output folder (default: Forward_simulations/outputs/spaghetti_<timestamp>/)')

    parser.add_argument('--selected_times', type=int, nargs='*', default=[30, 50, 70, 90],
                        help='Time indices for distribution plots (default: 30 50 70 90)')
    parser.add_argument('--xmax_dist', type=int, default=100, help='Max cluster size to show')
    parser.add_argument('--dist_mode', choices=['counts', 'prob'], default='counts',
                        help='Plot distribution as raw counts or probability')

    parser.add_argument('--ci_level', type=float, default=0.95, help='CI level, e.g. 0.95 for 95%%')
    parser.add_argument('--ci_method', choices=['percentile', 'sem'], default='percentile', help='CI method')

    parser.add_argument('--time_units', choices=['min', 'h'], default='h', help='Time units')
    parser.add_argument('--n_jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help='Parallel workers (default: CPU count - 1)')
    parser.add_argument('--fix_t0', choices=['copy_next', 'drop', 'none'], default='copy_next',
                        help='Fix empty t=0 logs')

    parser.add_argument('--grid_timepoints', type=int, nargs=2, default=[30, 70],
                        help='Two timepoints for 2x2 distribution grid (default: 30 70)')

    args = parser.parse_args()

    repeats = int(args.repeats)
    steps = int(args.steps)
    n0 = int(args.n0)
    selected_times = sorted(set(int(t) for t in args.selected_times))
    xmax = int(args.xmax_dist)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_base = os.path.join(THIS_DIR, 'outputs', f'spaghetti_{timestamp}') if args.outdir is None else args.outdir
    ensure_outdir(out_base)

    params = prepare_params_phase2_invasive(DEFAULTS, n_clusters=n0, steps=steps)
    T = steps + 1

    n_clusters_arr = np.full((repeats, T), np.nan, dtype=np.float32)
    mean_size_arr = np.full((repeats, T), np.nan, dtype=np.float32)
    var_size_arr = np.full((repeats, T), np.nan, dtype=np.float32)
    med_nnd_arr = np.full((repeats, T), np.nan, dtype=np.float32)
    dist_arr = np.zeros((repeats, len(selected_times), xmax), dtype=np.float32)

    seeds = [int(args.seed) + i for i in range(repeats)]

    print(f"Running {repeats} realisations with {int(args.n_jobs)} workers…")
    dt_used = None

    with ProcessPoolExecutor(max_workers=int(args.n_jobs)) as ex:
        futures = {
            ex.submit(worker_simulate_and_summarise, seed, params, steps, selected_times, xmax, args.dist_mode, args.fix_t0): i
            for i, seed in enumerate(seeds)
        }
        completed = 0
        for fut in as_completed(futures):
            i = futures[fut]
            res = fut.result()
            if dt_used is None:
                dt_used = float(res['dt'][0])
            n_clusters_arr[i, :] = res['n_clusters']
            mean_size_arr[i, :] = res['mean_size']
            var_size_arr[i, :] = res['var_size']
            med_nnd_arr[i, :] = res['median_nnd']
            dist_arr[i, :, :] = res['dist']
            completed += 1
            if completed % max(1, repeats // 10) == 0:
                print(f"  {completed}/{repeats} completed")

    if dt_used is None:
        dt_used = 1.0

    # Distributions
    dist_dir = ensure_outdir(os.path.join(out_base, 'distributions_selected_times'))
    sizes_axis = np.arange(1, xmax + 1)
    ylab = 'Count' if args.dist_mode == 'counts' else 'Probability'

    for j, t_sel in enumerate(selected_times):
        dist_by_run = dist_arr[:, j, :]
        time_minutes = t_sel * dt_used * MIN_PER_DT
        title_base = f"Size distribution at t={t_sel} ({time_minutes:.0f} min)"

        plot_spaghetti_all(dist_by_run, sizes_axis,
                           title=title_base + f" — spaghetti ({repeats} realisations)",
                           out_png=os.path.join(dist_dir, f'spaghetti_dist_t{t_sel:03d}.png'),
                           ylabel=ylab)

        plot_spaghetti_first10(dist_by_run, sizes_axis,
                               title=title_base + " — first 10 realisations (coloured)",
                               out_png=os.path.join(dist_dir, f'spaghetti_first10_dist_t{t_sel:03d}.png'),
                               ylabel=ylab)

        plot_mean_ci_curve(dist_by_run, sizes_axis,
                           title=title_base + f" — mean and {int(args.ci_level*100)}% CI",
                           out_png=os.path.join(dist_dir, f'mean_ci_dist_t{t_sel:03d}.png'),
                           ylabel=ylab,
                           ci_level=float(args.ci_level),
                           ci_method=str(args.ci_method))

    # 2x2 distribution grid (first10 vs mean+CI) for two timepoints
    grid_tp = [int(args.grid_timepoints[0]), int(args.grid_timepoints[1])]
    out_grid_dist = os.path.join(dist_dir, f'grid_2x2_first10_and_meanCI_t{grid_tp[0]:03d}_t{grid_tp[1]:03d}.png')
    plot_distribution_grid_timepoints(dist_arr, selected_times,
                                      timepoints=grid_tp,
                                      sizes_axis=sizes_axis,
                                      dt=dt_used,
                                      out_png=out_grid_dist,
                                      ylab=ylab,
                                      ci_level=float(args.ci_level),
                                      ci_method=str(args.ci_method))

    # Summary statistics
    t_axis = time_axis(T, dt_used, units=args.time_units)
    summ_dir = ensure_outdir(os.path.join(out_base, 'summary_statistics'))

    def do_summary(series_by_run: np.ndarray, name: str, ylabel: str) -> None:
        plot_time_spaghetti_all(series_by_run, t_axis,
                                title=f"{ylabel} over time — spaghetti ({repeats} realisations)",
                                out_png=os.path.join(summ_dir, f'spaghetti_{name}.png'),
                                ylabel=ylabel,
                                units=args.time_units)

        plot_time_spaghetti_first10(series_by_run, t_axis,
                                    title=f"{ylabel} over time — first 10 realisations (coloured)",
                                    out_png=os.path.join(summ_dir, f'spaghetti_first10_{name}.png'),
                                    ylabel=ylabel,
                                    units=args.time_units)

        plot_time_mean_ci(series_by_run, t_axis,
                          title=f"{ylabel} over time — mean and CI",
                          out_png=os.path.join(summ_dir, f'mean_ci_{name}.png'),
                          ylabel=ylabel,
                          units=args.time_units,
                          ci_level=float(args.ci_level),
                          ci_method=str(args.ci_method))

    do_summary(n_clusters_arr, 'n_clusters', 'Number of clusters')
    do_summary(mean_size_arr, 'mean_cluster_size', 'Mean cluster size (cells)')
    do_summary(var_size_arr, 'var_cluster_size', 'Variance of cluster size')
    do_summary(med_nnd_arr, 'median_nnd', 'Median nearest-neighbour distance')

    grid_titles = {
        'n_clusters': 'Number of clusters',
        'mean_size': 'Mean cluster size',
        'var_size': 'Variance of cluster size',
        'median_nnd': 'Median nearest-neighbour distance',
    }
    grid_ylabels = {
        'n_clusters': 'Number of clusters',
        'mean_size': 'Mean cluster size (cells)',
        'var_size': 'Variance',
        'median_nnd': 'Median NND',
    }
    grid_series = {
        'n_clusters': n_clusters_arr,
        'mean_size': mean_size_arr,
        'var_size': var_size_arr,
        'median_nnd': med_nnd_arr,
    }

    plot_summary_grid_2x2(t_axis, grid_series,
                          titles=grid_titles,
                          ylabels=grid_ylabels,
                          mode='spaghetti_all',
                          out_png=os.path.join(summ_dir, 'grid_2x2_spaghetti_all.png'),
                          units=args.time_units,
                          repeats=repeats,
                          ci_level=float(args.ci_level),
                          ci_method=str(args.ci_method))

    plot_summary_grid_2x2(t_axis, grid_series,
                          titles=grid_titles,
                          ylabels=grid_ylabels,
                          mode='spaghetti_first10',
                          out_png=os.path.join(summ_dir, 'grid_2x2_spaghetti_first10.png'),
                          units=args.time_units,
                          repeats=repeats,
                          ci_level=float(args.ci_level),
                          ci_method=str(args.ci_method))

    plot_summary_grid_2x2(t_axis, grid_series,
                          titles=grid_titles,
                          ylabels=grid_ylabels,
                          mode='mean_ci',
                          out_png=os.path.join(summ_dir, 'grid_2x2_mean_ci.png'),
                          units=args.time_units,
                          repeats=repeats,
                          ci_level=float(args.ci_level),
                          ci_method=str(args.ci_method))

    np.savez_compressed(
        os.path.join(out_base, 'analysis_arrays.npz'),
        n_clusters=n_clusters_arr,
        mean_size=mean_size_arr,
        var_size=var_size_arr,
        median_nnd=med_nnd_arr,
        dt=np.array([dt_used], dtype=np.float32),
        min_per_dt=np.array([MIN_PER_DT], dtype=np.float32),
        selected_times=np.array(selected_times, dtype=int),
        xmax_dist=np.array([xmax], dtype=int),
        dist_mode=np.array([0 if args.dist_mode == 'counts' else 1], dtype=int),
        dist=dist_arr,
    )

    print('Done. Outputs saved to:')
    print('  ' + out_base)


if __name__ == '__main__':
    main()
