# sobol_sensitivity/plot_sobol_neat_labels.py
# Reads Sobol results from a sibling folder 'sobol_results' and writes neat extra plots
# into each timestamped run directory. You can also supply a specific run dir on CLI.

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Project-relative locations
# ------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]        # repo root that contains sobol_sensitivity/ and sobol_results/
DEFAULT_RESULTS_DIR = PROJECT_ROOT / 'sobol_results'

# ------------------------------
# Utilities for labels
# ------------------------------
DEFAULT_SHORT_MAP = {
    'merge.p_merge': 'p_merge',
    'phenotypes.proliferative.prolif_rate': 'prolif',
    'phenotypes.proliferative.fragment_rate': 'fragment',
    'movement_v2.phase2.speed_dist.params.a': 'gamma_shape',
    'movement_v2.phase2.speed_dist.params.scale': 'gamma_scale',
    'movement_v2.phase2.turning.kappa': 'kappa',
}

def _auto_short_label(long_key: str) -> str:
    parts = long_key.split('.')
    if len(parts) <= 2:
        return long_key
    last2 = '.'.join(parts[-2:])
    if len(last2) <= 16:
        return last2
    return parts[-1]

def make_short_labels(params: List[str], custom_map: Optional[Dict[str, str]] = None) -> List[str]:
    m = dict(DEFAULT_SHORT_MAP)
    if custom_map:
        m.update(custom_map)
    out = []
    for p in params:
        out.append(m.get(p, _auto_short_label(p)))
    seen = {}
    uniq = []
    for label in out:
        if label not in seen:
            seen[label] = 1
            uniq.append(label)
        else:
            k = seen[label]
            seen[label] += 1
            uniq.append(f"{label}#{k}")
    return uniq

# ------------------------------
# Reading helpers
# ------------------------------
def read_run_dir(run_dir: Path):
    cfg_path = run_dir / 'config.json'
    npz_path = run_dir / 'sobol_arrays.npz'
    if not cfg_path.exists():
        raise FileNotFoundError(f'Missing config.json in {run_dir}')
    if not npz_path.exists():
        raise FileNotFoundError(f'Missing sobol_arrays.npz in {run_dir}')
    meta = json.loads(cfg_path.read_text())
    z = np.load(npz_path)
    arrays = {
        'S1': z['first_order'],
        'ST': z['total_order'],
        'S1_lo': z.get('S1_CI_low'),
        'S1_hi': z.get('S1_CI_high'),
        'ST_lo': z.get('ST_CI_low'),
        'ST_hi': z.get('ST_CI_high'),
    }
    return meta, arrays

# ------------------------------
# Plotting
# ------------------------------
def _errbars(center: np.ndarray, lo: Optional[np.ndarray], hi: Optional[np.ndarray]):
    if lo is None or hi is None:
        return None
    return np.vstack([center - lo, hi - center])

def plot_indices_grid(out_png: Path,
                      param_names: List[str],
                      output_labels: List[str],
                      S1: np.ndarray, ST: np.ndarray,
                      S1_lo: Optional[np.ndarray] = None, S1_hi: Optional[np.ndarray] = None,
                      ST_lo: Optional[np.ndarray] = None, ST_hi: Optional[np.ndarray] = None,
                      custom_label_map: Optional[Dict[str, str]] = None,
                      title: Optional[str] = None,
                      style: str = 'default'):
    plt.style.use(style)
    short = make_short_labels(param_names, custom_map=custom_label_map)
    d = len(param_names)
    s = len(output_labels)
    x = np.arange(d)
    ncols = 2
    nrows = int(np.ceil(s / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    for i in range(s):
        ax = axes[i]
        s1 = S1[i]
        st = ST[i]
        e1 = _errbars(s1, None if S1_lo is None else S1_lo[i], None if S1_hi is None else S1_hi[i])
        et = _errbars(st, None if ST_lo is None else ST_lo[i], None if ST_hi is None else ST_hi[i])
        ax.errorbar(x - 0.12, s1, yerr=e1, fmt='o', capsize=3, label='S1')
        ax.errorbar(x + 0.12, st, yerr=et, fmt='o', capsize=3, label='ST')
        ax.set_xticks(x, short, rotation=25, ha='right')
        ax.set_ylabel('Sobol index')
        ax.set_title(output_labels[i])
        ax.set_ylim(0, max(1.0, np.nanmax([S1, ST]) * 1.1))
        ax.grid(True, alpha=0.25)
        ax.legend()
    for j in range(s, len(axes)):
        axes[j].axis('off')
    if title:
        fig.suptitle(title, y=1.02, fontsize=12)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

def plot_heatmaps(out_png: Path,
                  param_names: List[str], output_labels: List[str],
                  S1: np.ndarray, ST: np.ndarray,
                  custom_label_map: Optional[Dict[str, str]] = None,
                  title: Optional[str] = None,
                  cmap: str = 'viridis'):
    short = make_short_labels(param_names, custom_map=custom_label_map)
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.7 * len(output_labels))), constrained_layout=True)
    for ax, mat, name in zip(axes, [S1, ST], ['S1 (first order)', 'ST (total order)']):
        im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=max(1.0, float(np.nanmax(mat))))
        ax.set_xticks(np.arange(len(short)), short, rotation=25, ha='right')
        ax.set_yticks(np.arange(len(output_labels)), output_labels)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.grid(False)
    if title:
        fig.suptitle(title, y=1.02)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

# ------------------------------
# Batch driver
# ------------------------------
def process_one_run(run_dir: Path, label_map: Optional[Dict[str, str]] = None, style: str = 'default'):
    meta, arrays = read_run_dir(run_dir)
    params = meta['parameters']
    outputs = meta['outputs']
    S1 = np.asarray(arrays['S1'], float)
    ST = np.asarray(arrays['ST'], float)
    S1_lo = arrays.get('S1_lo')
    S1_hi = arrays.get('S1_hi')
    ST_lo = arrays.get('ST_lo')
    ST_hi = arrays.get('ST_hi')
    title = f"Sobol indices: {run_dir.name}"
    out_dir = run_dir / 'extra_plots'
    out_dir.mkdir(exist_ok=True)
    plot_indices_grid(out_dir / 'indices_by_output_neat.png', params, outputs, S1, ST,
                      S1_lo, S1_hi, ST_lo, ST_hi, custom_label_map=label_map, title=title, style=style)
    plot_heatmaps(out_dir / 'heatmaps_S1_ST_neat.png', params, outputs, S1, ST,
                  custom_label_map=label_map, title=title)
    short = make_short_labels(params, label_map)
    pd.DataFrame({'parameter': params, 'short_label': short}).to_csv(out_dir / 'parameter_short_labels.csv', index=False)
    print(f"✓ Wrote: {out_dir / 'indices_by_output_neat.png'}")
    print(f"✓ Wrote: {out_dir / 'heatmaps_S1_ST_neat.png'}")
    print(f"✓ Wrote: {out_dir / 'parameter_short_labels.csv'}")

def _find_timestamp_dirs(parent: Path):
    return [p for p in parent.iterdir() if p.is_dir() and (p / 'config.json').exists() and (p / 'sobol_arrays.npz').exists()]

def main():
    # Behaviour:
    #   1) If a path is provided, use it (single run dir or parent dir with many runs).
    #   2) If no path is provided, default to PROJECT_ROOT/sobol_results.
    if len(sys.argv) >= 2:
        target = Path(sys.argv[1]).resolve()
    else:
        target = DEFAULT_RESULTS_DIR
    if not target.exists():
        print(f"Path not found: {target}")
        sys.exit(1)
    if (target / 'config.json').exists():
        process_one_run(target)
        return
    runs = _find_timestamp_dirs(target)
    if not runs:
        print('No Sobol run directories found under', target)
        sys.exit(1)
    for r in sorted(runs):
        print('Processing:', r)
        process_one_run(r)

if __name__ == '__main__':
    main()