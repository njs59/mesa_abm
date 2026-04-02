#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Colours (consistent with previous)
PRO_COLOR = (186/256, 29/256, 186/256)  # purple
INV_COLOR = (70/256, 158/256, 44/256)   # green
PCT_COLOR = 'C3'                        # red/orange

METRICS = ['n_clusters','mean_size','var_size','median_nnd','R','z']

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _compute_spatial(df_t: pd.DataFrame, W: float, H: float, torus: bool=True):
    if df_t.empty or len(df_t) < 2:
        return dict(median_nnd=np.nan, R=np.nan, z=np.nan)
    x = df_t['x'].to_numpy(float)
    y = df_t['y'].to_numpy(float)
    n = len(x); area = W * H
    dx = x[:,None] - x[None,:]
    dy = y[:,None] - y[None,:]
    if torus:
        dx = (dx + W/2) % W - W/2
        dy = (dy + H/2) % H - H/2
    d2 = dx*dx + dy*dy
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))
    r_obs = float(nn.mean())
    lam = n / area
    if lam <= 0: return dict(median_nnd=np.nan, R=np.nan, z=np.nan)
    r_exp = 1/(2*np.sqrt(lam))
    se = 0.26136 / np.sqrt(n*lam)
    if se <= 0: return dict(median_nnd=np.nan, R=np.nan, z=np.nan)
    z = (r_obs - r_exp) / se
    R = r_obs / r_exp
    return dict(median_nnd=float(np.median(nn)), R=float(R), z=float(z))

def _compute_timeseries(files: List[Path], W: float, H: float, torus: bool=True):
    if not files:
        raise RuntimeError('No files provided to compute timeseries')
    # Determine T
    T = 0
    for f in files:
        df = pd.read_csv(f, usecols=['step'])
        if len(df):
            T = max(T, int(df['step'].max())+1)
    per_rep = {m: [] for m in METRICS}
    for f in files:
        df = pd.read_csv(f)
        arr = {m: np.full(T, np.nan) for m in METRICS}
        for t in range(T):
            df_t = df[df['step'] == t]
            if df_t.empty: continue
            arr['n_clusters'][t] = len(df_t)
            sizes = df_t['size'].to_numpy(float)
            arr['mean_size'][t] = float(np.mean(sizes))
            arr['var_size'][t] = float(np.var(sizes, ddof=1)) if len(sizes) > 1 else 0.0
            spatial = _compute_spatial(df_t, W, H, torus)
            arr['median_nnd'][t] = spatial['median_nnd']
            arr['R'][t] = spatial['R']
            arr['z'][t] = spatial['z']
        for m in METRICS:
            per_rep[m].append(arr[m])
    out = {m: np.nanmean(np.vstack(per_rep[m]), axis=0) for m in METRICS}
    # dt from any file
    df0 = pd.read_csv(files[0], usecols=['step','time_min'])
    df0 = df0[df0.step > 0]
    dt = float(np.median(df0['time_min']/df0['step'])) if len(df0) else 1.0
    return out, T, dt

def _load_timeseries_for(results_dir: Path, scenario: str, cond: str, W: float, H: float, torus: bool=True):
    files = sorted((results_dir / 'trajectories' / scenario / cond).glob('repeat_*.csv'))
    if not files:
        raise RuntimeError(f'No trajectories found for {scenario}/{cond}')
    return _compute_timeseries(files, W, H, torus)

def plot_all_scenarios(*, inv_scenarios: List[str], pro_scenario: str, results_dir: Path, plots_dir: Path) -> None:
    from abm.utils import DEFAULTS
    W = float(DEFAULTS['space']['width'])
    H = float(DEFAULTS['space']['height'])
    torus = bool(DEFAULTS['space'].get('torus', True))

    # Load the single PRO baseline
    base_pro_ts, base_T, base_dt = _load_timeseries_for(results_dir, pro_scenario, 'PRO', W, H, torus)

    for sc in inv_scenarios:
        inv_ts, T_inv, _ = _load_timeseries_for(results_dir, sc, 'INV', W, H, torus)
        T = max(base_T, T_inv)
        x_axis = np.arange(T) * base_dt

        def pad(a, T):
            b = np.full(T, np.nan, dtype=float)
            b[:min(T, len(a))] = a[:min(T, len(a))]
            return b

        summary_dir = _ensure_dir(results_dir / 'summary')
        plots_s_dir = _ensure_dir(plots_dir / sc)
        pct_dir = _ensure_dir(plots_s_dir / 'pctdiff')

        # Save combined CSV (PRO baseline vs INV)
        rows = []
        for t in range(T):
            d = {'scenario': sc, 'time_min': x_axis[t], 'step': t}
            for m in METRICS:
                d[f'PRO_{m}'] = pad(base_pro_ts[m], T)[t]
                d[f'INV_{m}'] = pad(inv_ts[m], T)[t]
            rows.append(d)
        pd.DataFrame(rows).to_csv(summary_dir / f'{sc}__timeseries_PRObaseline_vs_INV.csv', index=False)

        # Raw plots
        for m in METRICS:
            y_pro = pad(base_pro_ts[m], T)
            y_inv = pad(inv_ts[m], T)
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(x_axis, y_inv, color=INV_COLOR, lw=2.2, label='INV')
            ax.plot(x_axis, y_pro, color=PRO_COLOR, lw=2.2, label='PRO baseline')
            ax.set_title(f'{sc}: {m} over time — INV vs PRO baseline')
            ax.set_xlabel('Time (min)'); ax.set_ylabel(m); ax.grid(alpha=0.3); ax.legend()
            fig.tight_layout()
            fig.savefig(plots_s_dir / f'{m}__INV_vs_PRObaseline.png', dpi=300)
            plt.close(fig)

        # % difference per scenario: INV vs baseline PRO
        for m in METRICS:
            y_pro = pad(base_pro_ts[m], T)
            y_inv = pad(inv_ts[m], T)
            with np.errstate(divide='ignore', invalid='ignore'):
                pct = 100.0 * (y_inv - y_pro) / y_pro
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(x_axis, pct, color=PCT_COLOR, lw=2.2)
            ax.axhline(0.0, color='black', lw=1.0, linestyle='--')
            ax.set_title(f'{sc}: % difference in {m} (INV vs PRO baseline)')
            ax.set_xlabel('Time (min)'); ax.set_ylabel('% difference (positive = INV > PRO)')
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(pct_dir / f'{m}__pctdiff_INV_vs_PRObaseline.png', dpi=300)
            plt.close(fig)

        print(f'[{sc}] Plots written to: {plots_s_dir}')

    # Combined % difference overlays (all INV variants vs PRO baseline)
    combo_dir = _ensure_dir(plots_dir / '_combined' / 'pctdiff')
    colours = ['C2', 'C1', 'C3', 'C4', 'C5']
    for m in METRICS:
        pct_all, labels, T_max = [], [], base_T
        for sc in inv_scenarios:
            inv_ts, T_inv, _ = _load_timeseries_for(results_dir, sc, 'INV', W, H, torus)
            T = max(base_T, T_inv); T_max = max(T_max, T)
            def pad(a, T):
                b = np.full(T, np.nan, dtype=float)
                b[:min(T, len(a))] = a[:min(T, len(a))]
                return b
            y_pro = pad(base_pro_ts[m], T)
            y_inv = pad(inv_ts[m], T)
            with np.errstate(divide='ignore', invalid='ignore'):
                pct_all.append(100.0 * (y_inv - y_pro) / y_pro)
            labels.append(sc)

        x_axis = np.arange(T_max) * base_dt
        fig, ax = plt.subplots(figsize=(10,6))
        for i, pct in enumerate(pct_all):
            ax.plot(x_axis, pct, lw=2.0, label=labels[i].replace('_',' '), color=colours[i % len(colours)])
        ax.axhline(0.0, color='black', lw=1.0, linestyle='--')
        ax.set_title(f'All scenarios: % difference in {m} (INV variants vs PRO baseline)')
        ax.set_xlabel('Time (min)'); ax.set_ylabel('% difference (positive = INV > PRO)')
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(combo_dir / f'{m}__pctdiff_INVvariants_vs_PRObaseline.png', dpi=300)
        plt.close(fig)