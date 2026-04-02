#!/usr/bin/env python3
"""
Row/column fits over 2D (p_merge, a) sweeps for time-averaged coagulation rate.

Usage (run from project ROOT):
    python Parameter_sweeps/fit_parameter_sweeps_2d_rowcol.py <DATE_TAG> \
        --window 25 --ycol P_pairs_smooth_mean --subdir coagulation_analysis_tagged

It expects scenario folders at:
  ./Parameter_sweeps_results/<DATE_TAG>/<subdir>/*/coag_prob.csv
with folder names containing tokens for p and a (e.g., ...__a_0p5__merge_0p7).

For each unique row (fixed a), it fits the p-sweep model:   Y = S * (b_p * p)/(1 + b_p * p)
For each unique column (fixed p), it fits the a-sweep model: Y = S * (b_a * a)/(1 + b_a * a)
(These are consistent with the finite-window identity \bar R = S * x/(1+x), where x = p*K(a)*N0*T,
and with small-x behaviour \bar R ≈ S * b * z) .

It outputs:
  - coag_aggregate_2d.csv              (tidy table of p, a, y0, y_win)
  - row_fit_summary.csv                (fit per fixed-a row)
  - col_fit_summary.csv                (fit per fixed-p column)
  - x_surface_from_rowfits.png         (2D heatmap of x_row(a,p) = b_p(a) * p)
  - x_surface_from_colfits.png         (2D heatmap of x_col(p,a) = b_a(p) * a)
  - regime_from_rowfits.png            (2D map of regimes from row fits)
  - regime_from_colfits.png            (2D map of regimes from col fits)

Region thresholds used:
  - linear: x <= 0.05
  - soft saturation: 0.05 < x <= 0.5
  - strong saturation: x >= 1

All plots overlay the observed points (p,a) coloured by observed y_win.
"""
from __future__ import annotations
import argparse
import glob
import os
import re
import sys
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.optimize import curve_fit

# ---------------- parsing helpers ----------------
DECIMAL_P = re.compile(r"^(\d+)p(\d+)$")

def token_to_float(tok: Optional[str]) -> Optional[float]:
    if tok is None:
        return None
    s = tok.strip().lower()
    m = DECIMAL_P.match(s)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    if s.startswith('p') and 'p' in s[1:] and s[1:].replace('p','').isdigit():
        head, tail = s[1:].split('p', 1)
        return float(f"{head}.{tail}")
    try:
        return float(s)
    except Exception:
        return None

P_PATTERNS = [
    re.compile(r"(?:^|[_\-/])(?:p|merge|mergeprob|pmerge)[_\-]?([0-9]+(?:p[0-9]+|\.[0-9]+)?)", re.I),
]
A_PATTERNS = [
    re.compile(r"(?:^|[_\-/])a[_\-]?([0-9]+(?:p[0-9]+|\.[0-9]+)?)", re.I),
    re.compile(r"(?:^|[_\-/])(?:amp|alpha|speed)[_\-]?([0-9]+(?:p[0-9]+|\.[0-9]+)?)", re.I),
]

def parse_params_from_folder(folder: str) -> Dict[str, Optional[float]]:
    p_val = None
    a_val = None
    for pat in P_PATTERNS:
        m = pat.search(folder)
        if m:
            p_val = token_to_float(m.group(1))
            break
    for pat in A_PATTERNS:
        m = pat.search(folder)
        if m:
            a_val = token_to_float(m.group(1))
            break
    return {'p': p_val, 'a': a_val}

# ---------------- data utils ----------------
def window_average(series: pd.Series, steps: int) -> float:
    steps = min(steps, len(series))
    return float(series.iloc[:steps].mean())

# ---------------- model & fitting ----------------
def sat_model(z, S, b):
    # Y = S * (b z)/(1 + b z)
    return S * (b*z) / (1.0 + b*z)

def fit_1d(z: np.ndarray, y: np.ndarray) -> Tuple[Dict[str,float], Dict[str,float]]:
    # robust initial guesses
    y_max = float(np.max(y)) if len(y) else 0.0
    S0 = max(y_max*1.2, 1e-12)
    if len(z) >= 2:
        slope0 = (y[1]-y[0]) / (z[1]-z[0] + 1e-12)
        b0 = max(slope0/(S0+1e-12), 1e-6)
    else:
        b0 = 1.0 / (np.max(z)+1e-9)
    popt, pcov = curve_fit(sat_model, z, y, p0=[S0, b0], bounds=(0, np.inf), maxfev=80000)
    S, b = map(float, popt)
    perr = np.sqrt(np.diag(pcov))
    S_lo, S_hi = S - 1.96*float(perr[0]), S + 1.96*float(perr[0])
    b_lo, b_hi = b - 1.96*float(perr[1]), b + 1.96*float(perr[1])
    yhat = sat_model(z, S, b)
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    R2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
    res = {'S':S, 'b':b, 'R2':R2}
    ci = {'S_CI_lo':S_lo, 'S_CI_hi':S_hi, 'b_CI_lo':b_lo, 'b_CI_hi':b_hi}
    return res, ci

# ---------------- plotting helpers ----------------
def regime_class(x: np.ndarray) -> np.ndarray:
    # 0 linear, 1 soft, 2 strong
    reg = np.zeros_like(x, dtype=int)
    reg[(x > 0.05) & (x <= 0.5)] = 1
    reg[x >= 1.0] = 2
    return reg

def plot_heatmap(X: np.ndarray, p_vals: np.ndarray, a_vals: np.ndarray, pts_df: pd.DataFrame,
                 out_png: str, title: str, cmap='viridis', vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=150)
    im = ax.imshow(X, origin='lower', extent=[p_vals.min(), p_vals.max(), a_vals.min(), a_vals.max()],
                   aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('$x$ value')
    sc = ax.scatter(pts_df['p'], pts_df['a'], c=pts_df['y_win'], s=35, cmap='magma', edgecolor='k', linewidth=0.3)
    cb2 = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb2.set_label('Observed $\\overline{R}(T)$')
    ax.set_xlabel('$p_{\\mathrm{merge}}$')
    ax.set_ylabel('speed parameter $a$')
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def plot_regime_map(REG: np.ndarray, p_vals: np.ndarray, a_vals: np.ndarray, pts_df: pd.DataFrame,
                    out_png: str, title: str):
    reg_cmap = ListedColormap(['#1b7837', '#f6e8c3', '#b2182b'])  # linear, soft, strong
    bounds = [0, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, reg_cmap.N)
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=150)
    im = ax.imshow(REG, origin='lower', extent=[p_vals.min(), p_vals.max(), a_vals.min(), a_vals.max()],
                   aspect='auto', cmap=reg_cmap, norm=norm, alpha=0.9)
    sc = ax.scatter(pts_df['p'], pts_df['a'], c=pts_df['y_win'], s=35, cmap='viridis', edgecolor='k', linewidth=0.3)
    cb = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('Observed $\\overline{R}(T)$')
    ax.set_xlabel('$p_{\\mathrm{merge}}$')
    ax.set_ylabel('speed parameter $a$')
    ax.set_title(title + "\nlinear (green): x\\le0.05; soft (beige): 0.05<x\\le0.5; strong (red): x\\ge1")
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description='Row/column 1D fits across a 2D (p,a) sweep')
    ap.add_argument('date_folder', type=str, help='Folder under Parameter_sweeps_results (e.g., 2026-03-25_163508)')
    ap.add_argument('--root', type=str, default='.', help='Project root')
    ap.add_argument('--subdir', type=str, default='coagulation_analysis_tagged', help='Scenario container')
    ap.add_argument('--ycol', type=str, default='P_pairs_smooth_mean', help='Rate column in coag_prob.csv')
    ap.add_argument('--window', type=int, default=25, help='Initial steps to average')
    ap.add_argument('--outdir', type=str, default=None, help='Output directory')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    results_root = os.path.join(args.root, 'Parameter_sweeps_results', args.date_folder, args.subdir)
    if not os.path.isdir(results_root):
        print('ERROR: folder not found:', results_root)
        return 2
    outdir = args.outdir or os.path.join(args.root, 'Parameter_sweeps', 'fit_outputs', args.date_folder)
    os.makedirs(outdir, exist_ok=True)

    # collect data
    rows = []
    scen_dirs = sorted([d for d in glob.glob(os.path.join(results_root, '*')) if os.path.isdir(d)])
    for scen in scen_dirs:
        csv_path = os.path.join(scen, 'coag_prob.csv')
        if not os.path.isfile(csv_path):
            continue
        pars = parse_params_from_folder(os.path.basename(scen))
        p = pars['p']; a = pars['a']
        if p is None or a is None:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print('WARN: cannot read', csv_path, e)
            continue
        if args.ycol not in df.columns:
            print('WARN: missing column', args.ycol, 'in', csv_path)
            continue
        y_series = df[args.ycol]
        y0 = float(y_series.iloc[0])
        y_win = window_average(y_series, args.window)
        rows.append({'scenario': os.path.basename(scen), 'p': float(p), 'a': float(a), 'y0': y0, 'y_win': y_win})

    if not rows:
        print('No (p,a) scenarios found with the expected naming. Aborting.')
        return 0

    ag = pd.DataFrame(rows)
    agg_csv = os.path.join(outdir, 'coag_aggregate_2d.csv')
    if not os.path.exists(agg_csv) or args.overwrite:
        ag.to_csv(agg_csv, index=False)
        print('Wrote', agg_csv)

    # build grids
    p_vals = np.sort(ag['p'].unique())
    a_vals = np.sort(ag['a'].unique())
    P_grid, A_grid = np.meshgrid(p_vals, a_vals)

    # --- row fits (fixed a, sweep p) ---
    row_records = []
    b_row = np.full_like(a_vals, np.nan, dtype=float)
    S_row = np.full_like(a_vals, np.nan, dtype=float)
    for i, aval in enumerate(a_vals):
        sub = ag[ag['a']==aval].sort_values('p')
        if len(sub) < 3:
            continue
        z = sub['p'].values.astype(float)
        y = sub['y_win'].values.astype(float)
        try:
            res, ci = fit_1d(z, y)
            b_row[i] = res['b']; S_row[i] = res['S']
            row_records.append({'a': float(aval), 'n_points': int(len(sub)), **res, **ci,
                                'x_min': float(res['b']*np.min(z)), 'x_max': float(res['b']*np.max(z))})
        except Exception as e:
            print(f'Row fit failed for a={aval}:', e)

    row_csv = os.path.join(outdir, 'row_fit_summary.csv')
    pd.DataFrame(row_records).to_csv(row_csv, index=False)
    print('Wrote', row_csv)

    # x surface from row fits: x_row(a,p) = b_row(a) * p
    X_row = np.zeros_like(P_grid, dtype=float)
    for i, aval in enumerate(a_vals):
        X_row[i, :] = (b_row[i] if np.isfinite(b_row[i]) else np.nan) * p_vals
    REG_row = regime_class(X_row)

    # --- column fits (fixed p, sweep a) ---
    col_records = []
    b_col = np.full_like(p_vals, np.nan, dtype=float)
    S_col = np.full_like(p_vals, np.nan, dtype=float)
    for j, pval in enumerate(p_vals):
        sub = ag[ag['p']==pval].sort_values('a')
        if len(sub) < 3:
            continue
        z = sub['a'].values.astype(float)
        y = sub['y_win'].values.astype(float)
        try:
            res, ci = fit_1d(z, y)
            b_col[j] = res['b']; S_col[j] = res['S']
            col_records.append({'p': float(pval), 'n_points': int(len(sub)), **res, **ci,
                                'x_min': float(res['b']*np.min(z)), 'x_max': float(res['b']*np.max(z))})
        except Exception as e:
            print(f'Column fit failed for p={pval}:', e)

    col_csv = os.path.join(outdir, 'col_fit_summary.csv')
    pd.DataFrame(col_records).to_csv(col_csv, index=False)
    print('Wrote', col_csv)

    # x surface from column fits: x_col(p,a) = b_col(p) * a
    X_col = np.zeros_like(P_grid, dtype=float)
    for j, pval in enumerate(p_vals):
        X_col[:, j] = (b_col[j] if np.isfinite(b_col[j]) else np.nan) * a_vals
    REG_col = regime_class(X_col)

    # observed points df
    pts_df = ag[['p','a','y_win']].copy()

    # --- plots ---
    plot_heatmap(X_row, p_vals, a_vals, pts_df,
                 out_png=os.path.join(outdir,'x_surface_from_rowfits.png'),
                 title='x-surface from row fits (fixed a, fit over p)')
    plot_regime_map(REG_row, p_vals, a_vals, pts_df,
                    out_png=os.path.join(outdir,'regime_from_rowfits.png'),
                    title='Regimes from row fits: linear (green), soft (beige), strong (red)')

    plot_heatmap(X_col, p_vals, a_vals, pts_df,
                 out_png=os.path.join(outdir,'x_surface_from_colfits.png'),
                 title='x-surface from column fits (fixed p, fit over a)')
    plot_regime_map(REG_col, p_vals, a_vals, pts_df,
                    out_png=os.path.join(outdir,'regime_from_colfits.png'),
                    title='Regimes from column fits: linear (green), soft (beige), strong (red)')

    print('Saved surfaces and regime maps to', outdir)
    return 0

if __name__ == '__main__':
    sys.exit(main())