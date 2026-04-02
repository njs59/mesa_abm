#!/usr/bin/env python3
"""
Fit saturating time-window coagulation-rate model across parameter sweeps.

Run this script from the project ROOT (the directory that contains `Parameter_sweeps`):

    python Parameter_sweeps/fit_parameter_sweeps.py 2026-03-25_163508 \
        --window 25 --ycol P_pairs_smooth_mean

It will look under:
  ./Parameter_sweeps_results/<DATE_TAG>/coagulation_analysis_tagged/*/coag_prob.csv
and produce outputs under:
  ./Parameter_sweeps/fit_outputs/<DATE_TAG>/

Model fitted per sweep (vary one knob while holding the other fixed):
    Y(z) = S * (b*z) / (1 + b*z),   where z ∈ {p, a},  x ≡ b z.
Outputs per sweep include S, b, R^2, and implied x-range [b*z_min, b*z_max].
"""
from __future__ import annotations
import argparse
import os
import re
import sys
import glob
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# --------------------------
# Parsing helpers
# --------------------------
DECIMAL_P = re.compile(r"^(\d+)p(\d+)$")

def token_to_float(tok: Optional[str]) -> Optional[float]:
    """Convert strings like '0.7', '0p7', '1p25' to float; return None if not parseable."""
    if tok is None:
        return None
    s = tok.strip().lower()
    # patterns like '0p7' or '1p25'
    m = DECIMAL_P.match(s)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    # patterns like 'p0p7' (rare): treat as '0.7'
    if s.startswith('p') and 'p' in s[1:] and s[1:].replace('p','').isdigit():
        head, tail = s[1:].split('p', 1)
        return float(f"{head}.{tail}")
    try:
        return float(s)
    except Exception:
        return None

# default patterns for parameter tokens inside folder names
P_PATTERNS = [
    re.compile(r"(?:^|[_\-/])(?:p|merge|mergeprob|pmerge)[_\-]?([0-9]+(?:p[0-9]+|\.[0-9]+)?)", re.I),
]
A_PATTERNS = [
    re.compile(r"(?:^|[_\-/])a[_\-]?([0-9]+(?:p[0-9]+|\.[0-9]+)?)", re.I),
    re.compile(r"(?:^|[_\-/])amp|alpha|speed[_\-]?([0-9]+(?:p[0-9]+|\.[0-9]+)?)", re.I),
]

def parse_params_from_folder(folder: str) -> Dict[str, Optional[float]]:
    """Extract p and a from a folder name (scenario path segment)."""
    p_val: Optional[float] = None
    a_val: Optional[float] = None
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

# --------------------------
# Data and model
# --------------------------
def saturating_model(z, S, b):
    return S * (b*z) / (1 + b*z)

def fit_saturation(z: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    # Initial guesses: S ~ 1.2*max(y); b from initial slope if possible
    S0 = max(float(np.max(y)), 1e-12) * 1.2
    if len(z) >= 2:
        slope0 = (y[1]-y[0]) / (z[1]-z[0] + 1e-12)
        b0 = max(float(slope0 / (S0 + 1e-12)), 1e-6)
    else:
        b0 = 0.1
    popt, pcov = curve_fit(saturating_model, z, y, p0=[S0, b0], bounds=(0, np.inf), maxfev=50000)
    perr = np.sqrt(np.diag(pcov))
    S, b = map(float, popt)
    ci = {
        'S': (S - 1.96*float(perr[0]), S + 1.96*float(perr[0])),
        'b': (b - 1.96*float(perr[1]), b + 1.96*float(perr[1])),
    }
    params = {'S': S, 'b': b}
    return params, ci

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    return 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')

def window_average(series: pd.Series, steps: int) -> float:
    steps = min(int(steps), len(series))
    return float(series.iloc[:steps].mean())

# --------------------------
# Pipeline
# --------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fit saturation model to coag_prob.csv across parameter sweeps")
    ap.add_argument('date_folder', type=str, help="Folder name under Parameter_sweeps_results (e.g., 2026-03-25_163508)")
    ap.add_argument('--root', type=str, default='.', help="Root directory (run from project root; default '.')")
    ap.add_argument('--window', type=int, default=25, help="Number of initial steps to average (time window)")
    ap.add_argument('--ycol', type=str, default='P_pairs_smooth_mean', help="Column in coag_prob.csv to use as rate")
    ap.add_argument('--subdir', type=str, default='coagulation_analysis_tagged', help="Subfolder containing scenario folders")
    ap.add_argument('--outdir', type=str, default=None, help="Optional output dir; default Parameter_sweeps/fit_outputs/<date>")
    ap.add_argument('--overwrite', action='store_true', help="Overwrite existing output files")
    args = ap.parse_args(argv)

    # Paths
    results_root = os.path.join(args.root, 'Parameter_sweeps_results', args.date_folder, args.subdir)
    if not os.path.isdir(results_root):
        print(f"ERROR: results folder not found: {results_root}")
        return 2

    outdir = args.outdir or os.path.join(args.root, 'Parameter_sweeps', 'fit_outputs', args.date_folder)
    os.makedirs(outdir, exist_ok=True)

    # Discover scenario folders
    scenario_dirs = sorted([d for d in glob.glob(os.path.join(results_root, '*')) if os.path.isdir(d)])
    if not scenario_dirs:
        print(f"WARNING: no scenario folders found in {results_root}")

    rows = []
    for scen in scenario_dirs:
        csv_path = os.path.join(scen, 'coag_prob.csv')
        if not os.path.isfile(csv_path):
            # skip silently if scenario has no CSV
            continue
        params = parse_params_from_folder(os.path.basename(scen))
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"ERROR reading {csv_path}: {e}")
            continue
        if args.ycol not in df.columns:
            print(f"SKIP {csv_path}: missing column {args.ycol}")
            continue
        y_series = df[args.ycol]
        # Extract instantaneous and window-averaged values
        y0 = float(y_series.iloc[0])
        y_win = window_average(y_series, args.window)
        rows.append({
            'scenario': os.path.basename(scen),
            'csv_path': csv_path,
            'p': params['p'],
            'a': params['a'],
            'y0': y0,
            'y_win': y_win,
            'window_steps': args.window,
        })

    if not rows:
        print("No data rows collected. Nothing to fit.")
        return 0

    ag = pd.DataFrame(rows)
    agg_csv = os.path.join(outdir, 'coag_aggregate.csv')
    if os.path.exists(agg_csv) and not args.overwrite:
        print(f"NOTE: {agg_csv} exists. Use --overwrite to replace.")
    else:
        ag.to_csv(agg_csv, index=False)
        print(f"Wrote aggregate to {agg_csv}")

    # Fit per sweep: varying 'p' (group by 'a'), and varying 'a' (group by 'p')
    fits = []
    def do_fit(varying: str):
        fixed = 'a' if varying == 'p' else 'p'
        for fixed_val, g in ag.groupby(fixed, dropna=False):
            sub = g.dropna(subset=[varying, 'y_win']).sort_values(varying)
            if len(sub) < 3:
                continue
            z = sub[varying].values.astype(float)
            y = sub['y_win'].values.astype(float)
            try:
                params, ci = fit_saturation(z, y)
                S, b = params['S'], params['b']
                yhat = saturating_model(z, S, b)
                r2 = r2_score(y, yhat)
                x_min, x_max = float(b*np.min(z)), float(b*np.max(z))
                fits.append({
                    'varying': varying,
                    f'{fixed}_held': fixed_val,
                    'n_points': int(len(sub)),
                    'S': S,
                    'S_CI_lo': ci['S'][0],
                    'S_CI_hi': ci['S'][1],
                    'b': b,
                    'b_CI_lo': ci['b'][0],
                    'b_CI_hi': ci['b'][1],
                    'R2': r2,
                    'x_min': x_min,
                    'x_max': x_max,
                })
            except Exception as e:
                print(f"Fit failed for {varying}-sweep with {fixed}={fixed_val}: {e}")
    do_fit('p')
    do_fit('a')

    if fits:
        fit_csv = os.path.join(outdir, 'fit_summary.csv')
        df_fits = pd.DataFrame(fits)
        if os.path.exists(fit_csv) and not args.overwrite:
            print(f"NOTE: {fit_csv} exists. Use --overwrite to replace.")
        else:
            df_fits.to_csv(fit_csv, index=False)
            print(f"Wrote fit summary to {fit_csv}")
        # Pretty print to console
        with pd.option_context('display.max_columns', None,
                               'display.width', 160,
                               'display.float_format', '{:0.6g}'.format):
            print('\n=== FIT SUMMARY ===')
            print(df_fits.sort_values(['varying', 'R2'], ascending=[True, False]).to_string(index=False))
    else:
        print('No valid sweep groups to fit (need ≥3 points and both z, y present).')

    return 0


if __name__ == '__main__':
    sys.exit(main())
