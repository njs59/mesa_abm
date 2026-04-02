#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process ABM sensitivity analysis outputs:
  • Read metrics_long.csv from the latest (or specified) analysis directory
  • Compare model-based K_eff to data-driven K_direct (window/full)
  • Fit a 2D surface model to K_direct over (a, p_merge)
  • Plot K_direct as a 3D surface (not a heatmap) and save figures
  • Save fitted predictions, residuals, and a short fit summary

Usage examples:
  python ABM_sensitivity/postprocess_k_direct.py --sweep-name Merge_speed_2x2 \
    --results-root ABM_sensitivity_results --which direct_window --model poly2

  python ABM_sensitivity/postprocess_k_direct.py --analysis-dir \
    ABM_sensitivity_results/Merge_speed_2x2/analysis_20260320_121500 \
    --which direct_full --model poly3

Outputs go to:
  <analysis-dir>/postfit_<timestamp>/
"""
from __future__ import annotations
import argparse
import os
import glob
import json
import datetime as _dt
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# -------------------------- helpers ------------------------------------

def find_latest_analysis_dir(results_root: str, sweep_name: str) -> str:
    base = os.path.join(results_root, sweep_name)
    cand = sorted([d for d in glob.glob(os.path.join(base, 'analysis_*')) if os.path.isdir(d)])
    if not cand:
        raise SystemExit(f"No analysis_* dirs found under {base}")
    return cand[-1]


def load_metrics(analysis_dir: str) -> pd.DataFrame:
    p = os.path.join(analysis_dir, 'tables', 'metrics_long.csv')
    if not os.path.isfile(p):
        raise SystemExit(f"metrics_long.csv not found at {p}")
    df = pd.read_csv(p)
    # normalise column names just in case
    df.columns = [c.strip() for c in df.columns]
    return df


def select_target(df: pd.DataFrame, which: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return (a, p, z, label) for the chosen target surface."""
    if which == 'direct_window':
        key = 'K_direct_window'
        label = 'K_direct (window)'
    elif which == 'direct_full':
        key = 'K_direct_full'
        label = 'K_direct (full-run)'
    elif which == 'keff':
        key = 'K_eff'
        label = 'K_eff (model-based)'
    else:
        raise SystemExit("--which must be one of: direct_window | direct_full | keff")
    if key not in df.columns:
        raise SystemExit(f"Column {key} not found in metrics_long.csv")
    sub = df[['a_shape','p_merge', key]].dropna()
    a = sub['a_shape'].to_numpy(dtype=float)
    p = sub['p_merge'].to_numpy(dtype=float)
    z = sub[key].to_numpy(dtype=float)
    return a, p, z, label


# -------------------------- models -------------------------------------

def design_matrix(a: np.ndarray, p: np.ndarray, model: str):
    a = a.astype(float); p = p.astype(float)
    if model == 'poly2':
        X = np.column_stack([
            np.ones_like(a),
            a, p,
            a*a, a*p, p*p,
        ])
        names = ('1','a','p','a2','ap','p2')
        return X, names
    elif model == 'poly3':
        X = np.column_stack([
            np.ones_like(a),
            a, p,
            a*a, a*p, p*p,
            a*a*a, (a*a)*p, a*(p*p), p*p*p,
        ])
        names = ('1','a','p','a2','ap','p2','a3','a2p','ap2','p3')
        return X, names
    elif model == 'loglin':
        # log K ≈ β0 + β1 log(a) + β2 log(p) + β3 log(a)log(p)
        X = np.column_stack([
            np.ones_like(a),
            np.log(np.clip(a, 1e-12, None)),
            np.log(np.clip(p, 1e-12, None)),
            np.log(np.clip(a, 1e-12, None)) * np.log(np.clip(p, 1e-12, None)),
        ])
        names = ('1','log_a','log_p','log_a*log_p')
        return X, names
    else:
        raise SystemExit("--model must be one of: poly2 | poly3 | loglin")


def fit_surface(a: np.ndarray, p: np.ndarray, z: np.ndarray, model: str) -> Dict[str, object]:
    if model == 'loglin':
        if np.any(z <= 0):
            raise SystemExit("loglin model requires positive target values")
        y = np.log(z)
    else:
        y = z
    X, names = design_matrix(a, p, model)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    if model == 'loglin':
        yhat_lin = np.exp(yhat)
        resid = z - yhat_lin
        rss = float(np.sum((resid)**2))
        tss = float(np.sum((z - np.mean(z))**2))
        r2 = 1.0 - rss/tss if tss > 0 else np.nan
        yhat_report = yhat_lin
    else:
        resid = z - yhat
        rss = float(np.sum((resid)**2))
        tss = float(np.sum((z - np.mean(z))**2))
        r2 = 1.0 - rss/tss if tss > 0 else np.nan
        yhat_report = yhat
    k = X.shape[1]; n = X.shape[0]
    aic = n*np.log(max(rss/n, 1e-300)) + 2*k
    bic = n*np.log(max(rss/n, 1e-300)) + k*np.log(max(n, 1))
    return {
        'coef': coef,
        'names': names,
        'r2': r2,
        'rss': rss,
        'aic': aic,
        'bic': bic,
        'pred': yhat_report,
        'resid': resid,
    }


def predict_on_grid(a_vals: np.ndarray, p_vals: np.ndarray, fit: Dict[str, object], model: str) -> np.ndarray:
    """
    Return Z on a grid with SHAPE = (len(p_vals), len(a_vals)) so that
    rows map to p (index) and columns map to a (columns).
    """
    A, P = np.meshgrid(a_vals, p_vals, indexing='xy')
    Xg, _ = design_matrix(A.ravel(), P.ravel(), model)
    yhat = Xg @ fit['coef']
    if model == 'loglin':
        yhat = np.exp(yhat)
    # reshape as rows=p, cols=a
    return yhat.reshape(len(p_vals), len(a_vals))


# -------------------------- plotting -----------------------------------

def plot_surface_3d(a: np.ndarray, p: np.ndarray, z: np.ndarray, title: str, out_png: str,
                    elev=30, azim=-60):
    fig = plt.figure(figsize=(8.0, 6.2), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    a_vals = np.unique(a); p_vals = np.unique(p)
    if a_vals.size * p_vals.size == a.size:
        A, P = np.meshgrid(a_vals, p_vals, indexing='xy')
        Z = z.reshape(len(p_vals), len(a_vals))  # rows=p, cols=a
        surf = ax.plot_surface(A, P, Z, cmap='viridis', edgecolor='none', alpha=0.95)
        fig.colorbar(surf, shrink=0.6, pad=0.1)
    else:
        pts = ax.scatter(a, p, z, c=z, cmap='viridis', s=18)
        fig.colorbar(pts, shrink=0.6, pad=0.1)
    ax.set_title(title)
    ax.set_xlabel('Gamma shape a')
    ax.set_ylabel('p_merge')
    ax.set_zlabel('K')
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


# ------------------------------- main ----------------------------------

def main():
    ap = argparse.ArgumentParser(description='Post-process K_direct surfaces and fit 2D models')
    ap.add_argument('--results-root', type=str, default='ABM_sensitivity_results')
    ap.add_argument('--sweep-name', type=str, default=None, help='Sweep folder name; used to auto-find latest analysis dir')
    ap.add_argument('--analysis-dir', type=str, default=None, help='Path to a specific analysis_*/ directory (overrides sweep-name)')
    ap.add_argument('--which', choices=['direct_window','direct_full','keff'], default='direct_window', help='Which target surface to fit/plot')
    ap.add_argument('--model', choices=['poly2','poly3','loglin'], default='poly2')
    ap.add_argument('--save-pred', action='store_true', help='Save fitted predictions as a CSV surface and PNG 3D surface')
    args = ap.parse_args()

    if args.analysis_dir is None:
        if args.sweep_name is None:
            raise SystemExit('Provide either --analysis-dir or --sweep-name')
        analysis_dir = find_latest_analysis_dir(args.results_root, args.sweep_name)
    else:
        analysis_dir = args.analysis_dir

    out_dir = os.path.join(analysis_dir, f'postfit_{_dt.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)

    df = load_metrics(analysis_dir)

    # Extract targets
    a, p, z, label = select_target(df, args.which)

    # Plot raw surface in 3D (not heatmap)
    plot_surface_3d(a, p, z, title=f'{label} surface (raw data)',
                    out_png=os.path.join(out_dir, f'{args.which}_surface_3d.png'))

    # Fit chosen model
    fit = fit_surface(a, p, z, args.model)

    # Save fit summary
    summary = {
        'target': args.which,
        'model': args.model,
        'r2': fit['r2'], 'rss': fit['rss'], 'aic': fit['aic'], 'bic': fit['bic'],
        'coef_names': fit['names'],
        'coef_values': [float(c) for c in fit['coef'].ravel().tolist()],
    }
    with open(os.path.join(out_dir, 'fit_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Residual diagnostics plot
    fig = plt.figure(figsize=(6.8, 4.8), dpi=150)
    plt.scatter(fit['pred'], fit['resid'], s=12, alpha=0.8)
    plt.axhline(0.0, color='k', lw=1)
    plt.xlabel('Predicted K')
    plt.ylabel('Residual (obs - pred)')
    plt.title(f'Residuals vs Predicted ({args.which}, {args.model})\nR2={fit["r2"]:.3f}, AIC={fit["aic"]:.1f}')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'residuals_vs_pred.png')); plt.close(fig)

    # Optionally save predicted surface on a grid and plot its 3D surface
    if args.save_pred:
        a_vals = np.sort(np.unique(a))
        p_vals = np.sort(np.unique(p))
        Zg = predict_on_grid(a_vals, p_vals, fit, args.model)   # rows=p, cols=a

        # Safety: if a downstream change ever flips orientation, transpose to match rows=p, cols=a
        if Zg.shape != (len(p_vals), len(a_vals)):
            Zg = Zg.T

        pred_csv = os.path.join(out_dir, f'{args.which}_{args.model}_pred_surface.csv')
        pd.DataFrame(Zg, index=p_vals, columns=a_vals).to_csv(pred_csv)

        # Plot fitted surface
        A, P = np.meshgrid(a_vals, p_vals, indexing='xy')
        plot_surface_3d(A.ravel(), P.ravel(), Zg.ravel(),
                        title=f'{label} fitted surface ({args.model})',
                        out_png=os.path.join(out_dir, f'{args.which}_{args.model}_surface_3d.png'))

    print(f'Post-processing complete. Outputs in: {out_dir}')


if __name__ == '__main__':
    main()