
#!/usr/bin/env python3
"""
Generate a folder of PNG files plotting posterior distributions for
PRO vs INV ABC runs (assumed same speed/motion model), one PNG per
parameter (overlayed histograms), plus a single combined panel PNG
containing all parameters.

Usage
-----
python make_posteriors_pngs.py \
  --pro_db results/abc_PRO_isotropic_constant_seed42_20260117_191004.db \
  --inv_db results/abc_INV_isotropic_constant_seed42_20260117_183012.db \
  --out_dir posteriors_pngs \
  --sharex --bins 40 --kde

Options
-------
--params p1 p2 ...   Restrict to these parameter names
--bins INT           Histogram bins (default 40)
--sharex             Use a shared x-range per parameter between PRO & INV
--kde                Try to overlay simple KDE curves (requires scipy); optional
--dpi INT            PNG resolution (default 200)
--transparent        PNGs with transparent background
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyabc


# ------------------------------ IO helpers ---------------------------------
def load_last_population(db_path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load last-population parameter samples and weights from a pyabc DB."""
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    hist = pyabc.History(f"sqlite:///{db_path}")
    tmax = hist.max_t
    if tmax is None:
        raise RuntimeError(f"No populations found in {db_path}")
    df, w = hist.get_distribution(t=tmax, m=0)
    w = np.asarray(w, dtype=float)
    # Drop any pyabc internal columns if present
    for col in list(df.columns):
        if col.startswith('_'):
            df = df.drop(columns=[col])
    return df, w


# ------------------------------ Plotting ------------------------------------
def overlay_hist(ax, name: str, pro_vals: np.ndarray, inv_vals: np.ndarray,
                 pro_w: np.ndarray, inv_w: np.ndarray, bins: int,
                 sharex: bool, kde: bool):
    """Overlay weighted histograms for PRO and INV on a single axis."""
    color_pro = '#d62728'
    color_inv = '#1f77b4'

    # Clean weights
    pro_w = np.clip(pro_w, 0, None)
    inv_w = np.clip(inv_w, 0, None)
    pro_w = pro_w / (pro_w.sum() if pro_w.sum() > 0 else 1.0)
    inv_w = inv_w / (inv_w.sum() if inv_w.sum() > 0 else 1.0)

    # Determine bin edges
    if sharex:
        lo = np.nanmin([np.nanmin(pro_vals), np.nanmin(inv_vals)])
        hi = np.nanmax([np.nanmax(pro_vals), np.nanmax(inv_vals)])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        bins_edges = np.linspace(lo, hi, bins + 1)
    else:
        bins_edges = bins  # let matplotlib pick per dataset

    ax.hist(inv_vals, bins=bins_edges, weights=inv_w, density=True,
            color=color_inv, alpha=0.55, edgecolor='white', label='INV')
    ax.hist(pro_vals, bins=bins_edges, weights=pro_w, density=True,
            color=color_pro, alpha=0.55, edgecolor='white', label='PRO')

    ax.set_title(name)
    ax.set_xlabel(name)
    ax.set_ylabel('Density')

    # Optionally overlay KDEs
    if kde:
        try:
            from scipy.stats import gaussian_kde
            if len(inv_vals) > 1:
                xs = np.linspace(*ax.get_xlim(), 256)
                kdi = gaussian_kde(inv_vals, weights=inv_w)
                ax.plot(xs, kdi(xs), color=color_inv, lw=1.3)
            if len(pro_vals) > 1:
                xs = np.linspace(*ax.get_xlim(), 256)
                kdp = gaussian_kde(pro_vals, weights=pro_w)
                ax.plot(xs, kdp(xs), color=color_pro, lw=1.3)
        except Exception:
            ax.text(0.02, 0.92, 'KDE unavailable', transform=ax.transAxes, fontsize=8)


def weighted_quantiles(vals: np.ndarray, w: np.ndarray, qs=(0.025, 0.5, 0.975)) -> List[float]:
    order = np.argsort(vals)
    v = np.asarray(vals)[order]
    w = np.clip(np.asarray(w)[order], 0, None)
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        cw = np.arange(1, len(cw) + 1)
    cw = cw / cw[-1]
    return [float(np.interp(q, cw, v)) for q in qs]


# ------------------------------ Main ----------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Create PNGs of posterior histograms for PRO vs INV.')
    ap.add_argument('--pro_db', required=True, type=Path)
    ap.add_argument('--inv_db', required=True, type=Path)
    ap.add_argument('--out_dir', type=Path, default=Path('posteriors_pngs'))
    ap.add_argument('--params', nargs='+', default=None)
    ap.add_argument('--bins', type=int, default=40)
    ap.add_argument('--sharex', action='store_true')
    ap.add_argument('--kde', action='store_true')
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('--transparent', action='store_true')
    args = ap.parse_args()

    # Load distributions
    pro_df, pro_w = load_last_population(args.pro_db)
    inv_df, inv_w = load_last_population(args.inv_db)

    # Common parameter set (preserve order from PRO)
    common = [p for p in pro_df.columns if p in inv_df.columns]
    if args.params:
        params = [p for p in args.params if p in common]
        missing = set(args.params) - set(params)
        if missing:
            print(f"[warn] Skipping unknown params: {sorted(missing)}")
    else:
        params = common

    if not params:
        raise RuntimeError('No overlapping parameters to plot.')

    # Output folder
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Styling (British English labels already set)
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1) Individual parameter PNGs (overlayed PRO/INV histograms)
    per_param_paths = []
    for p in params:
        fig, ax = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)
        overlay_hist(ax, p,
                     pro_df[p].values.astype(float), inv_df[p].values.astype(float),
                     pro_w, inv_w, bins=args.bins, sharex=args.sharex, kde=args.kde)
        ax.legend(loc='best', frameon=True)
        out_png = args.out_dir / f"{p}_PROvsINV.png"
        fig.savefig(out_png, dpi=args.dpi, transparent=args.transparent)
        plt.close(fig)
        per_param_paths.append(out_png)

    # 2) Combined grid PNG (all parameters together)
    n = len(params)
    ncols = 3 if n >= 3 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.2*nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])

    for i, p in enumerate(params):
        ax = axes[i]
        overlay_hist(ax, p,
                     pro_df[p].values.astype(float), inv_df[p].values.astype(float),
                     pro_w, inv_w, bins=args.bins, sharex=args.sharex, kde=args.kde)
        if i == 0:
            ax.legend(loc='best', frameon=True)

    # Hide any extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    grid_png = args.out_dir / 'all_params_PROvsINV.png'
    fig.savefig(grid_png, dpi=args.dpi, transparent=args.transparent)
    plt.close(fig)

    # 3) Save a small CSV summary with medians & 95% intervals (handy for captions)
    rows = []
    for p in params:
        pro_q = weighted_quantiles(pro_df[p].values.astype(float), pro_w)
        inv_q = weighted_quantiles(inv_df[p].values.astype(float), inv_w)
        rows.append({
            'param': p,
            'PRO_q2.5': pro_q[0], 'PRO_median': pro_q[1], 'PRO_q97.5': pro_q[2],
            'INV_q2.5': inv_q[0], 'INV_median': inv_q[1], 'INV_q97.5': inv_q[2],
        })
    pd.DataFrame(rows).to_csv(args.out_dir / 'posteriors_summary.csv', index=False)

    print({'out_dir': str(args.out_dir),
           'per_param_pngs': [str(p) for p in per_param_paths],
           'all_params_png': str(grid_png)})


if __name__ == '__main__':
    main()
