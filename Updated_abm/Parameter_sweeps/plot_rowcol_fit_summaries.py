#!/usr/bin/env python3
# Plot row/column fit summaries + extra x heatmaps and annotated regime maps.
# Usage:
#   python Updated_abm/Parameter_sweeps/plot_rowcol_fit_summaries.py <DATE_TAG>

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ---------- CLI ----------
if len(sys.argv) < 2:
    print("Usage: python Updated_abm/Parameter_sweeps/plot_rowcol_fit_summaries.py <DATE_TAG>")
    sys.exit(2)

DATE_TAG = sys.argv[1]
OUT_DIR  = os.path.join("Parameter_sweeps", "fit_outputs", DATE_TAG)
print(f"[plot_rowcol] Using OUT_DIR = {OUT_DIR}")

col_csv = os.path.join(OUT_DIR, "col_fit_summary.csv")
row_csv = os.path.join(OUT_DIR, "row_fit_summary.csv")
agg_csv = os.path.join(OUT_DIR, "coag_aggregate_2d.csv")

# Existence checks
missing = [p for p in (col_csv, row_csv, agg_csv) if not os.path.exists(p)]
if missing:
    print("[plot_rowcol] Missing the following required files:")
    for p in missing:
        print("  -", p)
    print("\nRun the fitter first with the same date-tag, e.g.:")
    print(f"  python Updated_abm/Parameter_sweeps/fit_parameter_sweeps_2d_rowcol.py {DATE_TAG} "
          "--window 25 --ycol P_pairs_smooth_mean --subdir coagulation_analysis_tagged --overwrite")
    sys.exit(2)

# ---------- Load data ----------
col = pd.read_csv(col_csv)  # p, n_points, S, b, R2, S_CI_lo, S_CI_hi, b_CI_lo, b_CI_hi, x_min, x_max
row = pd.read_csv(row_csv)  # a, n_points, S, b, R2, S_CI_lo, S_CI_hi, b_CI_lo, b_CI_hi, x_min, x_max
agg = pd.read_csv(agg_csv)  # p, a, y0, y_win

# Observed grid
p_vals = np.sort(agg['p'].unique().astype(float))
a_vals = np.sort(agg['a'].unique().astype(float))
p_min, p_max = float(p_vals.min()), float(p_vals.max())
a_min, a_max = float(a_vals.min()), float(a_vals.max())

# ---------- Helpers ----------
def safe_div(num, den):
    """
    Divide scalar/array 'num' by array-like 'den' with broadcasting.
    Returns an array like 'den'; where den<=0 or NaN, returns NaN.
    """
    den = np.asarray(den, dtype=float)
    out = np.full(den.shape, np.nan, dtype=float)
    mask = np.isfinite(den) & (den > 0)
    if np.isscalar(num):
        out[mask] = float(num) / den[mask]
    else:
        num_arr = np.asarray(num, dtype=float)
        try:
            out[mask] = (num_arr / den)[mask]
        except Exception:
            pass
    return out

def add_conf_band(ax, x, lo, hi, color, alpha=0.18, floor=1e-30):
    lo = np.maximum(np.asarray(lo, dtype=float), floor)
    hi = np.maximum(np.asarray(hi, dtype=float), floor)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)

def regime_class(x):
    """Return integer regime map: 0=linear (x<=0.05), 1=soft (0.05<x<=0.5), 2=strong (x>=1)."""
    reg = np.zeros_like(x, dtype=int)
    reg[(x > 0.05) & (x <= 0.5)] = 1
    reg[x >= 1.0] = 2
    return reg

def format_x_value(x):
    """Compact x label for annotation."""
    if not np.isfinite(x):
        return ""
    # Show with two sig figs; switch to scientific if >=100 or <0.01
    if (x >= 100) or (x < 0.01 and x > 0):
        return f"{x:.1e}"
    return f"{x:.2g}"

def plot_x_heatmap(X, p_vals, a_vals, pts_df, out_png, title, x_cmap='viridis', vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7.6, 6.0), dpi=150)
    im = ax.imshow(X, origin='lower',
                   extent=[p_vals.min(), p_vals.max(), a_vals.min(), a_vals.max()],
                   aspect='auto', cmap=x_cmap, vmin=vmin, vmax=vmax)
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('$x$ value')
    # observed points coloured by observed \bar R
    sc = ax.scatter(pts_df['p'], pts_df['a'], c=pts_df['y_win'], s=35,
                    cmap='magma', edgecolor='k', linewidth=0.3, alpha=0.9)
    cb2 = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb2.set_label('Observed $\\overline{R}(T)$')
    ax.set_xlabel('$p_{\\mathrm{merge}}$')
    ax.set_ylabel('speed parameter $a$')
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def plot_regime_with_text(X, p_vals, a_vals, pts_df, out_png, title):
    reg_cmap = ListedColormap(['#1b7837', '#f6e8c3', '#b2182b'])  # linear, soft, strong
    bounds = [0, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, reg_cmap.N)
    REG = regime_class(X)

    fig, ax = plt.subplots(figsize=(7.6, 6.0), dpi=150)
    ax.imshow(REG, origin='lower',
              extent=[p_vals.min(), p_vals.max(), a_vals.min(), a_vals.max()],
              aspect='auto', cmap=reg_cmap, norm=norm, alpha=0.95)
    # annotate with x values
    for i, a in enumerate(a_vals):
        for j, p in enumerate(p_vals):
            x_val = X[i, j]
            if not np.isfinite(x_val):
                continue
            label = format_x_value(x_val)
            # pick text colour for contrast
            tcolor = 'white' if REG[i, j] == 2 else 'black'
            ax.text(p, a, label, ha='center', va='center', fontsize=7.5, color=tcolor)
    # overlay observed points (for reference)
    sc = ax.scatter(pts_df['p'], pts_df['a'], c=pts_df['y_win'], s=20,
                    cmap='magma', edgecolor='k', linewidth=0.3, alpha=0.85)
    cb = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('Observed $\\overline{R}(T)$')
    ax.set_xlabel('$p_{\\mathrm{merge}}$')
    ax.set_ylabel('speed parameter $a$')
    ax.set_title(title + "\nlinear (green): x≤0.05; soft (beige): 0.05<x≤0.5; strong (red): x≥1")
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

# ---------- Prepare inputs for surfaces ----------
# sort by p / a
col = col.sort_values('p').reset_index(drop=True)
row = row.sort_values('a').reset_index(drop=True)

# Column-derived gain b_a(p) for each unique p in the grid
# Map p -> b (if a column exists)
b_col_map = {float(rp): float(rb) for rp, rb in zip(col['p'].values, col['b'].values)}
b_col = np.array([b_col_map.get(float(p), np.nan) for p in p_vals], dtype=float)  # shape [P]

# Row-derived gain b_p(a) for each unique a in the grid
b_row_map = {float(ra): float(rb) for ra, rb in zip(row['a'].values, row['b'].values)}
b_row = np.array([b_row_map.get(float(a), np.nan) for a in a_vals], dtype=float)  # shape [A]

# Build X surfaces:
# X_col[i,j] = b_col[j] * a_vals[i]   (fixed p, vary a)
# X_row[i,j] = b_row[i] * p_vals[j]   (fixed a, vary p)
A, P = len(a_vals), len(p_vals)
X_col = np.zeros((A, P), dtype=float)
X_row = np.zeros((A, P), dtype=float)
for j, p in enumerate(p_vals):
    X_col[:, j] = b_col[j] * a_vals
for i, a in enumerate(a_vals):
    X_row[i, :] = b_row[i] * p_vals

# observed points (for overlays)
pts_df = agg[['p','a','y_win']].copy()

# ---------- Existing overview/threshold plots (unchanged core) ----------
# (1) Column overview S(p), b_a(p), slope
col['slope']  = col['S'] * col['b']
col['a_lin']  = safe_div(0.05, col['b'])
col['a_star'] = safe_div(1.00,  col['b'])

fig, axes = plt.subplots(3, 1, figsize=(7.0, 10.0), dpi=150, sharex=True)
p = col['p'].values
ax = axes[0]
ax.semilogy(p, np.maximum(col['b'], 1e-30), marker='o', color='#1f77b4', label=r'$b_a(p)$')
if {'b_CI_lo','b_CI_hi'}.issubset(col.columns):
    add_conf_band(ax, p, col['b_CI_lo'], col['b_CI_hi'], '#1f77b4')
ax.set_ylabel(r'$b_a(p)$'); ax.grid(True, alpha=0.3); ax.legend(loc='best')
ax.set_title('Column fits (fixed $p$, fit over $a$): gain, plateau, and small-$a$ slope')
ax = axes[1]
ax.semilogy(p, np.maximum(col['S'], 1e-30), marker='s', color='#2ca02c', label=r'$S(p)$')
if {'S_CI_lo','S_CI_hi'}.issubset(col.columns):
    add_conf_band(ax, p, col['S_CI_lo'], col['S_CI_hi'], '#2ca02c')
ax.set_ylabel(r'$S(p)$'); ax.grid(True, alpha=0.3); ax.legend(loc='best')
ax = axes[2]
ax.plot(p, col['slope'], marker='d', color='#d62728', label=r'$m(p)=S\,b_a(p)$')
ax.set_xlabel(r'$p_{\mathrm{merge}}$'); ax.set_ylabel(r'$m(p)$ (rate per unit $a$)')
ax.grid(True, alpha=0.3); ax.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'col_fit_overview.png'))
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)
ax.plot(p, col['a_lin'],  color='#9467bd', marker='o', label=r'$a_{\mathrm{lin}}(p)=0.05/b_a$')
ax.plot(p, col['a_star'], color='#8c564b', marker='s', label=r'$a_\star(p)=1/b_a$')
ax.fill_between(p, a_min, a_max, color='#dddddd', alpha=0.3, label='observed $a$-range')
ax.set_ylim(bottom=0); ax.set_xlabel(r'$p_{\mathrm{merge}}$'); ax.set_ylabel(r'$a$ threshold')
ax.set_title('Column fits: regime thresholds vs $p$'); ax.grid(True, alpha=0.3); ax.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'col_thresholds.png'))
plt.close(fig)

# (2) Row overview S(a), b_p(a), slope
row['slope']  = row['S'] * row['b']
row['p_lin']  = safe_div(0.05, row['b'])
row['p_star'] = safe_div(1.00,  row['b'])

fig, axes = plt.subplots(3, 1, figsize=(7.0, 10.0), dpi=150, sharex=True)
a = row['a'].values
ax = axes[0]
ax.semilogy(a, np.maximum(row['b'], 1e-30), marker='o', color='#1f77b4', label=r'$b_p(a)$')
if {'b_CI_lo','b_CI_hi'}.issubset(row.columns):
    add_conf_band(ax, a, row['b_CI_lo'], row['b_CI_hi'], '#1f77b4')
ax.set_ylabel(r'$b_p(a)$'); ax.grid(True, alpha=0.3); ax.legend(loc='best')
ax.set_title('Row fits (fixed $a$, fit over $p$): gain, plateau, and small-$p$ slope')
ax = axes[1]
ax.semilogy(a, np.maximum(row['S'], 1e-30), marker='s', color='#2ca02c', label=r'$S(a)$')
if {'S_CI_lo','S_CI_hi'}.issubset(row.columns):
    add_conf_band(ax, a, row['S_CI_lo'], row['S_CI_hi'], '#2ca02c')
ax.set_ylabel(r'$S(a)$'); ax.grid(True, alpha=0.3); ax.legend(loc='best')
ax = axes[2]
ax.plot(a, row['slope'], marker='d', color='#d62728', label=r'$m(a)=S\,b_p(a)$')
ax.set_xlabel(r'speed parameter $a$'); ax.set_ylabel(r'$m(a)$ (rate per unit $p$)')
ax.grid(True, alpha=0.3); ax.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'row_fit_overview.png'))
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)
ax.plot(a, row['p_lin'],  color='#9467bd', marker='o', label=r'$p_{\mathrm{lin}}(a)=0.05/b_p$')
ax.plot(a, row['p_star'], color='#8c564b', marker='s', label=r'$p_\star(a)=1/b_p$')
ax.fill_between(a, p_min, p_max, color='#dddddd', alpha=0.3, label='observed $p$-range')
ax.set_ylim(bottom=0); ax.set_xlabel(r'speed parameter $a$'); ax.set_ylabel(r'$p$ threshold')
ax.set_title('Row fits: regime thresholds vs $a$'); ax.grid(True, alpha=0.3); ax.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'row_thresholds.png'))
plt.close(fig)

# ---------- NEW: x heatmaps + annotated regime maps ----------
# continuous x heatmap from column fits (x = b_a(p)*a)
plot_x_heatmap(X_col, p_vals, a_vals, pts_df,
               out_png=os.path.join(OUT_DIR, 'x_continuous_from_colfits.png'),
               title='x heatmap from column fits (fixed p, vary a)')

# continuous x heatmap from row fits (x = b_p(a)*p)
plot_x_heatmap(X_row, p_vals, a_vals, pts_df,
               out_png=os.path.join(OUT_DIR, 'x_continuous_from_rowfits.png'),
               title='x heatmap from row fits (fixed a, vary p)')

# annotated regime map (column fits)
plot_regime_with_text(X_col, p_vals, a_vals, pts_df,
                      out_png=os.path.join(OUT_DIR, 'x_regime_annotated_from_colfits.png'),
                      title='Regime map (from column fits) with per-cell x labels')

# annotated regime map (row fits)
plot_regime_with_text(X_row, p_vals, a_vals, pts_df,
                      out_png=os.path.join(OUT_DIR, 'x_regime_annotated_from_rowfits.png'),
                      title='Regime map (from row fits) with per-cell x labels')

print("Saved plots in:", OUT_DIR)