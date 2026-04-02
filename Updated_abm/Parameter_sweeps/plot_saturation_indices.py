#!/usr/bin/env python3
"""
Compute and plot:
  (1) Saturation (elasticity-based, model-free)
  (2) Saturation (x-fraction) continuous
  (3) Saturation (x-fraction) binned into 4 semantic colours
  (4) Step-change comparison at interior points (both orientations):
      compare (prev step) y(u_j)-y(u_{j-1}) vs (next step) y(u_{j+1})-y(u_j)
      and plot a scalar at u_j (e.g., at p=0.2 for 0.1→0.2 vs 0.2→0.3).

Inputs (in fit_outputs/<DATE_TAG>/):
  - col_fit_summary.csv : columns [p, b]  → b_a(p)
  - row_fit_summary.csv : columns [a, b]  → b_p(a)
  - coag_aggregate_2d.csv: columns [p, a, <ycol>]  (default ycol=y_win)

Outputs (all written to the same directory):
  - sat_elasticity_from_colfits.png
  - sat_elasticity_from_rowfits.png
  - sat_xfrac_from_colfits.png
  - sat_xfrac_from_rowfits.png
  - sat_xfrac_bins_from_colfits.png
  - sat_xfrac_bins_from_rowfits.png
  - step_asym_from_colfits.png           (fixed p, vary a; metric=--step_metric)
  - step_asym_from_rowfits.png           (fixed a, vary p; metric=--step_metric)

Note: This mirrors the grid and x=b*control construction used by your
      `plot_rowcol_fit_2.py` (which created x_text_mag4_from_colfits.png).  # [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/plot_rowcol_fit_2.py)
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

# -------------------------- CLI --------------------------
ap = argparse.ArgumentParser(description="Saturation indices + step-change comparison plots for ABM parameter sweeps.")
ap.add_argument("date_tag", help="Subfolder under Updated_abm/Parameter_sweeps/fit_outputs/")
ap.add_argument("--ycol", default="y_win",
                help="Measured response column in coag_aggregate_2d.csv (default: y_win).")
ap.add_argument("--hill", type=float, default=1.0,
                help="Hill exponent h for Sx = x^h/(1+x^h) (default: 1.0).")
ap.add_argument("--smooth", type=int, default=1,
                help="Optional moving-average window (odd int) before differencing; 1 disables smoothing.")
ap.add_argument("--show_labels", action="store_true",
                help="Overlay labels in cells (percent for Sx; numeric for others).")
ap.add_argument("--fontsize", type=float, default=8.5,
                help="Font size for optional numeric cell labels.")
ap.add_argument("--sx_bins", nargs=3, type=float, default=[0.01, 0.10, 0.50],
                help="Three Sx thresholds for 4-bin colours (default: 0.01 0.10 0.50).")
ap.add_argument("--step_metric", choices=["asym", "diff", "ratio"], default="asym",
                help="Metric for step-change comparison (default: asym).")
args = ap.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # Updated_abm/Parameter_sweeps
OUT_DIR = os.path.join(SCRIPT_DIR, "fit_outputs", args.date_tag)
print(f"[plots] OUT_DIR = {OUT_DIR}")

# -------------------------- Inputs --------------------------
col_csv = os.path.join(OUT_DIR, "col_fit_summary.csv")   # p, b  => b_a(p)
row_csv = os.path.join(OUT_DIR, "row_fit_summary.csv")   # a, b  => b_p(a)
agg_csv = os.path.join(OUT_DIR, "coag_aggregate_2d.csv") # p, a, y...

for f in (col_csv, row_csv, agg_csv):
    if not os.path.exists(f):
        sys.exit(f"[plots] Missing required file: {f}\n"
                 "These are produced by your fitter and used by plot_rowcol_fit_2.py.  # [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/plot_rowcol_fit_2.py)")

# -------------------------- Load & grid --------------------------
col = pd.read_csv(col_csv).sort_values('p').reset_index(drop=True)  # columns: p, b
row = pd.read_csv(row_csv).sort_values('a').reset_index(drop=True)  # columns: a, b
agg = pd.read_csv(agg_csv)

# unique axes
p_vals = np.sort(agg['p'].astype(float).unique())
a_vals = np.sort(agg['a'].astype(float).unique())
P, A = len(p_vals), len(a_vals)

# maps for gains
b_a_map = {float(p): float(b) for p, b in zip(col['p'].values, col['b'].values)}  # fixed p -> b_a(p)
b_p_map = {float(a): float(b) for a, b in zip(row['a'].values, row['b'].values)}  # fixed a -> b_p(a)

# Build x surfaces as in your plotting script  # [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/plot_rowcol_fit_2.py)
X_col = np.zeros((A, P), dtype=float)  # fixed p: x = b_a(p) * a   (rows: a, cols: p)
for j, p in enumerate(p_vals):
    bj = b_a_map.get(float(p), np.nan)
    X_col[:, j] = bj * a_vals

X_row = np.zeros((A, P), dtype=float)  # fixed a: x = b_p(a) * p
for i, a in enumerate(a_vals):
    bi = b_p_map.get(float(a), np.nan)
    X_row[i, :] = bi * p_vals

# Build Y grid (pivot measured response)
if args.ycol not in agg.columns:
    # attempt fallbacks
    candidates = [c for c in ["P_pairs_smooth_mean", "y_win", "y", "y0"] if c in agg.columns]
    if not candidates:
        sys.exit(f"[plots] ycol '{args.ycol}' not in {agg_csv} and no fallback column is available.")
    print(f"[plots] WARNING: ycol '{args.ycol}' not found; using '{candidates[0]}' instead.")
    args.ycol = candidates[0]

Y_grid = (agg.pivot_table(index="a", columns="p", values=args.ycol)
          .reindex(index=a_vals, columns=p_vals).values.astype(float))

# -------------------------- Helpers --------------------------
def centres_to_edges(centres: np.ndarray) -> np.ndarray:
    """Convert bin centres to edges (uneven spacing OK)."""
    c = np.asarray(centres, dtype=float)
    if c.size == 1:
        w = 0.5 * (abs(c[0]) + 1.0) * 1e-3
        return np.array([c[0] - w, c[0] + w], dtype=float)
    mids = 0.5 * (c[:-1] + c[1:])
    first_edge = c[0] - (c[1] - c[0]) / 2.0
    last_edge  = c[-1] + (c[-1] - c[-2]) / 2.0
    return np.concatenate([[first_edge], mids, [last_edge]])

def smooth_1d(y, win: int):
    if win <= 1: return y.copy()
    win = int(max(1, win))
    if win % 2 == 0: win += 1
    k = np.ones(win, dtype=float)/win
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode='reflect')
    ys = np.convolve(ypad, k, mode='valid')
    return ys

# -------------------------- (1) Elasticity-based saturation --------------------------
def elasticity_saturation_from_cols(a_vals, p_vals, Y, smooth_win=1):
    """Fixed p_j, vary a: E = (a/y)*dy/da; S = 1 - E/E0 (clip [0,1])."""
    A, P = Y.shape
    S = np.zeros_like(Y, dtype=float)
    eps = 1e-12
    for j in range(P):
        y = Y[:, j].astype(float)
        y_s = smooth_1d(y, smooth_win)
        dy_da = np.gradient(y_s, a_vals)
        E = (a_vals / np.maximum(y_s, eps))*dy_da
        mask = np.isfinite(E)
        if not np.any(mask):
            S[:, j] = 0.0
            continue
        valid_idx = np.where(mask)[0]
        ref_idx = valid_idx[:min(2, len(valid_idx))]
        E0 = np.mean(E[ref_idx])
        if abs(E0) < 1e-8: E0 = 1.0
        S[:, j] = np.clip(1.0 - E/E0, 0.0, 1.0)
    return S

def elasticity_saturation_from_rows(a_vals, p_vals, Y, smooth_win=1):
    """Fixed a_i, vary p: E = (p/y)*dy/dp; S = 1 - E/E0 (clip [0,1])."""
    A, P = Y.shape
    S = np.zeros_like(Y, dtype=float)
    eps = 1e-12
    for i in range(A):
        y = Y[i, :].astype(float)
        y_s = smooth_1d(y, smooth_win)
        dy_dp = np.gradient(y_s, p_vals)
        E = (p_vals / np.maximum(y_s, eps))*dy_dp
        mask = np.isfinite(E)
        if not np.any(mask):
            S[i, :] = 0.0
            continue
        valid_idx = np.where(mask)[0]
        ref_idx = valid_idx[:min(2, len(valid_idx))]
        E0 = np.mean(E[ref_idx])
        if abs(E0) < 1e-8: E0 = 1.0
        S[i, :] = np.clip(1.0 - E/E0, 0.0, 1.0)
    return S

S_el_col = elasticity_saturation_from_cols(a_vals, p_vals, Y_grid, smooth_win=args.smooth)
S_el_row = elasticity_saturation_from_rows(a_vals, p_vals, Y_grid, smooth_win=args.smooth)

# -------------------------- (2) x-fraction saturation (continuous) --------------------------
def sat_xfrac(X, h=1.0):
    X = np.asarray(X, dtype=float)
    h = max(h, 1e-9)
    Xh = np.power(np.maximum(X, 0.0), h)
    return Xh/(1.0 + Xh)

Sx_col = sat_xfrac(X_col, h=args.hill)  # fixed p, vary a
Sx_row = sat_xfrac(X_row, h=args.hill)  # fixed a, vary p

# -------------------------- (3) x-fraction saturation (binned) --------------------------
def sx_to_categories(Sx: np.ndarray, thr=(0.01, 0.10, 0.50)) -> np.ndarray:
    Sx = np.asarray(Sx, dtype=float)
    cat = np.zeros_like(Sx, dtype=int)
    cat[(Sx >= thr[0]) & (Sx < thr[1])] = 1
    cat[(Sx >= thr[1]) & (Sx < thr[2])] = 2
    cat[Sx >= thr[2]] = 3
    return cat

def build_satx_legend_handles(thr=(0.01, 0.10, 0.50)):
    colors = ['#b2182b', '#f46d43', '#fee08b', '#1a9850']  # red, orange, yellow, green
    labels = [
        f"Red   : Sx < {thr[0]:.2g} — deep linear",
        f"Orange: {thr[0]:.2g}–{thr[1]:.2g} — early saturation",
        f"Yellow: {thr[1]:.2g}–{thr[2]:.2g} — clear concavity",
        f"Green : ≥ {thr[2]:.2g} — strong saturation (≥ half‑plateau)",
    ]
    return [Patch(facecolor=c, edgecolor='none', label=lab) for c, lab in zip(colors, labels)], colors

# -------------------------- (4) Step-change comparison --------------------------
def step_compare_1d(y: np.ndarray, metric: str = "asym", eps: float = 1e-12):
    """
    Given a 1D series y[u_0..u_{K-1}], compute a step comparison value per interior point k=1..K-2:
      delta_prev = y[k]   - y[k-1]
      delta_next = y[k+1] - y[k]
    metric:
      - 'asym' : (delta_next - delta_prev) / (|delta_next| + |delta_prev| + eps)  ∈ [-1,1]
      - 'diff' : delta_next - delta_prev
      - 'ratio': (|delta_next| + eps) / (|delta_prev| + eps)
    Returns an array same length as y with NaN at the edges and metric at interior points.
    """
    y = np.asarray(y, dtype=float)
    K = len(y)
    out = np.full(K, np.nan, dtype=float)
    if K < 3: return out
    dp = y[1:-1] - y[:-2]     # delta_prev at positions 1..K-2  (y[k]-y[k-1])
    dn = y[2:]   - y[1:-1]    # delta_next at positions 1..K-2  (y[k+1]-y[k])
    if metric == "asym":
        denom = np.abs(dn) + np.abs(dp) + eps
        out[1:-1] = (dn - dp)/denom
    elif metric == "diff":
        out[1:-1] = dn - dp
    elif metric == "ratio":
        out[1:-1] = (np.abs(dn) + eps)/(np.abs(dp) + eps)
    else:
        out[1:-1] = (dn - dp)/(np.abs(dn) + np.abs(dp) + eps)
    return out

def step_compare_from_cols(a_vals, p_vals, Y, metric="asym", smooth_win=1):
    """
    Fixed p_j, vary a along column → compute step comparison per interior a_i and place it at a_i.
    Returns A x P matrix with NaN in first/last rows (no interior point).
    """
    A, P = Y.shape
    M = np.full_like(Y, np.nan, dtype=float)
    for j in range(P):
        y = Y[:, j].astype(float)
        y_s = smooth_1d(y, smooth_win)
        M[:, j] = step_compare_1d(y_s, metric=metric)
    return M

def step_compare_from_rows(a_vals, p_vals, Y, metric="asym", smooth_win=1):
    """
    Fixed a_i, vary p along row → compute step comparison per interior p_j and place it at p_j.
    Returns A x P matrix with NaN in first/last columns.
    """
    A, P = Y.shape
    M = np.full_like(Y, np.nan, dtype=float)
    for i in range(A):
        y = Y[i, :].astype(float)
        y_s = smooth_1d(y, smooth_win)
        M[i, :] = step_compare_1d(y_s, metric=metric)
    return M

M_step_col = step_compare_from_cols(a_vals, p_vals, Y_grid, metric=args.step_metric, smooth_win=args.smooth)
M_step_row = step_compare_from_rows(a_vals, p_vals, Y_grid, metric=args.step_metric, smooth_win=args.smooth)

# -------------------------- Plot helpers --------------------------
def plot_matrix_continuous(Z, p_vals, a_vals, title, outfile, xlabel, ylabel,
                           vmin=None, vmax=None, cmap="viridis",
                           show_labels=False, fmt=".2f", fontsize=8.0, cbar_label="Value"):
    p_edges = centres_to_edges(p_vals); a_edges = centres_to_edges(a_vals)
    fig, ax = plt.subplots(figsize=(8.8, 6.9), dpi=150)
    m = ax.pcolormesh(p_edges, a_edges, Z, cmap=cmap,
                      norm=Normalize(vmin=vmin, vmax=vmax), shading='flat')
    for x in p_edges: ax.axvline(x, color='0.4', lw=0.5, alpha=0.25)
    for y in a_edges: ax.axhline(y, color='0.4', lw=0.5, alpha=0.25)
    ax.set_xticks(p_vals); ax.set_yticks(a_vals)
    ax.set_xlim(p_edges[0], p_edges[-1]); ax.set_ylim(a_edges[0], a_edges[-1])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8, fontsize=12, fontweight="bold")
    if show_labels:
        for i, a in enumerate(a_vals):
            for j, p in enumerate(p_vals):
                val = Z[i, j]
                if np.isfinite(val):
                    ax.text(p, a, format(val, fmt), ha='center', va='center',
                            fontsize=fontsize, color='white' if (vmin is not None and vmax is not None and
                                                                 (val > 0.66*(vmax-vmin)+vmin)) else 'black')
    cb = fig.colorbar(ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                      ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print("Saved:", outfile)

def plot_satx_binned(Sx, p_vals, a_vals, outfile, title, xlabel, ylabel,
                     thr=(0.01, 0.10, 0.50), show_labels=False, fontsize=8.0, alpha_grid=0.25):
    CAT = sx_to_categories(Sx, thr=thr)
    handles, colors = build_satx_legend_handles(thr=thr)
    cmap = ListedColormap(colors); bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)
    p_edges = centres_to_edges(p_vals); a_edges = centres_to_edges(a_vals)
    fig, ax = plt.subplots(figsize=(8.8, 6.9), dpi=150)
    ax.pcolormesh(p_edges, a_edges, CAT, cmap=cmap, norm=norm, shading='flat')
    for x in p_edges: ax.axvline(x, color='0.35', lw=0.6, alpha=alpha_grid)
    for y in a_edges: ax.axhline(y, color='0.35', lw=0.6, alpha=alpha_grid)
    ax.set_xticks(p_vals); ax.set_yticks(a_vals)
    ax.set_xlim(p_edges[0], p_edges[-1]); ax.set_ylim(a_edges[0], a_edges[-1])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8, fontsize=12, fontweight='bold')
    if show_labels:
        for i, a in enumerate(a_vals):
            for j, p in enumerate(p_vals):
                val = Sx[i, j]
                if np.isfinite(val):
                    txt = f"{100*val:.0f}%"
                    tcol = 'black' if CAT[i, j] in (1, 2) else 'white'
                    ax.text(p, a, txt, ha='center', va='center', fontsize=fontsize, color=tcol)
    # Legend at bottom
    ax_leg = fig.add_axes([0.10, 0.02, 0.80, 0.10])
    ax_leg.set_axis_off()
    ax_leg.legend(handles=handles, ncol=1, loc='center', frameon=True, framealpha=0.95,
                  fontsize=8, handlelength=1.8, columnspacing=0.8, labelspacing=0.6,
                  title="Saturation fraction  (Sx = x^h / (1 + x^h))", title_fontsize=9)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print("Saved:", outfile)

# -------------------------- Render all plots --------------------------

# (1) Elasticity-based saturation
plot_matrix_continuous(
    Z=S_el_col, p_vals=p_vals, a_vals=a_vals,
    title=f"Saturation (elasticity) — fixed p, vary a — ycol={args.ycol}, smooth={args.smooth}",
    outfile=os.path.join(OUT_DIR, "sat_elasticity_from_colfits.png"),
    xlabel=r"$p_{\mathrm{merge}}$ (fixed column)", ylabel=r"speed parameter $a$ (varied)",
    vmin=0.0, vmax=1.0, cmap="viridis", show_labels=args.show_labels,
    fmt=".2f", fontsize=args.fontsize, cbar_label="Saturation (1 − elasticity / ref)"
)
plot_matrix_continuous(
    Z=S_el_row, p_vals=p_vals, a_vals=a_vals,
    title=f"Saturation (elasticity) — fixed a, vary p — ycol={args.ycol}, smooth={args.smooth}",
    outfile=os.path.join(OUT_DIR, "sat_elasticity_from_rowfits.png"),
    xlabel=r"$p_{\mathrm{merge}}$ (varied)", ylabel=r"speed parameter $a$ (fixed row)",
    vmin=0.0, vmax=1.0, cmap="viridis", show_labels=args.show_labels,
    fmt=".2f", fontsize=args.fontsize, cbar_label="Saturation (1 − elasticity / ref)"
)

# (2) x-fraction saturation (continuous)
plot_matrix_continuous(
    Z=Sx_col, p_vals=p_vals, a_vals=a_vals,
    title=f"Saturation (x-frac, continuous) — fixed p, vary a — h={args.hill}",
    outfile=os.path.join(OUT_DIR, "sat_xfrac_from_colfits.png"),
    xlabel=r"$p_{\mathrm{merge}}$ (fixed column)", ylabel=r"speed parameter $a$ (varied)",
    vmin=0.0, vmax=1.0, cmap="plasma", show_labels=args.show_labels,
    fmt=".2f", fontsize=args.fontsize, cbar_label="Saturation fraction Sx"
)
plot_matrix_continuous(
    Z=Sx_row, p_vals=p_vals, a_vals=a_vals,
    title=f"Saturation (x-frac, continuous) — fixed a, vary p — h={args.hill}",
    outfile=os.path.join(OUT_DIR, "sat_xfrac_from_rowfits.png"),
    xlabel=r"$p_{\mathrm{merge}}$ (varied)", ylabel=r"speed parameter $a$ (fixed row)",
    vmin=0.0, vmax=1.0, cmap="plasma", show_labels=args.show_labels,
    fmt=".2f", fontsize=args.fontsize, cbar_label="Saturation fraction Sx"
)

# (3) x-fraction saturation (binned)
plot_satx_binned(
    Sx=Sx_col, p_vals=p_vals, a_vals=a_vals,
    outfile=os.path.join(OUT_DIR, "sat_xfrac_bins_from_colfits.png"),
    title=f"Saturation (x-frac, binned) — fixed p, vary a — h={args.hill}",
    xlabel=r"$p_{\mathrm{merge}}$ (fixed column)", ylabel=r"speed parameter $a$ (varied)",
    thr=tuple(args.sx_bins), show_labels=args.show_labels, fontsize=args.fontsize
)
plot_satx_binned(
    Sx=Sx_row, p_vals=p_vals, a_vals=a_vals,
    outfile=os.path.join(OUT_DIR, "sat_xfrac_bins_from_rowfits.png"),
    title=f"Saturation (x-frac, binned) — fixed a, vary p — h={args.hill}",
    xlabel=r"$p_{\mathrm{merge}}$ (varied)", ylabel=r"speed parameter $a$ (fixed row)",
    thr=tuple(args.sx_bins), show_labels=args.show_labels, fontsize=args.fontsize
)

# (4) Step-change comparison (asymmetry by default)
if args.step_metric == "asym":
    vmin, vmax, cmap, cbar = -1.0, 1.0, "coolwarm", "Step asymmetry  (Δ+−Δ−)/( |Δ+|+|Δ−| )"
elif args.step_metric == "diff":
    # auto-range from data spread
    finite_vals = np.concatenate([M_step_col[np.isfinite(M_step_col)], M_step_row[np.isfinite(M_step_row)]])
    if finite_vals.size:
        m = np.nanpercentile(finite_vals, [5, 95])
        vmin, vmax = float(m[0]), float(m[1])
    else:
        vmin, vmax = -1.0, 1.0
    cmap, cbar = "coolwarm", "Step difference  (Δ+ − Δ−)"
else:  # ratio
    # ratio can be large; show log10 ratio centred at 0
    def log10_safe(Z, eps=1e-12):
        Z = np.asarray(Z, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.log10(Z)
        return out
    M_step_col = log10_safe(M_step_col)
    M_step_row = log10_safe(M_step_row)
    finite_vals = np.concatenate([M_step_col[np.isfinite(M_step_col)], M_step_row[np.isfinite(M_step_row)]])
    if finite_vals.size:
        r = np.nanpercentile(finite_vals, [5, 95]); vmax = max(abs(r[0]), abs(r[1])); vmin = -vmax
    else:
        vmin, vmax = -1.0, 1.0
    cmap, cbar = "PuOr", "log10 step ratio  log10( |Δ+| / |Δ−| )"

plot_matrix_continuous(
    Z=M_step_col, p_vals=p_vals, a_vals=a_vals,
    title=f"Step change ({args.step_metric}) — fixed p, vary a — ycol={args.ycol}",
    outfile=os.path.join(OUT_DIR, "step_asym_from_colfits.png"),
    xlabel=r"$p_{\mathrm{merge}}$ (fixed column)", ylabel=r"speed parameter $a$ (varied)",
    vmin=vmin, vmax=vmax, cmap=cmap, show_labels=args.show_labels,
    fmt=".2f", fontsize=args.fontsize, cbar_label=cbar
)
plot_matrix_continuous(
    Z=M_step_row, p_vals=p_vals, a_vals=a_vals,
    title=f"Step change ({args.step_metric}) — fixed a, vary p — ycol={args.ycol}",
    outfile=os.path.join(OUT_DIR, "step_asym_from_rowfits.png"),
    xlabel=r"$p_{\mathrm{merge}}$ (varied)", ylabel=r"speed parameter $a$ (fixed row)",
    vmin=vmin, vmax=vmax, cmap=cmap, show_labels=args.show_labels,
    fmt=".2f", fontsize=args.fontsize, cbar_label=cbar
)