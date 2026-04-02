#!/usr/bin/env python3
"""
Cell-centred x maps with 4-colour magnitude bins for (p_merge, a) sweeps.

Legend is rendered in its OWN bottom strip (vertical stack) with an extra spacer
between plot and legend so it never overlaps axis labels. Helper arrow + label
are OUTSIDE the plot and compact:
  - fixed a (horizontal reading): helper ABOVE the plot (below title), compact.
  - fixed p (vertical reading):   helper to the RIGHT of the plot, compact & rotated.

Colour bins:
  red    : x < 1e-2
  orange : 1e-2 <= x < 1e-1
  yellow : 1e-1 <= x < 1
  green  : x >= 1

Outputs (under fit_outputs/<DATE_TAG>/):
  - x_text_mag4_from_colfits.png   (fixed p, vary a)  — READ VERTICALLY (↑)
  - x_text_mag4_from_rowfits.png   (fixed a, vary p)  — READ HORIZONTALLY (→)
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# ---------- CLI ----------
ap = argparse.ArgumentParser(
    description="Cell-centred x maps (4-bin colouring). Legend below (with spacer); helper arrow outside."
)
ap.add_argument("date_tag", help="Folder under Updated_abm/Parameter_sweeps/fit_outputs/")
ap.add_argument("--fmt", default=".2g", help="Numeric format for the x labels (e.g., .2g, .3f)")
ap.add_argument("--fontsize", type=float, default=8.0, help="Font size for x labels")
ap.add_argument("--alpha_grid", type=float, default=0.25, help="Grid-line alpha")
args = ap.parse_args()

DATE_TAG   = args.date_tag
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))               # Updated_abm/Parameter_sweeps
OUT_DIR    = os.path.join(SCRIPT_DIR, "fit_outputs", DATE_TAG)
print(f"[plot_x_mag4_outside] Using OUT_DIR = {OUT_DIR}")

# ---------- Inputs ----------
col_csv = os.path.join(OUT_DIR, "col_fit_summary.csv")
row_csv = os.path.join(OUT_DIR, "row_fit_summary.csv")
agg_csv = os.path.join(OUT_DIR, "coag_aggregate_2d.csv")

missing = [p for p in (col_csv, row_csv, agg_csv) if not os.path.exists(p)]
if missing:
    print("[plot_x_mag4_outside] Missing required files:")
    for p in missing: print("  -", p)
    print("\nRun the fitter first with:")
    print(f"  python {os.path.join(SCRIPT_DIR, 'fit_parameter_sweeps_2d_rowcol.py')} {DATE_TAG} "
          "--window 25 --ycol P_pairs_smooth_mean --subdir coagulation_analysis_tagged --overwrite")
    sys.exit(2)

# ---------- Load & grid ----------
col = pd.read_csv(col_csv).sort_values('p').reset_index(drop=True)    # p, b
row = pd.read_csv(row_csv).sort_values('a').reset_index(drop=True)    # a, b
agg = pd.read_csv(agg_csv)                                            # p, a, y0, y_win

p_vals = np.sort(agg['p'].unique().astype(float))
a_vals = np.sort(agg['a'].unique().astype(float))
A, P   = len(a_vals), len(p_vals)

# Gains
b_col_map = {float(rp): float(rb) for rp, rb in zip(col['p'].values, col['b'].values)}  # b_a(p)
b_row_map = {float(ra): float(rb) for ra, rb in zip(row['a'].values, row['b'].values)}  # b_p(a)

# X surfaces
X_col = np.zeros((A, P), dtype=float)  # fixed p → x = b_a(p) * a
for j, p in enumerate(p_vals):
    bj = b_col_map.get(float(p), np.nan)
    X_col[:, j] = bj * a_vals

X_row = np.zeros((A, P), dtype=float)  # fixed a → x = b_p(a) * p
for i, a in enumerate(a_vals):
    bi = b_row_map.get(float(a), np.nan)
    X_row[i, :] = bi * p_vals

# ---------- helpers ----------
def centres_to_edges(centres: np.ndarray) -> np.ndarray:
    """Convert bin centres to edges (works with uneven spacing)."""
    c = np.asarray(centres, dtype=float)
    if c.size == 1:
        w = 0.5 * (abs(c[0]) + 1.0) * 1e-3
        return np.array([c[0] - w, c[0] + w], dtype=float)
    mids = 0.5 * (c[:-1] + c[1:])
    first_edge = c[0] - (c[1] - c[0]) / 2.0
    last_edge  = c[-1] + (c[-1] - c[-2]) / 2.0
    return np.concatenate([[first_edge], mids, [last_edge]])

def mag4_category(X):
    """Return 4-bin category array for x values."""
    X = np.asarray(X, dtype=float)
    cat = np.zeros_like(X, dtype=int)  # red by default (x<1e-2 or non-finite)
    mask1 = np.isfinite(X) & (X >= 1e-2) & (X < 1e-1)  # orange
    mask2 = np.isfinite(X) & (X >= 1e-1) & (X < 1.0)   # yellow
    mask3 = np.isfinite(X) & (X >= 1.0)                # green
    cat[mask1] = 1
    cat[mask2] = 2
    cat[mask3] = 3
    return cat

def build_legend_handles():
    colors = ['#b2182b', '#f46d43', '#fee08b', '#1a9850']  # red, orange, yellow, green
    return [
        Patch(facecolor=colors[0], edgecolor='none',
              label="Red   : x < 1e-2   — deep linear (≈ proportional response; far from plateau)"),
        Patch(facecolor=colors[1], edgecolor='none',
              label="Orange: 1e-2–1e-1 — early soft saturation"),
        Patch(facecolor=colors[2], edgecolor='none',
              label="Yellow: 1e-1–1    — clear concavity (soft saturation)"),
        Patch(facecolor=colors[3], edgecolor='none',
              label="Green : ≥ 1       — strong saturation (≥ half-plateau; diminishing returns)"),
    ]

def draw_one(
    X, p_vals, a_vals, out_png, title, fixed_param: str,
    fmt=".2g", fontsize=8.0, alpha_grid=0.25
):
    """
    Draw one plot with:
      - main heatmap (centre x labels),
      - helper arrow + label OUTSIDE (right or above),
      - legend BELOW the plot in a TALL, dedicated strip (vertical stack) with a spacer strip above it.
    """
    p_edges = centres_to_edges(p_vals)
    a_edges = centres_to_edges(a_vals)
    CAT     = mag4_category(X)

    colors = ['#b2182b', '#f46d43', '#fee08b', '#1a9850']  # red, orange, yellow, green
    cmap   = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm   = BoundaryNorm(bounds, cmap.N)

    # ---------- Figure layout ----------
    if fixed_param == 'a':
        # helper ABOVE, spacer, then legend BELOW
        fig = plt.figure(figsize=(9.8, 8.6), dpi=150)
        gs  = fig.add_gridspec(
            nrows=4, ncols=1, height_ratios=[3, 25, 3, 6],  # [helper, main, spacer, legend]
            left=0.08, right=0.92, top=0.92, bottom=0.10, hspace=0.06
        )
        ax_helper = fig.add_subplot(gs[0])
        ax_main   = fig.add_subplot(gs[1])
        ax_space  = fig.add_subplot(gs[2])
        ax_legend = fig.add_subplot(gs[3])
        ax_space.set_axis_off()  # blank spacer
        fig.suptitle(title, y=0.975, fontsize=12, fontweight='bold')
    else:
        # helper RIGHT, spacer row, then legend BELOW (spanning both columns)
        fig = plt.figure(figsize=(10.7, 8.6), dpi=150)
        gs  = fig.add_gridspec(
            nrows=3, ncols=2,
            height_ratios=[26, 3, 6], width_ratios=[28, 5],  # [main+helper], [spacer], [legend]
            left=0.08, right=0.92, top=0.92, bottom=0.10, hspace=0.06, wspace=0.05
        )
        ax_main   = fig.add_subplot(gs[0, 0])
        ax_helper = fig.add_subplot(gs[0, 1])
        ax_space  = fig.add_subplot(gs[1, :])
        ax_legend = fig.add_subplot(gs[2, :])
        ax_space.set_axis_off()
        ax_main.set_title(title, pad=6, fontsize=12, fontweight='bold')

    # ---------- Main heatmap ----------
    pc = ax_main.pcolormesh(p_edges, a_edges, CAT, cmap=cmap, norm=norm, shading='flat')

    # grid lines
    for x in p_edges: ax_main.axvline(x, color='0.25', lw=0.6, alpha=alpha_grid)
    for y in a_edges: ax_main.axhline(y, color='0.25', lw=0.6, alpha=alpha_grid)

    # centre labels (numeric x)
    for i, a in enumerate(a_vals):
        for j, p in enumerate(p_vals):
            val = X[i, j]
            if not np.isfinite(val): 
                continue
            if (val >= 100) or (0 < val < 0.01):
                label = f"{val:.1e}"
            else:
                label = format(val, fmt)
            tcol = 'black' if CAT[i, j] in (1, 2) else 'white'
            ax_main.text(p, a, label, ha='center', va='center', fontsize=fontsize, color=tcol)

    # axes ticks/labels — extra padding for xlabel
    ax_main.set_xticks(p_vals); ax_main.set_yticks(a_vals)
    ax_main.set_xlim(p_edges[0], p_edges[-1]); ax_main.set_ylim(a_edges[0], a_edges[-1])
    if fixed_param == 'p':
        ax_main.set_xlabel(r"$p_{\mathrm{merge}}$   (fixed)", labelpad=18)
        ax_main.set_ylabel(r"speed parameter $a$   (varied)")
    else:
        ax_main.set_xlabel(r"$p_{\mathrm{merge}}$   (varied)", labelpad=18)
        ax_main.set_ylabel(r"speed parameter $a$   (fixed)")

    # ---------- Helper (outside, compact) ----------
    ax_helper.set_axis_off()
    ax_helper.set_xlim(0, 1); ax_helper.set_ylim(0, 1)
    if fixed_param == 'a':
        ax_helper.annotate("", xy=(0.85, 0.52), xytext=(0.15, 0.52),
                           arrowprops=dict(arrowstyle="<->", lw=1.6, color='k'))
        ax_helper.text(0.50, 0.80, "READ HORIZONTALLY  (vary p at fixed a)",
                       ha="center", va="center", fontsize=9, fontweight='bold', color='k')
    else:
        ax_helper.annotate("", xy=(0.50, 0.85), xytext=(0.50, 0.15),
                           arrowprops=dict(arrowstyle="<->", lw=1.6, color='k'))
        ax_helper.text(0.72, 0.50, "READ VERTICALLY  (vary a at fixed p)",
                       rotation=90, ha="center", va="center",
                       fontsize=9, fontweight='bold', color='k')

    # ---------- Legend BELOW (own tall strip; vertical stack) ----------
    ax_legend.set_axis_off()
    handles = build_legend_handles()
    # One column (vertical stack). Place at the BOTTOM of the legend strip; large strip + spacer prevents overlap.
    leg = ax_legend.legend(
        handles=handles, ncol=1, loc='lower center', frameon=True, framealpha=0.95,
        fontsize=8, handlelength=1.8, columnspacing=0.8, labelspacing=0.7,
        title="x magnitude (finite-window load)", title_fontsize=9
    )

    # Save with tight bbox to include outside elements
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ---------- Make the two figures ----------
out1 = os.path.join(OUT_DIR, "x_text_mag4_from_colfits.png")
draw_one(
    X_col, p_vals, a_vals, out_png=out1,
    title="Cell-centred x — fixed p, vary a", fixed_param='p',
    fmt=args.fmt, fontsize=args.fontsize, alpha_grid=args.alpha_grid
)
print("Saved:", out1)

out2 = os.path.join(OUT_DIR, "x_text_mag4_from_rowfits.png")
draw_one(
    X_row, p_vals, a_vals, out_png=out2,
    title="Cell-centred x — fixed a, vary p", fixed_param='a',
    fmt=args.fmt, fontsize=args.fontsize, alpha_grid=args.alpha_grid
)
print("Saved:", out2)