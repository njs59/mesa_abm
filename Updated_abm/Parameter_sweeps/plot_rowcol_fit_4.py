#!/usr/bin/env python3
"""
Empirical R(T) surface plotter for (p_merge, a).

Reads empirical window-averaged rate values from:
  Updated_abm/Parameter_sweeps/fit_outputs/<DATE_TAG>/coag_aggregate_2d.csv
and builds:
  - A static 3D surface (Matplotlib) saved as PNG.
  - An interactive 3D surface (Plotly) saved as HTML and auto-opened in your browser.

Axes:
  X = p_merge (p)
  Y = a (speed parameter)
  Z = empirical R(T) (chosen column from coag_aggregate_2d.csv)

Usage:
  python Updated_abm/Parameter_sweeps/plot_empirical_surface.py <DATE_TAG> \
      [--outsub surfaces] [--ycol auto|y_win|P_pairs_smooth_mean|P_pairs_mean|y0] [--grid 120]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import webbrowser
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import griddata

# ---------------- CLI ----------------
ap = argparse.ArgumentParser(description="Empirical R(T) surface for (p_merge, a).")
ap.add_argument("date_tag", help="Folder under Updated_abm/Parameter_sweeps/fit_outputs/")
ap.add_argument("--outsub", default="surfaces", help="Subfolder for outputs (default: surfaces)")
ap.add_argument("--ycol", default="auto", help="Empirical column (or 'auto'): y_win, P_pairs_smooth_mean, P_pairs_mean, y0")
ap.add_argument("--grid", type=int, default=120, help="Grid density for interpolation (default: 120)")
args = ap.parse_args()

DATE_TAG = args.date_tag
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Updated_abm/Parameter_sweeps
IN_DIR  = os.path.join(SCRIPT_DIR, "fit_outputs", DATE_TAG)
OUT_DIR = os.path.join(IN_DIR, args.outsub)
os.makedirs(OUT_DIR, exist_ok=True)

agg_csv = os.path.join(IN_DIR, "coag_aggregate_2d.csv")
if not os.path.exists(agg_csv):
    print("[empirical_surface] ERROR: Not found:", agg_csv)
    sys.exit(2)

# ---------------- Load data ----------------
agg = pd.read_csv(agg_csv)

# Choose empirical Y column
preferred_order = ["y_win", "P_pairs_smooth_mean", "P_pairs_mean", "y0"]
if args.ycol != "auto":
    if args.ycol in agg.columns:
        YCOL = args.ycol
    else:
        print(f"[empirical_surface] WARNING: requested --ycol '{args.ycol}' not found in CSV. Falling back to auto.")
        YCOL = next((c for c in preferred_order if c in agg.columns), None)
else:
    YCOL = next((c for c in preferred_order if c in agg.columns), None)

if YCOL is None:
    print("[empirical_surface] ERROR: No suitable empirical column found.",
          "Available:", ", ".join(agg.columns))
    sys.exit(2)

print(f"[empirical_surface] Using empirical column: {YCOL}")

# Extract points
if not {"p", "a"}.issubset(agg.columns):
    print("[empirical_surface] ERROR: CSV must contain 'p' and 'a' columns.")
    sys.exit(2)

p = agg["p"].values.astype(float)
a = agg["a"].values.astype(float)
z_emp = agg[YCOL].values.astype(float)

# ---------------- Build a regular grid via interpolation ----------------
# Grid extents and density
p_lin = np.linspace(np.nanmin(p), np.nanmax(p), max(10, int(args.grid)))
a_lin = np.linspace(np.nanmin(a), np.nanmax(a), max(10, int(args.grid)))
PP, AA = np.meshgrid(p_lin, a_lin)

# Interpolate to grid (linear where possible, nearest to fill gaps)
pts = np.column_stack([p, a])
Z_lin = griddata(pts, z_emp, (PP, AA), method="linear")
Z_nn  = griddata(pts, z_emp, (PP, AA), method="nearest")
Z = np.where(np.isnan(Z_lin), Z_nn, Z_lin)

# ---------------- Save a static 3D surface (Matplotlib) ----------------
fig = plt.figure(figsize=(8.2, 6.6), dpi=150)
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(PP, AA, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
# overlay original points for reference
ax.scatter(p, a, z_emp, c="k", s=8, alpha=0.6)

ax.set_xlabel(r"$p_{\mathrm{merge}}$")
ax.set_ylabel(r"speed parameter $a$")
ax.set_zlabel(f"Empirical {YCOL}")
ax.set_title("Empirical R(T) surface")

# A sensible z limit (optional): auto from data
# ax.set_zlim(np.nanmin(Z), np.nanmax(Z))

m = plt.cm.ScalarMappable(cmap="viridis")
m.set_array(Z)
cb = fig.colorbar(m, ax=ax, shrink=0.65, pad=0.08)
cb.set_label(f"Empirical {YCOL}")

png_path = os.path.join(OUT_DIR, "empirical_surface_static.png")
fig.tight_layout()
fig.savefig(png_path)
plt.close(fig)
print("[empirical_surface] Saved static surface:", png_path)

# ---------------- Save an interactive surface (Plotly) and auto-open ----------------
html_path = os.path.join(OUT_DIR, "empirical_surface_interactive.html")
try:
    import plotly.graph_objects as go
    fig3d = go.Figure(
        data=[
            go.Surface(
                x=p_lin, y=a_lin, z=Z,
                colorscale="Viridis", showscale=True, opacity=1.0
            )
        ]
    )
    fig3d.update_layout(
        title="Empirical R(T) surface (interactive)",
        scene=dict(
            xaxis_title="p_merge",
            yaxis_title="a",
            zaxis_title=f"Empirical {YCOL}"
        ),
        margin=dict(l=10, r=10, b=10, t=40)
    )
    fig3d.write_html(html_path, auto_open=True)
    print("[empirical_surface] Wrote and opened interactive surface:", html_path)
except Exception as e:
    print("[empirical_surface] NOTE: Could not create interactive Plotly surface:", e)
    print("                     Install plotly to enable:  pip install plotly")
    # still save the grid points for external tools
    pass

# ---------------- (Optional) Save the gridded surface points as CSV ----------------
grid_df = pd.DataFrame({
    "p": PP.ravel(),
    "a": AA.ravel(),
    f"{YCOL}_grid": Z.ravel()
})
grid_csv = os.path.join(OUT_DIR, "empirical_surface_grid_points.csv")
grid_df.to_csv(grid_csv, index=False)
print("[empirical_surface] Saved gridded points CSV:", grid_csv)