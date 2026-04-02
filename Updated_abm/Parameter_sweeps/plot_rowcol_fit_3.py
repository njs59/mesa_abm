#!/usr/bin/env python3
"""
Diagnostic plots for row/column fits of the time-averaged coagulation rate.

Reads:
  Updated_abm/Parameter_sweeps/fit_outputs/<DATE_TAG>/
    ├─ row_fit_summary.csv        # per-a fits: a, S, b, R2, (CIs optional), x_min, x_max
    ├─ col_fit_summary.csv        # per-p fits: p, S, b, R2, (CIs optional), x_min, x_max
    └─ coag_aggregate_2d.csv      # tidy grid: scenario, p, a, y0, y_win (empirical per-run)

Writes into:
  Updated_abm/Parameter_sweeps/fit_outputs/<DATE_TAG>/<OUTSUB>/

Row fits (fixed a, vary p):
  - rows_S_vs_a.png
  - rows_b_vs_a.png
  - rows_R2_vs_a.png
  - rows_parity/row_parity_a_<value>.png                (empirical vs predicted parity)
  - rows_curves/row_curve_a_<value>.png                 (empirical vs p + predicted curve vs p)
  - rows_parity_points.csv, rows_curve_points.csv       (CSV exports)

Column fits (fixed p, vary a):
  - cols_S_vs_p.png
  - cols_b_vs_p.png
  - cols_R2_vs_p.png
  - cols_parity/col_parity_p_<value>.png                (empirical vs predicted parity)
  - cols_curves/col_curve_p_<value>.png                 (empirical vs a + predicted curve vs a)
  - cols_parity_points.csv, cols_curve_points.csv       (CSV exports)

Usage:
  python Updated_abm/Parameter_sweeps/plot_fit_diagnostics.py <DATE_TAG> \
      [--outsub diagnostics] [--ycol auto|y_win|P_pairs_smooth_mean|P_pairs_mean|y0]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CLI ----------------
ap = argparse.ArgumentParser(description="Diagnostic plots for fitted S,b and parity/curve checks.")
ap.add_argument("date_tag", help="Folder under Updated_abm/Parameter_sweeps/fit_outputs/")
ap.add_argument("--outsub", default="diagnostics", help="Subfolder name for outputs (default: diagnostics)")
ap.add_argument("--ycol", default="auto", help="Empirical rate column in coag_aggregate_2d.csv (or 'auto')")
args = ap.parse_args()

DATE_TAG = args.date_tag
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Updated_abm/Parameter_sweeps
IN_DIR  = os.path.join(SCRIPT_DIR, "fit_outputs", DATE_TAG)
OUT_DIR = os.path.join(IN_DIR, args.outsub)

row_csv = os.path.join(IN_DIR, "row_fit_summary.csv")
col_csv = os.path.join(IN_DIR, "col_fit_summary.csv")
agg_csv = os.path.join(IN_DIR, "coag_aggregate_2d.csv")

missing = [p for p in (row_csv, col_csv, agg_csv) if not os.path.exists(p)]
if missing:
    print("[plot_fit_diagnostics] Missing required files:")
    for p in missing:
        print("  -", p)
    sys.exit(2)

# Prepare output folders
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "rows_parity"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "cols_parity"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "rows_curves"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "cols_curves"), exist_ok=True)

# ---------------- Load data ----------------
row = pd.read_csv(row_csv).sort_values("a").reset_index(drop=True)
col = pd.read_csv(col_csv).sort_values("p").reset_index(drop=True)
agg = pd.read_csv(agg_csv)

# ---------------- Choose empirical Y column ----------------
preferred_order = ["y_win", "P_pairs_smooth_mean", "P_pairs_mean", "y0"]
available = list(agg.columns)
if args.ycol != "auto":
    if args.ycol in agg.columns:
        YCOL = args.ycol
    else:
        print(f"[plot_fit_diagnostics] WARNING: requested --ycol '{args.ycol}' not found.")
        YCOL = next((c for c in preferred_order if c in agg.columns), None)
        if YCOL is None:
            print("[plot_fit_diagnostics] No suitable y-column found. Available:", ", ".join(available))
            sys.exit(2)
        else:
            print(f"[plot_fit_diagnostics] Falling back to '{YCOL}'.")
else:
    YCOL = next((c for c in preferred_order if c in agg.columns), None)
    if YCOL is None:
        print("[plot_fit_diagnostics] No suitable y-column found. Available:", ", ".join(available))
        sys.exit(2)

print(f"[plot_fit_diagnostics] Using empirical column: {YCOL}")

# ---------------- Helpers ----------------
def add_ci_band(ax, x, lo, hi, color, alpha=0.18):
    x = np.asarray(x, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.any(hi > lo):
        ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)

def sane_name(v, key):
    """Make a filesystem-friendly token like a_0p5 or p_0p1 for filenames."""
    try:
        fv = float(v)
        s = ("%.6g" % fv).replace(".", "p")
        return f"{key}_{s}"
    except Exception:
        return f"{key}_{str(v)}"

# ---------------- 1) Row fits: S(a), b(a), R2(a) ----------------
def plot_rows_S_b_R2():
    if len(row) == 0:
        return
    a = row["a"].values.astype(float)

    # S vs a
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    ax.plot(a, row["S"].values, marker="o", color="#2ca02c", label="S(a)")
    if {"S_CI_lo","S_CI_hi"}.issubset(row.columns):
        add_ci_band(ax, a, row["S_CI_lo"].values, row["S_CI_hi"].values, "#2ca02c")
    ax.set_xlabel("speed parameter a")
    ax.set_ylabel("Fitted S")
    ax.set_title("Row fits: S(a) vs a (fixed a, vary p)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "rows_S_vs_a.png")); plt.close(fig)

    # b vs a (log-y)
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    ax.semilogy(a, np.maximum(row["b"].values, 1e-30), marker="o", color="#1f77b4", label="b_p(a)")
    if {"b_CI_lo","b_CI_hi"}.issubset(row.columns):
        add_ci_band(ax, a, np.maximum(row["b_CI_lo"].values, 1e-30),
                    np.maximum(row["b_CI_hi"].values, 1e-30), "#1f77b4")
    ax.set_xlabel("speed parameter a")
    ax.set_ylabel("Fitted b_p(a)")
    ax.set_title("Row fits: b_p(a) vs a (fixed a, vary p)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "rows_b_vs_a.png")); plt.close(fig)

    # R^2 vs a
    if "R2" in row.columns:
        fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
        ax.plot(a, row["R2"].values, marker="d", color="#d62728")
        ax.set_xlabel("speed parameter a"); ax.set_ylabel(r"$R^2$"); ax.set_ylim(0, 1.02)
        ax.set_title("Row fits: $R^2$ vs a (fixed a, vary p)")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "rows_R2_vs_a.png")); plt.close(fig)

# ---------------- 2) Column fits: S(p), b(p), R2(p) ----------------
def plot_cols_S_b_R2():
    if len(col) == 0:
        return
    p = col["p"].values.astype(float)

    # S vs p
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    ax.plot(p, col["S"].values, marker="o", color="#2ca02c", label="S(p)")
    if {"S_CI_lo","S_CI_hi"}.issubset(col.columns):
        add_ci_band(ax, p, col["S_CI_lo"].values, col["S_CI_hi"].values, "#2ca02c")
    ax.set_xlabel(r"$p_{\mathrm{merge}}$"); ax.set_ylabel("Fitted S")
    ax.set_title("Column fits: S(p) vs p (fixed p, vary a)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "cols_S_vs_p.png")); plt.close(fig)

    # b vs p (log-y)
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    ax.semilogy(p, np.maximum(col["b"].values, 1e-30), marker="o", color="#1f77b4", label="b_a(p)")
    if {"b_CI_lo","b_CI_hi"}.issubset(col.columns):
        add_ci_band(ax, p, np.maximum(col["b_CI_lo"].values, 1e-30),
                    np.maximum(col["b_CI_hi"].values, 1e-30), "#1f77b4")
    ax.set_xlabel(r"$p_{\mathrm{merge}}$"); ax.set_ylabel("Fitted b_a(p)")
    ax.set_title("Column fits: b_a(p) vs p (fixed p, vary a)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "cols_b_vs_p.png")); plt.close(fig)

    # R^2 vs p
    if "R2" in col.columns:
        fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
        ax.plot(p, col["R2"].values, marker="d", color="#d62728")
        ax.set_xlabel(r"$p_{\mathrm{merge}}$"); ax.set_ylabel(r"$R^2$"); ax.set_ylim(0, 1.02)
        ax.set_title("Column fits: $R^2$ vs p (fixed p, vary a)")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "cols_R2_vs_p.png")); plt.close(fig)

# ---------------- 3) Parity plots + CSV exports ----------------
def parity_rows_and_curves():
    """For each fixed a (row fit), produce parity and curve-vs-p plots. Export combined CSVs."""
    fit_row = row.set_index("a")
    parity_rows_out = []
    curve_rows_out  = []
    for aval, g in agg.groupby("a"):
        aval_f = float(aval)
        if aval_f not in fit_row.index:
            continue
        S = fit_row.loc[aval_f, "S"]; b = fit_row.loc[aval_f, "b"]
        sub = g.sort_values("p")
        p = sub["p"].values.astype(float)
        y_emp = sub[YCOL].values.astype(float)
        x = b * p
        y_pred = S * (x / (1.0 + x))

        # 3a) Parity (empirical vs predicted)
        fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
        ax.scatter(y_emp, y_pred, s=22, c="#1f77b4", edgecolor="k", linewidth=0.3, alpha=0.9)
        xy_min = np.nanmin([y_emp.min(), y_pred.min()])
        xy_max = np.nanmax([y_emp.max(), y_pred.max()])
        if not np.isfinite(xy_min) or not np.isfinite(xy_max) or xy_min == xy_max:
            xy_min, xy_max = 0.0, max(1e-8, np.nanmax(y_emp))
        pad = 0.05 * (xy_max - xy_min) if xy_max > xy_min else 1e-8
        ax.plot([xy_min-pad, xy_max+pad], [xy_min-pad, xy_max+pad], "k--", lw=1)
        ax.set_xlim(xy_min-pad, xy_max+pad); ax.set_ylim(xy_min-pad, xy_max+pad)
        ax.set_xlabel(f"Empirical {YCOL}"); ax.set_ylabel(f"Predicted {YCOL}")
        ax.set_title(f"Row parity (fixed a={aval_f:g})"); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "rows_parity", f"row_parity_{sane_name(aval_f, 'a')}.png"))
        plt.close(fig)

        # Collect parity points
        parity_rows_out.append(pd.DataFrame({"a": aval_f, "p": p, "y_emp": y_emp, "y_pred": y_pred}))

        # 3b) Curve over parameter p (empirical vs p, predicted curve vs p)
        fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
        ax.scatter(p, y_emp, s=28, c="#1f77b4", edgecolor="k", linewidth=0.3, alpha=0.9, label="Empirical")
        # For a smooth curve, optionally densify p; here we plot at observed points + a thin line through them:
        p_dense = np.linspace(p.min(), p.max(), max(100, len(p)*5))
        y_pred_dense = S * ((b * p_dense) / (1.0 + b * p_dense))
        ax.plot(p_dense, y_pred_dense, color="#d62728", lw=1.5, label="Predicted (fit)")
        ax.set_xlabel(r"$p_{\mathrm{merge}}$"); ax.set_ylabel(f"{YCOL}")
        ax.set_title(f"Row fit curve: a={aval_f:g} (empirical vs predicted over p)")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "rows_curves", f"row_curve_{sane_name(aval_f, 'a')}.png"))
        plt.close(fig)

        # Collect curve points (observed grid + densified predicted, marked)
        curve_rows_out.append(pd.DataFrame({
            "a": aval_f,
            "p": p,
            "y_emp": y_emp,
            "y_pred": S * ((b * p) / (1.0 + b * p)),
            "type": "observed"
        }))
        curve_rows_out.append(pd.DataFrame({
            "a": aval_f,
            "p": p_dense,
            "y_emp": np.nan,
            "y_pred": y_pred_dense,
            "type": "pred_curve"
        }))

    # export combined CSVs
    if parity_rows_out:
        pd.concat(parity_rows_out, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "rows_parity_points.csv"), index=False
        )
    if curve_rows_out:
        pd.concat(curve_rows_out, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "rows_curve_points.csv"), index=False
        )

def parity_cols_and_curves():
    """For each fixed p (column fit), produce parity and curve-vs-a plots. Export combined CSVs."""
    fit_col = col.set_index("p")
    parity_cols_out = []
    curve_cols_out  = []
    for pval, g in agg.groupby("p"):
        pval_f = float(pval)
        if pval_f not in fit_col.index:
            continue
        S = fit_col.loc[pval_f, "S"]; b = fit_col.loc[pval_f, "b"]
        sub = g.sort_values("a")
        a = sub["a"].values.astype(float)
        y_emp = sub[YCOL].values.astype(float)
        x = b * a
        y_pred = S * (x / (1.0 + x))

        # 4a) Parity (empirical vs predicted)
        fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
        ax.scatter(y_emp, y_pred, s=22, c="#ff7f0e", edgecolor="k", linewidth=0.3, alpha=0.9)
        xy_min = np.nanmin([y_emp.min(), y_pred.min()])
        xy_max = np.nanmax([y_emp.max(), y_pred.max()])
        if not np.isfinite(xy_min) or not np.isfinite(xy_max) or xy_min == xy_max:
            xy_min, xy_max = 0.0, max(1e-8, np.nanmax(y_emp))
        pad = 0.05 * (xy_max - xy_min) if xy_max > xy_min else 1e-8
        ax.plot([xy_min-pad, xy_max+pad], [xy_min-pad, xy_max+pad], "k--", lw=1)
        ax.set_xlim(xy_min-pad, xy_max+pad); ax.set_ylim(xy_min-pad, xy_max+pad)
        ax.set_xlabel(f"Empirical {YCOL}"); ax.set_ylabel(f"Predicted {YCOL}")
        ax.set_title(f"Column parity (fixed p={pval_f:g})"); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "cols_parity", f"col_parity_{sane_name(pval_f, 'p')}.png"))
        plt.close(fig)

        # Collect parity points
        parity_cols_out.append(pd.DataFrame({"p": pval_f, "a": a, "y_emp": y_emp, "y_pred": y_pred}))

        # 4b) Curve over parameter a (empirical vs a, predicted curve vs a)
        fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
        ax.scatter(a, y_emp, s=28, c="#ff7f0e", edgecolor="k", linewidth=0.3, alpha=0.9, label="Empirical")
        a_dense = np.linspace(a.min(), a.max(), max(100, len(a)*5))
        y_pred_dense = S * ((b * a_dense) / (1.0 + b * a_dense))
        ax.plot(a_dense, y_pred_dense, color="#2ca02c", lw=1.5, label="Predicted (fit)")
        ax.set_xlabel("speed parameter a"); ax.set_ylabel(f"{YCOL}")
        ax.set_title(f"Column fit curve: p={pval_f:g} (empirical vs predicted over a)")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "cols_curves", f"col_curve_{sane_name(pval_f, 'p')}.png"))
        plt.close(fig)

        # Collect curve points
        curve_cols_out.append(pd.DataFrame({
            "p": pval_f,
            "a": a,
            "y_emp": y_emp,
            "y_pred": S * ((b * a) / (1.0 + b * a)),
            "type": "observed"
        }))
        curve_cols_out.append(pd.DataFrame({
            "p": pval_f,
            "a": a_dense,
            "y_emp": np.nan,
            "y_pred": y_pred_dense,
            "type": "pred_curve"
        }))

    # export combined CSVs
    if parity_cols_out:
        pd.concat(parity_cols_out, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "cols_parity_points.csv"), index=False
        )
    if curve_cols_out:
        pd.concat(curve_cols_out, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "cols_curve_points.csv"), index=False
        )

# ---------------- Run all plots ----------------
plot_rows_S_b_R2()
plot_cols_S_b_R2()
parity_rows_and_curves()
parity_cols_and_curves()
print("[plot_fit_diagnostics] Done. Wrote outputs to:", OUT_DIR)