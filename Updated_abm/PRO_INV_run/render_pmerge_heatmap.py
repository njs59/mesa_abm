#!/usr/bin/env python3
"""
Render a heatmap from a p-merge sweep summary CSV produced by sweep_pmerge.py.

This script:
  * Accepts either a direct path to `sweep_pmerge_summary.csv` (via --csv)
    or a run directory that contains it (via --run_dir).
  * Builds a p_PRO (rows) × p_INV (cols) matrix of a chosen metric
    (default: pct_diff_INV_vs_PRO_final).
  * Renders a heatmap with hatch on the lower triangle (where p_INV < p_PRO),
    labels, colourbar, optional annotations, and saves to PNG
    (and optionally PDF/SVG).

Usage examples:
  # Point to the timestamped run directory (auto-finds the CSV)
  python -m PRO_INV_run.render_pmerge_heatmap \
      --run_dir PRO_INV_results/sweep_pmerge/20260326_150942

  # Or give the CSV directly and choose explicit output folder
  python -m PRO_INV_run.render_pmerge_heatmap \
      --csv PRO_INV_results/sweep_pmerge/20260326_150942/summary/sweep_pmerge_summary.csv \
      --annot --vmin 143 --vmax 562 --fmt ".0f" \
      --out_dir PRO_INV_results/sweep_pmerge/20260326_150942/plots

Outputs:
  <out_dir>/mean_size_pctdiff_heatmap.png  (and optionally .pdf/.svg)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def find_summary_csv(run_dir: Path) -> Path:
    """Return <run_dir>/summary/sweep_pmerge_summary.csv or raise if missing."""
    candidate = run_dir / "summary" / "sweep_pmerge_summary.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find sweep_pmerge_summary.csv under {run_dir}")


def build_matrix(df: pd.DataFrame, metric: str):
    """
    Build p_PRO × p_INV matrix (M) for the chosen metric.

    Returns:
        p_pro_vals (sorted unique row labels),
        p_inv_vals (sorted unique column labels),
        M (2D numpy array with NaNs where missing)
    """
    required = {"p_merge_PRO", "p_merge_INV", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Summary CSV missing required columns: {missing}")

    # Normalise grid keys to 1 decimal place (as produced by the sweep)
    df = df.copy()
    df["p_merge_PRO"] = df["p_merge_PRO"].astype(float).round(1)
    df["p_merge_INV"] = df["p_merge_INV"].astype(float).round(1)

    # If duplicates exist for a given pair, average them
    df = df.groupby(["p_merge_PRO", "p_merge_INV"], as_index=False)[metric].mean()

    p_pro_vals = np.sort(df["p_merge_PRO"].unique())
    p_inv_vals = np.sort(df["p_merge_INV"].unique())

    M = np.full((len(p_pro_vals), len(p_inv_vals)), np.nan, dtype=float)
    idx_pro = {v: i for i, v in enumerate(p_pro_vals)}
    idx_inv = {v: i for i, v in enumerate(p_inv_vals)}

    for _, row in df.iterrows():
        i = idx_pro[row["p_merge_PRO"]]
        j = idx_inv[row["p_merge_INV"]]
        M[i, j] = float(row[metric]) if pd.notna(row[metric]) else np.nan

    return p_pro_vals, p_inv_vals, M


def render_heatmap(
    p_pro_vals,
    p_inv_vals,
    M,
    *,
    title: str,
    out_dir: Path,
    vmin: float | None,
    vmax: float | None,
    cmap: str,
    annot: bool,
    fmt: str,
    dpi: int,
    save_pdf: bool,
    save_svg: bool,
    hatch_lower: bool = True,
):
    """Render and save the heatmap; return path to the PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose vmin/vmax from data if not provided
    data = M[np.isfinite(M)]
    if data.size == 0:
        raise ValueError("All values are NaN; cannot render heatmap.")
    if vmin is None:
        vmin = float(np.nanpercentile(data, 5))
    if vmax is None:
        vmax = float(np.nanpercentile(data, 95))
    if vmin == vmax:
        # widen slightly to avoid mpl warnings
        vmin -= 1.0
        vmax += 1.0

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a copy of the colormap if possible; set NaN colour to light grey
    cmap_obj = plt.get_cmap(cmap)
    if hasattr(cmap_obj, "copy"):
        cmap_obj = cmap_obj.copy()
    cmap_obj.set_bad(color=(0.95, 0.95, 0.95, 1.0))

    im = ax.imshow(M, origin="lower", cmap=cmap_obj, vmin=vmin, vmax=vmax)

    # Axis ticks/labels
    ax.set_xticks(range(len(p_inv_vals)))
    ax.set_yticks(range(len(p_pro_vals)))
    ax.set_xticklabels([f"{v:.1f}" for v in p_inv_vals])
    ax.set_yticklabels([f"{v:.1f}" for v in p_pro_vals])
    ax.set_xlabel("p_merge (INV)")
    ax.set_ylabel("p_merge (PRO)")
    ax.set_title(title)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% difference (positive = INV > PRO)")

    # Hatch lower triangle (j < i)
    if hatch_lower:
        for i in range(len(p_pro_vals)):
            for j in range(len(p_inv_vals)):
                if j < i:
                    ax.add_patch(
                        Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=False,
                            hatch="//",
                            edgecolor="k",
                            linewidth=0,
                        )
                    )

    # Optional annotations
    if annot:
        for i in range(len(p_pro_vals)):
            for j in range(len(p_inv_vals)):
                val = M[i, j]
                if np.isfinite(val):
                    ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8, color="black")

    fig.tight_layout()
    out_png = out_dir / "mean_size_pctdiff_heatmap.png"
    fig.savefig(out_png, dpi=dpi)
    if save_pdf:
        fig.savefig(out_dir / "mean_size_pctdiff_heatmap.pdf")
    if save_svg:
        fig.savefig(out_dir / "mean_size_pctdiff_heatmap.svg")
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser(description="Render heatmap from sweep_pmerge_summary.csv")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=str, help="Path to sweep_pmerge_summary.csv")
    src.add_argument(
        "--run_dir",
        type=str,
        help="Path to a timestamped run directory that contains summary/sweep_pmerge_summary.csv",
    )

    ap.add_argument(
        "--metric",
        type=str,
        default="pct_diff_INV_vs_PRO_final",
        help="Metric column name to plot (default: pct_diff_INV_vs_PRO_final)",
    )
    ap.add_argument(
        "--title",
        type=str,
        default="% difference in mean cluster size (final) — INV vs PRO",
        help="Figure title",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save the heatmap (default: run's plots/ folder or CSV's parent/../plots)",
    )

    # >>> Updated defaults here (you can still override via CLI) <<<
    ap.add_argument("--vmin", type=float, default=143.0, help="Lower colour bound")
    ap.add_argument("--vmax", type=float, default=562.0, help="Upper colour bound")
    ap.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap")

    ap.add_argument("--annot", action="store_true", help="Annotate cells with values")
    ap.add_argument("--fmt", type=str, default=".0f", help="Annotation number format (e.g., .0f, .1f)")
    ap.add_argument("--dpi", type=int, default=300, help="PNG DPI")
    ap.add_argument("--pdf", action="store_true", help="Also save PDF")
    ap.add_argument("--svg", action="store_true", help="Also save SVG")

    args = ap.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        default_out = csv_path.parent.parent / "plots"  # <run>/plots if following our layout
    else:
        run_dir = Path(args.run_dir)
        csv_path = find_summary_csv(run_dir)
        default_out = run_dir / "plots"

    out_dir = Path(args.out_dir) if args.out_dir else default_out

    df = pd.read_csv(csv_path)
    p_pro_vals, p_inv_vals, M = build_matrix(df, metric=args.metric)

    out_png = render_heatmap(
        p_pro_vals,
        p_inv_vals,
        M,
        title=args.title,
        out_dir=out_dir,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        annot=args.annot,
        fmt=args.fmt,
        dpi=args.dpi,
        save_pdf=args.pdf,
        save_svg=args.svg,
        hatch_lower=True,
    )

    print(f"Saved heatmap: {out_png}")


if __name__ == "__main__":
    main()