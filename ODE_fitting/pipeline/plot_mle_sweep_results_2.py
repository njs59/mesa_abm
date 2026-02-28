#!/usr/bin/env python3
"""
Plot ODE parameter (theta_hat) heatmaps for EACH model across a 2-D sweep,
and overlay the GLOBAL best-model decision boundary on every heatmap.

Usage:
  python -m pipeline.plot_mle_sweep_results_2 --root <SWEEP_ROOT> [--param-names N0,p,b]

What it does:
  1) Loads ODE/MLE outputs from <SWEEP_ROOT>/scenarios.json and the per-model
     <scenario_dir>/model_*/mle/mle_results.yaml files (AIC + theta_hat).
  2) Detects the two swept parameters (p1, p2) across scenarios.
  3) Builds the best-model grid Z (min AIC across models for each (p1,p2)).
  4) For EACH model, for EACH theta index, makes a heatmap of that parameter's
     fitted value over the (p1,p2) grid, with the GLOBAL Z boundary overlaid.

Outputs:
  <root>/extra_plots/best_model_2d.png
  <root>/extra_plots/best_model_grid.csv
  <root>/extra_plots/params_heatmaps/<model_name>/heatmap_<param>.png
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

sns.set(style="whitegrid")
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# ---------------------------------------------------------------------
# Manifest + data collection
# ---------------------------------------------------------------------

def load_manifest(run_root: str):
    path = os.path.join(run_root, "scenarios.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing scenarios.json at: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def flatten_dict(d, parent_key=""):
    out = {}
    for k, v in (d or {}).items():
        nk = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, nk))
        else:
            out[nk] = v
    return out

def collect_mle_outputs(run_root, scenarios):
    """
    Return DataFrame with columns:
      scenario | scenario_dir? | model | AIC | params (theta_hat list) | <flattened overrides...>
    """
    rows = []
    for sc in scenarios:
        scen_name = sc.get("scenario_name")
        scen_dir = sc.get("scenario_dir")
        overrides = (sc.get("abm_params", {}) or {}).get("overrides", {}) or {}
        flat_over = flatten_dict(overrides)

        if not scen_dir or not os.path.isdir(scen_dir):
            continue

        for item in os.listdir(scen_dir):
            if not item.startswith("model_"):
                continue
            model_id = item.replace("model_", "")
            res_yaml = os.path.join(scen_dir, item, "mle", "mle_results.yaml")
            if not os.path.isfile(res_yaml):
                continue
            try:
                with open(res_yaml, "r") as f:
                    mle = yaml.safe_load(f) or {}
            except Exception:
                mle = {}
            aic = mle.get("AIC", np.nan)
            theta = mle.get("theta_hat", None)
            row = {
                "scenario": scen_name,
                "model": model_id,
                "AIC": aic,
                "params": theta,
            }
            row.update(flat_over)
            rows.append(row)
    return pd.DataFrame(rows)

def detect_swept_params(df: pd.DataFrame):
    exclude = {"scenario", "model", "AIC", "params"}
    cols = [c for c in df.columns if c not in exclude]
    swept = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return swept

def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _pretty_param_name(name: str) -> str:
    parts = name.split(".")
    if len(parts) <= 2:
        return name.replace("_", " ")
    return ".".join(parts[-3:]).replace("_", " ")

def _model_to_mathlabel(name: str) -> str:
    """Optional prettifier for model ids -> mathtext. Fallback to raw name."""
    try:
        parts = name.split("_prolif_")
        left = parts[0]  # e.g. 'Cell_num'
        right = parts[1]  # e.g. 'no_shed', 'cst_shed', ...
        middle = right.split("_shed")[0]  # 'no' / 'cst' / 'SA'
        first_map = {"Cell_num": 3}
        second_map = {"no": 0, "cst": 1, "SA": 2}
        a = first_map[left]
        b = second_map[middle]
        return rf"$M_{{{a},{b}}}$"
    except Exception:
        return name

def _safe_folder(s: str) -> str:
    # keep it simple and OS-safe
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_.")
    return out or "model"


# ---------------------------------------------------------------------
# Build best-model grid (winner codes Z) + boundary segments
# ---------------------------------------------------------------------

def build_best_model_grid(df: pd.DataFrame, p1: str, p2: str):
    """
    From all models per (p1,p2), compute:
      - Z (ny × nx): integer code of the best model (lowest AIC),
      - Delta (ny × nx): runner-up ΔAIC,
      - model_names (list of model ids),
      - model_math (list of nice labels),
      - x_unique, y_unique
    """
    df_xy = df.copy()
    df_xy["_x"] = _to_numeric_series(df_xy[p1])
    df_xy["_y"] = _to_numeric_series(df_xy[p2])
    df_xy = df_xy[np.isfinite(df_xy["_x"]) & np.isfinite(df_xy["_y"])]

    x_unique = np.sort(df_xy["_x"].unique())
    y_unique = np.sort(df_xy["_y"].unique())
    nx, ny = len(x_unique), len(y_unique)
    if nx == 0 or ny == 0:
        raise ValueError("Empty 2D grid after coercion to numeric.")

    model_names = sorted(df_xy["model"].unique())
    model_math = [_model_to_mathlabel(m) for m in model_names]
    code_of = {m: i for i, m in enumerate(model_names)}

    Z = np.full((ny, nx), np.nan, dtype=float)
    Delta = np.full((ny, nx), np.nan, dtype=float)

    # At each (x,y), choose model with lowest AIC
    for i, xv in enumerate(x_unique):
        for j, yv in enumerate(y_unique):
            sub = df_xy[(df_xy["_x"] == xv) & (df_xy["_y"] == yv)]
            if sub.empty or sub["AIC"].isna().all():
                continue
            sub = sub.dropna(subset=["AIC"]).sort_values("AIC")
            Z[j, i] = code_of[sub.iloc[0]["model"]]
            if len(sub) > 1:
                Delta[j, i] = sub.iloc[1]["AIC"] - sub.iloc[0]["AIC"]

    return Z, Delta, model_names, model_math, x_unique, y_unique

def extract_model_boundary_segments(Z: np.ndarray):
    """
    Given Z (ny × nx) of integer codes, return boundary line segments
    in index-space (for overlay on pcolormesh heatmaps).
    """
    ny, nx = Z.shape
    segments = []
    # Horizontal boundaries
    for j in range(ny - 1):
        for i in range(nx):
            if Z[j, i] != Z[j + 1, i]:
                segments.append(([i, i + 1], [j + 1, j + 1]))
    # Vertical boundaries
    for j in range(ny):
        for i in range(nx - 1):
            if Z[j, i] != Z[j, i + 1]:
                segments.append(([i + 1, i + 1], [j, j + 1]))
    return segments


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_best_model_2d(Z, x_unique, y_unique, model_names, model_math, p1, p2, outdir):
    cmap = mcolors.ListedColormap(sns.color_palette("tab10", len(model_names)))
    bounds = np.arange(-0.5, len(model_names) + 0.5, 1.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    nx, ny = len(x_unique), len(y_unique)
    ix = np.arange(nx + 1)
    iy = np.arange(ny + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    mesh = ax.pcolormesh(ix, iy, Z, cmap=cmap, norm=norm,
                         shading="flat", edgecolors="white", linewidth=0.6)
    ax.set_aspect("equal")
    ax.set_xlabel(_pretty_param_name(p1)); ax.set_ylabel(_pretty_param_name(p2))
    ax.set_title("Best-fitting ODE model on parameter grid")

    # centre ticks
    ax.set_xticks(np.arange(nx) + 0.5)
    ax.set_yticks(np.arange(ny) + 0.5)
    ax.set_xticklabels([f"{v:g}" for v in x_unique], ha="center")
    ax.set_yticklabels([f"{v:g}" for v in y_unique])

    cbar = fig.colorbar(mesh, ax=ax, ticks=np.arange(len(model_names)), fraction=0.035, pad=0.02)
    cbar.ax.set_yticklabels(model_math)
    cbar.set_label("Best model (lowest AIC)")

    sns.despine(left=False, bottom=False)
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "best_model_2d.png"), bbox_inches="tight")
    plt.close(fig)

def plot_parameter_heatmaps_per_model(df_all, Z, x_unique, y_unique,
                                      p1, p2, outdir_root, param_names=None):
    """
    For EACH model, for EACH theta index, build a heatmap of the fitted parameter
    across the (x_unique, y_unique) grid. Overlay the global best-model boundary.

    df_all: table with columns ['model', '_x', '_y', 'params'] + AIC etc.
            It should contain one row per (scenario, model).
    Z: best-model codes (ny × nx) across models.
    """
    # prepare boundary segments once
    segments = extract_model_boundary_segments(Z)

    # discover per-model maximum theta length (can vary by model)
    def len_theta(v):
        return len(v) if isinstance(v, (list, tuple)) else 0

    models = sorted(df_all["model"].unique())
    nx, ny = len(x_unique), len(y_unique)
    ix = np.arange(nx + 1)
    iy = np.arange(ny + 1)

    # optional naming
    def names_for_len(L):
        if param_names:
            base = [s.strip() for s in param_names.split(",")]
            if len(base) < L:
                base = base + [f"param_{k+1}" for k in range(len(base), L)]
            return base[:L]
        return [f"param_{k+1}" for k in range(L)]

    for model in models:
        df_m = df_all[df_all["model"] == model].copy()
        if df_m.empty:
            continue

        # ensure numeric coords
        df_m = df_m[np.isfinite(df_m["_x"]) & np.isfinite(df_m["_y"])]

        # how many parameters does this model have?
        max_len = df_m["params"].apply(len_theta).max()
        if not max_len or np.isnan(max_len):
            print(f"[INFO] Skipping {model}: no theta_hat values.")
            continue
        names = names_for_len(int(max_len))

        # create output folder for this model
        model_folder = _safe_folder(model)
        outdir = os.path.join(outdir_root, "params_heatmaps", model_folder)
        os.makedirs(outdir, exist_ok=True)

        # expand theta into columns for speed
        for k in range(int(max_len)):
            df_m[f"_theta_{k}"] = df_m["params"].apply(
                lambda v: (v[k] if isinstance(v, (list, tuple)) and len(v) > k else np.nan)
            )

        # build and plot a grid for each parameter index
        for k in range(int(max_len)):
            # grid of parameter values for this model
            STAT = np.full((ny, nx), np.nan, dtype=float)
            for _, r in df_m.iterrows():
                # indices for this (x,y)
                i = int(np.where(x_unique == r["_x"])[0][0])
                j = int(np.where(y_unique == r["_y"])[0][0])
                STAT[j, i] = r[f"_theta_{k}"]

            fig, ax = plt.subplots(figsize=(7.8, 6.2))
            mesh = ax.pcolormesh(ix, iy, STAT, cmap="viridis",
                                 shading="flat", edgecolors="white", linewidth=0.4)

            # overlay the GLOBAL boundary
            for xs, ys in segments:
                ax.plot(xs, ys, color="black", linewidth=2)

            # centred ticks
            ax.set_xticks(np.arange(nx) + 0.5)
            ax.set_yticks(np.arange(ny) + 0.5)
            ax.set_xticklabels([f"{v:g}" for v in x_unique], rotation=45, ha="right")
            ax.set_yticklabels([f"{v:g}" for v in y_unique])

            ax.set_xlabel(_pretty_param_name(p1))
            ax.set_ylabel(_pretty_param_name(p2))
            ax.set_title(f"{model} — {names[k]}")
            ax.set_aspect("equal")

            cbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label(names[k])

            fig.tight_layout()
            out_png = os.path.join(outdir, f"heatmap_{names[k]}.png")
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"[OK] wrote {out_png}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="ODE per-model parameter heatmaps + global best-model boundary")
    ap.add_argument("--root", required=True, help="Sweep root folder (contains scenarios.json)")
    ap.add_argument("--param-names", default=None,
                    help="Comma-separated names for theta_hat entries (applies to all models; "
                         "defaults to param_1, param_2, ...).")
    args = ap.parse_args()

    run_root = os.path.abspath(args.root)
    outdir = os.path.join(run_root, "extra_plots")
    os.makedirs(outdir, exist_ok=True)

    print(f"Loading scenarios from: {run_root}")
    scenarios = load_manifest(run_root)
    print("Collecting MLE outputs...")
    df = collect_mle_outputs(run_root, scenarios)

    if df.empty:
        print("[ERROR] No MLE outputs found. Exiting.")
        return

    # Identify swept parameters (must be exactly 2 for 2-D grids)
    swept = detect_swept_params(df)
    print(f"Detected swept parameters: {swept}")
    if len(swept) != 2:
        print("[ERROR] Exactly 2 swept parameters are required for 2-D heatmaps.")
        return
    p1, p2 = swept[0], swept[1]

    # Numeric coordinates
    df["_x"] = _to_numeric_series(df[p1])
    df["_y"] = _to_numeric_series(df[p2])
    df = df[np.isfinite(df["_x"]) & np.isfinite(df["_y"])]

    # Build global best-model grid Z (to overlay everywhere)
    Z, Delta, model_names, model_math, x_unique, y_unique = build_best_model_grid(df, p1, p2)

    # Save the reference best-model plot & CSV (handy for QA)
    plot_best_model_2d(Z, x_unique, y_unique, model_names, model_math, p1, p2, outdir)

    grid_rows = []
    for j, yv in enumerate(y_unique):
        for i, xv in enumerate(x_unique):
            code = Z[j, i]
            name = model_names[int(code)] if np.isfinite(code) else None
            dAIC = Delta[j, i] if np.isfinite(Delta[j, i]) else None
            grid_rows.append({p1: float(xv), p2: float(yv), "best_model": name, "delta_AIC": dAIC})
    pd.DataFrame(grid_rows).to_csv(os.path.join(outdir, "best_model_grid.csv"), index=False)

    # Make per-model parameter heatmaps (with the SAME boundary overlaid)
    plot_parameter_heatmaps_per_model(
        df_all=df,
        Z=Z,
        x_unique=x_unique,
        y_unique=y_unique,
        p1=p1,
        p2=p2,
        outdir_root=outdir,
        param_names=args.param_names
    )


if __name__ == "__main__":
    main()