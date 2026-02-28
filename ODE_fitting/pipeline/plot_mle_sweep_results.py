#!/usr/bin/env python3
"""
Clean visualisations for MLE-only sweep results.

Usage:
    python -m pipeline.plot_mle_sweep_results --run-root <RUN_ROOT>

Outputs (under <RUN_ROOT>/extra_plots/):
    - best_model_barplot.png
    - aic_heatmap.png
    - best_model_1d.png          (if 1 swept param)
    - best_model_2d.png          (if 2 swept params; ticks centred, single gridline set)
    - best_model_grid.csv        (data table for 2D)
    - param_*.png                (best-model theta_hat, when available)
"""

import os
import yaml
import argparse
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

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_manifest(run_root: str):
    path = os.path.join(run_root, "scenarios.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing scenarios.json at: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def flatten_dict(d, parent_key=""):
    out = {}
    for k, v in (d or {}).items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, new_key))
        else:
            out[new_key] = v
    return out

def collect_mle_outputs(run_root, scenarios):
    rows = []
    for sc in scenarios:
        scenario_name = sc["scenario_name"]
        scenario_dir = sc["scenario_dir"]

        overrides = (sc.get("abm_params", {}) or {}).get("overrides", {}) or {}
        flat_over = flatten_dict(overrides)

        if not os.path.isdir(scenario_dir):
            continue

        for item in os.listdir(scenario_dir):
            if not item.startswith("model_"):
                continue
            model_id = item.replace("model_", "")
            yaml_path = os.path.join(scenario_dir, item, "mle", "mle_results.yaml")
            if not os.path.isfile(yaml_path):
                continue

            with open(yaml_path, "r") as f:
                mle_out = yaml.safe_load(f) or {}

            aic = mle_out.get("AIC", np.nan)
            theta_hat = mle_out.get("theta_hat", None)

            row = {"scenario": scenario_name, "model": model_id, "AIC": aic, "params": theta_hat}
            row.update(flat_over)
            rows.append(row)

    return pd.DataFrame(rows)

def detect_swept_params(df: pd.DataFrame):
    exclude = {"scenario", "model", "AIC", "params"}
    param_cols = [c for c in df.columns if c not in exclude]
    swept = [c for c in param_cols if df[c].nunique(dropna=True) > 1]
    return swept

def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _pretty_param_name(name: str) -> str:
    """Shorten nested keys for cleaner axis labels."""
    parts = name.split(".")
    if len(parts) <= 2:
        return name.replace("_", " ")
    short = ".".join(parts[-3:])
    return short.replace("_", " ")

def _model_to_mathlabel(name: str) -> str:
    """
    Convert names like 'Cell_num_prolif_no_shed' to mathtext 'M_{3,0}'.

    Mapping:
      α (first index): token BEFORE '_prolif_' — {'Cell_num': 3}
      β (second index): token BETWEEN prolif/shed — {'no': 0, 'cst': 1, 'SA': 2}
    Fallback: original string if parsing fails.
    """
    try:
        parts = name.split("_prolif_")
        left = parts[0]                  # e.g. 'Cell_num'
        right = parts[1]                 # e.g. 'no_shed'
        middle = right.split("_shed")[0] # 'no' | 'cst' | 'SA'
        first_map = {"Cell_num": 3}
        second_map = {"no": 0, "cst": 1, "SA": 2}
        a = first_map[left]
        b = second_map[middle]
        return rf"$M_{{{a},{b}}}$"
    except Exception:
        return name

def _save_png(fig, path_no_ext: str):
    fig.savefig(path_no_ext + ".png", bbox_inches="tight")

# ------------------------------------------------------------
# Generic plots
# ------------------------------------------------------------

def plot_best_fit_counts(df: pd.DataFrame, outdir: str):
    idx = df.groupby("scenario")["AIC"].idxmin()
    best = df.loc[idx].copy()
    best["model_math"] = best["model"].apply(_model_to_mathlabel)
    counts = best["model_math"].value_counts()

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    sns.barplot(x=counts.index, y=counts.values, palette="tab10", ax=ax)
    ax.set_xlabel("ODE model")
    ax.set_ylabel("Number of scenarios (best by AIC)")
    ax.set_title("Best-fitting ODE model per scenario")
    sns.despine()
    _save_png(fig, os.path.join(outdir, "best_model_barplot"))
    plt.close(fig)

def plot_aic_heatmap(df: pd.DataFrame, outdir: str):
    tmp = df.copy()
    tmp["model_math"] = tmp["model"].apply(_model_to_mathlabel)
    pivot = tmp.pivot(index="scenario", columns="model_math", values="AIC")
    fig_h = max(4.2, 0.24 * len(pivot))
    fig, ax = plt.subplots(figsize=(8, fig_h))
    sns.heatmap(pivot, cmap="viridis", cbar_kws={"label": "AIC"}, ax=ax)
    ax.set_title("AIC across models and scenarios")
    ax.set_xlabel("Model"); ax.set_ylabel("Scenario")
    plt.tight_layout()
    _save_png(fig, os.path.join(outdir, "aic_heatmap"))
    plt.close(fig)

def plot_best_params(df: pd.DataFrame, outdir: str):
    idx = df.groupby("scenario")["AIC"].idxmin()
    best = df.loc[idx].copy()
    best = best.dropna(subset=["params"])
    if best.empty:
        return

    best["model_math"] = best["model"].apply(_model_to_mathlabel)

    max_len = best["params"].apply(lambda v: len(v) if isinstance(v, (list, tuple)) else 0).max()
    for i in range(max_len):
        best[f"param_{i+1}"] = best["params"].apply(
            lambda v: v[i] if (isinstance(v, (list, tuple)) and len(v) > i) else np.nan
        )

    param_cols = [c for c in best.columns if c.startswith("param_")]
    for param in param_cols:
        fig_h = max(4.2, 0.26 * best["scenario"].nunique())
        fig, ax = plt.subplots(figsize=(8.5, fig_h))
        order = best["scenario"].sort_values().unique()
        sns.barplot(data=best, x="scenario", y=param, hue="model_math",
                    dodge=False, palette="tab10", order=order, ax=ax)
        ax.set_title(f"Best-model fitted value: {param}")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=90)
        ax.legend(frameon=False, loc="best", title="model")
        sns.despine()
        _save_png(fig, os.path.join(outdir, f"{param}"))
        plt.close(fig)

# ------------------------------------------------------------
# Parameter landscapes
# ------------------------------------------------------------

def plot_landscape_1d(df: pd.DataFrame, param: str, outdir: str):
    idx = df.groupby("scenario")["AIC"].idxmin()
    best = df.loc[idx].copy()

    xs = _to_numeric_series(best[param])
    best = best.assign(_x=xs)
    best = best[np.isfinite(best["_x"])]

    # Keep one per unique param value (lowest AIC)
    best = best.sort_values(["_x", "AIC"]).drop_duplicates(subset=["_x"], keep="first")
    best = best.sort_values("_x")
    best["model_math"] = best["model"].apply(_model_to_mathlabel)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    sns.barplot(
        data=best,
        x="_x", y="AIC",
        hue="model_math", dodge=False,
        palette="tab10", ax=ax
    )
    ax.set_title(f"Best-fitting model across {_pretty_param_name(param)}")
    ax.set_xlabel(_pretty_param_name(param))
    ax.set_ylabel("Best AIC (lower is better)")
    ax.legend(frameon=False, loc="best", title="model")
    vals = best["_x"].to_numpy()
    if len(vals) > 12:
        step = max(1, len(vals)//12)
        ax.set_xticks(vals[::step])
        ax.set_xticklabels([f"{v:g}" for v in vals[::step]])
    else:
        ax.set_xticklabels([f"{v:g}" for v in vals])
    sns.despine()
    _save_png(fig, os.path.join(outdir, "best_model_1d"))
    plt.close(fig)

def plot_landscape_2d(df: pd.DataFrame, p1: str, p2: str, outdir: str, weak_win_delta: float = 2.0):
    """
    Robust 2D landscape plot that works regardless of the numerical scale of p1/p2.
    It plots on an index grid so cells are always square, then labels ticks with
    the actual parameter values (scientific formatting where needed).
    """

    # ------------------------------------------------------------------
    # Determine best model per (x,y)
    # ------------------------------------------------------------------
    idx = df.groupby("scenario")["AIC"].idxmin()
    best = df.loc[idx].copy()

    x_vals = _to_numeric_series(best[p1])
    y_vals = _to_numeric_series(best[p2])
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    best = best.loc[mask].copy()
    best["_x"] = x_vals.loc[mask]
    best["_y"] = y_vals.loc[mask]

    # Deduplicate by (x,y): lowest AIC only
    best = (
        best.sort_values(["_x", "_y", "AIC"])
             .drop_duplicates(subset=["_x", "_y"], keep="first")
    )

    x_unique = np.sort(best["_x"].unique())
    y_unique = np.sort(best["_y"].unique())
    nx, ny = len(x_unique), len(y_unique)
    if nx == 0 or ny == 0:
        return

    model_names = sorted(best["model"].unique())
    model_math = [_model_to_mathlabel(m) for m in model_names]
    code_of = {m: i for i, m in enumerate(model_names)}

    # ------------------------------------------------------------------
    # Build grids Z (winner index) and Delta (runner-up AIC)
    # ------------------------------------------------------------------
    Z = np.full((ny, nx), np.nan)
    Delta = np.full((ny, nx), np.nan)

    df_xy = df.copy()
    df_xy["_x"] = _to_numeric_series(df_xy[p1])
    df_xy["_y"] = _to_numeric_series(df_xy[p2])
    df_xy = df_xy[np.isfinite(df_xy["_x"]) & np.isfinite(df_xy["_y"])]

    for i, xv in enumerate(x_unique):
        for j, yv in enumerate(y_unique):
            sub = df_xy[(df_xy["_x"] == xv) & (df_xy["_y"] == yv)]
            if sub.empty or sub["AIC"].isna().all():
                continue
            sub = sub.dropna(subset=["AIC"]).sort_values("AIC")
            Z[j, i] = code_of[sub.iloc[0]["model"]]
            if len(sub) > 1:
                Delta[j, i] = sub.iloc[1]["AIC"] - sub.iloc[0]["AIC"]

    # ------------------------------------------------------------------
    # Plot in INDEX SPACE so cells stay square
    # ------------------------------------------------------------------
    ix = np.arange(nx + 1)
    iy = np.arange(ny + 1)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))

    cmap = mcolors.ListedColormap(sns.color_palette("tab10", len(model_names)))
    bounds = np.arange(-0.5, len(model_names) + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    mesh = ax.pcolormesh(
        ix, iy, Z,
        cmap=cmap, norm=norm,
        shading="flat",
        edgecolors="white", linewidth=0.6
    )

    ax.set_aspect("equal")



    # Use scientific formatting for compact non-overlapping labels
    import math

    def fmt(v):
        # Choose between fixed or scientific based on magnitude
        if v != 0 and (abs(v) < 1e-2 or abs(v) > 1e3):
            return f"{v:.3e}"
        return f"{v:g}"
    
    x_centres = np.arange(nx) + 0.5
    y_centres = np.arange(ny) + 0.5

    ax.set_xticks(x_centres)
    ax.set_yticks(y_centres)

    ax.set_xticklabels([fmt(v) for v in x_unique], ha="center")
    ax.set_yticklabels([fmt(v) for v in y_unique], va="center")

    # ax.set_xticklabels([fmt(v) for v in x_unique], ha="center")
    # ax.set_yticklabels([fmt(v) for v in y_unique], va="center")

    # Thin ticks if too many
    max_ticks = 12
    if nx > max_ticks:
        sel = np.linspace(0, nx - 1, max_ticks).astype(int)
        ax.set_xticks(sel)
        ax.set_xticklabels([fmt(x_unique[i]) for i in sel], ha="center")

    if ny > max_ticks:
        sel = np.linspace(0, ny - 1, max_ticks).astype(int)
        ax.set_yticks(sel)
        ax.set_yticklabels([fmt(y_unique[i]) for i in sel], va="center")

    # ------------------------------------------------------------------
    # Labels and colourbar
    # ------------------------------------------------------------------
    ax.set_xlabel(_pretty_param_name(p1))
    ax.set_ylabel(_pretty_param_name(p2))
    ax.set_title("Best-fitting ODE model on parameter grid")

    cbar = fig.colorbar(
        mesh, ax=ax,
        ticks=np.arange(len(model_names)),
        fraction=0.035, pad=0.02
    )
    cbar.ax.set_yticklabels(model_math)
    cbar.set_label("Best model (lowest AIC)")

    sns.despine(left=False, bottom=False)

    _save_png(fig, os.path.join(outdir, "best_model_2d"))
    plt.close(fig)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    rows = []
    for j, yv in enumerate(y_unique):
        for i, xv in enumerate(x_unique):
            code = Z[j, i]
            name = model_names[int(code)] if np.isfinite(code) else None
            dAIC = Delta[j, i] if np.isfinite(Delta[j, i]) else None
            rows.append({p1: float(xv), p2: float(yv), "best_model": name, "delta_AIC": dAIC})

    pd.DataFrame(rows).to_csv(os.path.join(outdir, "best_model_grid.csv"), index=False)
# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cleaner extra visualisations for MLE-only results")
    parser.add_argument("--run-root", required=True,
                        help="Folder containing results from pipeline_mle_only")
    args = parser.parse_args()

    run_root = os.path.abspath(args.run_root)
    outdir = os.path.join(run_root, "extra_plots")
    os.makedirs(outdir, exist_ok=True)

    print(f"Loading scenarios from: {run_root}")
    scenarios = load_manifest(run_root)

    print("Collecting MLE outputs...")
    df = collect_mle_outputs(run_root, scenarios)
    if df.empty:
        print("No MLE outputs found. Exiting.")
        return

    # Generic plots
    print("Plotting best-model counts...")
    plot_best_fit_counts(df, outdir)

    print("Plotting AIC heatmap...")
    plot_aic_heatmap(df, outdir)

    print("Plotting best-model parameter bars (if theta_hat available)...")
    plot_best_params(df, outdir)

    # Landscapes
    swept = detect_swept_params(df)
    print(f"Detected swept parameters: {swept}")

    if len(swept) == 1:
        print("Generating 1D landscape...")
        plot_landscape_1d(df, swept[0], outdir)

    elif len(swept) == 2:
        print("Generating 2D landscape...")
        plot_landscape_2d(df, swept[0], swept[1], outdir)

    else:
        print("No 1D/2D landscape produced (swept params: 0 or >2).")

if __name__ == "__main__":
    main()