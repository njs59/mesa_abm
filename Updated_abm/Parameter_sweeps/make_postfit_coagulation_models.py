#!/usr/bin/env python3
"""
postfit_coagulation_models.py
-----------------------------
Post-fitting for coagulation-vs-distance curves produced by
make_1d_coagulation_plots_infer.py.

Reads (from the inference step):
  Parameter_sweeps_results/<RUN>/coagulation_1d_infer/combined/coag_prob_by_distance_ALL.csv
  # columns: param_key, param_value, d_center, n_pairs, n_merged, prob

Fits, per parameter value:
  - exp       : A * exp(-B d) + C
  - strexp    : A * exp(-(d/lam)^k) + C
  - power     : A * (d + x0)^(-k) + C
  - exp_pure  : A * exp(-B d)
  - exp_half  : A * 2^(-d / d_half)
  - recip     : A / (1 + B d)
  - invlin    : A / (d + d0)
  - hill      : A / (1 + (d/lam)^k)
  - ratquad   : A / (1 + (d/lam)^2)

Saves:
  - Per-parameter CSV of parameters & metrics (SSE, AIC, BIC, R^2, KS)
  - Per-parameter CSV of predictions on the full x grid
  - All-model overlays (linear & log-y)
  - EXP vs STREXP overlays (linear & log-y)
  - Parameter trends for EXP and STREXP (individual plots)
  - Subplot grids of EXP and STREXP parameters vs the sweep value
  - Global summary CSV across all param values/models (includes param_key)

Outputs under:
  <run>/coagulation_1d_infer/model_fits_all/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------------- Model functions -----------------------
def f_exp_off(x, A, B, C):      # A * exp(-B d) + C
    return A * np.exp(-B * x) + C

def f_strexp(x, A, lam, k, C):  # A * exp(-(d/lam)^k) + C
    return A * np.exp(-np.power(np.maximum(x, 1e-12) / lam, k)) + C

def f_power(x, A, x0, k, C):    # A * (d + x0)^(-k) + C
    return A * np.power(np.maximum(x + x0, 1e-9), -k) + C

# simpler models (1–6)
def f_exp_pure(x, A, B):        # A * exp(-B d)
    return A * np.exp(-B * x)

def f_exp_half(x, A, d_half):   # A * 2^(-d/d_half)
    return A * np.power(2.0, -x / np.maximum(d_half, 1e-12))

def f_recip(x, A, B):           # A / (1 + B d)
    return A / np.maximum(1.0 + B * x, 1e-12)

def f_invlin(x, A, d0):         # A / (d + d0)
    return A / np.maximum(x + d0, 1e-12)

def f_hill(x, A, lam, k):       # A / (1 + (d/lam)^k)
    return A / (1.0 + np.power(np.maximum(x, 0.0) / np.maximum(lam, 1e-12), np.maximum(k, 1e-12)))

def f_ratquad(x, A, lam):       # A / (1 + (d/lam)^2)
    z = np.maximum(x, 0.0) / np.maximum(lam, 1e-12)
    return A / (1.0 + z * z)

MODEL_SPECS: Dict[str, dict] = {
    "exp":     dict(func=f_exp_off,  pnames=["A","B","C"],
                    p0=lambda x,y: [float(y[0]) if len(y) else 1.0, 0.01, 0.0],
                    bounds=([0.0, 0.0, 0.0], [1.5, 10.0, 0.5])),
    "strexp":  dict(func=f_strexp,   pnames=["A","lam","k","C"],
                    p0=lambda x,y: [float(y[0]) if len(y) else 1.0,
                                    max(1.0, float(np.median(x))), 1.0, 0.0],
                    bounds=([0.0, 1e-3, 0.2, 0.0], [2.0, 1e4, 6.0, 0.5])),
    "power":   dict(func=f_power,    pnames=["A","x0","k","C"],
                    p0=lambda x,y: [float(y[0]) if len(y) else 1.0, 1.0, 1.0, 0.0],
                    bounds=([0.0, 0.0, 0.2, 0.0], [2.0, 1e3, 6.0, 0.5])),

    "exp_pure":dict(func=f_exp_pure, pnames=["A","B"],
                    p0=lambda x,y: [min(1.0, float(y[0]) if len(y) else 1.0), 0.02],
                    bounds=([0.0, 0.0], [1.5, 10.0])),
    "exp_half":dict(func=f_exp_half, pnames=["A","d_half"],
                    p0=lambda x,y: [min(1.0, float(y[0]) if len(y) else 1.0),
                                    max(1.0, float(np.median(x)) or 1.0)],
                    bounds=([0.0, 1e-3], [1.5, 1e5])),
    "recip":   dict(func=f_recip,    pnames=["A","B"],
                    p0=lambda x,y: [min(1.0, float(np.max(y)) if len(y) else 1.0), 0.01],
                    bounds=([0.0, 0.0], [1.5, 10.0])),
    "invlin":  dict(func=f_invlin,   pnames=["A","d0"],
                    p0=lambda x,y: [min(1.0, float(np.max(y)) if len(y) else 1.0), 1.0],
                    bounds=([0.0, 1e-6], [1.5, 1e5])),
    "hill":    dict(func=f_hill,     pnames=["A","lam","k"],
                    p0=lambda x,y: [min(1.0, float(np.max(y)) if len(y) else 1.0),
                                    max(1.0, float(np.median(x)) or 1.0), 1.0],
                    bounds=([0.0, 1e-3, 0.2], [2.0, 1e5, 6.0])),
    "ratquad": dict(func=f_ratquad,  pnames=["A","lam"],
                    p0=lambda x,y: [min(1.0, float(np.max(y)) if len(y) else 1.0),
                                    max(1.0, float(np.median(x)) or 1.0)],
                    bounds=([0.0, 1e-3], [2.0, 1e5])),
}

# ----------------------- Metrics & fitting -----------------------
def clip01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)

def fit_one(x: np.ndarray, y: np.ndarray, model_name: str) -> dict:
    spec = MODEL_SPECS[model_name]
    func   = spec["func"]
    p0     = spec["p0"](x, y)
    bounds = spec["bounds"]
    try:
        popt, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=20000)
        yhat = clip01(func(x, *popt))
        resid = y - yhat
        sse = float(np.sum(resid**2))
        n = len(y)
        k = len(popt)
        if sse > 0 and n > k:
            aic = n * np.log(sse / n) + 2 * k
            bic = n * np.log(sse / n) + k * np.log(n)
        else:
            aic = bic = np.inf
        sst = float(np.sum((y - np.mean(y))**2))
        r2 = 1.0 - sse / sst if sst > 0 else np.nan
        return dict(model=model_name, pnames=spec["pnames"], params=popt, yhat=yhat,
                    sse=sse, aic=aic, bic=bic, r2=r2)
    except Exception:
        return dict(model=model_name, pnames=spec["pnames"], params=None, yhat=None,
                    sse=np.inf, aic=np.inf, bic=np.inf, r2=np.nan)

def ks_on_discrete_cdfs(n_merged: np.ndarray, n_pairs: np.ndarray, yhat_prob: np.ndarray) -> float:
    obs = np.clip(n_merged.astype(float), 0.0, None)
    mod = np.clip(yhat_prob.astype(float) * n_pairs.astype(float), 0.0, None)
    if obs.sum() <= 0 or mod.sum() <= 0:
        return np.nan
    P_obs = obs / obs.sum()
    P_mod = mod / mod.sum()
    cdf_obs = np.cumsum(P_obs)
    cdf_mod = np.cumsum(P_mod)
    return float(np.nanmax(np.abs(cdf_obs - cdf_mod)))

# ----------------------- IO & small helpers -----------------------
def find_run_dir(results_root: Path, run: str) -> Path:
    if run in (None, "latest", "LAST", "Latest"):
        runs = sorted([p for p in results_root.iterdir() if p.is_dir()])
        if not runs:
            raise FileNotFoundError(f"No runs found under {results_root}")
        return runs[-1]
    rd = results_root / run
    if not rd.exists():
        raise FileNotFoundError(f"Run folder not found: {rd}")
    return rd

def load_combined_csv(run_dir: Path) -> pd.DataFrame:
    fp = run_dir / "coagulation_1d_infer" / "combined" / "coag_prob_by_distance_ALL.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Combined CSV not found: {fp}")
    return pd.read_csv(fp)

def detect_param_key(df: pd.DataFrame) -> str:
    """Prefer the param_key column if it is present and unique."""
    if "param_key" in df.columns:
        keys = [str(k) for k in df["param_key"].dropna().unique().tolist() if str(k).strip() != ""]
        if len(keys) == 1:
            return keys[0]
    return "ABM sweep parameter"

def as_numeric_with_labels(vals: List[Any]) -> Tuple[np.ndarray, List[str]]:
    """Return numeric x positions and display labels for categorical values."""
    xnum = []
    try_numeric = True
    for v in vals:
        try:
            xnum.append(float(v))
        except Exception:
            try_numeric = False
            break
    if try_numeric:
        return np.array(xnum, dtype=float), [str(v) for v in vals]
    labs = [str(v) for v in vals]
    return np.arange(len(labs), dtype=float), labs

def sort_key(z):
    try:
        return float(z)
    except Exception:
        return str(z)

# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser(description="Fit multiple decay models to coagulation-vs-distance curves and save all results.")
    ap.add_argument("--results-root", type=str, default="Parameter_sweeps_results")
    ap.add_argument("--run", type=str, default="latest")
    ap.add_argument("--models", type=str, nargs="+",
                    default=["exp","strexp","power","exp_pure","exp_half","recip","invlin","hill","ratquad"],
                    help="Which models to fit")
    ap.add_argument("--pmin", type=float, default=0.02,
                    help="Use only bins where ALL parameter curves have prob >= pmin")
    ap.add_argument("--min-pairs-per-bin", type=int, default=50,
                    help="Use only bins where ALL parameter curves have n_pairs >= this")
    ap.add_argument("--dpi", type=int, default=260)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    run_dir = find_run_dir(results_root, args.run)
    df = load_combined_csv(run_dir)  # contains param_key & param_value from the inference step
    if df.empty:
        raise RuntimeError("Combined CSV is empty; run the inference step first.")

    # Detect the sweep parameter name robustly (e.g., 'p_merge' for Merge_sweep)
    sweep_key = detect_param_key(df)  # prefer the single unique param_key from CSV
    print(f"[INFO] Detected sweep parameter: {sweep_key}")  # e.g. p_merge

    outdir = run_dir / "coagulation_1d_infer" / "model_fits_all"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Common reliable fit window across parameter values ---
    keep_prob   = df.assign(ok=(df["prob"]     >= args.pmin))\
                    .pivot(index="d_center", columns="param_value", values="ok").fillna(False)
    keep_counts = df.assign(ok=(df["n_pairs"]  >= args.min_pairs_per_bin))\
                    .pivot(index="d_center", columns="param_value", values="ok").fillna(False)
    mask_common_by_center = (keep_prob.all(axis=1)) & (keep_counts.all(axis=1))
    x_common = np.array(mask_common_by_center[mask_common_by_center].index.values, dtype=float)
    use_common = x_common.size >= 4

    models = [m for m in args.models if m in MODEL_SPECS]
    if not models:
        raise ValueError("No valid models requested.")

    summary_rows: List[dict] = []

    # ---------------- Per-parameter-value fits ----------------
    for pval, g in sorted(df.groupby("param_value"), key=lambda kv: sort_key(kv[0])):
        g = g.sort_values("d_center")
        x        = g["d_center"].to_numpy(float)
        y        = g["prob"].to_numpy(float)
        n_pairs  = g["n_pairs"].to_numpy(float)
        n_merge  = g["n_merged"].to_numpy(float)

        if use_common:
            mask = np.isin(x, x_common)
        else:
            mask = (y >= args.pmin) & (n_pairs >= args.min_pairs_per_bin)
            if np.count_nonzero(mask) < 4:
                mask = np.ones_like(x, dtype=bool)

        xf, yf = x[mask], y[mask]
        npf, nmf = n_pairs[mask], n_merge[mask]

        fit_results: List[dict] = []
        for name in models:
            fr = fit_one(xf, yf, name)
            if fr["yhat"] is not None:
                ks = ks_on_discrete_cdfs(nmf, npf, fr["yhat"])
            else:
                ks = np.nan
            fr["ks"] = float(ks) if np.isfinite(ks) else np.nan
            # evaluate on FULL x for outputs/plots
            if fr["params"] is not None:
                yhat_full = clip01(MODEL_SPECS[name]["func"](x, *fr["params"]))
            else:
                yhat_full = np.full_like(x, np.nan, dtype=float)
            fr["yhat_full"] = yhat_full
            fit_results.append(fr)

        # 1) Save per-parameter parameters/metrics (ALL models)
        rows = []
        for fr in fit_results:
            row = {
                "param_key": sweep_key,
                "param_value": pval,
                "model": fr["model"],
                "n_points_fit": int(len(xf)),
                "sse": fr["sse"],
                "aic": fr["aic"],
                "bic": fr["bic"],
                "r2": fr["r2"],
                "ks": fr["ks"],
                "params_json": "[]" if fr["params"] is None else json.dumps([float(v) for v in fr["params"]]),
            }
            if fr["params"] is not None:
                for name_i, val in zip(MODEL_SPECS[fr["model"]]["pnames"], fr["params"]):
                    row[name_i] = float(val)
            rows.append(row)
            summary_rows.append(row)
        Path(outdir, f"params__{sweep_key}__param={pval}.csv").write_text(
            pd.DataFrame(rows).to_csv(index=False), encoding="utf-8"
        )

        # 2) Save predictions (ALL models) on the full grid
        pred = {"d_center": x, "observed_prob": y, "n_pairs": n_pairs, "n_merged": n_merge}
        for fr in fit_results:
            pred[f"pred__{fr['model']}"] = fr["yhat_full"]
        pd.DataFrame(pred).to_csv(outdir / f"predictions__{sweep_key}__param={pval}.csv", index=False)

        # 3) Plots: ALL models together (linear & log-y)
        def plot_all(logy: bool):
            fig, ax = plt.subplots(figsize=(8.6, 5.2))
            ax.plot(x, y, "o", color="black", ms=4.5, label="Observed")
            palette = ["C1","C2","C3","C4","C5","C6","C7","C8","C9","tab:brown","tab:pink"]
            for i, fr in enumerate(fit_results):
                label = f"{fr['model']} (AIC={fr['aic']:.1f}, KS={fr['ks']:.3f})"
                ax.plot(x, fr["yhat_full"], "-", lw=2.0, color=palette[i % len(palette)], label=label)
            ax.set_xlabel("Initial distance")
            ax.set_ylabel("P(pair lineage merge)")
            if logy:
                ax.set_yscale("log")
            ax.set_title(f"All models — {sweep_key}={pval}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            out = outdir / (f"all_models__{sweep_key}={pval}__logy.png" if logy else f"all_models__{sweep_key}={pval}.png")
            fig.savefig(out, dpi=args.dpi)
            plt.close(fig)
        plot_all(logy=False)
        plot_all(logy=True)

        # 4) EXP vs STREXP overlays (linear & log-y)
        def plot_exp_vs_strexp(logy: bool):
            fig, ax = plt.subplots(figsize=(8.6, 5.2))
            ax.plot(x, y, "o", color="black", ms=4.5, label="Observed")
            for mdl, col in [("exp", "C1"), ("strexp", "C2")]:
                fr = next((r for r in fit_results if r["model"] == mdl), None)
                if fr and fr["params"] is not None:
                    label = f"{mdl} (AIC={fr['aic']:.1f}, KS={fr['ks']:.3f})"
                    ax.plot(x, fr["yhat_full"], "-", lw=2.4, color=col, label=label)
            ax.set_xlabel("Initial distance")
            ax.set_ylabel("P(pair lineage merge)")
            if logy:
                ax.set_yscale("log")
            ax.set_title(f"exp vs strexp — {sweep_key}={pval}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=9)
            fig.tight_layout()
            out = outdir / (f"exp_vs_strexp__{sweep_key}={pval}__logy.png" if logy else f"exp_vs_strexp__{sweep_key}={pval}.png")
            fig.savefig(out, dpi=args.dpi)
            plt.close(fig)
        plot_exp_vs_strexp(logy=False)
        plot_exp_vs_strexp(logy=True)

    # ---------------- Global summary & sweep parameter trends -------------
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "summary_all_models.csv", index=False)

    # Make a nice numeric x for plotting the sweep values and keep labels
    def make_x_for_param(vals: List[Any]):
        xnum, xlabs = as_numeric_with_labels(vals)
        return xnum, xlabs

    # EXP parameter trends
    exp_df = summary_df[summary_df["model"] == "exp"].copy()
    if not exp_df.empty:
        exp_df = exp_df.sort_values("param_value", key=lambda s: s.map(sort_key))
        xvals = exp_df["param_value"].tolist()
        xnum, xlabs = make_x_for_param(xvals)

        # Individual parameter plots
        for p in ["A", "B", "C"]:
            if p in exp_df.columns:
                fig, ax = plt.subplots(figsize=(7.6, 4.4))
                ax.plot(xnum, exp_df[p].astype(float), "-o", lw=2)
                ax.set_xticks(xnum); ax.set_xticklabels(xlabs, rotation=0)
                ax.set_xlabel(sweep_key)
                ax.set_ylabel(f"exp: {p}")
                ax.set_title(f"exp parameter {p} vs {sweep_key}")
                ax.grid(alpha=0.25)
                fig.tight_layout()
                fig.savefig(outdir / f"exp_param_{p}_vs_{sweep_key}.png", dpi=args.dpi)
                plt.close(fig)

        # Subplot grid (exp: A,B,C)
        params_order = [p for p in ["A","B","C"] if p in exp_df.columns]
        if params_order:
            fig, axes = plt.subplots(1, len(params_order), figsize=(5.0*len(params_order), 4.2), squeeze=False)
            for i, p in enumerate(params_order):
                ax = axes[0, i]
                ax.plot(xnum, exp_df[p].astype(float), "-o", lw=2)
                ax.set_xticks(xnum); ax.set_xticklabels(xlabs, rotation=0)
                ax.set_xlabel(sweep_key); ax.set_ylabel(p); ax.grid(alpha=0.25)
                ax.set_title(f"exp: {p}")
            fig.tight_layout()
            fig.savefig(outdir / f"exp_params_grid_vs_{sweep_key}.png", dpi=args.dpi)
            plt.close(fig)

    # STREXP parameter trends
    sx_df = summary_df[summary_df["model"] == "strexp"].copy()
    if not sx_df.empty:
        sx_df = sx_df.sort_values("param_value", key=lambda s: s.map(sort_key))
        xvals = sx_df["param_value"].tolist()
        xnum, xlabs = make_x_for_param(xvals)

        # Individual parameter plots
        for p in ["A", "lam", "k", "C"]:
            if p in sx_df.columns:
                fig, ax = plt.subplots(figsize=(7.6, 4.4))
                ax.plot(xnum, sx_df[p].astype(float), "-o", lw=2)
                ax.set_xticks(xnum); ax.set_xticklabels(xlabs, rotation=0)
                ax.set_xlabel(sweep_key)
                ax.set_ylabel(f"strexp: {p}")
                ax.set_title(f"strexp parameter {p} vs {sweep_key}")
                ax.grid(alpha=0.25)
                fig.tight_layout()
                fig.savefig(outdir / f"strexp_param_{p}_vs_{sweep_key}.png", dpi=args.dpi)
                plt.close(fig)

        # Subplot grid (strexp: A,lam,k,C)
        params_order = [p for p in ["A","lam","k","C"] if p in sx_df.columns]
        if params_order:
            n = len(params_order)
            ncols = min(2, n)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 4.2 * nrows), squeeze=False)
            for i, p in enumerate(params_order):
                r, c = divmod(i, ncols)
                ax = axes[r, c]
                ax.plot(xnum, sx_df[p].astype(float), "-o", lw=2)
                ax.set_xticks(xnum); ax.set_xticklabels(xlabs, rotation=0)
                ax.set_xlabel(sweep_key); ax.set_ylabel(p); ax.grid(alpha=0.25)
                ax.set_title(f"strexp: {p}")
            # hide any unused axes
            for j in range(n, nrows*ncols):
                r, c = divmod(j, ncols); axes[r, c].axis("off")
            fig.tight_layout()
            fig.savefig(outdir / f"strexp_params_grid_vs_{sweep_key}.png", dpi=args.dpi)
            plt.close(fig)

    print(f"[DONE] Wrote per-param tables & plots to: {outdir}")
    print(f"[DONE] Summary saved to:                  {outdir / 'summary_all_models.csv'}")

if __name__ == "__main__":
    main()