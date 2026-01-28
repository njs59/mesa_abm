#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare ABC–SMC posteriors (pyABC) for PRO vs INV and run Posterior Predictive Checks (PPC).

Inputs (in results/ by default):
  - abc_maxabs_PRO_test_1_no_gr_500_8.db
  - abc_maxabs_INV_test_1_no_gr_500_8.db

Outputs (written to results/):
  - posterior_summary_PRO.csv / posterior_summary_INV.csv
  - posterior_marginals_overlay.png
  - posterior_pair_overlay.png
  - ess_over_time.png
  - distance_over_time.png
  - epsilon_over_time.png            (if present)
  - ppc_nnd_ts_PRO.png / ppc_nnd_ts_INV.png
  - ppc_final_summaries.png
  - ppc_summaries_PRO.csv / ppc_summaries_INV.csv
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: seaborn (nicer KDEs); script degrades gracefully if missing
try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

# ------------------------------------------------------------
# Script-relative imports so abm/ is visible
# ------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ABM
from abm.clusters_model import ClustersModel
from abm.utils import DEFAULTS

# pyABC
from pyabc import History


# ============================================================
# Path / IO helpers
# ============================================================

def resolve_results_dir(results_dir: str) -> Path:
    """Interpret relative paths relative to the script (not shell CWD)."""
    p = Path(results_dir)
    if not p.is_absolute():
        p = (SCRIPT_DIR.parent / p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_db_exists(db_path: Path, label: str):
    if not db_path.exists():
        parent = db_path.parent
        candidates = sorted([f.name for f in parent.glob("*.db")])
        msg = [
            f"{label} DB not found at:\n  {db_path}",
            f"Looked in:\n  {parent}",
            f"Available .db files here: {candidates if candidates else '(none found)'}",
            "\nTry:  python scripts/compare_abc_posteriors.py --out results",
        ]
        raise FileNotFoundError("\n".join(msg))


# ============================================================
# pyABC → posterior extraction
# ============================================================

def load_history(db_path: Path) -> History:
    return History(f"sqlite:///{db_path}")


def _select_param_columns(df: pd.DataFrame) -> List[str]:
    """
    Heuristically choose numeric parameter columns from a pyABC population DF.
    Excludes typical meta columns (t, w, epsilon, distance, etc.).
    """
    ignore = {"t", "w", "epsilon", "distance", "accept", "particle", "m", "model"}
    cols = []
    for c in df.columns:
        if c in ignore:
            continue
        if np.issubdtype(df[c].dtype, np.number):
            cols.append(c)
    return cols


def get_final_posterior(history: History) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    Returns (params_df, weights, t_max) for the final population using
    History.get_distribution(m=0, t=t_max) → (df, w).
    """
    # last population index
    t_max = int(history.max_t)

    # df: parameters; w: weights. Default model m=0.
    params_df, weights = history.get_distribution(t=t_max)  # df, w
    weights = np.asarray(weights, dtype=float)
    sw = weights.sum()
    if sw > 0:
        weights = weights / sw

    # Keep only numeric parameter columns
    param_cols = _select_param_columns(params_df)
    params_df = params_df[param_cols].copy()
    print(f"Final population t={t_max}: n_particles={len(params_df)}; parameters={param_cols}")
    return params_df, weights, t_max


def weighted_summary(df: pd.DataFrame, w: np.ndarray) -> pd.DataFrame:
    """Weighted mean, sd, and credible intervals per parameter."""
    out = []
    for col in df.columns:
        x = df[col].to_numpy(float)
        m = np.average(x, weights=w)
        v = np.average((x - m) ** 2, weights=w)
        s = np.sqrt(v)

        # weighted quantiles
        idx = np.argsort(x)
        xs = x[idx]
        ws = w[idx]
        cdf = np.cumsum(ws)
        cdf /= cdf[-1] if cdf[-1] > 0 else 1.0

        def wq(p): return np.interp(p, cdf, xs)

        out.append({
            "parameter": col,
            "mean": m,
            "sd": s,
            "q2.5": wq(0.025),
            "q50":  wq(0.5),
            "q97.5": wq(0.975),
        })
    return pd.DataFrame(out)


# ============================================================
# Plotters: posterior overlays & SMC diagnostics
# ============================================================

def plot_marginals_overlay(pro_df: pd.DataFrame, pro_w: np.ndarray,
                           inv_df: pd.DataFrame, inv_w: np.ndarray,
                           out_path: Path, max_cols: int = 8):
    """Overlay 1D marginals for parameters common to both runs."""
    pro_df = pro_df.select_dtypes(include=[np.number])
    inv_df = inv_df.select_dtypes(include=[np.number])

    common = [c for c in pro_df.columns if c in inv_df.columns]
    if not common:
        print("No common parameters; skipping marginal overlay.")
        return
    if len(common) > max_cols:
        print(f"Many parameters ({len(common)}). Limiting to first {max_cols}.")
        common = common[:max_cols]

    n = len(common)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    plt.figure(figsize=(4*ncols, 3*nrows))
    for i, p in enumerate(common, 1):
        ax = plt.subplot(nrows, ncols, i)
        xp, xi = pro_df[p].to_numpy(float), inv_df[p].to_numpy(float)
        if HAS_SNS:
            sns.kdeplot(x=xp, weights=pro_w, fill=False, lw=2, color="tab:blue", label="PRO", ax=ax)
            sns.kdeplot(x=xi, weights=inv_w, fill=False, lw=2, color="tab:red",  label="INV", ax=ax)
        else:
            ax.hist(xp, weights=pro_w, bins=30, density=True, histtype="step",
                    color="tab:blue", label="PRO")
            ax.hist(xi, weights=inv_w, bins=30, density=True, histtype="step",
                    color="tab:red",  label="INV")
        ax.set_title(p)
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[saved] {out_path}")


def _weighted_resample(df: pd.DataFrame, w: np.ndarray, n: int = 4000, rng=None) -> pd.DataFrame:
    rng = np.random.default_rng(None if rng is None else rng)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=True, p=w/w.sum())
    return df.iloc[idx].reset_index(drop=True)


def plot_pair_overlay(pro_df: pd.DataFrame, pro_w: np.ndarray,
                      inv_df: pd.DataFrame, inv_w: np.ndarray,
                      out_path: Path, max_cols: int = 6, draws: int = 3000):
    """Overlay 2D pairs using weighted resampling."""
    pro_df = pro_df.select_dtypes(include=[np.number])
    inv_df = inv_df.select_dtypes(include=[np.number])

    common = [c for c in pro_df.columns if c in inv_df.columns]
    if not common:
        print("No common parameters; skipping pair overlay.")
        return
    if len(common) > max_cols:
        print(f"Many parameters ({len(common)}). Limiting to first {max_cols}.")
        common = common[:max_cols]

    pro_draws = _weighted_resample(pro_df[common], pro_w, n=draws)
    inv_draws = _weighted_resample(inv_df[common], inv_w, n=draws)

    if HAS_SNS:
        pro_draws["__group__"] = "PRO"
        inv_draws["__group__"] = "INV"
        both = pd.concat([pro_draws, inv_draws], axis=0, ignore_index=True)
        g = sns.pairplot(both, vars=common, hue="__group__", corner=True,
                         plot_kws=dict(alpha=0.35, s=10, linewidth=0),
                         diag_kws=dict(common_norm=False))
        g.fig.suptitle("Pairwise posterior overlay (resampled)", y=1.02)
        g.fig.set_size_inches(2.6*len(common), 2.6*len(common))
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
    else:
        # Minimal fallback
        plt.figure(figsize=(12, 4))
        pairs = [(common[i], common[i+1]) for i in range(0, min(len(common)-1, 3))]
        for j, (xv, yv) in enumerate(pairs, 1):
            ax = plt.subplot(1, len(pairs), j)
            ax.scatter(pro_draws[xv], pro_draws[yv], s=8, alpha=0.35, color="tab:blue", label="PRO")
            ax.scatter(inv_draws[xv], inv_draws[yv], s=8, alpha=0.35, color="tab:red",  label="INV")
            ax.set_xlabel(xv); ax.set_ylabel(yv); ax.grid(True, alpha=0.3)
            if j == 1: ax.legend()
        plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()
    print(f"[saved] {out_path}")


def compute_smc_diagnostics(history: History) -> pd.DataFrame:
    """
    Per-population diagnostics using documented accessors:
      - get_distribution(t)  for weights (ESS) & n_particles
      - get_weighted_distances(t) for distances (min/median/mean)
      - get_all_populations() for epsilon
    """
    # epsilon series from all populations
    pops = history.get_all_populations()  # columns include: t, epsilon, samples, ...
    epsl = pops.set_index("t")["epsilon"] if "epsilon" in pops.columns else None

    rows = []
    t_max = int(history.max_t)
    for t in range(0, t_max + 1):
        # df, w for population t
        df_t, w_t = history.get_distribution(t=t)  # df, weights
        w_t = np.asarray(w_t, dtype=float)
        w_t = w_t / w_t.sum() if w_t.sum() > 0 else w_t
        n_particles = len(df_t)
        ess = 1.0 / np.sum(np.square(w_t)) if w_t.size else np.nan

        # distances for population t
        try:
            ddf = history.get_weighted_distances(t=t)  # columns: w, distance
            d = ddf["distance"].to_numpy(float)
            dist_min = float(np.min(d)) if d.size else np.nan
            dist_med = float(np.median(d)) if d.size else np.nan
            dist_mean = float(np.mean(d)) if d.size else np.nan
        except Exception:
            dist_min = dist_med = dist_mean = np.nan

        rows.append({
            "t": t,
            "n_particles": n_particles,
            "ess": ess,
            "dist_min": dist_min,
            "dist_med": dist_med,
            "dist_mean": dist_mean,
            "epsilon": float(epsl.loc[t]) if epsl is not None and t in epsl.index else np.nan,
        })
    return pd.DataFrame(rows)


def overlay_lines(df_pro: pd.DataFrame, df_inv: pd.DataFrame,
                  y: str, ylabel: str, out_path: Path, title: str):
    plt.figure(figsize=(8, 5))
    plt.plot(df_pro["t"], df_pro[y], "-o", label="PRO", color="tab:blue")
    plt.plot(df_inv["t"], df_inv[y], "-o", label="INV", color="tab:red")
    plt.xlabel("Population (t)")
    plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()
    print(f"[saved] {out_path}")


def print_key_differences(summary_pro: pd.DataFrame, summary_inv: pd.DataFrame):
    dfp = summary_pro.set_index("parameter")
    dfi = summary_inv.set_index("parameter")
    common = dfp.index.intersection(dfi.index)
    if len(common) == 0:
        print("No common parameters to compare in summaries.")
        return
    comp = []
    for p in common:
        comp.append({
            "parameter": p,
            "PRO_q50": dfp.loc[p, "q50"],
            "INV_q50": dfi.loc[p, "q50"],
            "abs_diff": abs(dfp.loc[p, "q50"] - dfi.loc[p, "q50"]),
            "PRO_q2.5": dfp.loc[p, "q2.5"], "PRO_q97.5": dfp.loc[p, "q97.5"],
            "INV_q2.5": dfi.loc[p, "q2.5"], "INV_q97.5": dfi.loc[p, "q97.5"],
        })
    comp_df = pd.DataFrame(comp).sort_values("abs_diff", ascending=False)
    print("\n=== Posterior median differences (largest first) ===")
    with pd.option_context("display.float_format", "{:,.4g}".format):
        print(comp_df.to_string(index=False))


# ============================================================
# NND helpers (ABM summaries)
# ============================================================

def compute_full_nnd_list(positions: np.ndarray, width: float, height: float, torus: bool) -> np.ndarray:
    """Return one NND per agent (cluster) at a timestep."""
    from scipy.spatial import cKDTree
    N = len(positions)
    if N < 2:
        return np.array([])
    if torus:
        tiles = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tiles.append(positions + np.array([dx * width, dy * height]))
        tiled_positions = np.vstack(tiles)
        tree = cKDTree(tiled_positions)
        dists, _ = tree.query(positions, k=2)
        return dists[:, 1]
    else:
        tree = cKDTree(positions)
        dists, _ = tree.query(positions, k=2)
        return dists[:, 1]


# ============================================================
# PPC: parameter mapping & ABM simulation
# ============================================================

def deepcopy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    import copy
    return copy.deepcopy(d)


def set_param_by_path(params: Dict[str, Any], path: str, value: float,
                      sep: str = ".") -> bool:
    """
    Set nested key in DEFAULTS-like dict using a dotted path, e.g.
      "phenotypes.proliferative.speed_base" = 1.2
    Returns True if set, False if path not found.
    """
    keys = path.split(sep)
    cur = params
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            return False
        cur = cur[k]
    last = keys[-1]
    if last in cur:
        cur[last] = float(value)
        return True
    return False


def apply_param_map(params: Dict[str, Any], sample_row: pd.Series,
                    param_prefix: Optional[str] = None,
                    param_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Given a posterior sample (Series), write values into a copy of params.
    Two mechanisms:
      * If param_map provided: map from DB name -> dotted path in params.
      * Else, if param_prefix is provided ('phenotypes.proliferative.'), prepend.
      * Else, attempt to set with name as dotted path directly.
    """
    p = deepcopy_dict(params)
    for name, val in sample_row.items():
        if not np.issubdtype(type(val), np.number):
            continue
        path = None
        if param_map and name in param_map:
            path = param_map[name]
        elif param_prefix is not None:
            path = f"{param_prefix}{name}"
        else:
            path = name
        ok = set_param_by_path(p, path, float(val))
        if not ok and "__" in path:
            set_param_by_path(p, path.replace("__", "."), float(val))
    return p


def run_abm_once(params: Dict[str, Any], phenotype: str, steps: int,
                 seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Run a single ABM simulation and return summary artefacts needed for PPC.
    Uses toroidal nearest neighbour distances (cluster centres).
    """
    # Ensure init phenotype & time horizon
    p = deepcopy_dict(params)
    p.setdefault("init", {})
    p["init"]["phenotype"] = phenotype
    p.setdefault("time", deepcopy_dict(DEFAULTS["time"]))
    p["time"]["steps"] = steps

    # Instantiate and run
    model = ClustersModel(params=p, seed=int(seed) if seed is not None else 42)
    width  = p["space"]["width"]
    height = p["space"]["height"]

    # collect time-series of toroidal median NND
    nnd_med_ts = []
    for _ in range(steps):
        model.step()
        pos = model.pos_log[-1]
        nnd_list = compute_full_nnd_list(pos, width, height, torus=True)
        nnd_med_ts.append(np.median(nnd_list) if len(nnd_list) else np.nan)

    # final-time summaries
    final_pos = model.pos_log[-1]
    nnd_list_final = compute_full_nnd_list(final_pos, width, height, torus=True)
    summary = {
        "nnd_med_ts": np.array(nnd_med_ts, dtype=float),
        "nnd_med_final": float(np.median(nnd_list_final)) if len(nnd_list_final) else np.nan,
        "nnd_p10_final": float(np.percentile(nnd_list_final, 10)) if len(nnd_list_final) else np.nan,
        "nnd_p90_final": float(np.percentile(nnd_list_final, 90)) if len(nnd_list_final) else np.nan,
        "cluster_count_final": int(len(final_pos)),
        "mean_size_final": float(np.mean(model.size_log[-1])) if len(model.size_log[-1]) else np.nan,
    }
    return summary


def posterior_predictive(
    params_base: Dict[str, Any],
    posterior_df: pd.DataFrame, posterior_w: np.ndarray,
    phenotype: str, draws: int, steps: int,
    results_dir: Path,
    label: str,
    param_prefix: Optional[str] = None,
    param_map: Optional[Dict[str, str]] = None,
    seed: Optional[int] = 12345
) -> Dict[str, Any]:
    """
    Run PPC by drawing 'draws' weighted samples from posterior, sim ABM, compute summaries.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(posterior_df), size=draws, replace=True, p=posterior_w/posterior_w.sum())
    draws_df = posterior_df.iloc[idx].reset_index(drop=True)

    nnd_ts_mat = np.zeros((draws, steps), dtype=float)
    nnd_med_final = np.zeros(draws, dtype=float)
    nnd_p10_final = np.zeros(draws, dtype=float)
    nnd_p90_final = np.zeros(draws, dtype=float)
    clus_final    = np.zeros(draws, dtype=float)
    msize_final   = np.zeros(draws, dtype=float)

    for i in range(draws):
        overrides = apply_param_map(params_base, draws_df.iloc[i, :],
                                    param_prefix=param_prefix, param_map=param_map)
        sim = run_abm_once(overrides, phenotype=phenotype, steps=steps,
                           seed=None if seed is None else int(seed + i))
        nnd_ts_mat[i, :]   = sim["nnd_med_ts"]
        nnd_med_final[i]   = sim["nnd_med_final"]
        nnd_p10_final[i]   = sim["nnd_p10_final"]
        nnd_p90_final[i]   = sim["nnd_p90_final"]
        clus_final[i]      = sim["cluster_count_final"]
        msize_final[i]     = sim["mean_size_final"]

    # Save csv
    df_out = pd.DataFrame({
        "nnd_med_final": nnd_med_final,
        "nnd_p10_final": nnd_p10_final,
        "nnd_p90_final": nnd_p90_final,
        "cluster_count_final": clus_final,
        "mean_size_final": msize_final,
    })
    out_csv = results_dir / f"ppc_summaries_{label}.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    # Plot time-series predictive band
    t = np.arange(steps)
    mu  = np.nanmedian(nnd_ts_mat, axis=0)
    lo  = np.nanpercentile(nnd_ts_mat, 2.5, axis=0)
    hi  = np.nanpercentile(nnd_ts_mat, 97.5, axis=0)

    plt.figure(figsize=(10,6))
    plt.title(f"PPC — median NND (toroidal) time-series — {label}")
    color = "tab:blue" if label == "PRO" else "tab:red"
    plt.fill_between(t, lo, hi, color=color, alpha=0.20, label="95% predictive band")
    plt.plot(t, mu, color=color, lw=2, label="Predictive median")
    plt.xlabel("Timestep"); plt.ylabel("Median NND (toroidal)")
    plt.grid(True, alpha=0.3); plt.legend()
    fig_path = results_dir / f"ppc_nnd_ts_{label}.png"
    plt.tight_layout(); plt.savefig(fig_path, dpi=180); plt.close()
    print(f"[saved] {fig_path}")

    return {
        "nnd_ts": nnd_ts_mat,
        "nnd_med_final": nnd_med_final,
        "nnd_p10_final": nnd_p10_final,
        "nnd_p90_final": nnd_p90_final,
        "cluster_count_final": clus_final,
        "mean_size_final": msize_final,
    }


# ============================================================
# Main driver
# ============================================================

def main(
    pro_db: str = "abc_maxabs_PRO_test_1_no_gr_500_8.db",
    inv_db: str = "abc_maxabs_INV_test_1_no_gr_500_8.db",
    results_dir: str = "results",
    # PPC knobs
    ppc_draws: int = 50,
    ppc_steps: int = None,
    # Parameter mapping helpers (optional)
    param_prefix_pro: Optional[str] = "phenotypes.proliferative.",
    param_prefix_inv: Optional[str] = "phenotypes.invasive.",
    param_map_json: Optional[str] = None,
):
    results_dir_path = resolve_results_dir(results_dir)
    pro_path = results_dir_path / pro_db
    inv_path = results_dir_path / inv_db

    print(f"Using results dir: {results_dir_path}")
    print(f"Looking for PRO DB: {pro_path.name}")
    print(f"Looking for INV DB: {inv_path.name}")
    ensure_db_exists(pro_path, "PRO")
    ensure_db_exists(inv_path, "INV")

    # Load histories
    print("Loading pyABC histories...")
    H_pro = load_history(pro_path)
    H_inv = load_history(inv_path)

    # Final posteriors (params DF + weights)
    print("Extracting final posteriors...")
    pro_df, pro_w, _ = get_final_posterior(H_pro)
    inv_df, inv_w, _ = get_final_posterior(H_inv)

    # Weighted summaries to CSV
    print("Summarising posteriors...")
    summ_pro = weighted_summary(pro_df, pro_w)
    summ_inv = weighted_summary(inv_df, inv_w)
    (results_dir_path / "posterior_summary_PRO.csv").write_text(summ_pro.to_csv(index=False))
    (results_dir_path / "posterior_summary_INV.csv").write_text(summ_inv.to_csv(index=False))
    print(f"[saved] {results_dir_path/'posterior_summary_PRO.csv'}")
    print(f"[saved] {results_dir_path/'posterior_summary_INV.csv'}")

    # Posterior overlays
    plot_marginals_overlay(
        pro_df, pro_w, inv_df, inv_w,
        out_path=results_dir_path / "posterior_marginals_overlay.png"
    )
    plot_pair_overlay(
        pro_df, pro_w, inv_df, inv_w,
        out_path=results_dir_path / "posterior_pair_overlay.png"
    )

    # SMC diagnostics (ESS, distances, epsilon)
    print("Computing SMC diagnostics over populations...")
    diag_pro = compute_smc_diagnostics(H_pro)
    diag_inv = compute_smc_diagnostics(H_inv)
    overlay_lines(diag_pro, diag_inv, "ess", "Effective Sample Size",
                  results_dir_path / "ess_over_time.png", "ESS over SMC populations")
    overlay_lines(diag_pro, diag_inv, "dist_med", "Median distance",
                  results_dir_path / "distance_over_time.png", "ABC distance over SMC populations")
    if (not diag_pro["epsilon"].isna().all()) or (not diag_inv["epsilon"].isna().all()):
        overlay_lines(diag_pro, diag_inv, "epsilon", "Epsilon",
                      results_dir_path / "epsilon_over_time.png", "Epsilon schedule over SMC populations")

    # -----------------------------
    # Posterior Predictive Checks
    # -----------------------------
    param_map = None
    if param_map_json:
        p = Path(param_map_json)
        if not p.is_absolute():
            p = (SCRIPT_DIR.parent / p).resolve()
        if p.exists():
            param_map = pd.read_json(p, typ="series").to_dict()
            print(f"Loaded param map with {len(param_map)} entries from: {p}")

    base_params = deepcopy_dict(DEFAULTS)
    steps_ppc = ppc_steps if ppc_steps is not None else int(DEFAULTS["time"]["steps"])

    print(f"Running PPC for PRO (draws={ppc_draws}, steps={steps_ppc}) ...")
    ppc_pro = posterior_predictive(
        base_params, pro_df, pro_w, phenotype="proliferative",
        draws=ppc_draws, steps=steps_ppc, results_dir=results_dir_path, label="PRO",
        param_prefix=param_prefix_pro, param_map=param_map
    )

    print(f"Running PPC for INV (draws={ppc_draws}, steps={steps_ppc}) ...")
    ppc_inv = posterior_predictive(
        base_params, inv_df, inv_w, phenotype="invasive",
        draws=ppc_draws, steps=steps_ppc, results_dir=results_dir_path, label="INV",
        param_prefix=param_prefix_inv, param_map=param_map
    )

    # Final-time predictive distributions overlay (violin or KDE)
    plt.figure(figsize=(10,6))
    plt.title("Posterior Predictive — final-time summaries (PRO vs INV)")

    labels = ["nnd_med_final", "cluster_count_final", "mean_size_final"]
    colours = {"PRO":"tab:blue", "INV":"tab:red"}

    def small_violin(ax, data_pro, data_inv, name):
        if HAS_SNS:
            d = pd.DataFrame({
                name: np.concatenate([data_pro, data_inv]),
                "Group": (["PRO"]*len(data_pro)) + (["INV"]*len(data_inv))
            })
            sns.violinplot(data=d, x="Group", y=name,
                           palette=[colours["PRO"], colours["INV"]],
                           inner="box", cut=0, ax=ax)
            ax.set_xlabel(""); ax.set_ylabel(name)
        else:
            ax.hist(data_pro, bins=30, alpha=0.5, color=colours["PRO"], density=True, label="PRO")
            ax.hist(data_inv, bins=30, alpha=0.5, color=colours["INV"], density=True, label="INV")
            ax.set_xlabel(name); ax.legend()

    for i, metric in enumerate(labels, 1):
        ax = plt.subplot(1, len(labels), i)
        small_violin(ax, ppc_pro[metric], ppc_inv[metric], metric)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_ppc_final = results_dir_path / "ppc_final_summaries.png"
    plt.savefig(out_ppc_final, dpi=180); plt.close()
    print(f"[saved] {out_ppc_final}")

    # Console differences
    print_key_differences(summ_pro, summ_inv)
    print("\nDone. All outputs written to:", results_dir_path)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Compare ABC–SMC posteriors for PRO vs INV and run posterior predictive checks."
    )
    ap.add_argument("--pro", type=str, default="abc_maxabs_PRO_test_1_no_gr_500_8.db",
                    help="PRO .db filename (relative to --out)")
    ap.add_argument("--inv", type=str, default="abc_maxabs_INV_test_1_no_gr_500_8.db",
                    help="INV .db filename (relative to --out)")
    ap.add_argument("--out", type=str, default="results",
                    help="Output dir. Relative paths are resolved wrt project root (script parent).")

    # PPC settings
    ap.add_argument("--ppc-draws", type=int, default=50,
                    help="Number of posterior draws per phenotype for PPC (simulation count).")
    ap.add_argument("--ppc-steps", type=int, default=None,
                    help="Override number of ABM steps in PPC; default uses DEFAULTS['time']['steps'].")

    # Parameter mapping helpers
    ap.add_argument("--param-prefix-pro", type=str, default="phenotypes.proliferative.",
                    help="Prefix prepended to PRO DB param names when writing into DEFAULTS.")
    ap.add_argument("--param-prefix-inv", type=str, default="phenotypes.invasive.",
                    help="Prefix prepended to INV DB param names when writing into DEFAULTS.")
    ap.add_argument("--param-map", type=str, default=None,
                    help="JSON file mapping DB parameter names -> dotted paths in DEFAULTS.")

    args = ap.parse_args()
    main(
        pro_db=args.pro, inv_db=args.inv, results_dir=args.out,
        ppc_draws=args.ppc_draws, ppc_steps=args.ppc_steps,
        param_prefix_pro=args.param_prefix_pro, param_prefix_inv=args.param_prefix_inv,
        param_map_json=args.param_map
    )