#!/usr/bin/env python3
"""
make_1d_coagulation_plots.py
----------------------------
Compute and plot **pairwise coagulation probability vs initial distance** for
**1-parameter sweeps**, then fit decaying models to each parameter's curve.

Definitions
-----------
Two initial clusters i and j "coagulate" if their **lineages eventually merge**
(i.e., through any chain of merges both belong to the same final lineage).
We estimate P(coag | initial distance = d) by:
  - building lineages per repeat from **merge events** (union-find over time)
  - forming all (or a subsample of) initial pairs (i,j)
  - binning their initial separations d0
  - computing n_merged/n_total in each distance bin
  - aggregating across repeats for each parameter value

Inputs (per sweep run)
----------------------
Expected sweep layout created by your runner:
  Parameter_sweeps_results/<RUN>/
    simulations/<condition_id>/repeat_XX/   # requires abm.save_raw=true
      t_0000.csv, t_0001.csv, ...
      merge_events.csv       # preferred: one CSV per repeat (see spec below)
    summaries/<condition_id>/summary_repeat_XX.csv  # not used here

**Merge events** CSV (one per repeat) must contain at least:
  - t         : integer step at which the merge occurred
  - id_a,id_b : the two cluster ids that merged at step t
  - id_new    : (optional) resulting cluster id after merge

If you enable the post-step 'coagulation' in your package entry point
(Parameter_sweeps/__main__.py) it can generate these files automatically,
provided abm.save_raw=true.  See your __main__.py for the hook. [ref]  # noqa

Outputs (per RUN)
-----------------
<run>/coagulation_1d/
  curves/
    coag_prob_by_distance__param=<value>.csv
  combined/
    coag_prob_by_distance_ALL.csv
    coag_vs_distance_ALL.png
    coag_vs_distance_ALL_logy.png
  fits/
    fit_report.csv            # one row per parameter, best-model summary
    <param_value>__fit.png    # data + best fit overlay
    <param_value>__fit_logy.png

Usage examples
--------------
# Latest run under default results root, auto-detect swept parameter
python Parameter_sweeps/make_1d_coagulation_plots.py --run latest

# Explicit run + param key and a tighter min-prob threshold for fitting
python Parameter_sweeps/make_1d_coagulation_plots.py \
  --run 2026-03-30_1540 --param-key merge.p_merge --pmin 0.03

Notes
-----
- This script **requires** merge event files; it will fail fast with
  instructions if missing.  Your sweep can produce them by enabling the
  coagulation post-step referenced in Parameter_sweeps/__main__.py. [ref]  # noqa
- Raw timestep CSVs exist only if abm.save_raw=true in your YAML. [ref]    # noqa
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# Parse condition ids (robust: split at last underscore; support '0p1' -> 0.1)
# -----------------------------
IGNORE_PARAMS = {"scenario"}

import re
_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?([eE]-?\d+)?$")
_P_DECIMAL_RE = re.compile(r"^-?\d+p\d+([eE]-?\d+)?$")


def _parse_value(token: str):
    t = token.strip()
    if _P_DECIMAL_RE.fullmatch(t):
        t = t.replace("p", ".")
    if _NUMERIC_RE.fullmatch(t):
        try:
            return float(t)
        except ValueError:
            return token
    return token


def parse_condition_id(condition_id: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    parts = condition_id.split("__")
    for part in parts:
        if "_" not in part:
            if part not in IGNORE_PARAMS:
                out[part] = part
            continue
        key, rest = part.rsplit("_", 1)  # <-- split at LAST underscore
        if key in IGNORE_PARAMS:
            continue
        out[key] = _parse_value(rest)
    return out


def find_run_dir(results_root: Path, run: Optional[str]) -> Path:
    if run in (None, "latest", "LAST", "Latest"):
        runs = sorted([p for p in results_root.iterdir() if p.is_dir()])
        if not runs:
            raise FileNotFoundError(f"No runs found under {results_root}")
        return runs[-1]
    rd = results_root / run
    if not rd.exists():
        raise FileNotFoundError(f"Run folder not found: {rd}")
    return rd


# -----------------------------
# Distance helpers (optionally toroidal)
# -----------------------------
def torus_delta(dx: np.ndarray, L: float) -> np.ndarray:
    return (dx + L / 2.0) % L - L / 2.0


def pairwise_min_image_dist(pos0: np.ndarray, W: float, H: float, torus: bool) -> np.ndarray:
    """
    Return condensed vector of pairwise distances between initial points (N x 2).
    """
    N = pos0.shape[0]
    dx = pos0[:, None, 0] - pos0[None, :, 0]
    dy = pos0[:, None, 1] - pos0[None, :, 1]
    if torus:
        dx = torus_delta(dx, W)
        dy = torus_delta(dy, H)
    d = np.sqrt(dx * dx + dy * dy)
    iu = np.triu_indices(N, k=1)
    return d[iu], iu


# -----------------------------
# Merge events → union-find over lineages
# -----------------------------
class DSU:
    def __init__(self):
        self.parent = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x):
        # path compression
        px = self.parent.get(x, x)
        if px != x:
            self.parent[x] = self.find(px)
        return self.parent.get(x, x)

    def union(self, a, b, new_id: Optional[int] = None):
        # Merge sets of a and b; optionally redirect root to new_id
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            root = ra
        else:
            # attach rb under ra
            self.parent[rb] = ra
            root = ra
        if new_id is not None:
            # relabel root to new_id by ensuring new_id points to itself, then map others
            self.add(new_id)
            rroot = self.find(root)
            self.parent[rroot] = new_id
            self.parent[new_id] = new_id
            root = new_id
        return root


def load_merge_events(rep_dir: Path) -> pd.DataFrame:
    """
    Try common file names for merge events, return DataFrame with columns:
      t, id_a, id_b, id_new (id_new optional)
    """
    candidates = [
        rep_dir / "merge_events.csv",
        rep_dir / "merges.csv",
        rep_dir / "coagulation_events.csv",
    ]
    for f in candidates:
        if f.exists():
            df = pd.read_csv(f)
            # basic schema checks
            cols = {c.lower(): c for c in df.columns}
            required = {"t", "id_a", "id_b"}
            if not required.issubset(set([c.lower() for c in df.columns])):
                continue
            # normalize column names
            out = pd.DataFrame()
            out["t"] = df[cols["t"]]
            out["id_a"] = df[cols["id_a"]]
            out["id_b"] = df[cols["id_b"]]
            out["id_new"] = df[cols["id_new"]] if "id_new" in cols else np.nan
            # sort by time
            return out.sort_values("t").reset_index(drop=True)
    raise FileNotFoundError(
        f"Merge events not found in {rep_dir}. "
        f"Enable your coagulation post-step to generate them, "
        f"and ensure abm.save_raw=true in the YAML. "
        f"(See Parameter_sweeps/__main__.py post hooks.)"
    )


def build_lineages_from_events(initial_ids: np.ndarray, events: pd.DataFrame) -> DSU:
    """
    Construct lineage DSU over the universe of ids that appear in the events + initial ids.
    Events should be chronological.
    """
    dsu = DSU()
    for id0 in initial_ids:
        dsu.add(int(id0))
    for _, row in events.iterrows():
        a, b = int(row["id_a"]), int(row["id_b"])
        id_new = row.get("id_new")
        id_new = int(id_new) if pd.notna(id_new) else None
        dsu.add(a)
        dsu.add(b)
        if id_new is not None:
            dsu.add(id_new)
        dsu.union(a, b, new_id=id_new)
    return dsu


# -----------------------------
# Binning & aggregation
# -----------------------------
@dataclass
class BinningConfig:
    nbins: int = 25
    max_pairs: Optional[int] = None  # subsample per repeat to cap O(N^2)
    pmin: float = 0.02               # min prob threshold for model fitting common range
    min_pairs_per_bin: int = 50      # ensure statistical reliability
    torus: bool = True
    W: Optional[float] = None        # domain width  (needed for torus)
    H: Optional[float] = None        # domain height (needed for torus)


def compute_pairwise_stats_for_repeat(
    rep_dir: Path,
    nbins: int,
    torus: bool,
    W: Optional[float],
    H: Optional[float],
    max_pairs: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bin_edges, merged_mask_counts) where:
       - bin_edges: length nbins+1
       - merged_mask_counts: tuple of two arrays (merged_counts, total_counts) per bin
    """
    # Initial snapshot (t_0000.csv or smallest t_*.csv)
    ts = sorted(rep_dir.glob("t_*.csv"))
    if not ts:
        raise FileNotFoundError(f"No timestep CSVs in {rep_dir}. Ensure abm.save_raw=true in YAML.")
    t0 = min(ts, key=lambda p: int(re.search(r"t_(\d+)", p.stem).group(1)))
    df0 = pd.read_csv(t0)
    if not {"id", "x", "y"}.issubset(df0.columns):
        raise ValueError(f"{t0} must contain columns: id,x,y")

    ids0 = df0["id"].to_numpy(int)
    pos0 = df0[["x", "y"]].to_numpy(float)

    # Distance matrix (minimum-image if torus)
    if torus:
        if W is None or H is None:
            raise ValueError("--torus requires --W and --H to compute minimum-image distances.")
        dists, iu = pairwise_min_image_dist(pos0, float(W), float(H), True)
        pairs = np.vstack(iu).T  # shape (M,2) index pairs into ids0
    else:
        N = len(ids0)
        dx = pos0[:, None, 0] - pos0[None, :, 0]
        dy = pos0[:, None, 1] - pos0[None, :, 1]
        d = np.sqrt(dx * dx + dy * dy)
        iu = np.triu_indices(N, k=1)
        dists = d[iu]
        pairs = np.vstack(iu).T

    # (Optional) Subsample pairs to limit O(N^2)
    if max_pairs is not None and len(pairs) > max_pairs:
        rng = np.random.default_rng(12345)
        sel = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = pairs[sel]
        dists = dists[sel]

    # Load merge events & build lineages
    ev = load_merge_events(rep_dir)
    dsu = build_lineages_from_events(ids0, ev)

    # Did pair (i,j) end up in same lineage?
    merged_mask = np.fromiter(
        (dsu.find(int(ids0[i])) == dsu.find(int(ids0[j])) for i, j in pairs),
        dtype=bool,
        count=len(pairs),
    )

    # Bin distances
    dmin, dmax = float(np.nanmin(dists)), float(np.nanmax(dists))
    if dmax <= dmin:
        dmax = dmin + 1.0
    edges = np.linspace(dmin, dmax, nbins + 1)
    idx = np.clip(np.digitize(dists, edges) - 1, 0, nbins - 1)

    merged_counts = np.bincount(idx, weights=merged_mask.astype(int), minlength=nbins)
    total_counts = np.bincount(idx, minlength=nbins)

    return edges, (merged_counts.astype(int), total_counts.astype(int))


# -----------------------------
# Models & fitting
# -----------------------------
def model_exp(x, A, B, C):            # A * exp(-B x) + C
    return A * np.exp(-B * x) + C


def model_strexp(x, A, lam, k, C):    # A * exp(-(x/lam)^k) + C
    return A * np.exp(-np.power(np.maximum(x, 1e-12) / lam, k)) + C


def model_power(x, A, x0, k, C):      # A * (x + x0)^(-k) + C
    return A * np.power(np.maximum(x + x0, 1e-9), -k) + C


def fit_one_curve(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """
    Try several decays; return best by AIC with parameters & scores.
    """
    results = []

    def _eval(model_name, func, p0, bounds):
        try:
            popt, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=20000)
            yhat = func(x, *popt)
            resid = y - yhat
            sse = float(np.sum(resid ** 2))
            n = len(y)
            k = len(popt)
            aic = n * np.log(sse / max(n, 1)) + 2 * k if sse > 0 and n > k else np.inf
            bic = n * np.log(sse / max(n, 1)) + k * np.log(max(n, 1)) if sse > 0 and n > k else np.inf
            # pseudo R^2
            sst = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1 - sse / sst if sst > 0 else np.nan
            results.append(
                dict(model=model_name, params=popt.tolist(), sse=sse, aic=aic, bic=bic, r2=r2)
            )
        except Exception:
            pass

    # Initial guesses based on crude stats
    y0 = float(y[0]) if len(y) else 1.0
    _eval("exp", model_exp, p0=[y0, 0.01, 0.0], bounds=([0.0, 0.0, 0.0], [1.5, 10.0, 0.5]))
    _eval("strexp", model_strexp, p0=[y0, max(1.0, np.median(x)), 1.0, 0.0],
          bounds=([0.0, 1e-3, 0.2, 0.0], [2.0, 1e4, 4.0, 0.5]))
    _eval("power", model_power, p0=[y0, 1.0, 1.0, 0.0],
          bounds=([0.0, 0.0, 0.2, 0.0], [2.0, 1e3, 5.0, 0.5]))

    if not results:
        return dict(model="none", params=[], sse=np.inf, aic=np.inf, bic=np.inf, r2=np.nan)
    # Best by AIC
    best = min(results, key=lambda r: r["aic"])
    return best


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="1D sweep coagulation analysis: P(merge) vs initial distance")
    ap.add_argument("--results-root", type=str, default=None,
                    help="Path to Parameter_sweeps_results (default: sibling of this file).")
    ap.add_argument("--run", type=str, default="latest", help="'latest' or a specific run folder")
    ap.add_argument("--param-key", type=str, default=None,
                    help="Override auto-detected x-axis (swept) parameter")
    ap.add_argument("--nbins", type=int, default=25, help="Number of initial-distance bins")
    ap.add_argument("--max-pairs", type=int, default=None, help="Cap pairs per repeat to this many")
    ap.add_argument("--torus", action="store_true", help="Use minimum-image distances (requires --W and --H)")
    ap.add_argument("--W", type=float, default=None, help="Domain width (torus)")
    ap.add_argument("--H", type=float, default=None, help="Domain height (torus)")
    ap.add_argument("--pmin", type=float, default=0.02,
                    help="Min probability; fit only on distances where **all** params exceed this")
    ap.add_argument("--min-pairs-per-bin", type=int, default=50,
                    help="Require at least this many pairs per bin (after aggregating repeats)")
    ap.add_argument("--dpi", type=int, default=260)
    args = ap.parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    default_results_root = script_dir.parent / "Parameter_sweeps_results"
    results_root = Path(args.results_root) if args.results_root else default_results_root
    run_dir = find_run_dir(results_root, args.run)

    sim_root = run_dir / "simulations"
    if not sim_root.exists():
        raise FileNotFoundError(
            f"No simulations/ in {run_dir}. Raw snapshots and merge events are required. "
            f"Ensure abm.save_raw=true in YAML. [ref]"
        )

    # Find conditions (parameter values)
    cond_dirs = sorted([p for p in sim_root.iterdir() if p.is_dir()])
    if not cond_dirs:
        raise FileNotFoundError(f"No condition folders under {sim_root}")

    # Map: param_value -> list of (edges, merged_counts, total_counts) accumulated over repeats
    per_param_accum = {}
    param_values = []

    # Determine param key (auto-detect if not provided)
    sample_params = []
    for cdir in cond_dirs:
        sample_params.append(parse_condition_id(cdir.name))
    # candidate keys varying
    keys = sorted(set(k for d in sample_params for k in d.keys()))
    varying = [k for k in keys if len(set(d.get(k, None) for d in sample_params)) > 1]
    if args.param_key:
        xkey = args.param_key
        if xkey not in keys:
            raise ValueError(f"--param-key '{xkey}' not found among parsed keys: {keys}")
    else:
        if len(varying) != 1:
            raise ValueError(f"Expected one swept parameter, found {len(varying)}: {varying}. "
                             f"Use --param-key to choose one.")
        xkey = varying[0]

    print(f"Using swept parameter: {xkey}")

    for cdir in cond_dirs:
        params = parse_condition_id(cdir.name)
        pval = params.get(xkey, None)
        if pval is None:
            print(f"[WARN] {cdir.name}: missing {xkey}; skipping.")
            continue

        # Accumulate across repeats
        merged_acc = None
        total_acc = None
        edges_ref = None

        rep_dirs = sorted([p for p in cdir.iterdir() if p.is_dir() and p.name.startswith("repeat_")])
        if not rep_dirs:
            print(f"[WARN] {cdir.name}: no repeat_* folders; skipping.")
            continue

        for rdir in rep_dirs:
            try:
                edges, (m_counts, t_counts) = compute_pairwise_stats_for_repeat(
                    rdir, args.nbins, args.torus, args.W, args.H, args.max_pairs
                )
            except FileNotFoundError as e:
                print(f"[WARN] {rdir}: {e}")
                continue

            if edges_ref is None:
                edges_ref = edges
                merged_acc = m_counts.astype(float)
                total_acc = t_counts.astype(float)
            else:
                if len(edges) != len(edges_ref) or not np.allclose(edges, edges_ref):
                    # Re-bin to common edges if needed (rare)
                    # For simplicity, skip inconsistent repeats
                    print(f"[WARN] {rdir}: bin edges mismatch; skipping repeat.")
                    continue
                merged_acc += m_counts
                total_acc += t_counts

        if edges_ref is None:
            print(f"[WARN] {cdir.name}: no usable repeats.")
            continue

        prob = np.divide(merged_acc, np.maximum(total_acc, 1.0), where=(total_acc > 0))
        centers = 0.5 * (edges_ref[:-1] + edges_ref[1:])

        # Save per-parameter curve
        out_param_dir = run_dir / "coagulation_1d" / "curves"
        out_param_dir.mkdir(parents=True, exist_ok=True)
        df_curve = pd.DataFrame({
            "param_key": xkey,
            "param_value": pval,
            "d_left": edges_ref[:-1],
            "d_center": centers,
            "d_right": edges_ref[1:],
            "n_pairs": total_acc.astype(int),
            "n_merged": merged_acc.astype(int),
            "prob": prob
        })
        df_curve.to_csv(out_param_dir / f"coag_prob_by_distance__param={pval}.csv", index=False)

        per_param_accum[pval] = dict(edges=edges_ref, centers=centers, n=total_acc, m=merged_acc, p=prob)
        param_values.append(pval)

    if not per_param_accum:
        raise RuntimeError("No curves accumulated; check that merge_events.csv exist in repeats.")

    # Combine and save ALL curves
    comb_rows = []
    for pv in sorted(per_param_accum.keys(), key=lambda z: float(z) if isinstance(z, (int, float)) else z):
        d = per_param_accum[pv]
        for k in range(len(d["centers"])):
            comb_rows.append({
                "param_key": xkey,
                "param_value": pv,
                "d_center": float(d["centers"][k]),
                "n_pairs": int(d["n"][k]),
                "n_merged": int(d["m"][k]),
                "prob": float(d["p"][k]),
            })
    out_comb_dir = run_dir / "coagulation_1d" / "combined"
    out_comb_dir.mkdir(parents=True, exist_ok=True)
    df_all = pd.DataFrame(comb_rows)
    df_all.to_csv(out_comb_dir / "coag_prob_by_distance_ALL.csv", index=False)

    # Plots: ALL params (linear & logy)
    # color map
    uniq_params = sorted(per_param_accum.keys(), key=lambda z: float(z) if isinstance(z, (int, float)) else z)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(uniq_params)))

    def _plot_all(logy=False):
        fig, ax = plt.subplots(figsize=(10.5, 6.0))
        for pv, col in zip(uniq_params, colors):
            d = per_param_accum[pv]
            ax.plot(d["centers"], d["p"], marker="o", lw=2.0, color=col, label=f"{xkey}={pv}")
        ax.set_title(f"Pairwise coagulation vs initial distance (all {xkey})")
        ax.set_xlabel("Initial distance")
        ax.set_ylabel("P(pair lineage merge)")
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(out_comb_dir / ("coag_vs_distance_ALL_logy.png" if logy else "coag_vs_distance_ALL.png"), dpi=args.dpi)
        plt.close(fig)

    _plot_all(logy=False)
    _plot_all(logy=True)

    # Fitting: use only distances where **all** params meet thresholds
    centers_ref = None
    mask_common = None
    for pv in uniq_params:
        d = per_param_accum[pv]
        if centers_ref is None:
            centers_ref = d["centers"]
            mask_common = (d["n"] >= args.min_pairs_per_bin) & (d["p"] >= args.pmin)
        else:
            if not np.allclose(centers_ref, d["centers"]):
                raise RuntimeError("Distance centers inconsistent across parameters; cannot fit jointly.")
            mask_common &= (d["n"] >= args.min_pairs_per_bin) & (d["p"] >= args.pmin)

    xfit = centers_ref[mask_common]
    if len(xfit) < 4:
        print("[WARN] Not enough common bins above thresholds to fit models; skipping fits.")
        return

    fit_rows = []
    fit_dir = run_dir / "coagulation_1d" / "fits"
    fit_dir.mkdir(parents=True, exist_ok=True)

    for pv in uniq_params:
        d = per_param_accum[pv]
        yfit = d["p"][mask_common]
        # Guard: ensure positive y for log-like behaviors
        yfit = np.clip(yfit, 1e-9, 1.0)

        best = fit_one_curve(xfit, yfit)
        fit_rows.append({
            "param_key": xkey,
            "param_value": pv,
            "best_model": best["model"],
            "params": json.dumps(best["params"]),
            "sse": best["sse"],
            "aic": best["aic"],
            "bic": best["bic"],
            "r2": best["r2"],
            "n_points": int(len(xfit)),
            "pmin": float(args.pmin),
            "min_pairs_per_bin": int(args.min_pairs_per_bin),
        })

        # Plot per-parameter data + **best fit** on linear and logy axes
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot(d["centers"], d["p"], "o", color="C0", label="data")
        if best["model"] == "exp":
            yhat = model_exp(xfit, *best["params"])
            ax.plot(xfit, yhat, "-", color="C1", lw=2.2, label="exp fit")
        elif best["model"] == "strexp":
            yhat = model_strexp(xfit, *best["params"])
            ax.plot(xfit, yhat, "-", color="C2", lw=2.2, label="stretched-exp fit")
        elif best["model"] == "power":
            yhat = model_power(xfit, *best["params"])
            ax.plot(xfit, yhat, "-", color="C3", lw=2.2, label="power-law fit")
        ax.set_title(f"Coagulation vs distance; {xkey}={pv} (best={best['model']})")
        ax.set_xlabel("Initial distance")
        ax.set_ylabel("P(pair lineage merge)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fit_dir / f"{pv}__fit.png", dpi=args.dpi)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot(d["centers"], d["p"], "o", color="C0", label="data")
        if best["model"] == "exp":
            yhat = model_exp(xfit, *best["params"])
            ax.plot(xfit, yhat, "-", color="C1", lw=2.2, label="exp fit")
        elif best["model"] == "strexp":
            yhat = model_strexp(xfit, *best["params"])
            ax.plot(xfit, yhat, "-", color="C2", lw=2.2, label="stretched-exp fit")
        elif best["model"] == "power":
            yhat = model_power(xfit, *best["params"])
            ax.plot(xfit, yhat, "-", color="C3", lw=2.2, label="power-law fit")
        ax.set_yscale("log")
        ax.set_title(f"Coagulation vs distance (logy); {xkey}={pv} (best={best['model']})")
        ax.set_xlabel("Initial distance")
        ax.set_ylabel("P(pair lineage merge)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fit_dir / f"{pv}__fit_logy.png", dpi=args.dpi)
        plt.close(fig)

    pd.DataFrame(fit_rows).to_csv(fit_dir / "fit_report.csv", index=False)
    print(f"Saved curves to {run_dir/'coagulation_1d/curves'}")
    print(f"Saved combined plots/CSV to {out_comb_dir}")
    print(f"Saved fit plots/report to {fit_dir}")


if __name__ == "__main__":
    main()