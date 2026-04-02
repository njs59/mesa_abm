#!/usr/bin/env python3
"""
make_1d_coagulation_plots_infer.py  (Parallelised)
--------------------------------------------------
Infer pairwise lineage merges directly from raw timestep CSVs (no pre-logged events),
then compute and plot P(coagulation | initial distance) for 1-parameter sweeps,
with ONE global set of distance bins across the run and multi-process parallelism.

Outputs (under the selected run folder):
  coagulation_1d_infer/
    curves/   (per-parameter CSVs)
    combined/ (all-parameters CSV + plots)
    fits/     (per-parameter fit overlays + fit_report.csv)
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
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------
# Parse condition ids (LAST underscore split; support '0p1' -> 0.1)
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
        key, rest = part.rsplit("_", 1)  # LAST underscore
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
# Geometry helpers
# -----------------------------
def torus_delta(dx: np.ndarray, L: float) -> np.ndarray:
    return (dx + L / 2.0) % L - L / 2.0


def min_image_dist(p: np.ndarray, Q: np.ndarray, W: float, H: float) -> np.ndarray:
    dx = p[0] - Q[:, 0]
    dy = p[1] - Q[:, 1]
    dx = torus_delta(dx, W)
    dy = torus_delta(dy, H)
    return np.sqrt(dx * dx + dy * dy)


def pairwise_min_image_dist(pos0: np.ndarray, W: float, H: float) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    N = pos0.shape[0]
    dx = pos0[:, None, 0] - pos0[None, :, 0]
    dy = pos0[:, None, 1] - pos0[None, :, 1]
    dx = torus_delta(dx, W)
    dy = torus_delta(dy, H)
    d = np.sqrt(dx * dx + dy * dy)
    iu = np.triu_indices(N, k=1)
    return d[iu], iu


# -----------------------------
# DSU (union–find) for lineages
# -----------------------------
class DSU:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def add(self, x: int):
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: int) -> int:
        px = self.parent.get(x, x)
        if px != x:
            self.parent[x] = self.find(px)
        return self.parent.get(x, x)

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        self.parent[rb] = ra
        return ra


# -----------------------------
# Loading snapshots
# -----------------------------
SNAP_REQUIRED = ("id", "x", "y", "size")
T_PATTERN = re.compile(r"t_(\d+)\.csv$")


def list_timesteps(rep_dir: Path) -> List[Path]:
    files = [p for p in rep_dir.glob("t_*.csv") if T_PATTERN.search(p.name)]
    files.sort(key=lambda p: int(T_PATTERN.search(p.name).group(1)))
    return files


def load_snapshot(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    for col in SNAP_REQUIRED:
        if col not in df.columns:
            raise ValueError(f"{fp} missing column '{col}' (required: {SNAP_REQUIRED})")
    out = df[["id", "x", "y", "size"]].copy()
    out["id"] = out["id"].astype(int)
    out["x"] = out["x"].astype(float)
    out["y"] = out["y"].astype(float)
    out["size"] = out["size"].astype(float)
    return out


# -----------------------------
# Inference of merge recipients at t+1 for lost IDs at t
# -----------------------------
@dataclass
class InferConfig:
    w_dist: float = 1.0
    w_size: float = 1.0
    min_pos_size_gain: float = 0.0
    torus: bool = False
    W: Optional[float] = None
    H: Optional[float] = None


def median_nnd(positions: np.ndarray, torus: bool, W: Optional[float], H: Optional[float]) -> float:
    if len(positions) < 2:
        return 1.0
    if torus:
        dists = []
        for i in range(len(positions)):
            dvec = min_image_dist(positions[i], np.delete(positions, i, axis=0), float(W), float(H))
            dists.append(np.min(dvec))
        return float(np.median(dists))
    else:
        dists = []
        for i in range(len(positions)):
            v = positions[i] - np.delete(positions, i, axis=0)
            d = np.sqrt(np.sum(v * v, axis=1))
            dists.append(np.min(d))
        return float(np.median(dists))


def infer_unions_at_step(
    df_t: pd.DataFrame,
    df_tp1: pd.DataFrame,
    dsu: DSU,
    cfg: InferConfig
) -> int:
    ids_t = df_t["id"].to_numpy(int)
    ids_tp1 = df_tp1["id"].to_numpy(int)
    set_t, set_tp1 = set(ids_t), set(ids_tp1)
    lost = np.array(sorted(list(set_t - set_tp1)), dtype=int)
    if lost.size == 0:
        return 0

    pos_t = df_t.set_index("id")[["x", "y"]].to_dict("index")
    size_t = df_t.set_index("id")["size"].to_dict()

    pos_tp1 = df_tp1[["x", "y"]].to_numpy(float)
    id_tp1 = ids_tp1
    size_tp1 = df_tp1.set_index("id")["size"]
    size_prev_tp1 = df_t.set_index("id")["size"].reindex(df_tp1["id"]).fillna(0.0)
    delta_size = (size_tp1 - size_prev_tp1).to_numpy(float)
    delta_pos = np.clip(delta_size, 0.0, None)
    size_norm = delta_pos / (delta_pos.max() + 1e-12)

    sigma = median_nnd(pos_tp1, cfg.torus, cfg.W, cfg.H) if len(pos_tp1) > 2 else 1.0

    unions = 0
    for lid in lost:
        dsu.add(int(lid))
        p = np.array([pos_t[lid]["x"], pos_t[lid]["y"]], dtype=float)

        if cfg.torus:
            dvec = min_image_dist(p, pos_tp1, float(cfg.W), float(cfg.H))
        else:
            dxy = pos_tp1 - p[None, :]
            dvec = np.sqrt(np.sum(dxy * dxy, axis=1))

        s_dist = np.exp(-dvec / max(1e-9, sigma))
        s_size = size_norm.copy()

        if cfg.min_pos_size_gain > 0:
            mask_ok = delta_pos >= cfg.min_pos_size_gain
            if np.any(mask_ok):
                s_dist = s_dist * (mask_ok.astype(float))
                s_size = s_size * (mask_ok.astype(float))

        score = cfg.w_dist * s_dist + cfg.w_size * s_size
        if not np.any(score > 0):
            score = s_dist

        rid = int(id_tp1[np.argmax(score)])
        dsu.add(rid)
        dsu.union(lid, rid)
        unions += 1

    return unions


# -----------------------------
# Repeat-level lineage building
# -----------------------------
@dataclass
class RepeatResult:
    initial_ids: np.ndarray
    initial_pos: np.ndarray
    dsu: DSU


def build_lineages_for_repeat(rep_dir: Path, infer_cfg: InferConfig) -> RepeatResult:
    t_files = list_timesteps(rep_dir)
    if not t_files:
        raise FileNotFoundError(f"No timestep CSVs in {rep_dir}. Ensure abm.save_raw=true.")
    df0 = load_snapshot(t_files[0])
    ids0 = df0["id"].to_numpy(int)
    pos0 = df0[["x", "y"]].to_numpy(float)

    dsu = DSU()
    for uid in ids0:
        dsu.add(int(uid))

    for k in range(len(t_files) - 1):
        df_t = load_snapshot(t_files[k])
        df_tp1 = load_snapshot(t_files[k + 1])
        infer_unions_at_step(df_t, df_tp1, dsu, infer_cfg)

    return RepeatResult(initial_ids=ids0, initial_pos=pos0, dsu=dsu)


# -----------------------------
# Distance binning (with fixed edges)
# -----------------------------
@dataclass
class Binning:
    nbins: int = 25
    max_pairs: Optional[int] = None
    torus: bool = False
    W: Optional[float] = None
    H: Optional[float] = None


def accumulate_pairs(rep: RepeatResult, bins: Binning, edges_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids0 = rep.initial_ids
    pos0 = rep.initial_pos
    N = len(ids0)

    iu = np.triu_indices(N, k=1)
    pairs = np.vstack(iu).T
    if bins.max_pairs is not None and len(pairs) > bins.max_pairs:
        rng = np.random.default_rng(12345)
        idx_sel = rng.choice(len(pairs), size=bins.max_pairs, replace=False)
        pairs = pairs[idx_sel]

    if bins.torus:
        if bins.max_pairs is not None and len(pairs) < (N * (N - 1)) // 2:
            dists = np.array(
                [min_image_dist(pos0[i], pos0[[j]], float(bins.W), float(bins.H))[0] for i, j in pairs],
                dtype=float
            )
        else:
            dvec_full, _ = pairwise_min_image_dist(pos0, float(bins.W), float(bins.H))
            dists = dvec_full
    else:
        dij = pos0[pairs[:, 0]] - pos0[pairs[:, 1]]
        dists = np.sqrt(np.sum(dij * dij, axis=1))

    merged_mask = np.fromiter(
        (rep.dsu.find(int(ids0[i])) == rep.dsu.find(int(ids0[j])) for i, j in pairs),
        dtype=bool,
        count=len(pairs),
    )

    edges = np.asarray(edges_ref, dtype=float)
    idx = np.clip(np.digitize(dists, edges) - 1, 0, len(edges) - 2)
    merged_counts = np.bincount(idx, weights=merged_mask.astype(int), minlength=len(edges) - 1).astype(int)
    total_counts = np.bincount(idx, minlength=len(edges) - 1).astype(int)
    centers = 0.5 * (edges[:-1] + edges[1:])
    prob = np.divide(merged_counts, np.maximum(1, total_counts), where=(total_counts > 0))

    return edges, centers, np.vstack([merged_counts, total_counts, prob])


# -----------------------------
# Global (RUN‑wide) edges selection (parallel)
# -----------------------------
def estimate_t0_distances_for_repeat(rep_dir: Path, torus: bool, W: Optional[float], H: Optional[float],
                                     q_sample_pairs: Optional[int]) -> np.ndarray:
    try:
        t_files = list_timesteps(rep_dir)
        if not t_files:
            return np.array([], dtype=float)
        df0 = load_snapshot(t_files[0])
        ids0 = df0["id"].to_numpy(int)
        pos0 = df0[["x", "y"]].to_numpy(float)

        N = len(ids0)
        iu = np.triu_indices(N, k=1)
        pairs = np.vstack(iu).T
        if q_sample_pairs is not None and len(pairs) > q_sample_pairs:
            rng = np.random.default_rng(2026)
            idx_sel = rng.choice(len(pairs), size=q_sample_pairs, replace=False)
            pairs = pairs[idx_sel]

        if torus:
            if q_sample_pairs is not None and len(pairs) < (N * (N - 1)) // 2:
                dists = np.array(
                    [min_image_dist(pos0[i], pos0[[j]], float(W), float(H))[0] for i, j in pairs],
                    dtype=float
                )
            else:
                dvec_full, _ = pairwise_min_image_dist(pos0, float(W), float(H))
                dists = dvec_full
        else:
            d = pos0[pairs[:, 0]] - pos0[pairs[:, 1]]
            dists = np.sqrt(np.sum(d * d, axis=1))
        return dists
    except Exception:
        return np.array([], dtype=float)


def choose_global_edges_across_run_parallel(
    cond_dirs: List[Path],
    nbins: int,
    *,
    mode: str,
    qmax: float,
    torus: bool,
    W: Optional[float],
    H: Optional[float],
    q_sample_pairs: Optional[int],
    fixed_max_distance: Optional[float],
    workers: int
) -> np.ndarray:
    if mode == "fixed":
        if fixed_max_distance is None:
            if torus and W is not None and H is not None:
                fixed_max_distance = float(math.hypot(W / 2.0, H / 2.0))
            else:
                raise ValueError("--edge-mode fixed requires --fixed-max-distance (or provide --torus --W --H)")
        dmin, dmax = 0.0, float(fixed_max_distance)
        return np.linspace(dmin, dmax, nbins + 1)

    # percentile mode: pool a subsample of all t0 distances across ALL repeats
    rep_dirs = []
    for cdir in cond_dirs:
        rep_dirs.extend([p for p in cdir.iterdir() if p.is_dir() and p.name.startswith("repeat_")])
    if not rep_dirs:
        raise RuntimeError("No repeats found to estimate distance range.")

    pool = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(estimate_t0_distances_for_repeat, rd, torus, W, H, q_sample_pairs)
            for rd in rep_dirs
        ]
        for fut in as_completed(futs):
            arr = fut.result()
            if arr.size:
                pool.append(arr)

    if not pool:
        raise RuntimeError("Could not estimate distance range from any repeat in the run.")
    pool = np.concatenate(pool).astype(float)

    dmin = 0.0
    dmax = float(np.nanquantile(pool, q=qmax))
    if not np.isfinite(dmax) or dmax <= dmin:
        dmax = float(np.nanmax(pool))
        if not np.isfinite(dmax) or dmax <= dmin:
            dmax = dmin + 1.0
    return np.linspace(dmin, dmax, nbins + 1)


# -----------------------------
# Curve models and fitting
# -----------------------------
def model_exp(x, A, B, C):         # A * exp(-B x) + C
    return A * np.exp(-B * x) + C


def model_strexp(x, A, lam, k, C): # A * exp(-(x/lam)^k) + C
    return A * np.exp(-np.power(np.maximum(x, 1e-12) / lam, k)) + C


def model_power(x, A, x0, k, C):   # A * (x + x0)^(-k) + C
    return A * np.power(np.maximum(x + x0, 1e-9), -k) + C


def fit_one_curve(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    results = []

    def _eval(name, func, p0, bounds):
        try:
            popt, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=20000)
            yhat = func(x, *popt)
            resid = y - yhat
            sse = float(np.sum(resid ** 2))
            n = len(y)
            k = len(popt)
            aic = n * np.log(sse / max(n, 1)) + 2 * k if sse > 0 and n > k else np.inf
            bic = n * np.log(sse / max(n, 1)) + k * np.log(max(n, 1)) if sse > 0 and n > k else np.inf
            sst = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1 - sse / sst if sst > 0 else np.nan
            results.append(dict(model=name, params=popt.tolist(), sse=sse, aic=aic, bic=bic, r2=r2))
        except Exception:
            pass

    y0 = float(y[0]) if len(y) else 1.0
    _eval("exp",    model_exp,    p0=[y0, 0.01, 0.0],            bounds=([0.0, 0.0, 0.0], [1.5, 10.0, 0.5]))
    _eval("strexp", model_strexp, p0=[y0, max(1.0, np.median(x)), 1.0, 0.0],
          bounds=([0.0, 1e-3, 0.2, 0.0], [2.0, 1e4, 4.0, 0.5]))
    _eval("power",  model_power,  p0=[y0, 1.0, 1.0, 0.0],        bounds=([0.0, 0.0, 0.2, 0.0], [2.0, 1e3, 5.0]))

    if not results:
        return dict(model="none", params=[], sse=np.inf, aic=np.inf, bic=np.inf, r2=np.nan)
    return min(results, key=lambda r: r["aic"])


# -----------------------------
# Worker wrappers (top-level for spawn pickling)
# -----------------------------
def worker_accumulate_repeat(rep_dir: str, infer_cfg_dict: dict, bins_dict: dict, edges_ref: np.ndarray):
    try:
        infer_cfg = InferConfig(**infer_cfg_dict)
        bins_cfg = Binning(**bins_dict)
        rep = build_lineages_for_repeat(Path(rep_dir), infer_cfg)
        _, _, arr = accumulate_pairs(rep, bins_cfg, edges_ref=edges_ref)
        return (arr[0], arr[1])  # merged_counts, total_counts
    except Exception as e:
        return None


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Infer pairwise lineage merges and plot P(merge|initial distance) for 1D sweeps (global bins, parallel)")
    ap.add_argument("--results-root", type=str, default=None, help="Path to Parameter_sweeps_results (default: sibling of this file).")
    ap.add_argument("--run", type=str, default="latest", help="'latest' or a specific run folder")
    ap.add_argument("--param-key", type=str, default=None, help="Choose the swept parameter (auto if unique)")

    # inference weights / rules
    ap.add_argument("--w-dist", type=float, default=1.0, help="Weight for distance in matching")
    ap.add_argument("--w-size", type=float, default=1.0, help="Weight for positive size-gain in matching")
    ap.add_argument("--min-size-gain", type=float, default=0.0, help="Minimum Δsize to count as a gain (recipient)")

    # geometry & binning (global edges)
    ap.add_argument("--torus", action="store_true", help="Use minimum-image distances (requires --W and --H)")
    ap.add_argument("--W", type=float, default=None, help="Domain width (torus)")
    ap.add_argument("--H", type=float, default=None, help="Domain height (torus)")
    ap.add_argument("--nbins", type=int, default=25, help="Number of initial-distance bins")
    ap.add_argument("--max-pairs", type=int, default=None, help="Subsample initial pairs per repeat for accumulation")
    ap.add_argument("--edge-mode", choices=["percentile", "fixed"], default="percentile", help="How to choose ONE global edges array across the run")
    ap.add_argument("--qmax", type=float, default=0.995, help="Upper quantile if edge-mode=percentile")
    ap.add_argument("--fixed-max-distance", type=float, default=None, help="Upper distance if edge-mode=fixed (falls back to sqrt((W/2)^2+(H/2)^2) on torus)")
    ap.add_argument("--q-sample-pairs", type=int, default=50000, help="Subsample pairs per repeat when estimating global quantile (edge-mode=percentile)")

    # fitting domain (reliability)
    ap.add_argument("--pmin", type=float, default=0.02, help="Fit only bins where ALL params have prob >= pmin")
    ap.add_argument("--min-pairs-per-bin", type=int, default=50, help="Fit only bins where ALL params have total pairs >= this")

    # parallelism
    ap.add_argument("--workers", type=int, default=None, help="Processes to use (default: CPU count - 1)")
    ap.add_argument("--dpi", type=int, default=260)
    args = ap.parse_args()

    # Paths
    script_dir = Path(__file__).resolve().parent
    default_results_root = script_dir.parent / "Parameter_sweeps_results"
    results_root = Path(args.results_root) if args.results_root else default_results_root
    run_dir = find_run_dir(results_root, args.run)
    sim_root = run_dir / "simulations"
    if not sim_root.exists():
        raise FileNotFoundError("No simulations/ in the run. Ensure abm.save_raw=true in your YAML.")

    cond_dirs = sorted([p for p in sim_root.iterdir() if p.is_dir()])
    if not cond_dirs:
        raise FileNotFoundError(f"No condition folders under {sim_root}")

    # Detect swept parameter if not specified
    sample_params = [parse_condition_id(d.name) for d in cond_dirs]
    keys = sorted(set(k for d in sample_params for k in d.keys()))
    varying = [k for k in keys if len(set(d.get(k) for d in sample_params)) > 1]
    if args.param_key:
        xkey = args.param_key
        if xkey not in keys:
            raise ValueError(f"--param-key '{xkey}' not found among parsed keys: {keys}")
    else:
        if len(varying) != 1:
            raise ValueError(f"Expected one swept parameter, found {len(varying)}: {varying}. Use --param-key.")
        xkey = varying[0]

    print(f"[INFO] Using swept parameter: {xkey}")

    # Process count
    if args.workers is None or args.workers <= 0:
        try:
            import os
            cpu = os.cpu_count() or 2
        except Exception:
            cpu = 2
        workers = max(1, cpu - 1)
    else:
        workers = int(args.workers)
    print(f"[INFO] Using workers={workers}")

    infer_cfg = InferConfig(
        w_dist=float(args.w_dist),
        w_size=float(args.w_size),
        min_pos_size_gain=float(args.min_size_gain),
        torus=bool(args.torus),
        W=args.W,
        H=args.H,
    )

    bins_cfg = Binning(
        nbins=int(args.nbins),
        max_pairs=args.max_pairs,
        torus=bool(args.torus),
        W=args.W,
        H=args.H,
    )

    # ---- ONE GLOBAL EDGES ARRAY ACROSS THE RUN (PARALLEL) ----
    edges_global = choose_global_edges_across_run_parallel(
        cond_dirs,
        nbins=bins_cfg.nbins,
        mode=args.edge_mode,
        qmax=args.qmax,
        torus=bins_cfg.torus,
        W=bins_cfg.W,
        H=bins_cfg.H,
        q_sample_pairs=args.q_sample_pairs,
        fixed_max_distance=args.fixed_max_distance,
        workers=workers
    )
    centers_global = 0.5 * (edges_global[:-1] + edges_global[1:])
    print(f"[INFO] Global edges: d in [{edges_global[0]:.3f}, {edges_global[-1]:.3f}]  nbins={len(edges_global)-1}")

    # Accumulate per-parameter curves (PARALLEL over repeats)
    per_param: Dict[object, dict] = {}
    for cdir in cond_dirs:
        params = parse_condition_id(cdir.name)
        pval = params.get(xkey, None)
        if pval is None:
            print(f"[WARN] {cdir.name}: missing {xkey}; skipping.")
            continue

        rep_dirs = sorted([p for p in cdir.iterdir() if p.is_dir() and p.name.startswith("repeat_")])
        if not rep_dirs:
            print(f"[WARN] {cdir.name}: no repeat_* folders; skipping.")
            continue

        merged_acc = np.zeros(len(edges_global) - 1, dtype=float)
        total_acc  = np.zeros(len(edges_global) - 1, dtype=float)

        infer_cfg_dict = dict(
            w_dist=infer_cfg.w_dist, w_size=infer_cfg.w_size,
            min_pos_size_gain=infer_cfg.min_pos_size_gain,
            torus=infer_cfg.torus, W=infer_cfg.W, H=infer_cfg.H
        )
        bins_dict = dict(nbins=bins_cfg.nbins, max_pairs=bins_cfg.max_pairs,
                         torus=bins_cfg.torus, W=bins_cfg.W, H=bins_cfg.H)

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(worker_accumulate_repeat, str(rd), infer_cfg_dict, bins_dict, edges_global)
                for rd in rep_dirs
            ]
            used = 0
            for fut in as_completed(futs):
                res = fut.result()
                if res is None:
                    continue
                m_counts, t_counts = res
                merged_acc += np.asarray(m_counts, float)
                total_acc  += np.asarray(t_counts, float)
                used += 1

        if used == 0:
            print(f"[WARN] {cdir.name}: no usable repeats.")
            continue

        prob = np.divide(merged_acc, np.maximum(1.0, total_acc), where=(total_acc > 0))
        per_param[pval] = dict(edges=edges_global, centers=centers_global, m=merged_acc, n=total_acc, p=prob)

        # Save per-parameter curve
        out_param_dir = run_dir / "coagulation_1d_infer" / "curves"
        out_param_dir.mkdir(parents=True, exist_ok=True)
        df_curve = pd.DataFrame({
            "param_key": xkey,
            "param_value": pval,
            "d_left": edges_global[:-1],
            "d_center": centers_global,
            "d_right": edges_global[1:],
            "n_pairs": total_acc.astype(int),
            "n_merged": merged_acc.astype(int),
            "prob": prob,
        })
        df_curve.to_csv(out_param_dir / f"coag_prob_by_distance__param={pval}.csv", index=False)

    if not per_param:
        raise RuntimeError("No curves accumulated. Check raw t_*.csv and inference settings.")

    # Combined CSV + all-parameter plots
    uniq_params = sorted(per_param.keys(), key=lambda z: float(z) if isinstance(z, (int, float)) else z)
    comb_rows = []
    for pv in uniq_params:
        d = per_param[pv]
        for k in range(len(d["centers"])):
            comb_rows.append({
                "param_key": xkey,
                "param_value": pv,
                "d_center": float(d["centers"][k]),
                "n_pairs": int(d["n"][k]),
                "n_merged": int(d["m"][k]),
                "prob": float(d["p"][k]),
            })
    out_comb = run_dir / "coagulation_1d_infer" / "combined"
    out_comb.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comb_rows).to_csv(out_comb / "coag_prob_by_distance_ALL.csv", index=False)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(uniq_params)))

    def plot_all(logy=False):
        fig, ax = plt.subplots(figsize=(10.5, 6.0))
        for pv, col in zip(uniq_params, colors):
            d = per_param[pv]
            ax.plot(d["centers"], d["p"], marker="o", lw=2.0, color=col, label=f"{xkey}={pv}")
        ax.set_title(f"Pairwise coagulation vs initial distance (all {xkey})")
        ax.set_xlabel("Initial distance")
        ax.set_ylabel("P(pair lineage merge)")
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(out_comb / ("coag_vs_distance_ALL_logy.png" if logy else "coag_vs_distance_ALL.png"), dpi=args.dpi)
        plt.close(fig)

    plot_all(logy=False)
    plot_all(logy=True)

    # Fitting domain: bins where ALL parameters meet thresholds
    centers_ref = centers_global
    mask_common = None
    for pv in uniq_params:
        d = per_param[pv]
        mask = (d["n"] >= args.min_pairs_per_bin) & (d["p"] >= args.pmin)
        mask_common = mask if mask_common is None else (mask_common & mask)

    xfit = centers_ref[mask_common] if mask_common is not None else np.array([])
    if xfit.size < 4:
        print("[WARN] Not enough common bins above thresholds to fit models; skipping fits.")
        return

    fit_dir = run_dir / "coagulation_1d_infer" / "fits"
    fit_dir.mkdir(parents=True, exist_ok=True)
    fit_rows = []

    for pv in uniq_params:
        d = per_param[pv]
        yfit = np.clip(d["p"][mask_common], 1e-9, 1.0)

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

        def overlay(logy=False):
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
            if logy:
                ax.set_yscale("log")
            ax.set_title(f"{xkey}={pv}  (best={best['model']})")
            ax.set_xlabel("Initial distance")
            ax.set_ylabel("P(pair lineage merge)")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            outname = f"{pv}__fit_logy.png" if logy else f"{pv}__fit.png"
            fig.savefig(fit_dir / outname, dpi=args.dpi)
            plt.close(fig)

        overlay(logy=False)
        overlay(logy=True)

    pd.DataFrame(fit_rows).to_csv(fit_dir / "fit_report.csv", index=False)
    print(f"[DONE] Curves → {run_dir/'coagulation_1d_infer/curves'}")
    print(f"[DONE] Combined plots/CSV → {out_comb}")
    print(f"[DONE] Fits → {fit_dir}")


if __name__ == "__main__":
    main()