#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABM sensitivity analysis — 1D sweep only (parallel, torus-aware, robust eligibility).

This script assumes your scenario folders vary ONLY ONE parameter across the sweep:
  • Either __a_<val> varies (and __p_merge_<val> is constant/absent), or
  • __p_merge_<val> varies (and __a_<val> is constant/absent).

Folder naming accepted (examples):
  ...__a_2, __a_2.5, __a_2p5
  ...__p_merge_1, __p_merge_0.5, __p_merge_0p5

If the fixed counterpart is missing from names, you can supply a constant via:
  --constant-a <value>   and/or   --constant-p <value>

Outputs under:
  ABM_sensitivity_results/<SWEEP>/analysis_<timestamp>/
    tables/metrics_long.csv
    tables/<metric>_1d.csv
    figures/<metric>_1d.png
    summary.json

Metrics include:
  D_eff, rho, R, k_enc, T_enc, e, (k_bg, k_loc, T_between),
  episode expectations (E[T_ep], E[T_succ], E[T_fail], P_succ_ep),
  T_merge components + T_merge, K_eff, dominance fractions.

NOTE: If p is not available (from names or --constant-p), episode terms are NaN.
"""

from __future__ import annotations
import argparse
import os
import re
import glob
import json
import datetime as _dt
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import os as _os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- helpers ---------------------------------

def radius_from_size_3d(n_cells: int, cell_volume: float = 1954.0) -> float:
    """r = ((3/(4π)) * n_cells * cell_volume)^(1/3)."""
    return float(((3.0/(4.0*np.pi)) * n_cells * cell_volume) ** (1.0/3.0))


@dataclass
class Config:
    results_root: str
    sweep_name: str
    width: float = 1344.0
    height: float = 1025.0
    dt: float = 1.0
    alpha: float = 1.0   # L = alpha * rho^{-1/2}
    xi: float = 1.0      # ln(xi/(rho R^2)) in k_enc
    c: float = 1.0       # r× = R + c * dt * <V>
    cell_volume: float = 1954.0
    pair_sample: int = 20000
    min_steps: int = 3
    n_eff: float = 1.0
    torus: bool = False
    workers: int = max((_os.cpu_count() or 2) - 1, 1)
    constant_a: Optional[float] = None
    constant_p: Optional[float] = None


# ---------------------- directory traversal ----------------------------
# tolerant patterns (integers, dotted decimals, 'p' decimals)
P_RE = re.compile(r"__p_merge_(\d+(?:p\d+|\.\d+)?)")
A_RE = re.compile(r"__a_(\d+(?:p\d+|\.\d+)?)")

def _tok2f(tok: str) -> float:
    return float(tok.replace('p', '.')) if 'p' in tok else float(tok)

def parse_params_from_name(name: str) -> Tuple[Optional[float], Optional[float]]:
    p_merge = None
    a_shape = None
    mp = P_RE.search(name)
    ma = A_RE.search(name)
    if mp:
        p_merge = _tok2f(mp.group(1))
    if ma:
        a_shape = _tok2f(ma.group(1))
    return a_shape, p_merge


# ---------------------- geometry utilities -----------------------------

def _wrap_delta(d: np.ndarray, L: float) -> np.ndarray:
    """Minimum-image wrapping for torus distances."""
    if not np.isfinite(L) or L <= 0:
        return d
    d = np.where(d >  +0.5*L, d - L, d)
    d = np.where(d <  -0.5*L, d + L, d)
    return d


# ---------------------- time-series estimators --------------------------

def load_timeseries(repeat_dir: str) -> Tuple[List[int], Dict[int, pd.DataFrame]]:
    paths = sorted(glob.glob(os.path.join(repeat_dir, 't_*.csv')))
    time_idx = []
    frames: Dict[int, pd.DataFrame] = {}
    for p in paths:
        m = re.match(r"t_(\d+)\.csv", os.path.basename(p))
        if not m:
            continue
        t = int(m.group(1))
        df = pd.read_csv(
            p,
            usecols=['id','x','y','size'],
            dtype={'id':'int64','x':'float64','y':'float64','size':'float32'}
        )
        frames[t] = df
        time_idx.append(t)
    time_idx.sort()
    return time_idx, frames


def estimate_speed_and_headings(time_idx: List[int], frames: Dict[int, pd.DataFrame],
                                dt: float, torus: bool, width: float, height: float) -> Tuple[np.ndarray, np.ndarray]:
    speeds = []
    headings = []
    for t0, t1 in zip(time_idx[:-1], time_idx[1:]):
        m = pd.merge(frames[t0], frames[t1], on='id', suffixes=('_0','_1'))
        if m.shape[0] == 0:
            continue
        dx = m['x_1'].to_numpy() - m['x_0'].to_numpy()
        dy = m['y_1'].to_numpy() - m['y_0'].to_numpy()
        if torus:
            dx = _wrap_delta(dx, width)
            dy = _wrap_delta(dy, height)
        step = np.vstack([dx, dy]).T
        n = np.linalg.norm(step, axis=1)
        valid = n > 1e-9
        if not np.any(valid):
            continue
        speeds.append(n[valid] / dt)
        headings.append(step[valid] / n[valid, None])
    return (np.concatenate(speeds, axis=0) if speeds else np.array([]),
            np.concatenate(headings, axis=0) if headings else np.empty((0,2)))


def estimate_heading_correlation(time_idx: List[int], frames: Dict[int, pd.DataFrame],
                                 dt: float, torus: bool, width: float, height: float) -> float:
    dots = []
    for t0, t1, t2 in zip(time_idx[:-2], time_idx[1:-1], time_idx[2:]):
        m01 = pd.merge(frames[t0], frames[t1], on='id', suffixes=('_0','_1'))
        m12 = pd.merge(frames[t1], frames[t2], on='id', suffixes=('_1','_2'))
        ids = np.intersect1d(m01['id'].values, m12['id'].values)
        if ids.size == 0:
            continue
        m01 = m01[m01['id'].isin(ids)].sort_values('id')
        m12 = m12[m12['id'].isin(ids)].sort_values('id')
        s01x = m01['x_1'].to_numpy() - m01['x_0'].to_numpy()
        s01y = m01['y_1'].to_numpy() - m01['y_0'].to_numpy()
        s12x = m12['x_2'].to_numpy() - m12['x_1'].to_numpy()
        s12y = m12['y_2'].to_numpy() - m12['y_1'].to_numpy()
        if torus:
            s01x = _wrap_delta(s01x, width); s01y = _wrap_delta(s01y, height)
            s12x = _wrap_delta(s12x, width); s12y = _wrap_delta(s12y, height)
        s01 = np.vstack([s01x, s01y]).T
        s12 = np.vstack([s12x, s12y]).T
        n01 = np.linalg.norm(s01, axis=1)
        n12 = np.linalg.norm(s12, axis=1)
        valid = (n01 > 1e-9) & (n12 > 1e-9)
        if not np.any(valid):
            continue
        u01 = s01[valid] / n01[valid, None]
        u12 = s12[valid] / n12[valid, None]
        dots.append(np.sum(u01 * u12, axis=1))
    if not dots:
        return np.nan
    d = np.clip(np.concatenate(dots, axis=0), -1.0, 1.0)
    return float(np.nanmean(d))


def estimate_density(time_idx: List[int], frames: Dict[int, pd.DataFrame], area: float) -> float:
    return float(np.mean([frames[t].shape[0] for t in time_idx]) / area) if time_idx else np.nan


def typical_contact_distance(cell_volume: float) -> float:
    return 2.0 * radius_from_size_3d(1, cell_volume)


def estimate_eligibility(time_idx: List[int], frames: Dict[int, pd.DataFrame], R: float,
                         pair_sample: int, rng: Optional[np.random.Generator],
                         torus: bool, width: float, height: float) -> Tuple[float, float]:
    """Same-ID overlap persistence: P(overlap at t -> still overlapped at t+dt)."""
    if rng is None:
        rng = np.random.default_rng(123)
    persists = []
    gaps = []
    for t0, t1 in zip(time_idx[:-1], time_idx[1:]):
        A = frames[t0]; B = frames[t1]
        nA, nB = A.shape[0], B.shape[0]
        if nA < 2 or nB < 2:
            continue
        id_to_idx_B = {int(k): i for i, k in enumerate(B['id'].values)}
        max_pairs = int(min(pair_sample, max(1000, 20*nA)))
        i = rng.integers(0, nA, size=max_pairs)
        j = rng.integers(0, nA, size=max_pairs)
        valid = i != j
        i = i[valid]; j = j[valid]
        if i.size == 0:
            continue
        idsA = A['id'].to_numpy(dtype=np.int64, copy=False)
        ax = A['x'].to_numpy(np.float64, copy=False); ay = A['y'].to_numpy(np.float64, copy=False)
        bx = B['x'].to_numpy(np.float64, copy=False); by = B['y'].to_numpy(np.float64, copy=False)
        dx0 = ax[i] - ax[j]; dy0 = ay[i] - ay[j]
        if torus:
            dx0 = _wrap_delta(dx0, width); dy0 = _wrap_delta(dy0, height)
        d0 = np.sqrt(dx0*dx0 + dy0*dy0)
        overlapped = np.where(d0 <= R)[0]
        if overlapped.size == 0:
            continue
        ii = i[overlapped]; jj = j[overlapped]
        id_i = idsA[ii]; id_j = idsA[jj]
        present = np.array([(k in id_to_idx_B) and (l in id_to_idx_B) for k, l in zip(id_i, id_j)], bool)
        if not np.any(present):
            continue
        bi = np.array([id_to_idx_B[int(k)] for k in id_i[present]], int)
        bj = np.array([id_to_idx_B[int(l)] for l in id_j[present]], int)
        dx1 = bx[bi] - bx[bj]; dy1 = by[bi] - by[bj]
        if torus:
            dx1 = _wrap_delta(dx1, width); dy1 = _wrap_delta(dy1, height)
        d1 = np.sqrt(dx1*dx1 + dy1*dy1)
        still = d1 <= R
        persists.append(np.mean(still))
        ended = ~still
        if np.any(ended):
            gap = np.maximum(d1[ended] - R, 0.0)
            if gap.size > 0:
                gaps.append(float(np.mean(gap)))
    if not persists:
        return np.nan, np.nan
    e = float(np.mean(persists))
    d_sep = float(np.nanmean(gaps)) if gaps else 0.05 * R
    return e, d_sep


# ------------------------ scenario aggregation -------------------------

def _analyse_one(args):
    scen, cfg_dict, seed = args
    cfg = Config(**cfg_dict)
    rng = np.random.default_rng(seed)
    res = analyse_scenario(scen, cfg, rng=rng)
    return os.path.basename(scen), res

def analyse_scenario(scenario_dir: str, cfg: Config, rng=None) -> Optional[Dict[str, float]]:
    name = os.path.basename(scenario_dir.rstrip(os.sep))
    a_name, p_name = parse_params_from_name(name)

    # Resolve effective parameters: use name if present; fallback to constants if provided
    a_eff = a_name if a_name is not None else cfg.constant_a
    p_eff = p_name if p_name is not None else cfg.constant_p

    # If neither a nor p provided anywhere (name/constant), skip
    if a_eff is None and p_eff is None:
        return None

    repeat_dirs = sorted(glob.glob(os.path.join(scenario_dir, 'repeat_*')))
    if not repeat_dirs:
        return None

    area = cfg.width * cfg.height
    vals = []
    for rdir in repeat_dirs:
        time_idx, frames = load_timeseries(rdir)
        if len(time_idx) < cfg.min_steps:
            continue

        speeds, _ = estimate_speed_and_headings(time_idx, frames, cfg.dt, cfg.torus, cfg.width, cfg.height)
        if speeds.size == 0:
            continue
        E_V = float(np.mean(speeds))

        C1 = estimate_heading_correlation(time_idx, frames, cfg.dt, cfg.torus, cfg.width, cfg.height)
        if not np.isfinite(C1) or C1 <= 1e-6:
            continue
        C1 = float(np.clip(C1, 1e-6, 0.999999))
        D_theta = -np.log(C1) / cfg.dt
        D_eff = (E_V ** 2) / (2.0 * D_theta)

        rho = estimate_density(time_idx, frames, area)
        R = typical_contact_distance(cfg.cell_volume)

        e, d_sep = estimate_eligibility(time_idx, frames, R, cfg.pair_sample, rng, cfg.torus, cfg.width, cfg.height)
        if not np.isfinite(e):
            continue

        # Hazards (2D logs)
        L = cfg.alpha * (rho ** -0.5) if rho > 0 else np.inf

        enc_arg = max(cfg.xi/(max(rho, 1e-12) * (R**2)), 1.000001)
        log_enc = np.log(enc_arg)
        k_enc = (4.0 * np.pi * D_eff * rho) / log_enc if np.isfinite(log_enc) else np.nan
        T_enc = 1.0 / (cfg.n_eff * k_enc) if (k_enc and k_enc > 0) else np.nan

        r0 = R + d_sep
        rx = R + cfg.c * cfg.dt * E_V
        log_bg = np.log(max((L/R) if np.isfinite(L) else 1.000001, 1.000001))
        log_loc = np.log(max(rx/r0, 1.000001))
        k_bg  = (4.0 * np.pi * D_eff * rho) / log_bg  if log_bg  > 0 else np.nan
        k_loc = (4.0 * np.pi * D_eff)        / log_loc if log_loc > 0 else np.nan
        kbkl = np.nansum([k_bg, k_loc])
        T_between = 1.0 / (cfg.n_eff * kbkl) if kbkl and kbkl > 0 else np.nan

        # Episode / renewal terms — require p
        p = p_eff
        if p is not None and np.isfinite(p):
            q = 1.0 - p
            T_sub = cfg.dt / 2.0
            denom = max(1.0 - q * e, 1e-9)
            P_succ_ep = p / denom
            E_T_ep    = T_sub * (1.0 + (q * (1.0 - e)) / denom)
            E_T_succ  = T_sub * (1.0 + (q * e) / denom)
            E_T_fail  = T_sub * (1.0 + 1.0 / denom)

            C_enc     = T_enc
            C_between = (q * (1.0 - e) / max(p, 1e-12)) * T_between
            C_ep      = T_sub * (1.0 + q * (1.0 - e)) / max(p, 1e-12)
            T_merge   = np.nansum([C_enc, C_between, C_ep])
            K_eff     = 1.0 / T_merge if (T_merge and T_merge > 0) else np.nan

            total = np.nansum([C_enc, C_between, C_ep])
            if np.isfinite(total) and total > 0:
                frac_enc = C_enc / total
                frac_between = C_between / total
                frac_ep = C_ep / total
            else:
                frac_enc = frac_between = frac_ep = np.nan
        else:
            # Without p we can still report transport/encounter/between metrics
            P_succ_ep = E_T_ep = E_T_succ = E_T_fail = np.nan
            C_enc = T_enc
            C_between = C_ep = T_merge = K_eff = np.nan
            frac_enc = frac_between = frac_ep = np.nan

        vals.append({
            'a_shape': a_eff if a_eff is not None else np.nan,
            'p_merge': p_eff if p_eff is not None else np.nan,
            'E_V': E_V, 'C1_heading': C1, 'D_theta': D_theta, 'D_eff': D_eff,
            'rho': rho, 'R': R, 'e': e, 'd_sep': d_sep,
            'k_enc': k_enc, 'T_enc': T_enc, 'k_bg': k_bg, 'k_loc': k_loc, 'T_between': T_between,
            'P_succ_ep': P_succ_ep, 'E_T_ep': E_T_ep, 'E_T_succ': E_T_succ, 'E_T_fail': E_T_fail,
            'C_enc': C_enc, 'C_between': C_between, 'C_ep': C_ep,
            'T_merge': T_merge, 'K_eff': K_eff,
            'frac_enc': frac_enc, 'frac_between': frac_between, 'frac_ep': frac_ep,
        })

    if not vals:
        return None

    # Average over repeats
    df = pd.DataFrame(vals)
    agg = df.mean(numeric_only=True).to_dict()
    agg['a_shape'] = float(df['a_shape'].iloc[0]) if not np.isnan(df['a_shape'].iloc[0]) else np.nan
    agg['p_merge'] = float(df['p_merge'].iloc[0]) if not np.isnan(df['p_merge'].iloc[0]) else np.nan
    return agg


# ----------------------------- plotting --------------------------------

def plot_line(x, y, title: str, xlabel: str, ylabel: str, out_path: str):
    plt.figure(figsize=(7.2, 5.0), dpi=150)
    plt.plot(x, y, marker='o')
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path); plt.close()


# ------------------------------- main ----------------------------------

def main():
    ap = argparse.ArgumentParser(description='ABM 1D sensitivity analysis (parallel)')
    ap.add_argument('--results-root', type=str, default='ABM_sensitivity_results')
    ap.add_argument('--sweep-name', type=str, required=True)
    ap.add_argument('--width', type=float, default=1344.0)
    ap.add_argument('--height', type=float, default=1025.0)
    ap.add_argument('--dt', type=float, default=1.0)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--xi', type=float, default=1.0)
    ap.add_argument('--c', type=float, default=1.0)
    ap.add_argument('--cell-volume', type=float, default=1954.0)
    ap.add_argument('--pair-sample', type=int, default=20000)
    ap.add_argument('--min-steps', type=int, default=3)
    ap.add_argument('--n-eff', type=float, default=1.0)
    ap.add_argument('--torus', action='store_true', help='Use minimum-image distances (periodic domain)')
    ap.add_argument('--workers', type=int, default=max((_os.cpu_count() or 2) - 1, 1))
    ap.add_argument('--constant-a', type=float, default=None, help='Supply fixed a when only p_merge varies or when a tag is absent')
    ap.add_argument('--constant-p', type=float, default=None, help='Supply fixed p_merge when only a varies or when p tag is absent')
    args = ap.parse_args()

    cfg = Config(results_root=args.results_root,
                 sweep_name=args.sweep_name,
                 width=args.width, height=args.height, dt=args.dt,
                 alpha=args.alpha, xi=args.xi, c=args.c,
                 cell_volume=args.cell_volume, pair_sample=args.pair_sample,
                 min_steps=args.min_steps, n_eff=args.n_eff,
                 torus=bool(args.torus), workers=int(max(args.workers, 1)),
                 constant_a=args.constant_a, constant_p=args.constant_p)

    sweep_dir = os.path.join(cfg.results_root, cfg.sweep_name, 'simulations')
    if not os.path.isdir(sweep_dir):
        raise SystemExit(f'No such simulations dir: {sweep_dir}')

    scenarios_all = sorted([d for d in glob.glob(os.path.join(sweep_dir, '*')) if os.path.isdir(d)])
    if not scenarios_all:
        raise SystemExit('No scenario folders found.')

    # Inspect names to detect which parameter varies
    a_vals, p_vals = [], []
    matched = []
    skipped = []
    for scen in scenarios_all:
        a_name, p_name = parse_params_from_name(os.path.basename(scen))
        if a_name is None and p_name is None:
            skipped.append(scen); continue
        # Effective values (name > constant > None)
        a_eff = a_name if a_name is not None else cfg.constant_a
        p_eff = p_name if p_name is not None else cfg.constant_p
        matched.append((scen, a_eff, p_eff))
        if a_eff is not None: a_vals.append(a_eff)
        if p_eff is not None: p_vals.append(p_eff)

    print(f"Found {len(scenarios_all)} scenarios. Matched: {len(matched)}; Skipped (no tags): {len(skipped)}")
    if not matched:
        raise SystemExit('No scenarios contain __a_ or __p_merge_ tags, and no constants provided.')

    # Check 1D assumption
    a_unique = np.unique(np.array(a_vals)) if a_vals else np.array([])
    p_unique = np.unique(np.array(p_vals)) if p_vals else np.array([])

    # Determine which axis varies
    vary_a = (a_unique.size > 1) and (p_unique.size <= 1)
    vary_p = (p_unique.size > 1) and (a_unique.size <= 1)

    if not (vary_a or vary_p):
        if (a_unique.size > 1) and (p_unique.size > 1):
            raise SystemExit('Detected variation in BOTH a and p_merge — this script is 1D only. Use the 2D script.')
        else:
            print('No variation detected (single scenario or constants only). Will still compute metrics_long.csv.')

    # Prepare parallel tasks
    cfg_dict = asdict(cfg)
    tasks = []
    for k, (scen, _, _) in enumerate(matched):
        tasks.append((scen, cfg_dict, 13579 + k))

    print(f'Analysing {len(tasks)} scenarios with {cfg.workers} workers...')
    rows: List[Dict[str, float]] = []
    done = 0; total = len(tasks)
    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        futs = [ex.submit(_analyse_one, t) for t in tasks]
        for fut in as_completed(futs):
            scen_name, res = fut.result()
            done += 1
            if res is not None:
                rows.append(res)
                print(f"[{done}/{total}] Finished: {scen_name}")
            else:
                print(f"[{done}/{total}] Finished (no data): {scen_name}")

    if not rows:
        raise SystemExit('No scenarios analysed successfully. Check files and flags.')

    # Output dirs
    out_root = os.path.join(cfg.results_root, cfg.sweep_name,
                            f'analysis_{_dt.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    tables_dir = os.path.join(out_root, 'tables')
    figs_dir = os.path.join(out_root, 'figures')
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # Long table
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(tables_dir, 'metrics_long.csv'), index=False)

    # Decide x-axis and label
    if vary_a:
        df_sorted = df.sort_values('a_shape')
        x = df_sorted['a_shape'].values
        xlab = 'Gamma shape a'
        dim = '1D_a'
    elif vary_p:
        df_sorted = df.sort_values('p_merge')
        x = df_sorted['p_merge'].values
        xlab = 'p_merge'
        dim = '1D_p'
    else:
        # no variation — just save long table
        summary = {
            'config': asdict(cfg),
            'n_scenarios_found': len(scenarios_all),
            'n_matched': len(matched),
            'n_skipped': len(skipped),
            'dimensionality': 'none',
            'note': 'Only metrics_long.csv written; no variation detected.'
        }
        os.makedirs(out_root, exist_ok=True)
        with open(os.path.join(out_root, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'Analysis complete. Outputs in: {out_root}')
        return

    # Metrics to export as 1D
    metrics = [
        ('K_eff','Effective coagulation rate K_eff'),
        ('k_enc','Encounter rate k_enc'),
        ('e','Continuation eligibility e'),
        ('E_T_ep','Expected time per episode'),
        ('E_T_succ','Expected time (successful episode)'),
        ('E_T_fail','Expected time (failed episode)'),
        ('T_enc','MFPT to first encounter T_enc'),
        ('T_between','Inter-episode mean time T_between'),
        ('T_merge','Total mean time to merge T_merge'),
        ('frac_enc','Fractional contribution: encounter'),
        ('frac_between','Fractional contribution: between-episodes'),
        ('frac_ep','Fractional contribution: episode cadence'),
    ]

    for key, title in metrics:
        sub = df_sorted[[('a_shape' if vary_a else 'p_merge'), key]].dropna()
        if sub.empty:
            continue
        sub.to_csv(os.path.join(tables_dir, f'{key}_1d.csv'), index=False)
        plot_line(sub.iloc[:,0].values, sub[key].values,
                  title=f'{title} vs {xlab}', xlabel=xlab, ylabel=key,
                  out_path=os.path.join(figs_dir, f'{key}_1d.png'))

    # Summary JSON
    summary = {
        'config': asdict(cfg),
        'n_scenarios_found': len(scenarios_all),
        'n_matched': len(matched),
        'n_skipped': len(skipped),
        'dimensionality': dim,
        'notes': {
            'eligibility': 'Same-agent-ID overlap persistence; torus-aware if --torus',
            'D_eff': 'From <V> and heading correlation at lag dt',
            'hazards': '2D logs with L=alpha*rho^{-1/2}, r0=R+d_sep, r×=R+c*dt*<V>',
            'T_merge': 'T_enc + ((1-p)(1-e)/p)*T_between + (dt/2)*((1+(1-p)(1-e))/p) when p is available'
        }
    }
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'Analysis complete. Outputs in: {out_root}')


if __name__ == '__main__':
    main()