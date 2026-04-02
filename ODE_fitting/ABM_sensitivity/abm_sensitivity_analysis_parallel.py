#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABM sweep analysis for the effective coagulation kernel — parallel, robust, and torus-aware.

Adds:
  • More tolerant scenario parser: accepts integers (e.g. 1), p-style decimals (2p5), and dotted decimals (0.5).
  • Progress markers: prints [done/total] and scenario name as each finishes.
  • Summary of matched vs skipped scenario folders.
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
    """Radius from size for a 3D volume-preserving sphere model.
       r = ((3/(4π)) * n_cells * cell_volume)^(1/3).
    """
    return float(((3.0/(4.0*np.pi)) * n_cells * cell_volume) ** (1.0/3.0))

@dataclass
class Config:
    results_root: str
    sweep_name: str
    width: float = 1344.0
    height: float = 1025.0
    dt: float = 1.0
    alpha: float = 1.0   # outer length scale L = alpha * rho^{-1/2}
    xi: float = 1.0      # encounter log constant in ln(xi/(rho R^2))
    c: float = 1.0       # r× = R + c * dt * <V>
    cell_volume: float = 1954.0
    pair_sample: int = 20000
    min_steps: int = 3
    n_eff: float = 1.0   # usability factor for encounters
    torus: bool = False  # minimum-image distances on a torus
    workers: int = max((_os.cpu_count() or 2) - 1, 1)

# ---------------------- directory traversal ----------------------------
# Accepts integers, p-style decimals (e.g. 2p5), and dotted decimals (e.g. 2.5)
SCENARIO_RE = re.compile(
    r"__p_merge_(?P<p>\d+(?:p\d+|\.\d+)?)__a_(?P<a>\d+(?:p\d+|\.\d+)?)"
)

def parse_ap_from_scenario(name: str) -> Optional[Tuple[float, float]]:
    m = SCENARIO_RE.search(name)
    if not m:
        return None
    def tok2f(s: str) -> float:
        if "p" in s:  # 2p5 -> 2.5
            return float(s.replace("p", "."))
        return float(s)
    p = tok2f(m.group("p"))
    a = tok2f(m.group("a"))
    return a, p

# ---------------------- geometry utilities -----------------------------

def _wrap_delta(d: np.ndarray, L: float) -> np.ndarray:
    """Minimum-image wrapping for torus distances, vectorised."""
    if not np.isfinite(L) or L <= 0:
        return d
    d = np.where(d >  +0.5*L, d - L, d)
    d = np.where(d <  -0.5*L, d + L, d)
    return d

# ---------------------- time-series estimators --------------------------

def load_timeseries(repeat_dir: str) -> Tuple[List[int], Dict[int, pd.DataFrame]]:
    """Load all t_*.csv in a repeat folder.
       Returns an increasing list of integer time indices and dataframes keyed by t.
       Each CSV must have columns: id, x, y, size (radius inferred if needed).
    """
    paths = sorted(glob.glob(os.path.join(repeat_dir, 't_*.csv')))
    time_idx = []
    frames: Dict[int, pd.DataFrame] = {}
    for p in paths:
        base = os.path.basename(p)
        m = re.match(r"t_(\d+)\.csv", base)
        if not m:
            continue
        t = int(m.group(1))
        # faster CSV read with usecols and dtypes
        df = pd.read_csv(
            p,
            usecols=['id','x','y','size'],
            dtype={'id':'int64','x':'float64','y':'float64','size':'float32'}
        )
        frames[t] = df
        time_idx.append(t)
    time_idx.sort()
    return time_idx, frames

def estimate_speed_and_headings(time_idx: List[int],
                                frames: Dict[int, pd.DataFrame],
                                dt: float,
                                torus: bool,
                                width: float,
                                height: float) -> Tuple[np.ndarray, np.ndarray]:
    """Speeds are ||Δr||/dt between consecutive snapshots (id must persist).
       Headings are unit displacement vectors (for heading correlation).
    """
    speeds = []
    headings = []
    for t0, t1 in zip(time_idx[:-1], time_idx[1:]):
        df0 = frames[t0]
        df1 = frames[t1]
        m = pd.merge(df0, df1, on='id', suffixes=('_0','_1'))
        if m.shape[0] == 0:
            continue
        dx = m['x_1'].to_numpy() - m['x_0'].to_numpy()
        dy = m['y_1'].to_numpy() - m['y_0'].to_numpy()
        if torus:
            dx = _wrap_delta(dx, width)
            dy = _wrap_delta(dy, height)
        step = np.vstack([dx, dy]).T
        step_norm = np.linalg.norm(step, axis=1)
        valid = step_norm > 1e-9
        if not np.any(valid):
            continue
        v = step_norm[valid] / dt
        u = (step[valid] / step_norm[valid, None])
        speeds.append(v)
        headings.append(u)
    if len(speeds) == 0:
        return np.array([]), np.empty((0,2))
    speeds = np.concatenate(speeds, axis=0)
    headings = np.concatenate(headings, axis=0)
    return speeds, headings

def estimate_heading_correlation(time_idx: List[int],
                                 frames: Dict[int, pd.DataFrame],
                                 dt: float,
                                 torus: bool,
                                 width: float,
                                 height: float) -> float:
    """Mean dot of unit displacements across three successive snapshots."""
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
            s01x = _wrap_delta(s01x, width)
            s01y = _wrap_delta(s01y, height)
            s12x = _wrap_delta(s12x, width)
            s12y = _wrap_delta(s12y, height)
        s01 = np.vstack([s01x, s01y]).T
        s12 = np.vstack([s12x, s12y]).T
        n01 = np.linalg.norm(s01, axis=1)
        n12 = np.linalg.norm(s12, axis=1)
        valid = (n01 > 1e-9) & (n12 > 1e-9)
        if not np.any(valid):
            continue
        u01 = s01[valid] / n01[valid, None]
        u12 = s12[valid] / n12[valid, None]
        dot = np.sum(u01 * u12, axis=1)
        dots.append(dot)
    if len(dots) == 0:
        return np.nan
    dots = np.concatenate(dots, axis=0)
    dots = np.clip(dots, -1.0, 1.0)
    return float(np.nanmean(dots))

def estimate_density(time_idx: List[int],
                     frames: Dict[int, pd.DataFrame],
                     area: float) -> float:
    N = [frames[t].shape[0] for t in time_idx]
    if len(N) == 0:
        return np.nan
    return float(np.mean(N) / area)

def typical_contact_distance(cell_volume: float) -> float:
    """Use two singlets as the typical contact scale: R = 2 * r(size=1)."""
    r1 = radius_from_size_3d(1, cell_volume)
    return 2.0 * r1

def estimate_eligibility(time_idx: List[int],
                         frames: Dict[int, pd.DataFrame],
                         R: float,
                         pair_sample: int,
                         rng: Optional[np.random.Generator],
                         torus: bool,
                         width: float,
                         height: float
                         ) -> Tuple[float, float]:
    """Continuation eligibility e as:
         P( pair overlapped at t  -->  still overlapped at t+dt ),
       using the same agent IDs across frames.
       Also returns d_sep (mean positive gap when overlap ends).
    """
    if rng is None:
        rng = np.random.default_rng(123)
    persists: List[float] = []
    sep_gaps: List[float] = []

    for t0, t1 in zip(time_idx[:-1], time_idx[1:]):
        A = frames[t0]
        B = frames[t1]
        nA = A.shape[0]; nB = B.shape[0]
        if nA < 2 or nB < 2:
            continue

        # map ids of B once
        id_to_idx_B = {int(k): i for i, k in enumerate(B['id'].values)}

        # sample pairs from A (cap work ~ linear in nA)
        max_pairs = int(min(pair_sample, max(1000, 20*nA)))
        i = rng.integers(0, nA, size=max_pairs)
        j = rng.integers(0, nA, size=max_pairs)
        valid = i != j
        i = i[valid]; j = j[valid]
        if i.size == 0:
            continue

        idsA = A['id'].to_numpy(dtype=np.int64, copy=False)
        ax = A['x'].to_numpy(np.float64, copy=False)
        ay = A['y'].to_numpy(np.float64, copy=False)
        bx = B['x'].to_numpy(np.float64, copy=False)
        by = B['y'].to_numpy(np.float64, copy=False)

        # distances at t
        dx0 = ax[i] - ax[j]
        dy0 = ay[i] - ay[j]
        if torus:
            dx0 = _wrap_delta(dx0, width)
            dy0 = _wrap_delta(dy0, height)
        d0 = np.sqrt(dx0*dx0 + dy0*dy0)
        overlapped_idx = np.where(d0 <= R)[0]
        if overlapped_idx.size == 0:
            continue

        ii = i[overlapped_idx]; jj = j[overlapped_idx]
        id_i = idsA[ii]; id_j = idsA[jj]
        present = np.array([(k in id_to_idx_B) and (l in id_to_idx_B)
                            for k, l in zip(id_i, id_j)], dtype=bool)
        if not np.any(present):
            continue

        bi = np.array([id_to_idx_B[int(k)] for k in id_i[present]], dtype=int)
        bj = np.array([id_to_idx_B[int(l)] for l in id_j[present]], dtype=int)

        dx1 = bx[bi] - bx[bj]
        dy1 = by[bi] - by[bj]
        if torus:
            dx1 = _wrap_delta(dx1, width)
            dy1 = _wrap_delta(dy1, height)
        d1 = np.sqrt(dx1*dx1 + dy1*dy1)

        still = d1 <= R
        persists.append(np.mean(still))

        ended = ~still
        if np.any(ended):
            gap = np.maximum(d1[ended] - R, 0.0)
            if gap.size > 0:
                sep_gaps.append(float(np.mean(gap)))

    if len(persists) == 0:
        return np.nan, np.nan

    e = float(np.mean(persists))
    d_sep = float(np.nanmean(sep_gaps)) if len(sep_gaps) > 0 else 0.05 * R
    return e, d_sep

# ------------------------ scenario aggregation -------------------------

def _analyse_one(args):
    scen, cfg_dict, seed = args
    cfg = Config(**cfg_dict)
    rng = np.random.default_rng(seed)
    res = analyse_scenario(scen, cfg, rng=rng)
    return os.path.basename(scen), res  # return scenario name for progress printing

def analyse_scenario(scenario_dir: str,
                     cfg: Config,
                     rng: Optional[np.random.Generator] = None
                     ) -> Optional[Dict[str, float]]:
    scenario_name = os.path.basename(scenario_dir.rstrip(os.sep))
    ap = parse_ap_from_scenario(scenario_name)
    if ap is None:
        return None
    a_shape, p_merge = ap

    repeat_dirs = sorted(glob.glob(os.path.join(scenario_dir, 'repeat_*')))
    if len(repeat_dirs) == 0:
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

        p = p_merge
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

        T_merge = np.nansum([C_enc, C_between, C_ep])
        K_eff = 1.0 / T_merge if (T_merge and T_merge > 0) else np.nan

        total = np.nansum([C_enc, C_between, C_ep])
        frac_enc = C_enc     / total if total and total > 0 else np.nan
        frac_between = C_between / total if total and total > 0 else np.nan
        frac_ep = C_ep      / total if total and total > 0 else np.nan
        dom_idx = int(np.nanargmax([frac_enc, frac_between, frac_ep])) if np.isfinite(total) else 2
        dom = ['encounter','between','episode'][dom_idx]

        vals.append({
            'a_shape': a_shape,
            'p_merge': p_merge,
            'E_V': E_V,
            'C1_heading': C1,
            'D_theta': D_theta,
            'D_eff': D_eff,
            'rho': rho,
            'R': R,
            'e': e,
            'd_sep': d_sep,
            'k_enc': k_enc,
            'T_enc': T_enc,
            'k_bg': k_bg,
            'k_loc': k_loc,
            'T_between': T_between,
            'P_succ_ep': P_succ_ep,
            'E_T_ep': E_T_ep,
            'E_T_succ': E_T_succ,
            'E_T_fail': E_T_fail,
            'C_enc': C_enc,
            'C_between': C_between,
            'C_ep': C_ep,
            'T_merge': T_merge,
            'K_eff': K_eff,
            'frac_enc': frac_enc,
            'frac_between': frac_between,
            'frac_ep': frac_ep,
            'dominant': dom,
        })

    if len(vals) == 0:
        return None

    df = pd.DataFrame(vals)
    agg = df.mean(numeric_only=True).to_dict()
    dominant = df['dominant'].value_counts().idxmax()
    agg['a_shape'] = float(df['a_shape'].iloc[0])
    agg['p_merge'] = float(df['p_merge'].iloc[0])
    agg['dominant'] = dominant
    return agg

# ----------------------------- plotting --------------------------------

def to_surfaces(rows: List[Dict[str, float]], xkey: str, ykey: str, metric: str):
    df = pd.DataFrame(rows)
    xs = np.sort(df[xkey].unique())
    ys = np.sort(df[ykey].unique())
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    Z = np.full_like(X, fill_value=np.nan, dtype=float)
    for i, a in enumerate(xs):
        for j, p in enumerate(ys):
            v = df[(df[xkey]==a) & (df[ykey]==p)][metric]
            if not v.empty:
                Z[j,i] = float(v.iloc[0])
    return xs, ys, Z

def plot_surface(xs, ys, Z, title: str, xlabel: str, ylabel: str, out_path: str, cmap='viridis'):
    plt.figure(figsize=(7.2, 5.6), dpi=150)
    im = plt.imshow(Z, origin='lower', aspect='auto',
                    extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                    cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ------------------------------- main ----------------------------------

def main():
    ap = argparse.ArgumentParser(description='ABM sensitivity sweep analysis (parallel)')
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
    ap.add_argument('--workers', type=int, default=max((_os.cpu_count() or 2) - 1, 1), help='Number of worker processes')
    args = ap.parse_args()

    cfg = Config(results_root=args.results_root,
                 sweep_name=args.sweep_name,
                 width=args.width,
                 height=args.height,
                 dt=args.dt,
                 alpha=args.alpha,
                 xi=args.xi,
                 c=args.c,
                 cell_volume=args.cell_volume,
                 pair_sample=args.pair_sample,
                 min_steps=args.min_steps,
                 n_eff=args.n_eff,
                 torus=bool(args.torus),
                 workers=int(max(args.workers, 1)))

    sweep_dir = os.path.join(cfg.results_root, cfg.sweep_name, 'simulations')
    if not os.path.isdir(sweep_dir):
        raise SystemExit(f'No such simulations dir: {sweep_dir}')

    scenarios_all = sorted([d for d in glob.glob(os.path.join(sweep_dir, '*')) if os.path.isdir(d)])
    matched, skipped = [], []
    for scen in scenarios_all:
        if parse_ap_from_scenario(os.path.basename(scen)) is None:
            skipped.append(scen)
        else:
            matched.append(scen)

    total = len(scenarios_all)
    print(f"Found {total} scenario folders. Matched pattern: {len(matched)}; Skipped: {len(skipped)}")
    if skipped:
        # show a few examples so it's clear what's being skipped
        examples = ", ".join(os.path.basename(s) for s in skipped[:5])
        print(f"Examples of skipped names (first 5): {examples}")

    if len(matched) == 0:
        raise SystemExit('No scenarios matched the naming pattern __p_merge_<val>__a_<val>.')

    # prepare tasks
    tasks = []
    cfg_dict = asdict(cfg)
    for k, scen in enumerate(matched):
        tasks.append((scen, cfg_dict, 12345 + k))

    print(f'Analysing {len(tasks)} scenarios with {cfg.workers} workers...')
    rows: List[Dict[str, float]] = []

    # progress mapping
    future_to_name = {}
    done = 0; total = len(tasks)
    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        futs = []
        for t in tasks:
            fut = ex.submit(_analyse_one, t)
            futs.append(fut)
            future_to_name[fut] = os.path.basename(t[0])

        for fut in as_completed(futs):
            scen_name, res = fut.result()
            done += 1
            if res is not None:
                rows.append(res)
                print(f"[{done}/{total}] Finished: {scen_name}")
            else:
                print(f"[{done}/{total}] Finished (no data/skip): {scen_name}")

    if len(rows) == 0:
        raise SystemExit('No scenarios analysed successfully. Check files and flags.')

    # Prepare output dirs
    out_root = os.path.join(cfg.results_root, cfg.sweep_name,
                            f'analysis_{_dt.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    tables_dir = os.path.join(out_root, 'tables')
    figs_dir = os.path.join(out_root, 'figures')
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # Save long table
    df = pd.DataFrame(rows)
    df.sort_values(['a_shape','p_merge'], inplace=True)
    df.to_csv(os.path.join(tables_dir, 'metrics_long.csv'), index=False)

    # Surfaces
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
        xs, ys, Z = to_surfaces(rows, 'a_shape', 'p_merge', key)
        # wide CSV
        wide = pd.DataFrame(Z, index=ys, columns=xs)
        wide.index.name = 'p_merge'
        wide.columns.name = 'a_shape'
        wide.to_csv(os.path.join(tables_dir, f'{key}_surface.csv'))
        # figure
        plot_surface(np.array(xs), np.array(ys), Z,
                     title=f'{title} over (a, p_merge)',
                     xlabel='Gamma shape a', ylabel='p_merge',
                     out_path=os.path.join(figs_dir, f'{key}_surface.png'),
                     cmap='viridis' if not key.startswith('frac_') else 'magma')

    # Save summary JSON
    summary = {
        'config': asdict(cfg),
        'n_scenarios_found': len(scenarios_all),
        'n_matched': len(matched),
        'n_skipped': len(skipped),
        'n_analysed': len(rows),
        'metrics_saved': [k for k,_ in metrics],
        'notes': {
            'eligibility_estimation': 'Same-agent-ID overlap persistence between t and t+dt; torus-aware if --torus is set.',
            'D_eff_method': 'From <V> and heading correlation at lag dt: D_eff = <V>^2 / (2*D_theta), D_theta = -ln(<u_t·u_{t+1}>)/dt.',
            'hazards': '2D log forms with L=alpha*rho^{-1/2}, r0=R+d_sep, r×=R+c*dt*<V>.',
            'K_eff_formula': 'T_merge = T_enc + ((1-p)(1-e)/p)*T_between + (dt/2)*((1 + (1-p)(1-e))/p); K_eff=1/T_merge.'
        }
    }
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'Analysis complete. Outputs in: {out_root}')
    if skipped:
        print(f'Note: {len(skipped)} scenario folders were skipped because their names did not match the pattern. See summary.json.')

if __name__ == '__main__':
    main()