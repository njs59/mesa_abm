#!/usr/bin/env python3
"""
Fast + parallel Encounter/Attempt Diagnostics from raw time-series CSVs,
with robust scenario parsing and dominance plots.

- Parallel across scenarios or repeats (ProcessPoolExecutor).
- Fast overlap via a uniform-grid spatial hash (near-linear in n).
- Robust parser that recognizes keys containing underscores (e.g. p_merge_0p5).
- Intermediate artefacts (per repeat) + progress.
- Visualisations:
  * heatmap_K_eff_pred_mean.png
  * heatmap_e_tick_mean.png
  * heatmap_p_star_mean.png
  * heatmap_D_eff_empirical.png
  * heatmap_dominant_part.png       # NEW: 1=encounter, 0=attempt
  * heatmap_attempt_fraction.png    # NEW: T_attempt_part / (T_enc_part + T_attempt_part)
  * slice_* plots for quick 1D inspection
  * scatter_Keff_vs_Deff.png

Semantics:
- 'TRUE' rule for merges by ID loss (parent ID missing at t+1).  (from your existing script)  # [2](blob:https://m365.cloud.microsoft/246d6923-5a45-476f-af73-d43cf950e715)
- Snapshot schema: id,x,y,size (as in your t_0002.csv).                                    # [2](blob:https://m365.cloud.microsoft/246d6923-5a45-476f-af73-d43cf950e715)
- Renewal & encounter formulas follow your derivation:                                      # [1](blob:https://m365.cloud.microsoft/0137585a-c4ef-47a4-9d45-e129859a430f)[2](blob:https://m365.cloud.microsoft/246d6923-5a45-476f-af73-d43cf950e715)
    T_enc = ln(1/(rho R^2)) / (4 D_eff rho eta_eff)
    T_enc-part     = ((1 - e(1-p) zeta)/p) * T_enc
    T_attempt-part = (dt/2) * (1 + (1-p) zeta (1-e)) / p
"""

from __future__ import annotations
import argparse, os, re
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?([eE]-?\d+)?$")
_P_DECIMAL_RE = re.compile(r"^-?\d+p\d+([eE]-?\d+)?$")

# ---------------------- utils ----------------------
def _ts() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def _parse_value(tok: str):
    t = tok.strip()
    if _P_DECIMAL_RE.fullmatch(t):
        t = t.replace("p", ".")
    if _NUMERIC_RE.fullmatch(t):
        try:
            return float(t)
        except Exception:
            return t
    return t

def parse_condition_parameters(name: str, known_keys: List[str] | None = None) -> Dict[str, object]:
    """
    Robust parser for tokens like `p_merge_0p5` and keys that contain underscores.
    Strategy:
      1) Split the scenario name by "__" to tokens.
      2) For each token:
         - If known_keys provided, try to match "<known_key>_<value>" (prefix match).
         - Else, split at the LAST underscore → key="_".join(parts[:-1]), value=parts[-1],
           and only accept if the value looks numeric ('p' decimal or numeric).
    """
    params: Dict[str, object] = {}
    tokens = name.split("__")
    known_keys = known_keys or []
    for tok in tokens:
        if "_" not in tok:
            continue

        # Prefer an exact known key prefix match: "<key>_<value>"
        matched = False
        for k in sorted(known_keys, key=len, reverse=True):
            prefix = k + "_"
            if tok.startswith(prefix):
                val = tok[len(prefix):]
                params[k] = _parse_value(val)
                matched = True
                break
        if matched:
            continue

        # Fallback: split at last underscore
        parts = tok.split("_")
        if len(parts) < 2:
            continue
        key_candidate = "_".join(parts[:-1])
        value_tok = parts[-1]
        if _P_DECIMAL_RE.fullmatch(value_tok) or _NUMERIC_RE.fullmatch(value_tok):
            params[key_candidate] = _parse_value(value_tok)

    return params

def write_progress(outroot: Path, msg: str, also_print: bool = True):
    outroot.mkdir(parents=True, exist_ok=True)
    with open(outroot / 'progress.log', 'a', encoding='utf-8') as f:
        f.write(f"[{_ts()}] {msg}\n")
    if also_print:
        print(msg, flush=True)

# ---------------------- geometry & fast overlap ----------------------
def radius_from_size(size: np.ndarray, C: float) -> np.ndarray:
    # r(s) = C * s^{1/3}
    return C * np.power(size, 1.0/3.0)

def build_grid(x: np.ndarray, y: np.ndarray, cell_size: float):
    gx = np.floor_divide(x, cell_size).astype(int)
    gy = np.floor_divide(y, cell_size).astype(int)
    grid: Dict[tuple, List[int]] = {}
    for i, (cx, cy) in enumerate(zip(gx, gy)):
        grid.setdefault((cx, cy), []).append(i)
    return grid, gx, gy

def neighbor_cells(cx: int, cy: int, r_cells: int):
    for dx in range(-r_cells, r_cells+1):
        for dy in range(-r_cells, r_cells+1):
            yield (cx+dx, cy+dy)

def overlap_pairs_grid(ids: np.ndarray, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> set:
    """
    Fast approximate-neighbour search with a uniform grid (exact for circular overlap check
    when searching a sufficient neighbourhood). Use cell_size = max(r) and 5x5 neighbours.
    """
    n = len(ids)
    if n < 2:
        return set()
    r_max = float(np.max(r)) if n else 0.0
    if r_max <= 0:
        return set()
    cell_size = r_max
    grid, gx, gy = build_grid(x, y, cell_size)
    pairs = set()
    for i in range(n):
        cx, cy = gx[i], gy[i]
        for nc in neighbor_cells(cx, cy, r_cells=2):  # 5x5 neighbourhood
            idxs = grid.get(nc, [])
            if not idxs:
                continue
            cand = [j for j in idxs if j > i]
            if not cand:
                continue
            dx = x[cand] - x[i]
            dy = y[cand] - y[i]
            dd = np.hypot(dx, dy)
            thresh = r[i] + r[cand]
            mask = dd <= thresh
            if np.any(mask):
                for j in np.array(cand)[mask]:
                    a, b = ids[i], ids[j]
                    pairs.add((a, b) if a < b else (b, a))
    return pairs

# ---------------------- MSD / D_eff ----------------------
def compute_msd(df_by_step: Dict[int, pd.DataFrame], max_lag: int, dt: float):
    idmaps = {k: grp.set_index('id')[['x','y']].to_dict('index') for k, grp in df_by_step.items()}
    steps = sorted(df_by_step.keys())
    max_lag = min(max_lag, steps[-1] - steps[0])
    lags, msd = [], []
    for lag in range(1, max_lag+1):
        sqs = []
        for t in steps:
            t2 = t + lag
            if t2 not in idmaps:
                continue
            M1 = idmaps[t]; M2 = idmaps[t2]
            common = set(M1.keys()) & set(M2.keys())
            if not common:
                continue
            for cid in common:
                dx = M2[cid]['x'] - M1[cid]['x']
                dy = M2[cid]['y'] - M1[cid]['y']
                sqs.append(dx*dx + dy*dy)
        if sqs:
            lags.append(lag * dt)
            msd.append(float(np.mean(sqs)))
    return np.array(lags), np.array(msd)

def estimate_D_eff_from_msd(lags: np.ndarray, msd: np.ndarray) -> float:
    if len(lags) < 2:
        return np.nan
    num = float(np.dot(lags, msd)); den = float(np.dot(lags, lags))
    if den <= 0:
        return np.nan
    return (num/den) / 4.0

# ---------------------- Eligibility / survival (with 'both-parents-gone') ----------------------
def overlap_pairs_df(df: pd.DataFrame, C: float) -> set:
    ids = df['id'].astype(str).to_numpy()
    x = df['x'].to_numpy(); y = df['y'].to_numpy()
    r = radius_from_size(df['size'].to_numpy(), C)
    return overlap_pairs_grid(ids, x, y, r)

def compute_tick_eligibility_and_survival(df_by_step: Dict[int, pd.DataFrame], C: float,
                                          save_path: Path | None = None):
    steps = sorted(df_by_step.keys())
    if len(steps) < 2:
        return float('nan'), float('nan'), 0
    rows = []
    e_hits = e_tot = 0
    z_hits = z_tot = 0
    for t in steps[:-1]:
        A = df_by_step[t]; B = df_by_step[t+1]
        A_id = A['id'].astype(str).to_numpy(); B_id = B['id'].astype(str).to_numpy()
        idsA = set(A_id); idsB = set(B_id)

        # All overlap pairs at t
        pairs_t_all = overlap_pairs_df(A, C)

        # Survivors at t+1
        surv_ids = idsA & idsB
        As = A[A['id'].astype(str).isin(surv_ids)].reset_index(drop=True)
        Bs = B[B['id'].astype(str).isin(surv_ids)].reset_index(drop=True)
        pairs_surv_t = overlap_pairs_df(As, C) if len(As) >= 2 else set()
        pairs_surv_t1 = overlap_pairs_df(Bs, C) if len(Bs) >= 2 else set()

        # Eligibility via survivors that persist
        surv_persist = sum(1 for p in pairs_surv_t if p in pairs_surv_t1)
        e_hits += surv_persist; e_tot += len(pairs_surv_t)
        # zeta: survivors only
        z_hits += surv_persist; z_tot += len(pairs_surv_t)

        # Eligibility via BOTH parents gone (eligible + success this tick, per your ID rules)
        disappeared = idsA - idsB
        both_gone = sum(1 for (i, j) in pairs_t_all if (i in disappeared and j in disappeared))
        e_hits += both_gone; e_tot += both_gone

        if save_path is not None:
            rows.append({
                'step': t,
                'n_pairs_t_all': len(pairs_t_all),
                'n_survivor_pairs_t': len(pairs_surv_t),
                'n_survivor_pairs_t1': len(pairs_surv_t1),
                'survivor_persist': surv_persist,
                'both_parents_gone_pairs': both_gone,
                'e_hits_cum': e_hits, 'e_tot_cum': e_tot,
                'z_hits_cum': z_hits, 'z_tot_cum': z_tot,
            })

    e_tick = (e_hits / e_tot) if e_tot > 0 else float('nan')
    zeta_tick = (z_hits / z_tot) if z_tot > 0 else float('nan')
    if save_path is not None and rows:
        pd.DataFrame(rows).to_csv(save_path, index=False)
    return e_tick, zeta_tick, e_tot

# ---------------------- Coagulation metrics (ID loss) ----------------------
def measure_coagulation_series(df_by_step: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []; steps = sorted(df_by_step.keys()); prev_ids = None
    for t in steps:
        ids = set(df_by_step[t]['id'].astype(str))
        lost = 0 if prev_ids is None else len(prev_ids - ids)
        nC = len(ids); pairs = nC*(nC-1)//2
        rows.append({
            'step': t, 'nC': nC, 'lost': lost,
            'lost_per_cluster': lost/max(1, nC),
            'lost_per_pairs'  : lost/max(1, pairs),
            'lost_per_n2'     : lost/max(1, nC*nC),
        })
        prev_ids = ids
    return pd.DataFrame(rows)

# ---------------------- Renewal formulas ----------------------
def T_encounter(rho: float, R: float, D_eff: float, eta_eff: float = 1.0) -> float:
    if rho <= 0 or R <= 0 or D_eff <= 0: return float('inf')
    return np.log(1.0/(rho*R*R)) / (4.0*D_eff*rho*max(eta_eff, 1e-9))  # [2](blob:https://m365.cloud.microsoft/246d6923-5a45-476f-af73-d43cf950e715)

def p_star_general(A: float, B: float, e: float, zeta: float) -> float:
    num = B*(1.0 + zeta*(1.0 - e)) - A*(1.0 - e*zeta)
    den = zeta*(A*e + B*(1.0 - e))
    if den == 0: return 0.0
    return float(np.clip(num/den, 0.0, 1.0))  # [1](blob:https://m365.cloud.microsoft/0137585a-c4ef-47a4-9d45-e129859a430f)

def K_parts_and_total(T_enc: float, dt: float, p: float, e: float, zeta: float):
    if p <= 0: return np.nan, np.nan, np.nan
    # Encounter & attempt time parts (renewal derivation)                  # [1](blob:https://m365.cloud.microsoft/0137585a-c4ef-47a4-9d45-e129859a430f)
    enc_part     = ((1.0 - e*(1.0 - p)*zeta) / p) * T_enc
    attempt_part = (dt/2.0) * (1.0 + (1.0 - p)*zeta*(1.0 - e)) / p
    T_merge = enc_part + attempt_part
    return enc_part, attempt_part, T_merge

def K_eff_from_total(T_merge: float) -> float:
    return 1.0/T_merge if (T_merge is not None and T_merge > 0 and np.isfinite(T_merge)) else np.nan

# ---------------------- IO helpers ----------------------
def load_time_series(repeat_dir: Path) -> Dict[int, pd.DataFrame]:
    files = sorted(repeat_dir.glob('t_*.csv'), key=lambda f: int(re.findall(r"t_(\d+)\.csv", f.name)[0]))
    series: Dict[int, pd.DataFrame] = {}
    for fp in files:
        df = pd.read_csv(fp)
        for col in ['id','x','y','size']:
            if col not in df.columns:
                raise RuntimeError(f"{fp} missing column '{col}'")
        df['id'] = df['id'].astype(str)
        step = int(re.findall(r"t_(\d+)\.csv", fp.name)[0])
        series[step] = df[['id','x','y','size']].copy()
    return series

def estimate_domain_area(series: Dict[int, pd.DataFrame]) -> float:
    xs, ys = [], []
    for df in series.values():
        xs.append(df['x'].to_numpy()); ys.append(df['y'].to_numpy())
    x = np.concatenate(xs); y = np.concatenate(ys)
    width  = float(np.max(x) - np.min(x))
    height = float(np.max(y) - np.min(y))
    return max(width*height, 1e-9)

# ---------------------- workers ----------------------
def process_repeat(args_tuple):
    scen_name, rep_dir, dt, R, C, eta_eff, max_msd_lag, save_intermediate, scen_out = args_tuple
    try:
        series = load_time_series(rep_dir)
        area = estimate_domain_area(series)
        t0 = min(series.keys()); nC0 = len(series[t0])
        rho = nC0 / area

        lags, msd = compute_msd(series, max_lag=max_msd_lag, dt=dt)
        D_eff = estimate_D_eff_from_msd(lags, msd)
        e_tick, z_tick, n_eval = compute_tick_eligibility_and_survival(
            series, C,
            save_path=(scen_out / f"overlap_tick_stats_{rep_dir.name}.csv") if save_intermediate else None
        )
        T_enc = T_encounter(rho, R, D_eff, eta_eff=eta_eff)
        B = dt/2.0

        if save_intermediate:
            scen_out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'lag': lags, 'msd': msd}).to_csv(scen_out / f"msd_{rep_dir.name}.csv", index=False)
            measure_coagulation_series(series).to_csv(scen_out / f"coag_series_{rep_dir.name}.csv", index=False)
            pd.DataFrame([{
                'repeat': rep_dir.name, 'D_eff': D_eff, 'rho': rho, 'T_enc': T_enc,
                'e_tick': e_tick, 'zeta_tick': z_tick, 'B_dt_over_2': B, 'pairs_counted': n_eval
            }]).to_csv(scen_out / f"summary_{rep_dir.name}.csv", index=False)

        return {
            'repeat': rep_dir.name, 'D_eff': D_eff, 'rho': rho, 'T_enc': T_enc,
            'e_tick': e_tick, 'zeta_tick': z_tick, 'B_dt_over_2': B, 'pairs_counted': n_eval
        }
    except Exception as ex:
        return {'repeat': rep_dir.name, 'error': repr(ex)}

def process_scenario(scen_dir: Path, run_out: Path, keys: dict, defaults: dict,
                     geom_C: float, eta_eff: float, max_msd_lag: int,
                     workers: int, parallel_by_repeats: bool,
                     save_intermediate: bool, verbose: bool) -> dict:
    name = scen_dir.name
    scen_out = run_out / name
    scen_out.mkdir(parents=True, exist_ok=True)

    # Robust parsing with known keys
    known = [keys[k] for k in ['a_key','p_key','theta_key','kappa_key','dt_key','R_key']]
    params = parse_condition_parameters(name, known_keys=known)

    # Pull params with defaults
    a       = float(params.get(keys['a_key'],     defaults['a']))
    p_merge = float(params.get(keys['p_key'],     defaults['p']))
    dt      = float(params.get(keys['dt_key'],    defaults['dt']))
    R       = float(params.get(keys['R_key'],     defaults['R']))
    theta   = float(params.get(keys['theta_key'], defaults['theta']))
    kappa   = float(params.get(keys['kappa_key'], defaults['kappa']))

    repeat_dirs = sorted([d for d in scen_dir.glob('repeat_*') if d.is_dir()])
    rep_args = [(name, rd, dt, R, geom_C, eta_eff, max_msd_lag, save_intermediate, scen_out) for rd in repeat_dirs]

    rep_results = []
    if parallel_by_repeats and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(process_repeat, arg) for arg in rep_args]
            for f in as_completed(futs):
                rep_results.append(f.result())
    else:
        for arg in rep_args:
            rep_results.append(process_repeat(arg))

    rep_df = pd.DataFrame(rep_results)
    if 'error' in rep_df.columns:
        rep_df = rep_df[~rep_df['error'].astype(str).str.len().fillna(0).gt(0)]

    D_list    = rep_df.get('D_eff',    pd.Series(dtype=float)).astype(float).tolist()
    rho_list  = rep_df.get('rho',      pd.Series(dtype=float)).astype(float).tolist()
    Tenc_list = rep_df.get('T_enc',    pd.Series(dtype=float)).astype(float).tolist()
    e_list    = rep_df.get('e_tick',   pd.Series(dtype=float)).astype(float).tolist()
    z_list    = rep_df.get('zeta_tick',pd.Series(dtype=float)).astype(float).tolist()

    # Aggregate e,zeta with fallbacks for prediction
    e_use = float(np.nanmean(e_list)) if len(e_list) else np.nan
    z_use = float(np.nanmean(z_list)) if len(z_list) else np.nan
    if np.isnan(e_use): e_use = defaults['e']
    if np.isnan(z_use): z_use = defaults['zeta']

    T_enc_mean = float(np.nanmean(Tenc_list)) if Tenc_list else np.nan
    enc_part, attempt_part, T_merge = (np.nan, np.nan, np.nan)
    K_pred = np.nan
    p_star = np.nan
    dominant = None
    attempt_fraction = np.nan

    if np.isfinite(T_enc_mean):
        enc_part, attempt_part, T_merge = K_parts_and_total(T_enc_mean, dt, p_merge, e_use, z_use)  # [1](blob:https://m365.cloud.microsoft/0137585a-c4ef-47a4-9d45-e129859a430f)
        K_pred = K_eff_from_total(T_merge)
        p_star = p_star_general(T_enc_mean, dt/2.0, e_use, z_use)
        if np.isfinite(enc_part) and np.isfinite(attempt_part):
            dominant = "encounter" if enc_part > attempt_part else "attempt"
            denom = enc_part + attempt_part
            attempt_fraction = attempt_part/denom if denom > 0 else np.nan

    rec = {
        **params,
        'scenario': name,
        'a': a, 'p_merge': p_merge, 'dt': dt, 'R': R, 'theta': theta, 'kappa': kappa,
        'geom_C': geom_C, 'eta_eff': eta_eff, 'repeats': int(len(repeat_dirs)),
        'D_eff_empirical': float(np.nanmean(D_list)) if D_list else np.nan,
        'rho_empirical':  float(np.nanmean(rho_list)) if rho_list else np.nan,
        'e_tick_mean':    float(np.nanmean(e_list))  if e_list else np.nan,
        'zeta_tick_mean': float(np.nanmean(z_list))  if z_list else np.nan,
        'T_enc_mean': T_enc_mean,
        'T_enc_part': enc_part,
        'T_attempt_part': attempt_part,
        'attempt_fraction': attempt_fraction,
        'dominant_part': dominant,
        'p_star_mean': p_star,
        'T_merge_pred': T_merge,
        'K_eff_pred_mean': K_pred,
    }

    # Save per-scenario traces
    rep_df.to_csv(scen_out / 'repeat_summaries.csv', index=False)
    pd.DataFrame([rec]).to_csv(scen_out / 'scenario_summary.csv', index=False)
    return rec

# ---------------------- plotting ----------------------
def plot_heatmap(df: pd.DataFrame, outdir: Path, value_col: str, a_col='a', p_col='p_merge'):
    try:
        pvt = df.pivot_table(index=a_col, columns=p_col, values=value_col, aggfunc='mean')
        pvt = pvt.sort_index(axis=0).sort_index(axis=1)
    except Exception:
        return
    plt.figure(figsize=(8, 6.2))
    im = plt.imshow(pvt.values, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(im, label=value_col)
    plt.xlabel(p_col); plt.ylabel(a_col)
    plt.title(f'{value_col} over ({a_col}, {p_col})')
    plt.xticks(range(len(pvt.columns)), [str(c) for c in pvt.columns], rotation=30, ha='right')
    plt.yticks(range(len(pvt.index)), [str(i) for i in pvt.index])
    plt.tight_layout(); plt.savefig(outdir / f'heatmap_{value_col}.png', dpi=200); plt.close()

def plot_slices(df: pd.DataFrame, outdir: Path, a_col='a', p_col='p_merge', y_col='K_eff_pred_mean'):
    # y vs a (coloured by p) and y vs p (coloured by a)
    plt.figure(figsize=(8.5, 5.2))
    for pval, sub in df.groupby(p_col):
        sub = sub.sort_values(a_col)
        plt.plot(sub[a_col], sub[y_col], '-o', label=f'{p_col}={pval}')
    plt.xlabel(a_col); plt.ylabel(y_col)
    plt.title(f'{y_col} vs {a_col} (lines by {p_col})')
    plt.legend(ncol=2, fontsize=8); plt.tight_layout()
    plt.savefig(outdir / f'slice_{y_col}_vs_{a_col}_by_{p_col}.png', dpi=200); plt.close()

    plt.figure(figsize=(8.5, 5.2))
    for aval, sub in df.groupby(a_col):
        sub = sub.sort_values(p_col)
        plt.plot(sub[p_col], sub[y_col], '-o', label=f'{a_col}={aval}')
    plt.xlabel(p_col); plt.ylabel(y_col)
    plt.title(f'{y_col} vs {p_col} (lines by {a_col})')
    plt.legend(ncol=2, fontsize=8); plt.tight_layout()
    plt.savefig(outdir / f'slice_{y_col}_vs_{p_col}_by_{a_col}.png', dpi=200); plt.close()

def plot_scatter_panel(df: pd.DataFrame, outdir: Path):
    plt.figure(figsize=(8.5, 5.5))
    sc = plt.scatter(df['D_eff_empirical'], df['K_eff_pred_mean'],
                     c=df['p_merge'], s=30+20*df['a'], cmap='viridis')
    plt.colorbar(sc, label='p_merge')
    plt.xlabel('D_eff_empirical'); plt.ylabel('K_eff_pred_mean')
    plt.title('Predicted K_eff vs empirical D_eff')
    plt.tight_layout(); plt.savefig(outdir / 'scatter_Keff_vs_Deff.png', dpi=200); plt.close()

def plot_dominance_heatmaps(df: pd.DataFrame, outdir: Path):
    # dominant_part → 1 for encounter, 0 for attempt
    dom = df.copy()
    dom['dom01'] = (dom['dominant_part'] == 'encounter').astype(float)
    plot_heatmap(dom, outdir, 'dom01')                 # heatmap_dominant_part.png
    # attempt fraction heatmap
    plot_heatmap(df, outdir, 'attempt_fraction')       # heatmap_attempt_fraction.png

# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', type=str, required=True)
    ap.add_argument('--run', type=str, default='simulations')

    # Keys in scenario names (robust parser will use these)
    ap.add_argument('--a-key', type=str, default='a')
    ap.add_argument('--p-key', type=str, default='p_merge')
    ap.add_argument('--theta-key', type=str, default='theta')
    ap.add_argument('--kappa-key', type=str, default='kappa')
    ap.add_argument('--dt-key', type=str, default='dt')
    ap.add_argument('--R-key', type=str, default='R')

    # Defaults if not encoded
    ap.add_argument('--a-default', type=float, default=1.0)
    ap.add_argument('--p-default', type=float, default=0.5)
    ap.add_argument('--theta-default', type=float, default=1.0)
    ap.add_argument('--kappa-default', type=float, default=1.0)
    ap.add_argument('--dt-default', type=float, default=1.0)
    ap.add_argument('--R-default', type=float, default=1.0)

    # Geometry + MSD
    ap.add_argument('--geom-C', type=float, default=1.0)
    ap.add_argument('--max-msd-lag', type=int, default=10)

    # Renewal & encounter options
    ap.add_argument('--eta-eff', type=float, default=1.0)
    ap.add_argument('--e-default', type=float, default=0.8)
    ap.add_argument('--zeta-default', type=float, default=1.0)

    # Parallelism & IO
    ap.add_argument('--save-intermediate', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count()//2))
    ap.add_argument('--parallel-by', type=str, choices=['scenarios','repeats','auto'], default='auto',
                    help='Parallelise across scenarios (default) or across repeats per scenario')

    args = ap.parse_args()

    root = Path(args.results_root).resolve()
    if not root.exists():
        raise SystemExit(f"results-root not found: {root}")

    run_dir = root / args.run if (root / args.run).exists() else root
    sim_root = run_dir if run_dir.name == 'simulations' else run_dir / 'simulations'
    outroot = run_dir / 'encounter_attempts_diagnostics'
    outroot.mkdir(exist_ok=True)

    scen_dirs = sorted([p for p in sim_root.iterdir() if p.is_dir()])
    print(f"[INFO] Found {len(scen_dirs)} scenarios under {sim_root}")

    keys = {'a_key': args.a_key, 'p_key': args.p_key, 'theta_key': args.theta_key,
            'kappa_key': args.kappa_key, 'dt_key': args.dt_key, 'R_key': args.R_key}
    defaults = {'a': args.a_default, 'p': args.p_default, 'theta': args.theta_default,
                'kappa': args.kappa_default, 'dt': args.dt_default, 'R': args.R_default,
                'e': args.e_default, 'zeta': args.zeta_default}

    parallel_by_repeats = (args.parallel_by == 'repeats')
    if args.parallel_by == 'auto':
        parallel_by_repeats = (len(scen_dirs) <= 2)

    # Process scenarios
    rows = []
    if not parallel_by_repeats and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_scenario, s, outroot, keys, defaults,
                               args.geom_C, args.eta_eff, args.max_msd_lag,
                               1,  # inner workers for repeats (outer pool handles parallelism)
                               False, args.save_intermediate, args.verbose)
                    for s in scen_dirs]
            for f in as_completed(futs):
                rows.append(f.result())
    else:
        for s in scen_dirs:
            rows.append(process_scenario(s, outroot, keys, defaults,
                                         args.geom_C, args.eta_eff, args.max_msd_lag,
                                         args.workers, True, args.save_intermediate, args.verbose))

    diag = pd.DataFrame(rows)
    diag.to_csv(outroot / 'diagnostics_empirical.csv', index=False)
    print(f"[DONE] diagnostics_empirical.csv -> {outroot / 'diagnostics_empirical.csv'}")

    # Plots
    for col in ['K_eff_pred_mean','e_tick_mean','p_star_mean','D_eff_empirical']:
        plot_heatmap(diag, outroot, col)
    plot_dominance_heatmaps(diag, outroot)           # NEW dominance maps
    plot_slices(diag, outroot, y_col='K_eff_pred_mean')
    plot_scatter_panel(diag, outroot)

if __name__ == '__main__':
    main()