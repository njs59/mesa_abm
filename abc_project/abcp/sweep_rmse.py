#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-stage sweep with **relative RMSE** on all four outputs (S0, S1, S2, NND_med),
with **equal weighting across all timepoints**.

Score definition (minimise):
  SCORE = mean( rRMSE_S0, rRMSE_S1, rRMSE_S2, rRMSE_NND )
where, for a series k(t),
  rRMSE_k = sqrt( mean_t( ((k_hat(t) - k(t)) / max(|k(t)|, 1e-9))^2 ) )

Other features kept from your sweeps:
- Multi-stage narrowing (10–90% quantiles with padding) & console summaries
- Per-stage overlays for NND, S0, S1, S2
- Final 95% CI overlays at best parameters (optional)
- Robust param tree handling and optional integer parameters
"""
from __future__ import annotations
import argparse, json, os, math, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp

# ---- ABM imports -------------------------------------------------------------
try:
    from abm.clusters_model import ClustersModel  # package style
except Exception:
    try:
        from clusters_model import ClustersModel  # local file style
    except Exception:
        raise ImportError("Could not import ClustersModel. Ensure clusters_model.py is importable and mesa is installed.")

# ---- Stats -------------------------------------------------------------------
def _plain_pairwise_dists(pos: np.ndarray) -> np.ndarray:
    if pos.size == 0:
        return np.empty((0, 0), dtype=float)
    diffs = pos[:, None, :] - pos[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2))

def compute_snapshot_summaries(model) -> Dict[str, float]:
    ids, pos, radii, sizes, speeds = model._log_alive_snapshot()  # noqa: SLF001
    n = len(sizes)
    if n == 0:
        return {"S0": 0.0, "S1": 0.0, "S2": 0.0, "NND_med": 0.0}
    S0 = float(n)
    S1 = float(np.mean(sizes))
    S2 = float(np.mean(np.square(sizes)))
    D = _plain_pairwise_dists(pos)
    if D.size == 0:
        nnd_med = 0.0
    else:
        D_no_self = D + np.eye(n) * 1e9
        nnd = D_no_self.min(axis=1)
        nnd_med = float(np.median(nnd))
    return {"S0": S0, "S1": S1, "S2": S2, "NND_med": nnd_med}

def simulate_timeseries(params: dict, total_steps: int, sample_steps: Tuple[int, ...], seed: int = 777) -> np.ndarray:
    m = ClustersModel(params=params, seed=seed)
    K = 4  # S0,S1,S2,NND_med
    T = len(sample_steps)
    out = np.zeros((T, K), dtype=float)
    current = 0
    steps_set = set(sample_steps)
    for step in range(total_steps + 1):
        if step in steps_set:
            s = compute_snapshot_summaries(m)
            out[current, :] = [s["S0"], s["S1"], s["S2"], s["NND_med"]]
            current += 1
            if current >= T:
                break
        if step < total_steps:
            m.step()
    return out

# ---- Parameter registry ------------------------------------------------------
@dataclass
class PInfo:
    path: Tuple[str, ...]
    lo: float
    hi: float
    scale: str = "linear"   # "linear" or "log"
    integer: bool = False

REGISTRY: Dict[str, PInfo] = {
    "merge_prob":   PInfo(("merge","prob_contact_merge"), 0.60, 1.00, "linear", False),
    "adhesion":     PInfo(("phenotypes","proliferative","adhesion"), 0.55, 1.00, "linear", False),
    "fragment_rate":PInfo(("phenotypes","proliferative","fragment_rate"), 2e-4, 1.5e-2, "log", False),
    "heading_sigma":PInfo(("movement","heading_sigma"), 0.10, 0.90, "linear", False),
    "dist_s":       PInfo(("movement","dist_params","s"), 0.40, 0.95, "linear", False),
    "dist_scale":   PInfo(("movement","dist_params","scale"), 1.50, 3.80, "linear", False),
    "prolif_rate":  PInfo(("phenotypes","proliferative","prolif_rate"), 0.002, 0.012, "linear", False),
    "init_n_clusters": PInfo(("init","n_clusters"), 400, 1400, "linear", True),
}

# ---- Helpers -----------------------------------------------------------------
def deep_set(d: dict, path: Tuple[str, ...], value):
    x = d
    for k in path[:-1]:
        v = x.get(k, None)
        if not isinstance(v, dict):
            v = {}
            x[k] = v
        x = v
    x[path[-1]] = value

def load_defaults(direction: str, speed: str) -> dict:
    try:
        from abm.utils import DEFAULTS as D
    except Exception:
        try:
            from utils import DEFAULTS as D
        except Exception:
            raise ImportError("Could not import utils. Place utils.py on PYTHONPATH.")
    base = json.loads(json.dumps(D))
    base.setdefault("movement", {}).setdefault("dist_params", {})
    base["movement"]["mode"] = "distribution"
    base["movement"]["direction"] = direction
    base["movement"]["distribution"] = speed
    base.setdefault("init", {}).setdefault("phenotype", "proliferative")
    return base

def params_from_vector(vec: dict, direction: str, speed: str) -> dict:
    p = load_defaults(direction, speed)
    for name, val in vec.items():
        if name not in REGISTRY: continue
        path = REGISTRY[name].path
        if REGISTRY[name].integer:
            val = int(round(val))
        deep_set(p, path, float(val) if not REGISTRY[name].integer else int(val))
    return p

# ---- Scoring (equal-time relative RMSE for all stats) -----------------------
def rel_rmse_equal_time(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(1e-9, np.abs(y_true))
    re = (y_pred - y_true) / denom
    return float(np.sqrt(np.mean(np.square(re))))

def evaluate_candidate(vec: dict, obs_mat: np.ndarray, stats_idx: dict, timesteps: List[int], total_steps: int,
                       direction: str, speed: str,
                       replicates: int, seed_base: int) -> dict:
    sims = []
    for r in range(replicates):
        seed = seed_base + r
        params = params_from_vector(vec, direction, speed)
        sim = simulate_timeseries(params, total_steps=total_steps, sample_steps=tuple(timesteps), seed=seed)
        sims.append(sim)
    sims = np.asarray(sims)  # R x T x 4
    med = np.median(sims, axis=0)  # T x 4

    # relative RMSE for each stat (equal-time)
    r_s0  = rel_rmse_equal_time(obs_mat[:, stats_idx['S0']],      med[:, stats_idx['S0']])
    r_s1  = rel_rmse_equal_time(obs_mat[:, stats_idx['S1']],      med[:, stats_idx['S1']])
    r_s2  = rel_rmse_equal_time(obs_mat[:, stats_idx['S2']],      med[:, stats_idx['S2']])
    r_nnd = rel_rmse_equal_time(obs_mat[:, stats_idx['NND_med']], med[:, stats_idx['NND_med']])

    score = float(np.mean([r_s0, r_s1, r_s2, r_nnd]))
    return {**vec, 'r_s0': r_s0, 'r_s1': r_s1, 'r_s2': r_s2, 'r_nnd': r_nnd, 'score': score}

# ---- Narrowing & overlays ----------------------------------------------------
@dataclass
class PInfoBox:
    path: Tuple[str, ...]
    lo: float
    hi: float
    scale: str
    integer: bool

def narrow_bounds(top_df: pd.DataFrame, infos: Dict[str, PInfo], pad_frac: float = 0.10) -> Dict[str, PInfo]:
    new = {}
    best = top_df.nsmallest(1, 'score').iloc[0]
    for name, info in infos.items():
        if name not in top_df.columns:
            new[name] = info
            continue
        lo_q = float(top_df[name].quantile(0.10))
        hi_q = float(top_df[name].quantile(0.90))
        best_val = float(best[name]) if name in best.index else None
        if info.integer:
            lo_i, hi_i = int(round(lo_q)), int(round(hi_q))
            if hi_i <= lo_i:
                lo_i, hi_i = int(math.floor(info.lo)), int(math.ceil(info.hi))
            pad = max(1, int(round(0.02 * max(1, hi_i - lo_i))))
            lo_i -= pad; hi_i += pad
            if best_val is not None:
                b = int(round(best_val))
                lo_i = min(lo_i, b); hi_i = max(hi_i, b)
            lo_p = max(int(math.floor(info.lo)), lo_i)
            hi_p = min(int(math.ceil(info.hi)),  hi_i)
            new[name] = PInfo(info.path, lo_p, hi_p, info.scale, True)
        else:
            span = max(1e-12, hi_q - lo_q)
            lo_p = max(info.lo, lo_q - pad_frac * span)
            hi_p = min(info.hi, hi_q + pad_frac * span)
            if best_val is not None:
                lo_p = min(lo_p, best_val)
                hi_p = max(hi_p, best_val)
            new[name] = PInfo(info.path, lo_p, hi_p, info.scale, False)
    return new

def print_stage_summary(stage_idx: int, df: pd.DataFrame, chosen: List[str], outdir: Path):
    best = df.nsmallest(1, 'score').iloc[0]
    msg = []
    msg.append(f"\n===== STAGE {stage_idx} =====")
    msg.append(f"Evaluated: {len(df)} candidates")
    msg.append(f"Best score: {best['score']:.4f}  |  rS0: {best['r_s0']:.3f}  |  rS1: {best['r_s1']:.3f}  |  rS2: {best['r_s2']:.3f}  |  rNND: {best['r_nnd']:.3f}")
    msg.append("Best params:")
    for n in chosen:
        if n in best.index:
            val = best[n]
            msg.append(f"  - {n:>15}: {val:.6g}")
    text = "\n".join(msg)
    print(text)
    (outdir / f"stage_{stage_idx}_summary.txt").write_text(text + "\n", encoding='utf-8')

def save_stage_overlays(stage_idx: int, df_stage: pd.DataFrame, chosen: List[str],
                        timesteps: List[int], obs_mat: np.ndarray, stats_idx: dict,
                        direction: str, speed: str, outdir: Path, seed: int, total_steps: int):
    top5 = df_stage.nsmallest(5, 'score')
    sims = []
    for j, (_, row) in enumerate(top5.iterrows()):
        v = {n: float(row[n]) for n in chosen if n in row.index}
        sim = simulate_timeseries(params_from_vector(v, direction, speed),
                                  total_steps=total_steps, sample_steps=tuple(timesteps),
                                  seed=seed*777 + stage_idx*10 + j)
        sims.append(sim)
    if not sims: return
    sims = np.asarray(sims)

    def _overlay(stat_key: str, fname: str, ylabel: str):
        plt.figure(figsize=(10,5))
        plt.plot(timesteps, obs_mat[:, stats_idx[stat_key]], color='k', lw=1.8, label='observed')
        for j in range(len(sims)):
            plt.plot(timesteps, sims[j,:, stats_idx[stat_key]], lw=1.2, alpha=0.9, label=f'candidate {j+1}')
        plt.xlabel('timestep'); plt.ylabel(ylabel)
        title_name = 'NND' if stat_key=='NND_med' else stat_key
        plt.title(f'{title_name} overlay (observed vs top-5, stage {stage_idx})')
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=160); plt.close()

    _overlay('NND_med', f'overlay_nnd_top5_stage{stage_idx}.png', 'NND_med')
    _overlay('S0',      f'overlay_S0_top5_stage{stage_idx}.png',      'S0')
    _overlay('S1',      f'overlay_S1_top5_stage{stage_idx}.png',      'S1')
    _overlay('S2',      f'overlay_S2_top5_stage{stage_idx}.png',      'S2')

# ---- Final CI ---------------------------------------------------------------
def save_final_uncertainty(best_vec: dict, timesteps: List[int], obs_mat: np.ndarray, stats_idx: dict,
                           direction: str, speed: str, total_steps: int,
                           outdir: Path, ci_reps: int, ci_seed: int):
    T = len(timesteps); K = 4
    sims = np.zeros((ci_reps, T, K), dtype=float)
    for r in range(ci_reps):
        sim = simulate_timeseries(params_from_vector(best_vec, direction, speed),
                                  total_steps=total_steps, sample_steps=tuple(timesteps),
                                  seed=ci_seed + r)
        sims[r] = sim

    # CSV ensemble
    rows = []
    for r in range(ci_reps):
        for i, t in enumerate(timesteps):
            s0, s1, s2, nnd = sims[r, i]
            rows.append({"replicate": r, "timestep": t, "S0": s0, "S1": s1, "S2": s2, "NND_med": nnd})
    pd.DataFrame(rows).to_csv(outdir / 'final_CI_ensemble.csv', index=False)

    def stats_band(arr):
        med = np.median(arr, axis=0)
        lo = np.percentile(arr, 2.5, axis=0)
        hi = np.percentile(arr, 97.5, axis=0)
        return med, lo, hi

    labels = [('S0','final_CI_S0.png','S0'),
              ('S1','final_CI_S1.png','S1'),
              ('S2','final_CI_S2.png','S2'),
              ('NND_med','final_CI_NND.png','NND_med')]

    for key, fname, ylabel in labels:
        k = stats_idx[key]
        med, lo, hi = stats_band(sims[:,:,k])
        plt.figure(figsize=(10,5))
        plt.plot(timesteps, obs_mat[:, k], color='k', lw=1.8, label='observed')
        plt.fill_between(timesteps, lo, hi, color='tab:blue', alpha=0.25, label='95% CI (best params)')
        plt.plot(timesteps, med, color='tab:blue', lw=2.0, label='median (best params)')
        plt.xlabel('timestep'); plt.ylabel(ylabel)
        title = 'NND' if key=='NND_med' else key
        plt.title(f'{title}: observed vs median±95% CI (best parameters)')
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=160); plt.close()

# ---- CLI & main -------------------------------------------------------------
def parse_cli():
    ap = argparse.ArgumentParser(description="Multi-stage sweep with equal-time relative RMSE over all 4 outputs")
    ap.add_argument('--observed_ts', type=str, required=True)
    ap.add_argument('--direction', type=str, choices=['isotropic','persistent'], default='persistent')
    ap.add_argument('--speed', type=str, choices=['lognorm','gamma','weibull','expon','rayleigh','invgauss','constant'], default='lognorm')
    ap.add_argument('--total_steps', type=int, default=300)
    ap.add_argument('--params', nargs='+', default=['merge_prob','adhesion','fragment_rate','heading_sigma','dist_s','dist_scale','init_n_clusters'])
    ap.add_argument('--stages', type=int, default=3)
    ap.add_argument('--n_per_stage', nargs='+', type=int, default=None)
    ap.add_argument('--replicates', type=int, default=4)
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count()//2))
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--bound', nargs=3, action='append', metavar=('NAME','LO','HI'), default=None,
                    help='Override bounds, e.g., --bound fragment_rate 0.0004 0.02')
    # Final CI
    ap.add_argument('--final_ci_replicates', type=int, default=50,
                    help='Number of simulations at best params for 95%% CI (default: 50)')
    ap.add_argument('--ci_seed', type=int, default=424200,
                    help='Base seed for final CI ensemble (default: 424200)')
    ap.add_argument('--outdir', type=str, default='results/sweep_relrmse_equal')
    return ap.parse_args()

def main():
    args = parse_cli()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    obs_df = pd.read_csv(args.observed_ts)
    needed = ['timestep','S0','S1','S2','NND_med']
    for c in needed:
        if c not in obs_df.columns:
            raise ValueError(f'Missing column {c} in observed_ts')
    timesteps = obs_df['timestep'].astype(int).to_list()
    stats = ['S0','S1','S2','NND_med']
    stats_idx = {s:i for i,s in enumerate(stats)}
    obs_mat = obs_df[stats].to_numpy(float)

    chosen, infos = [], {}
    for n in args.params:
        if n not in REGISTRY:
            raise KeyError(f'Unknown parameter: {n}')
        if n == 'heading_sigma' and args.direction != 'persistent':
            continue
        chosen.append(n)
        infos[n] = PInfo(**REGISTRY[n].__dict__)

    if args.bound:
        for (name, lo, hi) in args.bound:
            if name not in infos:
                raise KeyError(f'--bound specified for {name} which is not in --params')
            pi = infos[name]
            infos[name] = PInfo(pi.path, float(lo), float(hi), pi.scale, pi.integer)

    if args.n_per_stage is None:
        stage_sizes = [120]*int(args.stages)
    else:
        stage_sizes = args.n_per_stage
        if len(stage_sizes) < args.stages:
            stage_sizes = stage_sizes + [stage_sizes[-1]]*(args.stages - len(stage_sizes))

    rng = np.random.default_rng(args.seed)

    print("\nMulti-stage equal-time relative-RMSE sweep starting...")
    print(f"Observed steps: {len(timesteps)} (from {min(timesteps)} to {max(timesteps)})")
    print(f"Chosen params: {', '.join(chosen)}")
    print(f"Stages: {args.stages} | per-stage: {stage_sizes} | replicates: {args.replicates} | workers: {args.workers}")

    all_rows = []
    current_infos = infos

    for s_idx, N in enumerate(stage_sizes, start=1):
        t0 = time.time()
        vecs: List[dict] = []
        for _ in range(N):
            v = {}
            for name, pi in current_infos.items():
                if pi.scale == "log":
                    lo, hi = math.log10(pi.lo), math.log10(pi.hi)
                    x = 10 ** rng.uniform(lo, hi)
                else:
                    x = rng.uniform(pi.lo, pi.hi)
                v[name] = int(round(x)) if pi.integer else float(x)
            vecs.append(v)

        jobs = []
        for i, v in enumerate(vecs):
            jobs.append((v, obs_mat, stats_idx, timesteps, args.total_steps,
                         args.direction, args.speed, args.replicates,
                         args.seed*1000 + s_idx*100 + i))

        if args.workers <= 1:
            results = [evaluate_candidate(*j) for j in jobs]
        else:
            with mp.Pool(processes=args.workers) as pool:
                results = pool.starmap(evaluate_candidate, jobs)

        df = pd.DataFrame(results); df.insert(0, 'stage', s_idx)
        df.to_csv(outdir / f'candidates_stage{s_idx}.csv', index=False)
        all_rows.append(df)

        topk = df.nsmallest(max(10, int(0.1*len(df))), 'score')
        topk.to_csv(outdir / f'topk_stage{s_idx}.csv', index=False)

        print_stage_summary(s_idx, df, chosen, outdir)
        save_stage_overlays(s_idx, df, chosen, timesteps, obs_mat, stats_idx,
                            args.direction, args.speed, outdir, args.seed, args.total_steps)

        if s_idx < len(stage_sizes):
            current_infos = narrow_bounds(topk, current_infos, pad_frac=0.10)
            print("Next-stage bounds:")
            for name, pi in current_infos.items():
                rng_type = "int" if pi.integer else ("log" if pi.scale=="log" else "lin")
                print(f"  - {name:>15} [{pi.lo:.6g}, {pi.hi:.6g}]  ({rng_type})")
            print()

        dt = time.time() - t0
        print(f"Stage {s_idx} wall time: {dt/60:.2f} min\n")

    all_df = pd.concat(all_rows, axis=0, ignore_index=True)
    all_df.to_csv(outdir / 'sweep_all_candidates.csv', index=False)

    best_overall = all_df.nsmallest(1, 'score').iloc[0]
    best_vec = {n: float(best_overall[n]) for n in chosen if n in best_overall.index}
    Path(outdir / 'best_params.json').write_text(json.dumps(best_vec, indent=2) + "\n", encoding='utf-8')

    if args.final_ci_replicates > 0:
        print(f"Running final CI ensemble: {args.final_ci_replicates} replicates at best params...")
        save_final_uncertainty(best_vec, timesteps, obs_mat, stats_idx,
                               args.direction, args.speed, args.total_steps,
                               outdir, args.final_ci_replicates, args.ci_seed)
        print("Saved final CI figures and ensemble CSV.")

    print("All done. Results in:", outdir.resolve())

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
