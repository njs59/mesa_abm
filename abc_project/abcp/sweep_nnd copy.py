#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NND-focused *multi-stage* sweep for the Mesa cluster ABM (INV/PRO etc.).

Update in this version
----------------------
- **Per-stage overlays for ALL stats**: after *each* stage, saves
  `overlay_nnd_top5_stage{S}.png`, `overlay_S0_top5_stage{S}.png`,
  `overlay_S1_top5_stage{S}.png`, `overlay_S2_top5_stage{S}.png`.
- Keeps all prior fixes: robust `load_defaults`, resilient `deep_set`, console
  summaries, narrowing, and final overlays for the last stage as well.

Semantics match ABC: S0, S1, S2 and **NND_med with NO spatial wrap**; the code
samples exactly at the `timestep` indices from your observed CSV.
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

# ---- ABM imports (project or local) -----------------------------------------
try:
    from abm.clusters_model import ClustersModel  # package style
except Exception:
    try:
        from clusters_model import ClustersModel  # local file style
    except Exception:
        raise ImportError(
            "Could not import ClustersModel. Ensure clusters_model.py is importable and mesa is installed.")

# -------------------- Summary statistics (match your ABC semantics) -----------

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

# ----------------------- Parameter registry ----------------------------------
@dataclass
class PInfo:
    path: Tuple[str, ...]
    lo: float
    hi: float
    scale: str = "linear"   # "linear" or "log"
    integer: bool = False

REGISTRY: Dict[str, PInfo] = {
    "merge_prob":   PInfo(("merge","prob_contact_merge"), 0.60, 1.00, "linear", False),
    "adhesion":     PInfo(("phenotypes","proliferative","adhesion"), 0.60, 1.00, "linear", False),
    "fragment_rate":PInfo(("phenotypes","proliferative","fragment_rate"), 2e-4, 1.2e-2, "log", False),
    "heading_sigma":PInfo(("movement","heading_sigma"), 0.10, 0.90, "linear", False),
    "dist_s":       PInfo(("movement","dist_params","s"), 0.50, 0.90, "linear", False),
    "dist_scale":   PInfo(("movement","dist_params","scale"), 2.00, 3.50, "linear", False),
    "prolif_rate":  PInfo(("phenotypes","proliferative","prolif_rate"), 0.002, 0.012, "linear", False),
    "init_n_clusters": PInfo(("init","n_clusters"), 500, 1200, "linear", True),
}

# ----------------------- Params helpers --------------------------------------

def deep_set(d: dict, path: Tuple[str, ...], value):
    """Set d[path] = value, creating/replacing intermediate parents as dicts."""
    x = d
    for k in path[:-1]:
        v = x.get(k, None)
        if not isinstance(v, dict):
            v = {}
            x[k] = v
        x = v
    x[path[-1]] = value


def load_defaults(direction: str, speed: str) -> dict:
    DEFAULTS = None
    try:
        from abm.utils import DEFAULTS as D1
        DEFAULTS = D1
    except Exception:
        try:
            from utils import DEFAULTS as D2
            DEFAULTS = D2
        except Exception:
            raise ImportError("Could not import utils. Place utils.py on PYTHONPATH.")

    base = json.loads(json.dumps(DEFAULTS))

    base.setdefault("movement", {})
    base["movement"].setdefault("dist_params", {})

    base["movement"]["mode"] = "distribution"
    base["movement"]["direction"] = direction
    base["movement"]["distribution"] = speed

    base.setdefault("init", {}).setdefault("phenotype", "proliferative")

    return base


def params_from_vector(vec: dict, direction: str, speed: str) -> dict:
    p = load_defaults(direction, speed)
    for name, val in vec.items():
        if name not in REGISTRY:
            continue
        path = REGISTRY[name].path
        if REGISTRY[name].integer:
            val = int(round(val))
        deep_set(p, path, float(val) if not REGISTRY[name].integer else int(val))
    return p

# ----------------------- Sampling & scoring ----------------------------------

def sample_param(rng: np.random.Generator, info: PInfo) -> float:
    if info.scale == "log":
        lo, hi = math.log10(info.lo), math.log10(info.hi)
        x = 10 ** rng.uniform(lo, hi)
    else:
        x = rng.uniform(info.lo, info.hi)
    if info.integer:
        return int(round(x))
    return float(x)


def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, float)
    w = w / (w.sum() if w.sum() > 0 else 1.0)
    e2 = w * np.square(y_pred - y_true)
    return float(np.sqrt(e2.sum()))


def evaluate_candidate(vec: dict, obs_mat: np.ndarray, stats_idx: dict, timesteps: List[int], total_steps: int,
                       direction: str, speed: str,
                       time_weights: np.ndarray, replicates: int, seed_base: int) -> dict:
    sims = []
    for r in range(replicates):
        seed = seed_base + r
        params = params_from_vector(vec, direction, speed)
        sim = simulate_timeseries(params, total_steps=total_steps, sample_steps=tuple(timesteps), seed=seed)
        sims.append(sim)
    sims = np.asarray(sims)  # R x T x 4
    med = np.median(sims, axis=0)  # T x 4

    nnd_true = obs_mat[:, stats_idx['NND_med']]
    nnd_pred = med[:, stats_idx['NND_med']]
    nnd_rmse = weighted_rmse(nnd_true, nnd_pred, time_weights)

    def rel_rrmse(kname: str) -> float:
        y = obs_mat[:, stats_idx[kname]]
        yp = med[:, stats_idx[kname]]
        denom = np.maximum(1e-9, np.abs(y))
        re = (yp - y) / denom
        return float(np.sqrt(np.mean(np.square(re))))

    s0_rrmse = rel_rrmse('S0')
    s1_rrmse = rel_rrmse('S1')
    s2_rrmse = rel_rrmse('S2')

    penalty = 0.0
    for rrmse in (s0_rrmse, s1_rrmse, s2_rrmse):
        if rrmse > 0.10:
            penalty += (rrmse - 0.10) * 10.0
    score = nnd_rmse + penalty

    return {**vec, 'nnd_rmse': nnd_rmse, 's0_rrmse': s0_rrmse, 's1_rrmse': s1_rrmse,
            's2_rrmse': s2_rrmse, 'penalty': penalty, 'score': score}

# ----------------------- Stage engine ----------------------------------------

def narrow_bounds(top_df: pd.DataFrame, infos: Dict[str, PInfo], pad_frac: float = 0.10) -> Dict[str, PInfo]:
    new = {}
    for name, info in infos.items():
        if name not in top_df.columns:
            new[name] = info
            continue
        lo = float(top_df[name].quantile(0.10))
        hi = float(top_df[name].quantile(0.90))
        if info.integer:
            lo_i, hi_i = int(round(lo)), int(round(hi))
            if hi_i <= lo_i:
                lo_i, hi_i = int(math.floor(info.lo)), int(math.ceil(info.hi))
            lo_p = max(info.lo, lo_i)
            hi_p = min(info.hi, hi_i)
            new[name] = PInfo(info.path, lo_p, hi_p, info.scale, True)
        else:
            span = max(1e-12, hi - lo)
            lo_p = max(info.lo, lo - pad_frac * span)
            hi_p = min(info.hi, hi + pad_frac * span)
            new[name] = PInfo(info.path, lo_p, hi_p, info.scale, False)
    return new


def print_stage_summary(stage_idx: int, df: pd.DataFrame, chosen: List[str], outdir: Path):
    best = df.nsmallest(1, 'score').iloc[0]
    msg = []
    msg.append(f"\n===== STAGE {stage_idx} =====")
    msg.append(f"Evaluated: {len(df)} candidates")
    msg.append(f"Best score: {best['score']:.3f}  |  NND RMSE: {best['nnd_rmse']:.3f}  |  Penalty: {best['penalty']:.3f}")
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
                        direction: str, speed: str, outdir: Path, seed: int):
    """Save overlays for NND_med, S0, S1, S2 using the top-5 of this stage."""
    top5 = df_stage.nsmallest(5, 'score')
    sims = []
    for j, (_, row) in enumerate(top5.iterrows()):
        v = {n: float(row[n]) for n in chosen if n in row.index}
        sim = simulate_timeseries(
            params_from_vector(v, direction, speed),
            total_steps=int(max(timesteps)),
            sample_steps=tuple(timesteps),
            seed=seed*777 + stage_idx*10 + j,
        )
        sims.append(sim)
    if not sims:
        return
    sims = np.asarray(sims)  # 5 x T x 4

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
        plt.savefig(outdir / fname, dpi=160)
        plt.close()

    _overlay('NND_med', f'overlay_nnd_top5_stage{stage_idx}.png', 'NND_med')
    _overlay('S0',      f'overlay_S0_top5_stage{stage_idx}.png',      'S0')
    _overlay('S1',      f'overlay_S1_top5_stage{stage_idx}.png',      'S1')
    _overlay('S2',      f'overlay_S2_top5_stage{stage_idx}.png',      'S2')

# ----------------------- CLI --------------------------------------------------

def parse_cli():
    ap = argparse.ArgumentParser(description="Multi-stage NND sweep with per-stage overlays for NND,S0,S1,S2")
    ap.add_argument('--observed_ts', type=str, required=True)
    ap.add_argument('--direction', type=str, choices=['isotropic','persistent'], default='persistent')
    ap.add_argument('--speed', type=str, choices=['lognorm','gamma','weibull','expon','rayleigh','invgauss','constant'], default='lognorm')
    ap.add_argument('--total_steps', type=int, default=300)
    ap.add_argument('--params', nargs='+', default=['merge_prob','adhesion','fragment_rate','heading_sigma'])
    ap.add_argument('--stages', type=int, default=2)
    ap.add_argument('--n_per_stage', nargs='+', type=int, default=None)
    ap.add_argument('--replicates', type=int, default=4)
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count()//2))
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--time_power', type=float, default=2.0)
    ap.add_argument('--bound', nargs=3, action='append', metavar=('NAME','LO','HI'), default=None)
    ap.add_argument('--outdir', type=str, default='results/sweep_nnd_plus')
    return ap.parse_args()

# ----------------------- Main -------------------------------------------------

def main():
    args = parse_cli()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    obs_df = pd.read_csv(args.observed_ts)
    needed = ['timestep','S0','S1','S2','NND_med']
    for c in needed:
        if c not in obs_df.columns:
            raise ValueError(f'Missing column {c} in observed_ts')
    timesteps = obs_df['timestep'].astype(int).to_list()
    stats = ['S0','S1','S2','NND_med']
    stats_idx = {s:i for i,s in enumerate(stats)}
    obs_mat = obs_df[stats].to_numpy(float)

    # Time weights
    t = np.asarray(timesteps, float)
    w = (t - t.min() + 1.0) ** float(args.time_power)
    w = w / w.sum()

    # Build info map
    chosen = []
    infos: Dict[str,PInfo] = {}
    for n in args.params:
        if n not in REGISTRY:
            raise KeyError(f'Unknown parameter short-name: {n}')
        if n == 'heading_sigma' and args.direction != 'persistent':
            continue
        chosen.append(n)
        infos[n] = PInfo(**REGISTRY[n].__dict__)

    if args.bound:
        for (name, lo, hi) in args.bound:
            if name not in infos:
                raise KeyError(f'--bound specified for {name} which is not in --params')
            lo_f, hi_f = float(lo), float(hi)
            pi = infos[name]
            infos[name] = PInfo(pi.path, lo_f, hi_f, pi.scale, pi.integer)

    # Stage sizes
    if args.n_per_stage is None:
        stage_sizes = [120]*int(args.stages)
    else:
        stage_sizes = args.n_per_stage
        if len(stage_sizes) < args.stages:
            stage_sizes = stage_sizes + [stage_sizes[-1]]*(args.stages - len(stage_sizes))

    rng = np.random.default_rng(args.seed)

    # Console header
    print("\nMulti-stage NND sweep starting...")
    print(f"Observed steps: {len(timesteps)} (from {min(timesteps)} to {max(timesteps)})")
    print(f"Chosen params: {', '.join(chosen)}")
    print(f"Stages: {args.stages} | per-stage: {stage_sizes} | replicates: {args.replicates} | workers: {args.workers}")

    all_rows = []
    current_infos = infos

    for s_idx, N in enumerate(stage_sizes, start=1):
        t0 = time.time()
        # Sample vectors
        vecs: List[dict] = []
        for _ in range(N):
            v = {name: sample_param(rng, pi) for name, pi in current_infos.items()}
            vecs.append(v)

        # Prepare jobs
        jobs = []
        for i, v in enumerate(vecs):
            jobs.append((v, obs_mat, stats_idx, timesteps, args.total_steps,
                         args.direction, args.speed, w, args.replicates, args.seed*1000 + s_idx*100 + i))

        # Evaluate
        if args.workers <= 1:
            results = [evaluate_candidate(*j) for j in jobs]
        else:
            with mp.Pool(processes=args.workers) as pool:
                results = pool.starmap(evaluate_candidate, jobs)

        df = pd.DataFrame(results)
        df.insert(0, 'stage', s_idx)
        df.to_csv(outdir / f'candidates_stage{s_idx}.csv', index=False)
        all_rows.append(df)

        # Top-k and summary
        topk = df.nsmallest(max(10, int(0.1*len(df))), 'score')
        topk.to_csv(outdir / f'topk_stage{s_idx}.csv', index=False)
        print_stage_summary(s_idx, df, chosen, outdir)

        # Per-stage overlays for NND,S0,S1,S2
        save_stage_overlays(s_idx, df, chosen, timesteps, obs_mat, stats_idx,
                            args.direction, args.speed, outdir, args.seed)

        # Narrow bounds
        if s_idx < len(stage_sizes):
            current_infos = narrow_bounds(topk, current_infos, pad_frac=0.10)
            print("Next-stage bounds:")
            for name, pi in current_infos.items():
                rng_type = "int" if pi.integer else ("log" if pi.scale=="log" else "lin")
                print(f"  - {name:>15} [{pi.lo:.6g}, {pi.hi:.6g}]  ({rng_type})")
            print()

        dt = time.time() - t0
        print(f"Stage {s_idx} wall time: {dt/60:.2f} min\n")

    # Collate
    all_df = pd.concat(all_rows, axis=0, ignore_index=True)
    all_df.to_csv(outdir / 'sweep_all_candidates.csv', index=False)

    # Final overlays from the last stage
    final_stage = len(stage_sizes)
    df_last = all_df[all_df['stage']==final_stage]
    save_stage_overlays(final_stage, df_last, chosen, timesteps, obs_mat, stats_idx,
                        args.direction, args.speed, outdir, args.seed)

    # Store best vector overall
    best_overall = all_df.nsmallest(1, 'score').iloc[0]
    best_vec = {n: float(best_overall[n]) for n in chosen if n in best_overall.index}
    Path(outdir / 'best_params.json').write_text(json.dumps(best_vec, indent=2) + "\n", encoding='utf-8')

    print("All done. Results in:", outdir.resolve())


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
