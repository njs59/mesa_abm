#!/usr/bin/env python3
"""
Fast ABC-SMC runner for the ABM.

Features
--------
• Strict timestep validation & alignment against observed CSV
• Phenotype-selectable seeding; optional g(r) statistics
• Robust epsilon export (supports 'epsilon' and legacy 'eps')
• Progress bars via pyabc's built-in tqdm (per population)
• Pre-run point estimate (default: naive prior mean) simulated once before ABC
• Post-run point estimate (MAP/mean/median) simulated once after ABC
• Faster simulation by disabling per-step logging and sampling on demand
"""
from __future__ import annotations
import os
import sys
import json
import copy
import datetime
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import pyabc
from sklearn.preprocessing import MaxAbsScaler

# Ensure project root for abm/abcp imports
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from abm.clusters_model import ClustersModel
from abm.utils import DEFAULTS
from abcp.compute_summary import simulate_timeseries  # fast path (disable_logging=True)
from abcp.priors import load_priors
from abcp.abc_model_wrapper import particle_to_params


# ----------------------------- Config dataclass -----------------------------
@dataclass
class RunConfig:
    observed_csv: str
    priors_yaml: str
    total_steps: int
    timesteps: List[int]
    stats: List[str]
    motion: str
    speed: str
    popsize: int
    maxgen: int
    min_eps: float
    seed: int
    workers: int
    run_tag: str
    phenotype: str
    point_estimator: str   # 'map' | 'mean' | 'median' (post-run)
    show_progress: bool

    # Pre-run point estimate (run BEFORE ABC; default: prior_mean)
    pre_point_mode: str       # 'none' | 'prior_mean' | 'prior_median' | 'posterior_map' | 'posterior_mean' | 'posterior_median' | 'file'
    pre_point_run: Optional[str]
    pre_point_file: Optional[str]


# ----------------------------- Helpers -----------------------------
def _normalize_stats(stats_in: List[str]) -> List[str]:
    """Normalize input stat names to the canonical compute_summary columns."""
    mapping = {
        's0': 'S0', 's1': 'S1', 's2': 'S2',
        'nnd_med': 'NND_med', 'nndmed': 'NND_med', 'nnd': 'NND_med',
        'g_r40': 'g_r40', 'g_r80': 'g_r80'
    }
    return [mapping.get(str(s).strip().lower(), s) for s in stats_in]


def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def ensure_outdirs(tag: str) -> Path:
    out_dir = THIS_DIR / 'outputs' / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'figs').mkdir(exist_ok=True)
    return out_dir


def make_model_factory(seed: int = 42):
    def factory(params_dict):
        merged = copy.deepcopy(DEFAULTS)
        if params_dict:
            _deep_update(merged, params_dict)
        return ClustersModel(params=merged, seed=seed)
    return factory


def _read_posterior_run(run_dir: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load posterior particles & weights from a previous run folder."""
    df = pd.read_csv(run_dir / 'posterior_particles.csv')
    w = pd.read_csv(run_dir / 'posterior_weights.csv')['weight'].to_numpy(float)
    w = np.clip(w, 0, np.inf)
    if w.sum() <= 0:
        raise ValueError("Posterior weights sum to zero.")
    return df, (w / w.sum())


def _prior_center(prior: pyabc.Distribution, which: str = 'mean') -> Dict[str, float]:
    """
    Return per-parameter centres from a pyabc.Distribution.
    Use analytic mean/median if available, else estimate by sampling.
    """
    out: Dict[str, float] = {}
    for k, rv in prior.items():
        try:
            val = float(rv.mean() if which == 'mean' else getattr(rv, 'median', lambda: rv.ppf(0.5))())
        except Exception:
            samp = rv.rvs(size=20000)
            val = float(np.mean(samp)) if which == 'mean' else float(np.median(samp))
        out[k] = val
    return out


# ----------------------------- Runner -----------------------------
def run_abc(cfg: RunConfig) -> Path:
    out_dir = ensure_outdirs(cfg.run_tag)
    (out_dir / 'CONFIG.json').write_text(json.dumps(asdict(cfg), indent=2), encoding='utf-8')

    # ----- Load & validate observed -----
    obs_df = pd.read_csv(cfg.observed_csv)
    if 'timestep' not in obs_df.columns:
        raise ValueError("Observed CSV must contain a 'timestep' column")  # required

    missing = [s for s in cfg.stats if s not in obs_df.columns]
    if missing:
        raise ValueError(f"Observed CSV is missing required columns: {missing}")

    # Timesteps: strictly increasing and within total_steps
    timesteps = list(map(int, cfg.timesteps)) if cfg.timesteps else obs_df['timestep'].astype(int).tolist()
    if any(t <= 0 for t in timesteps):
        raise ValueError(f"Timesteps must be positive integers, got: {timesteps[:8]}...")
    if any(t1 >= t2 for t1, t2 in zip(timesteps, timesteps[1:])):
        raise ValueError(f"Timesteps must be strictly increasing, got: {timesteps[:8]}...")
    if max(timesteps) > int(cfg.total_steps):
        raise ValueError(f"Max timestep ({max(timesteps)}) exceeds total_steps ({cfg.total_steps}).")

    # Align observed rows to chosen timesteps (exact order)
    obs_sorted = obs_df.sort_values('timestep').reset_index(drop=True)
    ts_set = set(timesteps)
    obs_keep = obs_sorted[obs_sorted['timestep'].isin(ts_set)].copy()
    obs_keep['_order'] = obs_keep['timestep'].apply(lambda v: timesteps.index(int(v)))
    obs_keep = obs_keep.sort_values('_order').drop(columns=['_order'])
    obs_mat = obs_keep[cfg.stats].to_numpy(float)  # (T, K)
    T, K = obs_mat.shape

    # ----- Scaling on observed only -----
    scaler = MaxAbsScaler().fit(obs_mat)
    obs_scaled_vec = scaler.transform(obs_mat).flatten()

    # ----- Precompute column indices once (speed) -----
    full_order = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
    sel_idx = [full_order.index(s) for s in cfg.stats]

    # ----- Simulator wrapper -----
    model_factory = make_model_factory(seed=cfg.seed)

    def abm_model(particle):
        params = particle_to_params(particle, motion=cfg.motion, speed_dist=cfg.speed)
        sim_mat = simulate_timeseries(
            model_factory=model_factory,
            params=params,
            total_steps=cfg.total_steps,
            sample_steps=tuple(timesteps),
            init_phenotype=cfg.phenotype,
            force_phase2=True,
            disable_logging=True,  # fast path
        )
        sim_sel = sim_mat[:, sel_idx]
        sim_scaled_vec = scaler.transform(sim_sel).flatten()
        return {f"y_{i}": float(v) for i, v in enumerate(sim_scaled_vec)}

    observation = {f"y_{i}": float(v) for i, v in enumerate(obs_scaled_vec)}

    def l2_distance(sim, obs):
        sim_v = np.array([sim[f"y_{i}"] for i in range(T * K)], float)
        obs_v = np.array([obs[f"y_{i}"] for i in range(T * K)], float)
        return float(np.linalg.norm(sim_v - obs_v))

    # ----- Multicore sampler with pyabc progress bars -----
    try:
        from pyabc.sampler import MulticoreEvalParallelSampler
        sampler = MulticoreEvalParallelSampler(
            n_procs=cfg.workers,
            show_progress=cfg.show_progress,   # <-- built-in tqdm
        )
    except Exception:
        sampler = None  # fallback to default sampler (still works, without pyabc bar)

    # ----- PRE-RUN point estimate (default: prior_mean) -----
    def _write_pre_point_outputs(pe_params: Dict[str, float], label: str):
        (out_dir / 'pre_point_estimate_params.json').write_text(
            json.dumps({"label": label, "params": pe_params}, indent=2), encoding='utf-8'
        )
        pe_abm_params = particle_to_params(pe_params, motion=cfg.motion, speed_dist=cfg.speed)
        sim_mat_pe = simulate_timeseries(
            model_factory=make_model_factory(seed=cfg.seed + 777),
            params=pe_abm_params,
            total_steps=cfg.total_steps,
            sample_steps=tuple(timesteps),
            init_phenotype=cfg.phenotype,
            force_phase2=True,
            disable_logging=True,
        )
        sim_sel = sim_mat_pe[:, sel_idx]
        pe_df = pd.DataFrame(sim_sel, columns=cfg.stats)
        pe_df.insert(0, 'timestep', timesteps)
        pe_df.to_csv(out_dir / 'pre_point_estimate_prediction.csv', index=False)

    mode = (cfg.pre_point_mode or 'none').lower().strip()
    try:
        if mode in ('prior_mean', 'prior_median'):
            centre = 'mean' if mode == 'prior_mean' else 'median'
            prior = load_priors(cfg.priors_yaml if Path(cfg.priors_yaml).exists() else None)
            pe = _prior_center(prior, which=centre)
            _write_pre_point_outputs(pe, label=mode)
        elif mode in ('posterior_map', 'posterior_mean', 'posterior_median'):
            prev = Path(cfg.pre_point_run or '')
            if not (prev / 'posterior_particles.csv').exists():
                raise ValueError("--pre-point-run must point to a folder with posterior CSVs")
            df_prev, w_prev = _read_posterior_run(prev)
            if mode == 'posterior_map':
                pe = df_prev.iloc[int(np.argmax(w_prev))].to_dict()
            elif mode == 'posterior_mean':
                pe = {c: float(np.average(df_prev[c].to_numpy(float), weights=w_prev)) for c in df_prev.columns}
            else:
                pe = {}
                for c in df_prev.columns:
                    x = df_prev[c].to_numpy(float)
                    idx = np.argsort(x); xs = x[idx]; ws = w_prev[idx]
                    cdf = np.cumsum(ws) / np.sum(ws)
                    pe[c] = float(np.interp(0.5, cdf, xs))
            _write_pre_point_outputs(pe, label=mode + (f":{prev.name}" if prev.name else ""))
        elif mode == 'file':
            pe = json.loads(Path(cfg.pre_point_file or '').read_text())
            _write_pre_point_outputs(pe, label='file')
        # 'none' -> skip
    except Exception as e:
        (out_dir / 'pre_point_estimate.ERROR.txt').write_text(str(e), encoding='utf-8')

    # ----- Prepare & run ABC -----
    prior = load_priors(cfg.priors_yaml if Path(cfg.priors_yaml).exists() else None)
    abc = pyabc.ABCSMC(models=abm_model,
                       parameter_priors=prior,
                       distance_function=l2_distance,
                       population_size=cfg.popsize,
                       sampler=sampler)
    db_path = out_dir / 'abc.db'
    abc.new(f"sqlite:///{db_path}", observation)

    history = abc.run(max_nr_populations=cfg.maxgen, minimum_epsilon=cfg.min_eps)

    # ----- Epsilon schedule (robust to 'epsilon' vs 'eps') -----
    try:
        pops = history.get_all_populations()
        col = 'epsilon' if 'epsilon' in pops.columns else ('eps' if 'eps' in pops.columns else None)
        if col is None:
            epsilons = []
            for t in range(history.max_t + 1):
                try:
                    pop = history.get_population(t=t)
                    epsilons.append(getattr(pop, "epsilon", np.nan))
                except Exception:
                    epsilons.append(np.nan)
            eps = pd.DataFrame({'t': np.arange(len(epsilons)), 'epsilon': epsilons})
        else:
            eps = pd.DataFrame({'t': np.arange(len(pops[col].values)), 'epsilon': pops[col].values})
        eps.to_csv(out_dir / 'epsilon_schedule.csv', index=False)
    except Exception as e:
        (out_dir / 'epsilon_schedule.ERROR.txt').write_text(str(e), encoding='utf-8')

    # ----- Posterior -----
    df, w = history.get_distribution(m=0, t=history.max_t)
    w = np.asarray(w, float)
    df.to_csv(out_dir / 'posterior_particles.csv', index=False)
    pd.DataFrame({'weight': w}).to_csv(out_dir / 'posterior_weights.csv', index=False)
    (out_dir / 'posterior_mean.json').write_text(
        json.dumps(df.mean(numeric_only=True).to_dict(), indent=2), encoding='utf-8'
    )

    # ----- Post-run point estimate (MAP/mean/median) -----
    def _point_params(est: str) -> Dict[str, float]:
        est = est.lower().strip()
        if est == 'map':
            return df.iloc[int(np.argmax(w))].to_dict()
        elif est == 'mean':
            return {c: float(np.average(df[c].to_numpy(float), weights=w)) for c in df.columns}
        elif est == 'median':
            out: Dict[str, float] = {}
            for c in df.columns:
                x = df[c].to_numpy(float)
                idx = np.argsort(x); xs = x[idx]; ws = w[idx]
                cdf = np.cumsum(ws) / np.sum(ws)
                out[c] = float(np.interp(0.5, cdf, xs))
            return out
        else:
            raise ValueError("point_estimator must be one of: map, mean, median")

    pe = _point_params(cfg.point_estimator)
    (out_dir / 'point_estimate_params.json').write_text(json.dumps(pe, indent=2), encoding='utf-8')

    pe_params = particle_to_params(pe, motion=cfg.motion, speed_dist=cfg.speed)
    sim_mat_pe = simulate_timeseries(
        model_factory=make_model_factory(seed=cfg.seed + 999),
        params=pe_params,
        total_steps=cfg.total_steps,
        sample_steps=tuple(timesteps),
        init_phenotype=cfg.phenotype,
        force_phase2=True,
        disable_logging=True,
    )
    sim_sel = sim_mat_pe[:, sel_idx]
    pe_df = pd.DataFrame(sim_sel, columns=cfg.stats)
    pe_df.insert(0, 'timestep', timesteps)
    pe_df.to_csv(out_dir / 'point_estimate_prediction.csv', index=False)

    # ----- Scaler dump & README -----
    import joblib
    joblib.dump(scaler, out_dir / 'maxabs_scaler.joblib')
    np.save(out_dir / 'observed_scaled_vec.npy', obs_scaled_vec)

    readme = [
        "ABC-SMC run complete.",
        f"DB: {out_dir / 'abc.db'}",
        f"Populations: {history.max_t + 1}",
        f"Phenotype: {cfg.phenotype}",
        f"Stats: {', '.join(cfg.stats)}",
        f"Timesteps (T={len(timesteps)}): {timesteps[:10]}{' ...' if len(timesteps)>10 else ''}",
        f"Pre-run point mode: {cfg.pre_point_mode}",
        f"Point estimator (post-run): {cfg.point_estimator}",
    ]
    try:
        eps_df = pd.read_csv(out_dir / 'epsilon_schedule.csv')
        readme.append(f"Final epsilon: {float(eps_df['epsilon'].iloc[-1])}")
    except Exception:
        pass
    (out_dir / 'README.txt').write_text("\n".join(readme) + "\n", encoding='utf-8')

    return out_dir


# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser(description='Run fast ABC-SMC and save outputs under abcp/outputs/<run_tag>/')
    ap.add_argument('--observed_csv', required=True, type=str)
    ap.add_argument('--priors_yaml', type=str, default='abcp/priors.yaml')
    ap.add_argument('--statset', type=str, default='s012_nnd', choices=['s012', 's012_nnd'])
    ap.add_argument('--include-gr', action='store_true', help='Append g_r40 & g_r80 to the statset')
    ap.add_argument('--stats', type=str, nargs='+', default=None, help='Explicit stats (override presets)')
    ap.add_argument('--timesteps', type=int, nargs='+', default=None)
    ap.add_argument('--total_steps', type=int, default=300)

    ap.add_argument('--motion', type=str, default='isotropic', choices=['isotropic', 'persistent'])
    ap.add_argument('--speed', type=str, default='gamma', choices=['constant','lognorm','gamma','weibull'])
    ap.add_argument('--phenotype', type=str, default='invasive')

    ap.add_argument('--popsize', type=int, default=200)
    ap.add_argument('--maxgen', type=int, default=12)
    ap.add_argument('--min_eps', type=float, default=0.5)

    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--run_tag', type=str, default=None)

    ap.add_argument('--point-estimator', type=str, default='map', choices=['map', 'mean', 'median'])
    ap.add_argument('--no-progress', action='store_true', help='Disable pyabcs built-in tqdm bars')

    # Pre-run point estimate (default: prior_mean)
    ap.add_argument('--pre-point', type=str, default='prior_mean',
                    choices=['none', 'prior_mean', 'prior_median',
                             'posterior_map', 'posterior_mean', 'posterior_median', 'file'])
    ap.add_argument('--pre-point-run', type=str, default=None,
                    help='Path to a previous run folder for posterior_* modes')
    ap.add_argument('--pre-point-file', type=str, default=None,
                    help='Path to JSON file with {param: value} for --pre-point file')

    args = ap.parse_args()

    # Choose stats
    if args.stats and len(args.stats) > 0:
        stats = _normalize_stats(args.stats)
    else:
        stats = ['S0', 'S1', 'S2'] if args.statset.lower() == 's012' else ['S0', 'S1', 'S2', 'NND_med']
        if args.include_gr:
            for s in ['g_r40', 'g_r80']:
                if s not in stats:
                    stats.append(s)

    timesteps = args.timesteps if args.timesteps else None
    run_tag = args.run_tag or datetime.datetime.now().strftime('abc_%Y%m%d_%H%M%S')

    cfg = RunConfig(
        observed_csv=args.observed_csv,
        priors_yaml=args.priors_yaml,
        total_steps=int(args.total_steps),
        timesteps=timesteps or [],
        stats=stats,
        motion=str(args.motion),
        speed=str(args.speed),
        popsize=int(args.popsize),
        maxgen=int(args.maxgen),
        min_eps=float(args.min_eps),
        seed=int(args.seed),
        workers=int(args.workers),
        run_tag=run_tag,
        phenotype=str(args.phenotype),
        point_estimator=str(args.point_estimator),
        show_progress=(not args.no_progress),
        pre_point_mode=str(args.pre_point),
        pre_point_run=args.pre_point_run,
        pre_point_file=args.pre_point_file,
    )

    out_dir = run_abc(cfg)
    print(f"\nABC-SMC finished. Outputs saved to: {out_dir}")


if __name__ == '__main__':
    main()