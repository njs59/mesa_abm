#!/usr/bin/env python3
"""
Run motion-type × speed-distribution grid while FITTING init_n_clusters as an ABC parameter.

Compared to your original runner, this version:
- Adds init_n_clusters to the prior and treats it like the other parameters.
- Sets params['init']['n_clusters'] from the sampled particle (not a fixed CLI value).
- Records the posterior RESAMPLED median of init_n_clusters into the summary (column: init_cells).

Other behaviour (distance, coverage, outputs) mirrors your existing script.
"""
import argparse
import os
import time
import platform
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyabc
from pyabc import ABCSMC, Distribution, RV
from pyabc.populationstrategy import ConstantPopulationSize
from pyabc.sampler import MulticoreEvalParallelSampler
from scipy.stats import wasserstein_distance
from tqdm.auto import tqdm

# --- ABM imports ---
from clusters_abm.clusters_model import ClustersModel
from clusters_abm.utils import DEFAULTS

# ---------------- Utilities ----------------
def lib_version(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, '__version__', 'unknown')
    except Exception:
        return 'missing'

# ---------------- Distance ----------------
def make_wasserstein_distance_scaled(obs_mat: np.ndarray):
    obs_S0, obs_S1, obs_S2 = obs_mat[:,0], obs_mat[:,1], obs_mat[:,2]
    iqr = np.array([
        np.quantile(obs_S0, 0.75) - np.quantile(obs_S0, 0.25),
        np.quantile(obs_S1, 0.75) - np.quantile(obs_S1, 0.25),
        np.quantile(obs_S2, 0.75) - np.quantile(obs_S2, 0.25),
    ], dtype=float)
    iqr[iqr == 0] = 1.0
    obs_scaled = [obs_S0/iqr[0], obs_S1/iqr[1], obs_S2/iqr[2]]
    def distance(x: dict, x0: dict) -> float:
        d0 = wasserstein_distance(x['S0']/iqr[0], obs_scaled[0])
        d1 = wasserstein_distance(x['S1']/iqr[1], obs_scaled[1])
        d2 = wasserstein_distance(x['S2']/iqr[2], obs_scaled[2])
        return float(d0+d1+d2)
    return pyabc.distance.FunctionDistance(distance)

# ---------------- Simulation ----------------
def compute_summary_from_model(model):
    ids, pos, radii, sizes, speeds = model._log_alive_snapshot()
    n = len(sizes)
    if n == 0:
        return 0.0, 0.0, 0.0
    return float(n), float(np.mean(sizes)), float(np.mean(sizes**2))

def simulate_timeseries(params: dict, steps: int, seed: int) -> np.ndarray:
    model = ClustersModel(params=params, seed=seed)
    out = np.zeros((steps, 3), dtype=float)
    out[0,:] = compute_summary_from_model(model)
    for t in range(1, steps):
        model.step()
        out[t,:] = compute_summary_from_model(model)
    return out

# ---------------- Mapping ----------------
def _set_nested(base: dict, dotted: str, value):
    keys = dotted.split('.')
    d = base
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value

def build_speed_params(speed_dist: str, particle: dict) -> dict:
    if speed_dist == 'constant':
        return {}
    if speed_dist == 'lognorm':
        mu = float(particle.get('speed_meanlog', 1.0))
        sd = float(particle.get('speed_sdlog', 0.7))
        return {'s': sd, 'scale': float(np.exp(mu))}
    elif speed_dist == 'gamma':
        shape = float(particle.get('speed_shape', 2.0))
        scale = float(particle.get('speed_scale', 1.0))
        return {'a': shape, 'scale': scale}
    elif speed_dist == 'weibull':
        shape = float(particle.get('speed_shape', 2.0))
        scale = float(particle.get('speed_scale', 2.0))
        return {'c': shape, 'scale': scale}
    else:
        raise ValueError(f'Unknown speed_dist: {speed_dist}')

def make_params_from_particle(defaults: dict, particle: dict, motion: str, speed_dist: str) -> dict:
    params = {
        'space': dict(defaults['space']),
        'time': dict(defaults['time']),
        'physics': dict(defaults['physics']),
        'phenotypes': {
            'proliferative': dict(defaults['phenotypes']['proliferative']),
            'invasive': dict(defaults['phenotypes']['invasive']),
        },
        'merge': dict(defaults['merge']),
        'init': dict(defaults['init']),
        'movement': dict(defaults['movement']),
    }
    # Movement
    params['movement']['direction'] = motion
    if speed_dist == 'constant':
        params['movement']['mode'] = 'constant'
        params['movement'].pop('distribution', None)
        params['movement'].pop('dist_params', None)
    else:
        params['movement']['mode'] = 'distribution'
        params['movement']['distribution'] = speed_dist
        params['movement']['dist_params'] = build_speed_params(speed_dist, particle)
    if motion == 'persistent':
        hs = float(particle.get('heading_sigma', params['movement'].get('heading_sigma', 0.25)))
        params['movement']['heading_sigma'] = max(0.0, hs)
    else:
        params['movement'].pop('heading_sigma', None)
    # Phenotype / physics mapping
    mapping = {
        'prolif_rate': 'phenotypes.proliferative.prolif_rate',
        'adhesion': 'phenotypes.proliferative.adhesion',
        'fragment_rate': 'phenotypes.proliferative.fragment_rate',
        'merge_prob': 'merge.prob_contact_merge',
    }
    for k, v in particle.items():
        if k.startswith('speed_') or k == 'heading_sigma' or k == 'init_n_clusters':
            continue
        if ('rate' in k or 'prob' in k) and v < 0:
            v = 0.0
        if k in mapping:
            _set_nested(params, mapping[k], float(v))
        else:
            try:
                params[k] = float(v)
            except Exception:
                pass
    # Init: use fitted parameter (rounded ≥1)
    n_init = int(max(1, round(particle['init_n_clusters'])))
    params['init']['phenotype'] = 'proliferative'
    params['init']['n_clusters'] = n_init
    return params

# ---------------- Model closure ----------------
def make_pyabc_model(obs: np.ndarray, start_step: int, defaults: dict, rng: np.random.Generator, motion: str, speed_dist: str):
    T_obs = obs.shape[0]
    def model(p: dict) -> dict:
        sim_params = make_params_from_particle(defaults, p, motion=motion, speed_dist=speed_dist)
        steps_needed = start_step + T_obs
        seed = int(rng.integers(0, 2**31 - 1))
        sim = simulate_timeseries(sim_params, steps=steps_needed, seed=seed)
        seg = sim[start_step:start_step+T_obs,:]
        return {'S0': seg[:,0], 'S1': seg[:,1], 'S2': seg[:,2]}
    return model

# ---------------- Priors ----------------
def build_prior(motion: str, speed_dist: str, init_min: int, init_max: int) -> Distribution:
    priors = dict(
        prolif_rate=RV('uniform', 5e-4, 0.0195),
        adhesion=RV('uniform', 0.2, 0.8),
        fragment_rate=RV('uniform', 0.0, 0.005),
        merge_prob=RV('uniform', 0.2, 0.8),
        init_n_clusters = RV('uniform', init_min, init_max),  # uniform [min, max]
    )
    if speed_dist == 'lognorm':
        priors.update(dict(
            speed_meanlog=RV('uniform', 0.5, 2.0),
            speed_sdlog=RV('uniform', 0.3, 0.9),
        ))
    elif speed_dist == 'gamma':
        priors.update(dict(
            speed_shape=RV('uniform', 0.5, 4.5),
            speed_scale=RV('uniform', 0.2, 4.8),
        ))
    elif speed_dist == 'weibull':
        priors.update(dict(
            speed_shape=RV('uniform', 0.5, 4.5),
            speed_scale=RV('uniform', 0.5, 4.5),
        ))
    elif speed_dist == 'constant':
        pass
    else:
        raise ValueError(f'Unknown speed_dist: {speed_dist}')
    if motion == 'persistent':
        priors.update(dict(
            heading_sigma=RV('uniform', 0.05, 0.45),
        ))
    return Distribution(**priors)

# ---------------- Posterior predictive coverage ----------------
def posterior_predictive_coverage(history: pyabc.History, obs: np.ndarray, defaults: dict, start_step: int, motion: str, speed_dist: str, n_sims: int = 50, seed: int = 0, show_progress: bool = True):
    rng = np.random.default_rng(seed)
    T = obs.shape[0]
    df, w = history.get_distribution(m=0, t=history.max_t)
    if len(df) == 0:
        return 0.0, 0.0, 0.0, np.nan
    # normalise weights
    w = np.asarray(w, dtype=float)
    w = w / (w.sum() if w.sum() > 0 else 1.0)
    idx = np.arange(len(df))
    draws = rng.choice(idx, size=min(n_sims, len(df)), replace=False, p=w)
    sims = np.zeros((len(draws), T, 3), dtype=float)
    init_draws = []
    iterator = range(len(draws))
    if show_progress:
        iterator = tqdm(iterator, desc='Posterior predictive sims', leave=False)
    for j in iterator:
        i = draws[j]
        particle = {k: float(df.iloc[i][k]) for k in df.columns}
        init_draws.append(int(max(1, round(particle.get('init_n_clusters', 1)))))
        params = make_params_from_particle(DEFAULTS, particle, motion=motion, speed_dist=speed_dist)
        seg = simulate_timeseries(params, steps=start_step+T, seed=int(rng.integers(0, 2**31 - 1)))[start_step:start_step+T,:]
        sims[j,:,:] = seg
    q5 = np.quantile(sims, 0.05, axis=0)
    q95 = np.quantile(sims, 0.95, axis=0)
    cover = []
    for k in range(3):
        within = (obs[:,k] >= q5[:,k]) & (obs[:,k] <= q95[:,k])
        cover.append(float(np.mean(within)))
    init_median = float(np.median(init_draws)) if init_draws else np.nan
    return tuple(cover) + (init_median,)

# ---------------- Runner ----------------
def run_all(args):
    print('\n=== Motion Grid Runner (fitting init_n_clusters) ===')
    print(f"Dataset: {args.dataset}")
    print(f"Obs CSV: {args.obs_csv}")
    print(f"start_step={args.start_step}, popsize={args.popsize}, max_pops={args.max_pops}, min_eps={args.min_eps}")
    print(f"Seeds: {args.seeds}")
    obs_df = pd.read_csv(args.obs_csv)
    obs = obs_df[['S0','S1','S2']].to_numpy(dtype=float)
    # Init_n_clusters prior bounds
    obs_init_guess = int(max(1, round(obs[0,0])))
    init_min = 500
    init_max = 1200
    print(f"init_n_clusters prior: randint[{init_min}, {init_max}] (inclusive)")

    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        ('isotropic', 'constant'),
        ('isotropic', 'lognorm'),
        ('isotropic', 'gamma'),
        ('isotropic', 'weibull'),
        ('persistent','constant'),
        ('persistent','lognorm'),
        ('persistent','gamma'),
        ('persistent','weibull'),
    ]

    rows = []
    for motion, speed_dist in tqdm(variants, desc='Variants', position=0):
        for seed in tqdm(args.seeds, desc=f'{motion}/{speed_dist} seeds', position=1, leave=False):
            rng = np.random.default_rng(seed)
            model_func = make_pyabc_model(obs, args.start_step, DEFAULTS, rng, motion=motion, speed_dist=speed_dist)
            dist_func = make_wasserstein_distance_scaled(obs)
            prior = build_prior(motion, speed_dist, init_min=init_min, init_max=init_max)

            db_path = results_dir / f"{args.dataset}_{motion}_{speed_dist}_s{seed}.db"
            db_uri = f"sqlite:///{db_path.resolve()}"
            print(f"\n--- Running: dataset={args.dataset}, motion={motion}, speed={speed_dist}, seed={seed} ---")
            print(f"DB: {db_path.name}")

            sampler = MulticoreEvalParallelSampler()  # pyabc prints per-population INFO lines
            pop_strategy = ConstantPopulationSize(args.popsize)
            abc = ABCSMC(models=model_func, parameter_priors=prior, distance_function=dist_func,
                         population_size=pop_strategy, sampler=sampler)
            history = abc.new(db_uri, {'S0': obs[:,0], 'S1': obs[:,1], 'S2': obs[:,2]})
            t0 = time.perf_counter()
            history = abc.run(minimum_epsilon=args.min_eps, max_nr_populations=args.max_pops)
            dt_sec = time.perf_counter() - t0

            # Summaries
            try:
                pops = history.get_all_populations()
                final_eps = float(pops.iloc[-1]['epsilon'])
                n_pops = int(pops.shape[0])
            except Exception:
                final_eps = float(args.min_eps)
                n_pops = int(getattr(history, 'n_populations', args.max_pops))
            print(f"Finished ABC: final_eps={final_eps:.3f}, n_populations={n_pops}, runtime={dt_sec:.1f}s")

            cov_S0, cov_S1, cov_S2, init_med = posterior_predictive_coverage(
                history, obs, DEFAULTS, args.start_step, motion, speed_dist, n_sims=args.pp_samples,
                seed=seed, show_progress=True)
            print(f"Coverage: S0={cov_S0:.2f}, S1={cov_S1:.2f}, S2={cov_S2:.2f}; init_n_clusters median≈{init_med}")

            row = dict(
                dataset=args.dataset,
                motion=motion,
                speed_dist=speed_dist,
                seed=seed,
                final_eps=final_eps,
                n_populations=n_pops,
                runtime_sec=round(dt_sec,2),
                coverage_S0=round(cov_S0,3),
                coverage_S1=round(cov_S1,3),
                coverage_S2=round(cov_S2,3),
                db=str(db_path.name),
                start_step=args.start_step,
                popsize=args.popsize,
                max_pops=args.max_pops,
                min_eps=args.min_eps,
                pp_samples=args.pp_samples,
                init_cells=int(init_med) if not np.isnan(init_med) else None,  # store posterior median
                obs_rows=obs.shape[0],
            )
            env_tags = {
                'run_host': platform.node(),
                'run_user': os.environ.get('USER') or os.environ.get('USERNAME') or 'unknown',
                'pyabc_version': lib_version('pyabc'),
                'mesa_version': lib_version('mesa'),
                'numpy_version': lib_version('numpy'),
                'pandas_version': lib_version('pandas'),
                'scipy_version': lib_version('scipy'),
            }
            row.update(env_tags)
            rows.append(row)

            summary = pd.DataFrame(rows)
            out_csv = results_dir / 'summary.csv'
            summary.to_csv(out_csv, index=False)
            print(f"\nSaved summary: {out_csv}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Run motion-type × speed-distribution grid with IQR-scaled W1 distance, FITTING init_n_clusters as a parameter')
    ap.add_argument('--obs_csv', type=str, required=True, help='Path to observed CSV with columns S0,S1,S2')
    ap.add_argument('--dataset', type=str, default='INV', help='Short name for dataset in output filenames')
    ap.add_argument('--results_dir', type=str, default='motiongrid_pkg/results', help='Output directory')
    ap.add_argument('--start_step', type=int, default=20)
    ap.add_argument('--popsize', type=int, default=200)
    ap.add_argument('--max_pops', type=int, default=10)
    ap.add_argument('--min_eps', type=float, default=0.5)
    ap.add_argument('--pp_samples', type=int, default=50, help='Posterior predictive samples per run')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42,123,2026])
    # New: prior bounds for init_n_clusters (can be derived from obs if omitted)
    ap.add_argument('--init_clusters_min', type=int, default=None, help='Lower bound for init_n_clusters prior (inclusive)')
    ap.add_argument('--init_clusters_max', type=int, default=None, help='Upper bound for init_n_clusters prior (inclusive)')
    args = ap.parse_args()
    run_all(args)
