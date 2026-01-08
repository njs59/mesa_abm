
#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyabc
from pyabc import ABCSMC, Distribution, RV
from pyabc.populationstrategy import ConstantPopulationSize
from pyabc.sampler import MulticoreEvalParallelSampler
from scipy.stats import wasserstein_distance

from clusters_abm.clusters_model import ClustersModel
from clusters_abm.utils import DEFAULTS

# --- IQR-scaled Wasserstein distance ---

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

# --- Summary stats and simulation ---

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

# --- Mapping ---

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

def make_params_from_particle(defaults: dict, particle: dict, motion: str, speed_dist: str, fixed_n_clusters: int) -> dict:
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
    mapping = {
        'prolif_rate': 'phenotypes.proliferative.prolif_rate',
        'adhesion': 'phenotypes.proliferative.adhesion',
        'fragment_rate': 'phenotypes.proliferative.fragment_rate',
        'merge_prob': 'merge.prob_contact_merge',
    }
    for k, v in particle.items():
        if k.startswith('speed_') or k == 'heading_sigma':
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
    params['init']['phenotype'] = 'proliferative'
    params['init']['n_clusters'] = int(max(1, round(fixed_n_clusters)))
    return params

# --- Model closure ---

def make_pyabc_model(obs: np.ndarray, start_step: int, defaults: dict, rng: np.random.Generator, motion: str, speed_dist: str, fixed_n_clusters: int):
    T_obs = obs.shape[0]
    def model(p: dict) -> dict:
        sim_params = make_params_from_particle(defaults, p, motion=motion, speed_dist=speed_dist, fixed_n_clusters=fixed_n_clusters)
        steps_needed = start_step + T_obs
        seed = int(rng.integers(0, 2**31 - 1))
        sim = simulate_timeseries(sim_params, steps=steps_needed, seed=seed)
        seg = sim[start_step:start_step+T_obs,:]
        return {'S0': seg[:,0], 'S1': seg[:,1], 'S2': seg[:,2]}
    return model

# --- Priors ---

def build_prior(motion: str, speed_dist: str) -> Distribution:
    priors = dict(
        prolif_rate=RV('uniform', 5e-4, 0.0195),
        adhesion=RV('uniform', 0.2, 0.8),
        fragment_rate=RV('uniform', 0.0, 0.005),
        merge_prob=RV('uniform', 0.2, 0.8),
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

# --- Posterior predictive coverage ---

def posterior_predictive_coverage(history: pyabc.History, obs: np.ndarray, defaults: dict, start_step: int, motion: str, speed_dist: str, fixed_n_clusters: int, n_sims: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    T = obs.shape[0]
    df, w = history.get_distribution(m=0, t=history.max_t)
    if len(df) == 0:
        return 0.0, 0.0, 0.0
    w = np.asarray(w, dtype=float)
    w = w / (w.sum() if w.sum() > 0 else 1.0)
    idx = np.arange(len(df))
    draws = rng.choice(idx, size=min(n_sims, len(df)), replace=False, p=w)
    sims = np.zeros((len(draws), T, 3), dtype=float)
    for j, i in enumerate(draws):
        particle = {k: float(df.iloc[i][k]) for k in df.columns}
        params = make_params_from_particle(defaults, particle, motion=motion, speed_dist=speed_dist, fixed_n_clusters=fixed_n_clusters)
        seg = simulate_timeseries(params, steps=start_step+T, seed=int(rng.integers(0, 2**31 - 1)))[start_step:start_step+T,:]
        sims[j,:,:] = seg
    q5 = np.quantile(sims, 0.05, axis=0)
    q95 = np.quantile(sims, 0.95, axis=0)
    cover = []
    for k in range(3):
        within = (obs[:,k] >= q5[:,k]) & (obs[:,k] <= q95[:,k])
        cover.append(float(np.mean(within)))
    return tuple(cover)

# --- Runner ---

def run_all(args):
    obs_df = pd.read_csv(args.obs_csv)
    obs = obs_df[['S0','S1','S2']].to_numpy(dtype=float)
    # fixed_n_clusters = args.init_cells_fixed if args.init_cells_fixed > 0 else int(max(1, round(obs[0,0])))
    fixed_n_clusters = 800
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
    for motion, speed_dist in variants:
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            model_func = make_pyabc_model(obs, args.start_step, DEFAULTS, rng, motion=motion, speed_dist=speed_dist, fixed_n_clusters=fixed_n_clusters)
            dist_func = make_wasserstein_distance_scaled(obs)
            prior = build_prior(motion, speed_dist)
            db_path = results_dir / f"{args.dataset}_{motion}_{speed_dist}_s{seed}.db"
            db_uri = f"sqlite:///{db_path.resolve()}"
            sampler = MulticoreEvalParallelSampler()
            pop_strategy = ConstantPopulationSize(args.popsize)
            abc = ABCSMC(models=model_func, parameter_priors=prior, distance_function=dist_func,
                         population_size=pop_strategy, sampler=sampler)
            history = abc.new(db_uri, {'S0': obs[:,0], 'S1': obs[:,1], 'S2': obs[:,2]})
            t0 = time.time()
            history = abc.run(minimum_epsilon=args.min_eps, max_nr_populations=args.max_pops)
            dt_sec = time.time() - t0
            try:
                pops = history.get_all_populations()
                final_eps = float(pops.iloc[-1]['epsilon'])
                n_pops = int(pops.shape[0])
            except Exception:
                final_eps = float(args.min_eps)
                n_pops = int(getattr(history, 'n_populations', args.max_pops))
            cov_S0, cov_S1, cov_S2 = posterior_predictive_coverage(history, obs, DEFAULTS, args.start_step, motion, speed_dist, fixed_n_clusters, n_sims=args.pp_samples, seed=seed)
            rows.append(dict(
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
            ))
            print(f"Done {args.dataset} {motion}/{speed_dist} seed={seed}: eps={final_eps:.3f}, coverage S0/S1/S2={cov_S0:.2f}/{cov_S1:.2f}/{cov_S2:.2f}")
    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / 'summary.csv', index=False)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Run motion-type × speed-distribution grid with IQR-scaled W1 distance')
    ap.add_argument('--obs_csv', type=str, required=True, help='Path to observed CSV with columns S0,S1,S2')
    ap.add_argument('--dataset', type=str, default='INV', help='Short name for dataset in output filenames')
    ap.add_argument('--results_dir', type=str, default='motiongrid_pkg/results', help='Output directory')
    ap.add_argument('--start_step', type=int, default=20)
    ap.add_argument('--popsize', type=int, default=200)
    ap.add_argument('--max_pops', type=int, default=10)
    ap.add_argument('--min_eps', type=float, default=0.5)
    ap.add_argument('--init_cells_fixed', type=int, default=0)
    ap.add_argument('--pp_samples', type=int, default=50, help='Posterior predictive samples per run')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42,123,2026])
    args = ap.parse_args()
    run_all(args)
