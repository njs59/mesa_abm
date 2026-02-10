#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyabc
from sklearn.preprocessing import MaxAbsScaler

from abm.clusters_model import ClustersModel
from abcp.compute_summary import simulate_timeseries
from abcp.priors import DEFAULT_PRIORS
from abcp.abc_model_wrapper import particle_to_params_angles

def make_model_factory(seed: int = 42):
    def factory(params_dict):
        return ClustersModel(params=params_dict, seed=seed)
    return factory

def run_one(db_path: Path, obs_df: pd.DataFrame, stats, timesteps, prior, popsize, maxgen, min_eps, total_steps,
            scenario: int, speed: str, seed: int, workers: int, show_progress: bool, fixed: dict):
    obs_sorted = obs_df.sort_values("timestep").reset_index(drop=True)
    obs_mat = obs_sorted[stats].to_numpy(float)
    T, K = obs_mat.shape
    scaler = MaxAbsScaler().fit(obs_mat)
    obs_scaled = scaler.transform(obs_mat).flatten()

    model_factory = make_model_factory(seed=seed)

    def abm_model(particle):
        params = particle_to_params_angles(particle, scenario=scenario, speed_dist=speed, fixed=fixed)
        sim_mat = simulate_timeseries(model_factory, params=params, total_steps=total_steps, sample_steps=tuple(timesteps))
        full_order = ["S0","S1","S2","SSNND_med"]
        idx = [full_order.index(s) for s in stats]
        sim_sel = sim_mat[:, idx]
        sim_scaled = scaler.transform(sim_sel).flatten()
        return {f"y_{i}": float(v) for i,v in enumerate(sim_scaled)}

    observation = {f"y_{i}": float(v) for i,v in enumerate(obs_scaled)}

    def l2(sim, obs):
        s = np.array([sim[f"y_{i}"] for i in range(T*K)], float)
        o = np.array([obs[f"y_{i}"] for i in range(T*K)], float)
        return float(np.sqrt(np.sum((s-o)**2)))

    sampler=None
    try:
        if workers and workers>1:
            from pyabc.sampler import MulticoreEvalParallelSampler
            sampler = MulticoreEvalParallelSampler(n_procs=workers)
        else:
            from pyabc.sampler import SingleCoreSampler
            sampler = SingleCoreSampler()
        sampler.show_progress = bool(show_progress)
    except Exception as e:
        print(f"[warn] sampler init: {e}")
        sampler=None

    abc = pyabc.ABCSMC(models=abm_model, parameter_priors=prior, distance_function=l2, population_size=popsize, sampler=sampler)
    db_url=f"sqlite:///{db_path}"
    abc.new(db_url, observation)
    return abc.run(max_nr_populations=maxgen, minimum_epsilon=min_eps)

SCENARIO_KEYS = {
    1: ["heading_sigma"], 2: [], 3: [], 4: ["kappa"], 5: [], 6: ["kappa"], 7: [], 8: ["kappa1","kappa2"], 9: ["t_switch"], 10:["kappa1","kappa2","t_switch"], 11: [], 12: ["kappa1","kappa2","switch_meanlog","switch_sdlog","switch_low","switch_high"],
}
BASE_KEYS = ["prolif_rate","fragment_rate","p_merge","init_n_clusters","speed_meanlog","speed_sdlog","heading_sigma"]

def subset_prior(bounds: dict, keys):
    keep = {k:bounds[k] for k in keys if k in bounds}
    return pyabc.Distribution(**{name: pyabc.RV("uniform", float(b[0]), float(b[1])-float(b[0])) for name,b in keep.items()})

def main():
    ap = argparse.ArgumentParser(description="ABC-SMC for angle scenarios 1..12 (S0,S1,S2,SSNND_med)")
    ap.add_argument("--db", type=str, default="results/abc_angles.db")
    ap.add_argument("--observed_ts", type=str, required=True)
    ap.add_argument("--priors_yaml", type=str, default=None)
    ap.add_argument("--scenario", type=int, required=True, choices=list(range(1,13)))
    ap.add_argument("--speed", type=str, default="constant", choices=["constant","lognorm","gamma","weibull"])
    ap.add_argument("--popsize", type=int, default=200)
    ap.add_argument("--maxgen", type=int, default=12)
    ap.add_argument("--min_eps", type=float, default=0.5)
    ap.add_argument("--total_steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--show_progress", action="store_true")
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--kappa1", type=float, default=None)
    ap.add_argument("--kappa2", type=float, default=None)
    ap.add_argument("--t_switch", type=float, default=None)
    ap.add_argument("--switch_dist", type=str, default="lognorm", choices=["lognorm","uniform"])
    ap.add_argument("--switch_meanlog", type=float, default=None)
    ap.add_argument("--switch_sdlog", type=float, default=None)
    ap.add_argument("--switch_low", type=float, default=None)
    ap.add_argument("--switch_high", type=float, default=None)

    args = ap.parse_args()
    Path("results").mkdir(exist_ok=True)

    obs_df = pd.read_csv(args.observed_ts)
    stats=["S0","S1","S2","SSNND_med"]
    for s in stats:
        if s not in obs_df.columns:
            raise ValueError(f"Missing {s} in observed CSV.")
    timesteps = obs_df["timestep"].astype(int).tolist()

    pri_bounds = dict(DEFAULT_PRIORS)
    if args.priors_yaml and Path(args.priors_yaml).exists():
        import yaml
        with open(args.priors_yaml,'r') as f:
            user = yaml.safe_load(f) or {}
        pri_bounds.update(user)

    keys = BASE_KEYS + SCENARIO_KEYS.get(args.scenario, [])
    prior = subset_prior(pri_bounds, keys)

    fixed={}
    for k in ["kappa","kappa1","kappa2","t_switch","switch_meanlog","switch_sdlog","switch_low","switch_high"]:
        v=getattr(args,k)
        if v is not None: fixed[k]=v
    fixed["switch_dist"]=args.switch_dist

    run_one(db_path=Path(args.db), obs_df=obs_df, stats=stats, timesteps=timesteps, prior=prior,
            popsize=args.popsize, maxgen=args.maxgen, min_eps=args.min_eps, total_steps=args.total_steps,
            scenario=args.scenario, speed=args.speed, seed=args.seed, workers=args.workers,
            show_progress=args.show_progress, fixed=fixed)

if __name__ == "__main__":
    main()
