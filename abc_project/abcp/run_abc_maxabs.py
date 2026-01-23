
#!/usr/bin/env python3
"""
ABC-SMC using MaxAbsScaler to rescale summary stats to [-1,1]
and using L2 norm between rescaled observed and simulated stats.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyabc

from sklearn.preprocessing import MaxAbsScaler

from abm.clusters_model import ClustersModel
from abcp.compute_summary import simulate_timeseries
from abcp.priors import load_priors
from abcp.abc_model_wrapper import particle_to_params


def make_model_factory(seed: int = 42):
    def factory(params_dict):
        return ClustersModel(params=params_dict, seed=seed)
    return factory


def run_one(
    db_path: Path,
    obs_df: pd.DataFrame,
    stats: list[str],
    timesteps: list[int],
    prior,
    popsize: int,
    maxgen: int,
    min_eps: float,
    total_steps: int,
    motion: str,
    speed: str,
    seed: int,
    workers: int,
):
    # Sort observed
    obs_sorted = obs_df.sort_values("timestep").reset_index(drop=True)
    obs_mat = obs_sorted[stats].to_numpy(float)      # shape (T,K)
    T, K = obs_mat.shape
    obs_vec = obs_mat.flatten()

    # --- Fit MaxAbsScaler on observed only ---
    scaler = MaxAbsScaler()
    scaler.fit(obs_mat)  # learns per-stat max abs values

    # Rescale observed
    obs_scaled_mat = scaler.transform(obs_mat)
    obs_scaled_vec = obs_scaled_mat.flatten()

    # Model factory
    model_factory = make_model_factory(seed=seed)

    # pyABC model wrapper
    def abm_model(particle):
        params = particle_to_params(particle, motion=motion, speed_dist=speed)

        sim_mat = simulate_timeseries(
            model_factory,
            params=params,
            total_steps=total_steps,
            sample_steps=tuple(timesteps),
        )

        full_order = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
        idx = [full_order.index(s) for s in stats]

        sim_sel = sim_mat[:, idx]                   # T x K
        sim_scaled = scaler.transform(sim_sel)      # apply same scaler
        sim_scaled_vec = sim_scaled.flatten()

        return {f"y_{i}": float(v) for i, v in enumerate(sim_scaled_vec)}

    # Observation vector (scaled) for pyABC API
    observation = {f"y_{i}": float(v) for i, v in enumerate(obs_scaled_vec)}

    # L2 distance in rescaled space
    def l2_distance(sim, obs):
        sim_v = np.array([sim[f"y_{i}"] for i in range(T * K)], float)
        obs_v = np.array([obs[f"y_{i}"] for i in range(T * K)], float)
        return float(np.sqrt(np.sum((sim_v - obs_v) ** 2)))

    # Optional parallel sampler
    try:
        from pyabc.sampler import MulticoreEvalParallelSampler
        sampler = MulticoreEvalParallelSampler(n_procs=workers) if workers > 1 else None
    except Exception:
        sampler = None

    abc = pyabc.ABCSMC(
        models=abm_model,
        parameter_priors=prior,
        distance_function=l2_distance,
        population_size=popsize,
        sampler=sampler,
    )

    db_url = f"sqlite:///{db_path}"
    abc.new(db_url, observation)
    history = abc.run(max_nr_populations=maxgen, minimum_epsilon=min_eps)
    return history


def main():
    ap = argparse.ArgumentParser(description="ABC-SMC with MaxAbsScaler + L2 norm")
    ap.add_argument("--db", type=str, default="results/abc_maxabs.db")
    ap.add_argument("--observed_ts", type=str, required=True)
    ap.add_argument("--priors_yaml", type=str, default="priors.yaml")
    ap.add_argument("--motion", type=str, default="isotropic",
                    choices=["isotropic", "persistent"])
    ap.add_argument("--speed", type=str, default="constant",
                    choices=["constant", "lognorm", "gamma", "weibull"])
    ap.add_argument("--popsize", type=int, default=200)
    ap.add_argument("--maxgen", type=int, default=12)
    ap.add_argument("--min_eps", type=float, default=0.5)
    ap.add_argument("--total_steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no_gr", action="store_true")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True)

    obs_df = pd.read_csv(args.observed_ts)
    if args.no_gr:
        stats = ["S0", "S1", "S2", "NND_med"]
    else:
        stats = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]

    for s in stats:
        if s not in obs_df.columns:
            raise ValueError(f"Missing {s} in observed CSV")

    timesteps = obs_df["timestep"].astype(int).tolist()

    prior = load_priors(args.priors_yaml if Path(args.priors_yaml).exists() else None)

    run_one(
        db_path=Path(args.db),
        obs_df=obs_df,
        stats=stats,
        timesteps=timesteps,
        prior=prior,
        popsize=args.popsize,
        maxgen=args.maxgen,
        min_eps=args.min_eps,
        total_steps=args.total_steps,
        motion=args.motion,
        speed=args.speed,
        seed=args.seed,
        workers=args.workers,
    )

    print(f"\nABC-SMC MaxAbsScaler run complete. DB saved to {args.db}")


if __name__ == "__main__":
    main()
