
#!/usr/bin/env python3
"""
ABC-SMC with Mahalanobis distance for the clustering ABM.

Usage:
    python run_abc_mahalanobis.py \
        --observed_ts observed/PRO_ABM_ready_summary.csv \
        --priors_yaml priors.yaml \
        --motion persistent \
        --speed lognorm \
        --db results/abc_mahalanobis.db \
        --popsize 200 --maxgen 12 --min_eps 0.5

This script:
- Loads observed (S0,S1,S2,NND_med, optionally g(r))
- Simulates the ABM at the same timepoints
- Computes a Mahalanobis distance between flattened trajectories
- Runs pyABC ABC-SMC using the Mahalanobis distance
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyabc

from abm.clusters_model import ClustersModel
from abcp.compute_summary import simulate_timeseries
from abcp.priors import load_priors
from abcp.abc_model_wrapper import particle_to_params


# ----------------------------------------------------------------------
# ABM model factory
# ----------------------------------------------------------------------
def make_model_factory(seed: int = 42):
    def factory(params_dict):
        return ClustersModel(params=params_dict, seed=seed)
    return factory


# ----------------------------------------------------------------------
# Mahalanobis distance helper
# ----------------------------------------------------------------------
def build_mahalanobis_inverse(obs_mat: np.ndarray) -> np.ndarray:
    """
    Build a block-diagonal covariance matrix over the flattened time-series,
    based on covariance across statistics at each timestep.

    obs_mat: shape (T, K) containing observed stats per timestep.
    """
    T, K = obs_mat.shape

    # Covariance over stats (S0,S1,S2,NND_med) across timesteps
    cov_stats = np.cov(obs_mat, rowvar=False)   # shape (K,K)

    # Create block-diagonal covariance: kron(I_T, cov_stats)
    cov_block = np.kron(np.eye(T), cov_stats)   # shape (T*K, T*K)

    # Regularise to avoid singularity
    eps = 1e-6
    cov_reg = cov_block + eps * np.eye(cov_block.shape[0])

    # Invert
    inv_cov = np.linalg.inv(cov_reg)
    return inv_cov


# ----------------------------------------------------------------------
# ABC runner
# ----------------------------------------------------------------------
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
    # Sort observed and extract values
    obs_sorted = obs_df.sort_values("timestep").reset_index(drop=True)
    obs_mat = obs_sorted[stats].to_numpy(float)     # shape (T,K)
    obs_vec = obs_mat.flatten()

    # Build Mahalanobis inverse covariance
    inv_cov = build_mahalanobis_inverse(obs_mat)

    # Factory for model creation
    model_factory = make_model_factory(seed)

    # Wrapper for pyABC: returns dict of y_i entries
    def abm_model(particle):
        params = particle_to_params(particle, motion=motion, speed_dist=speed)
        sim_mat = simulate_timeseries(
            model_factory,
            params=params,
            total_steps=total_steps,
            sample_steps=tuple(timesteps),
        )

        # Select columns in the order of 'stats'
        full_order = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
        idx = [full_order.index(s) for s in stats]
        sim_sel = sim_mat[:, idx]
        sim_vec = sim_sel.flatten()

        return {f"y_{i}": float(v) for i, v in enumerate(sim_vec)}

    # Observation vector for pyABC
    observation = {f"y_{i}": float(v) for i, v in enumerate(obs_vec)}

    # Mahalanobis distance
    def mahalanobis(sim, obs):
        sim_v = np.array([sim[f"y_{i}"] for i in range(len(obs_vec))], float)
        obs_v = np.array([obs[f"y_{i}"] for i in range(len(obs_vec))], float)
        diff = sim_v - obs_v
        return float(np.sqrt(diff @ inv_cov @ diff))

    # Optional parallel sampler
    try:
        from pyabc.sampler import MulticoreEvalParallelSampler
        sampler = MulticoreEvalParallelSampler(n_procs=workers) if workers > 1 else None
    except Exception:
        sampler = None

    # Construct ABC-SMC object
    abc = pyabc.ABCSMC(
        models=abm_model,
        parameter_priors=prior,
        distance_function=mahalanobis,
        population_size=popsize,
        sampler=sampler,
    )

    db_url = f"sqlite:///{db_path}"
    abc.new(db_url, observation)

    history = abc.run(max_nr_populations=maxgen, minimum_epsilon=min_eps)
    return history


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ABC-SMC using Mahalanobis distance")
    ap.add_argument("--db", type=str, default="results/abc_mahalanobis.db")
    ap.add_argument("--observed_ts", type=str, required=True)
    ap.add_argument("--popsize", type=int, default=200)
    ap.add_argument("--maxgen", type=int, default=12)
    ap.add_argument("--min_eps", type=float, default=0.5)
    ap.add_argument("--total_steps", type=int, default=300)
    ap.add_argument("--priors_yaml", type=str, default="priors.yaml")
    ap.add_argument("--motion", type=str, default="isotropic",
                    choices=["isotropic", "persistent"])
    ap.add_argument("--speed", type=str, default="constant",
                    choices=["constant", "lognorm", "gamma", "weibull"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--no_gr", action="store_true",
        help="Use S0,S1,S2,NND only (drop g(r) stats)"
    )

    args = ap.parse_args()
    Path("results").mkdir(exist_ok=True)

    # Load observed
    obs_df = pd.read_csv(args.observed_ts)
    if args.no_gr:
        stats = ["S0", "S1", "S2", "NND_med"]
    else:
        stats = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]

    # Ensure required columns exist
    for s in stats:
        if s not in obs_df.columns:
            raise ValueError(f"Missing column {s} in observed CSV")

    timesteps = obs_df["timestep"].astype(int).tolist()

    # Load priors (YAML or default)
    prior = load_priors(args.priors_yaml if Path(args.priors_yaml).exists() else None)

    # Run ABC
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

    print(f"\nABC-SMC with Mahalanobis distance finished.\nDB saved to {args.db}\n")


if __name__ == "__main__":
    main()
