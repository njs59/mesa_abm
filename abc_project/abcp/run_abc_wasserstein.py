
#!/usr/bin/env python3
"""
ABC-SMC using a Wasserstein (Earth-Mover) distance over per-stat trajectories.

Distance:
  For each statistic k, compute W1 between the simulated and observed time series
  treated as empirical distributions with optional time weights. Normalise each
  W1 by a robust per-stat scale (IQR/MAD/Std), then aggregate across stats via
  a Euclidean norm:

      d = sqrt( sum_k ( W1_k / scale_k )^2 )

Usage (four stats only, time-emphasis on late points):
  python run_abc_wasserstein.py \
    --observed_ts observed/INV_ABM_ready_summary.csv \
    --priors_yaml priors.yaml \
    --motion persistent \
    --speed lognorm \
    --stats4 \
    --time_power 3 \
    --scale iqr \
    --workers 8 \
    --popsize 200 \
    --maxgen 12 \
    --min_eps 0.5 \
    --db results/abc_wasserstein.db
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd

import pyabc
from scipy.stats import wasserstein_distance

# Your project hooks (same as in your other scripts)
from abm.clusters_model import ClustersModel
from abcp.compute_summary import simulate_timeseries
from abcp.priors import load_priors
from abcp.abc_model_wrapper import particle_to_params

# -------------------------------------------
# Utilities
# -------------------------------------------
def make_model_factory(seed: int = 42):
    def factory(params_dict):
        return ClustersModel(params=params_dict, seed=seed)
    return factory

def time_weights_from_ts(timesteps: List[int], time_power: float) -> np.ndarray:
    """Normalised non-negative time weights (sum=1)."""
    t = np.asarray(timesteps, dtype=float)
    w = (t - t.min() + 1.0) ** float(time_power)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)

def robust_scale(values: np.ndarray, how: str = "iqr") -> float:
    """Return a robust positive scale for 1D values."""
    x = np.asarray(values, dtype=float)
    if how == "iqr":
        q75, q25 = np.percentile(x, [75, 25])
        sc = float(q75 - q25)
    elif how == "mad":
        med = np.median(x)
        sc = float(np.median(np.abs(x - med))) * 1.4826  # normal consistency
    elif how == "std":
        sc = float(np.std(x, ddof=1))
    else:
        raise ValueError(f"Unknown scale method: {how}")
    return sc if sc > 1e-12 else 1.0  # avoid divide-by-zero

# -------------------------------------------
# ABC runner
# -------------------------------------------
def run_one(
    db_path: Path,
    obs_df: pd.DataFrame,
    stats: List[str],
    timesteps: List[int],
    prior,
    popsize: int,
    maxgen: int,
    min_eps: float,
    total_steps: int,
    motion: str,
    speed: str,
    seed: int,
    workers: int,
    time_power: float,
    scale_method: str,
):
    # Sort/align observed
    obs_sorted = obs_df.sort_values("timestep").reset_index(drop=True)
    obs_mat = obs_sorted[stats].to_numpy(dtype=float)   # shape (T, K)
    T, K = obs_mat.shape
    # Per-stat robust scales (based on observed series per stat)
    scales = np.array([robust_scale(obs_mat[:, k], how=scale_method) for k in range(K)], dtype=float)

    # Time weights (shared across stats)
    w_time = time_weights_from_ts(timesteps, time_power)  # shape (T,)

    # Model factory (seed fixed per run)
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
        # Select requested stats in consistent order
        full_order = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
        idx = [full_order.index(s) for s in stats]
        sim_sel = sim_mat[:, idx]  # T x K
        # Flatten for pyABC interface (not used by our distance directly)
        sim_vec = sim_sel.flatten()
        return {f"y_{i}": float(v) for i, v in enumerate(sim_vec)}

    # Observation vector for pyABC (to satisfy API)
    obs_vec = obs_mat.flatten()
    observation = {f"y_{i}": float(v) for i, v in enumerate(obs_vec)}

    # Precompute slice indices for each stat in the flattened vector
    # so the distance can pull per-stat time series back out quickly.
    stat_slices: Dict[int, slice] = {}
    for k in range(K):
        stat_slices[k] = slice(k, T * K, K)

    # Wasserstein distance over per-stat time series (with time weights), scaled and aggregated
    def wdist(sim: Dict[str, float], obs: Dict[str, float]) -> float:
        # Rehydrate flattened vectors
        sim_v = np.array([sim[f"y_{i}"] for i in range(T * K)], dtype=float)
        obs_v = np.array([obs[f"y_{i}"] for i in range(T * K)], dtype=float)
        # Accumulate per-stat W1
        acc = 0.0
        for k in range(K):
            s_slice = stat_slices[k]
            sim_series = sim_v[s_slice]
            obs_series = obs_v[s_slice]
            # SciPy's wasserstein_distance supports per-sample weights
            Wk = wasserstein_distance(obs_series, sim_series,
                                      u_weights=w_time, v_weights=w_time)
            Wk_norm = Wk / scales[k]
            acc += (Wk_norm ** 2)
        return float(np.sqrt(acc))

    # Optional parallel sampler
    try:
        from pyabc.sampler import MulticoreEvalParallelSampler
        sampler = MulticoreEvalParallelSampler(n_procs=workers) if workers > 1 else None
    except Exception:
        sampler = None

    abc = pyabc.ABCSMC(
        models=abm_model,
        parameter_priors=prior,
        distance_function=wdist,
        population_size=popsize,
        sampler=sampler,
    )
    db_url = f"sqlite:///{db_path}"
    abc.new(db_url, observation)
    history = abc.run(max_nr_populations=maxgen, minimum_epsilon=min_eps)
    return history

# -------------------------------------------
# CLI
# -------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ABC-SMC with Wasserstein distance over stat trajectories")
    ap.add_argument("--db", type=str, default="results/abc_wasserstein.db")
    ap.add_argument("--observed_ts", type=str, required=True)
    ap.add_argument("--priors_yaml", type=str, default="priors.yaml")
    ap.add_argument("--motion", type=str, default="isotropic", choices=["isotropic", "persistent"])
    ap.add_argument("--speed", type=str, default="constant", choices=["constant", "lognorm", "gamma", "weibull"])
    ap.add_argument("--total_steps", type=int, default=300)
    ap.add_argument("--popsize", type=int, default=200)
    ap.add_argument("--maxgen", type=int, default=12)
    ap.add_argument("--min_eps", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)

    # Stats selection
    ap.add_argument("--stats4", action="store_true",
                    help="Use only S0, S1, S2, NND_med (drop g(r) stats).")

    # Wasserstein options
    ap.add_argument("--time_power", type=float, default=0.0,
                    help="Time weighting exponent; 0 = equal weights; larger emphasises later timepoints.")
    ap.add_argument("--scale", type=str, default="iqr", choices=["iqr", "mad", "std"],
                    help="Per-stat robust scale for normalising W1 distances (default: iqr).")

    args = ap.parse_args()
    Path("results").mkdir(parents=True, exist_ok=True)

    # Load observed dataframe
    obs_df = pd.read_csv(args.observed_ts)

    # Decide stats set
    if args.stats4:
        stats = ["S0", "S1", "S2", "NND_med"]
    else:
        # If present, include g(r) columns too
        cand = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
        stats = [c for c in cand if c in obs_df.columns]

    # Validate columns
    needed = ["timestep"] + stats
    for c in needed:
        if c not in obs_df.columns:
            raise ValueError(f"Missing column '{c}' in observed CSV.")

    timesteps = obs_df["timestep"].astype(int).tolist()

    # Load priors
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
        time_power=args.time_power,
        scale_method=args.scale,
    )

    print(f"\nABC-SMC (Wasserstein) finished.\nDB saved to {args.db}\n")

if __name__ == "__main__":
    main()
