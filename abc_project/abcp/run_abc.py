
#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyabc

from abm.clusters_model import ClustersModel
from abcp.compute_summary import simulate_timeseries
from abcp.priors import load_priors
from abcp.abc_model_wrapper import particle_to_params


def make_model_factory(seed: int = 42):
    def factory(params_dict):
        return ClustersModel(params=params_dict, seed=seed)
    return factory


def main():
    ap = argparse.ArgumentParser(
        description="Run ABC-SMC for the clustering ABM (supports dropping g(r) stats)"
    )

    ap.add_argument("--db", type=str, default="results/abc_run.db")
    ap.add_argument("--popsize", type=int, default=200)
    ap.add_argument("--maxgen", type=int, default=12)
    ap.add_argument("--min_eps", type=float, default=0.5)

    ap.add_argument("--observed_ts", type=str, default="observed/INV_ABM_ready_summary.csv")
    ap.add_argument("--t_start", type=int, default=22)
    ap.add_argument("--total_steps", type=int, default=300)

    ap.add_argument("--priors_yaml", type=str, default="priors.yaml")
    ap.add_argument("--motion", type=str, default="isotropic",
                    choices=["isotropic", "persistent"])
    ap.add_argument("--speed", type=str, default="constant",
                    choices=["constant", "lognorm", "gamma", "weibull"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)

    # NEW: toggle to exclude g(r)
    ap.add_argument("--no_gr", action="store_true",
                    help="Use only S0, S1, S2, NND_med (drop g_r40, g_r80)")

    args = ap.parse_args()
    Path("results").mkdir(exist_ok=True, parents=True)

    # ----------------------------
    # Load observed time-series
    # ----------------------------
    obs_df = pd.read_csv(args.observed_ts)
    timesteps = obs_df["timestep"].astype(int).to_list()

    # Stats list: with or without g(r)
    if args.no_gr:
        stats = ["S0", "S1", "S2", "NND_med"]
    else:
        stats = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]

    # Matrix T x K and flattened target vector
    obs_mat = obs_df[stats].to_numpy(float)
    obs_vec = obs_mat.flatten()

    # ----------------------------
    # Z-score normalisation
    # ----------------------------
    means = obs_mat.mean(axis=0)             # (K,)
    stds = obs_mat.std(axis=0)               # (K,)
    stds = np.where(stds < 1e-12, 1.0, stds)
    stds_mat = np.tile(stds, (len(timesteps), 1))     # (T, K)
    stds_rep = stds_mat.flatten()                      # (T*K,)

    # ----------------------------
    # Weights (Solution 1)
    # Full weight for S0,S1,S2,NND; g(r) down-weighted when present
    # ----------------------------
    if args.no_gr:
        base_weights = np.array([1.0, 1.0, 1.0, 1.0])      # S0 S1 S2 NND
    else:
        base_weights = np.array([1.0, 1.0, 1.0, 1.0, 0.3, 0.3])  # + g40 g80
    weights_rep = np.tile(base_weights, len(timesteps))     # (T*K,)

    # ----------------------------
    # Priors
    # ----------------------------
    prior = load_priors(args.priors_yaml if Path(args.priors_yaml).exists() else None)

    # ----------------------------
    # Model callback for pyABC
    # ----------------------------
    model_factory = make_model_factory(seed=args.seed)

    def abm_model(particle):
        params = particle_to_params(particle, motion=args.motion, speed_dist=args.speed)
        sim_mat = simulate_timeseries(
            model_factory,
            params=params,
            total_steps=args.total_steps,
            sample_steps=tuple(timesteps)
        )
        # Keep only the selected columns in the same order
        # sim_mat is T x 6 (S0,S1,S2,NND,g40,g80) in that exact order
        # Build a column index map:
        full_order = ["S0","S1","S2","NND_med","g_r40","g_r80"]
        col_idx = [full_order.index(s) for s in stats]
        sim_mat_sel = sim_mat[:, col_idx]

        sim_vec = sim_mat_sel.flatten()
        return {f"y_{i}": float(v) for i, v in enumerate(sim_vec)}

    observation = {f"y_{i}": float(v) for i, v in enumerate(obs_vec)}

    # ----------------------------
    # Weighted Z-score distance
    # ----------------------------
    def distance(sim, obs):
        sim_v = np.array([sim[f"y_{i}"] for i in range(len(obs_vec))], float)
        obs_v = np.array([obs[f"y_{i}"] for i in range(len(obs_vec))], float)
        z = (sim_v - obs_v) / stds_rep
        z = z * weights_rep
        return float(np.sqrt(np.sum(z * z)))

    # ----------------------------
    # ABC-SMC
    # ----------------------------
    abc = pyabc.ABCSMC(
        models=abm_model,
        parameter_priors=prior,
        distance_function=distance,
        population_size=args.popsize,
    )
    db_url = f"sqlite:///{args.db}"
    abc.new(db_url, observation)
    abc.run(max_nr_populations=args.maxgen, minimum_epsilon=args.min_eps)
    print(f"\nABC finished. DB saved to {args.db}")


if __name__ == "__main__":
    main()
