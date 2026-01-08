
# pyabc_fit_mesa_wasserstein_scaled.py
# ABC–SMC for Mesa ABM using SciPy Wasserstein distance, scaled by observed IQR per stat.

import argparse
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import pyabc
from pyabc import ABCSMC, Distribution, RV
from pyabc.populationstrategy import ConstantPopulationSize
from pyabc.sampler import MulticoreEvalParallelSampler
from scipy.stats import wasserstein_distance

try:
    from .clusters_model import ClustersModel  # type: ignore
    from .utils import DEFAULTS                 # type: ignore
except Exception:
    from clusters_model import ClustersModel
    from utils import DEFAULTS

# --- Summary statistics ---
def compute_summary_from_model(model) -> Tuple[float, float, float]:
    ids, pos, radii, sizes, speeds = model._log_alive_snapshot()
    n = len(sizes)
    if n == 0:
        return 0.0, 0.0, 0.0
    return float(n), float(np.mean(sizes)), float(np.mean(sizes ** 2))

def simulate_timeseries(params: Dict, steps: int, seed: int) -> np.ndarray:
    model = ClustersModel(params=params, seed=seed)
    out = np.zeros((steps, 3), dtype=float)
    out[0, :] = compute_summary_from_model(model)
    for t in range(1, steps):
        model.step()
        out[t, :] = compute_summary_from_model(model)
    return out

# --- Parameter mapping helpers ---
def _set_nested(base: Dict, dotted: str, value):
    keys = dotted.split(".")
    d = base
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value

def make_params_from_particle(defaults: Dict, particle: Dict[str, float]) -> Dict:
    params = {
        "space": dict(defaults["space"]),
        "time": dict(defaults["time"]),
        "physics": dict(defaults["physics"]),
        "phenotypes": {
            "proliferative": dict(defaults["phenotypes"]["proliferative"]),
            "invasive": dict(defaults["phenotypes"]["invasive"]),
        },
        "merge": dict(defaults["merge"]),
        "init": dict(defaults["init"]),
        "movement": dict(defaults["movement"]),
    }
    # Movement: persistent + lognormal step magnitudes
    params["movement"]["mode"] = "distribution"
    params["movement"]["distribution"] = "lognorm"
    params["movement"]["direction"] = "persistent"
    params["movement"]["dist_params"] = {
        "s": float(particle.get("speed_sdlog", params["movement"]["dist_params"].get("s", 0.6))),
        "scale": float(np.exp(particle.get("speed_meanlog", np.log(params["movement"]["dist_params"].get("scale", 2.0))))),
    }
    if "heading_sigma" in particle:
        params["movement"]["heading_sigma"] = float(max(0.0, particle["heading_sigma"]))
    mapping = {
        "prolif_rate": "phenotypes.proliferative.prolif_rate",
        "adhesion": "phenotypes.proliferative.adhesion",
        "fragment_rate": "phenotypes.proliferative.fragment_rate",
        "merge_prob": "merge.prob_contact_merge",
        "init_cells": "init.n_clusters",
    }
    for k, v in particle.items():
        if k in ("speed_meanlog", "speed_sdlog", "heading_sigma"):
            continue
        if ("rate" in k or "prob" in k) and v < 0:
            v = 0.0
        if k in mapping:
            if k == "init_cells":
                v_int = int(max(1, round(v)))
                _set_nested(params, mapping[k], v_int)
            else:
                _set_nested(params, mapping[k], float(v))
        else:
            try:
                params[k] = float(v)
            except Exception:
                pass
    params["init"]["phenotype"] = "proliferative"
    params["init"]["n_clusters"] = int(max(1, round(params["init"]["n_clusters"])))
    return params

# --- Model closure ---
def make_pyabc_model(obs: np.ndarray, start_step: int, base_defaults: Dict, rng: np.random.Generator):
    T_obs = obs.shape[0]
    def model(p: Dict) -> Dict:
        sim_params = make_params_from_particle(base_defaults, p)
        steps_needed = start_step + T_obs
        seed = int(rng.integers(0, 2**31 - 1))
        sim = simulate_timeseries(sim_params, steps=steps_needed, seed=seed)
        seg = sim[start_step : start_step + T_obs, :]
        if seg.shape[0] != T_obs:
            raise RuntimeError(f"Simulated segment length {seg.shape[0]} != observed length {T_obs}")
        return {"S0": seg[:, 0], "S1": seg[:, 1], "S2": seg[:, 2]}
    return model

# --- NEW: IQR-normalised Wasserstein distance ---
def make_wasserstein_distance_scaled(obs: np.ndarray):
    # compute IQRs from observed series per stat
    obs_S0, obs_S1, obs_S2 = obs[:, 0], obs[:, 1], obs[:, 2]
    iqr = np.array([
        np.quantile(obs_S0, 0.75) - np.quantile(obs_S0, 0.25),
        np.quantile(obs_S1, 0.75) - np.quantile(obs_S1, 0.25),
        np.quantile(obs_S2, 0.75) - np.quantile(obs_S2, 0.25),
    ])
    iqr[iqr == 0] = 1.0  # guard
    # pre-scale observed
    obs_scaled = [obs_S0 / iqr[0], obs_S1 / iqr[1], obs_S2 / iqr[2]]

    def distance(x: Dict, _x0: Dict) -> float:
        d0 = wasserstein_distance(x["S0"] / iqr[0], obs_scaled[0])
        d1 = wasserstein_distance(x["S1"] / iqr[1], obs_scaled[1])
        d2 = wasserstein_distance(x["S2"] / iqr[2], obs_scaled[2])
        return float(d0 + d1 + d2)

    return pyabc.distance.FunctionDistance(distance)

# --- CLI ---
def main():
    parser = argparse.ArgumentParser(description="ABC–SMC (IQR-scaled Wasserstein) for Mesa ABM.")
    parser.add_argument("--obs_csv", type=str, default="INV_summary_stats.csv")
    parser.add_argument("--start_step", type=int, default=20)
    parser.add_argument("--popsize", type=int, default=200)
    parser.add_argument("--max_pops", type=int, default=8)
    parser.add_argument("--min_eps", type=float, default=0.5)
    parser.add_argument("--db_file", type=str, default="clusters_abm/results/pyabc_runs_wasserstein_scaled.db")
    parser.add_argument("--results_dir", type=str, default="clusters_abm/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_cells_min", type=int, default=200)
    parser.add_argument("--init_cells_max", type=int, default=4000)
    args = parser.parse_args()

    obs_df = pd.read_csv(args.obs_csv)
    if not all(c in obs_df.columns for c in ("S0", "S1", "S2")):
        raise ValueError("Observed CSV must contain S0,S1,S2")
    obs = obs_df[["S0", "S1", "S2"]].to_numpy(dtype=float)
    obs_full_dict = {"S0": obs[:, 0], "S1": obs[:, 1], "S2": obs[:, 2]}

    prior = Distribution(
        speed_meanlog=RV("uniform", 0.5, 2.0),
        speed_sdlog=RV("uniform", 0.7, 2.3),
        heading_sigma=RV("uniform", 0.05, 0.45),
        prolif_rate=RV("uniform", 5e-4, 0.0195),
        adhesion=RV("uniform", 0.2, 0.8),
        fragment_rate=RV("uniform", 0.0, 0.005),
        merge_prob=RV("uniform", 0.2, 0.8),
        init_cells=RV("uniform", float(args.init_cells_min), float(args.init_cells_max - args.init_cells_min)),
    )

    rng = np.random.default_rng(args.seed)
    model_func = make_pyabc_model(obs, args.start_step, DEFAULTS, rng)
    dist_func = make_wasserstein_distance_scaled(obs)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    db_file = Path(args.db_file)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    db_uri = f"sqlite:///{db_file.resolve()}"

    sampler = MulticoreEvalParallelSampler()
    pop_strategy = ConstantPopulationSize(args.popsize)
    abc = ABCSMC(models=model_func, parameter_priors=prior, distance_function=dist_func,
                 population_size=pop_strategy, sampler=sampler)
    history = abc.new(db_uri, obs_full_dict)
    print(f"pyABC IQR-scaled Wasserstein run started. DB: {db_uri}")
    history = abc.run(minimum_epsilon=args.min_eps, max_nr_populations=args.max_pops)
    print(f"Done. n_populations={history.n_populations}")

if __name__ == "__main__":
    main()
