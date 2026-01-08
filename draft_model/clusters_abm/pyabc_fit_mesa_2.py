
# pyabc_fit_mesa.py
# ABC–SMC with pyabc for Mesa ABM, now using speed distribution parameters.

import argparse
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import pyabc
from pyabc import ABCSMC, Distribution, RV
from pyabc.populationstrategy import ConstantPopulationSize
from pyabc.sampler import MulticoreEvalParallelSampler

try:
    from .clusters_model import ClustersModel  # type: ignore
    from .utils import DEFAULTS  # type: ignore
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

# --- Parameter mapping ---
def set_in_params(base: Dict, dotted: str, value: float) -> None:
    keys = dotted.split(".")
    d = base
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = float(value)


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

    # Switch to distribution mode for movement
    params["movement"]["mode"] = "distribution"
    params["movement"]["distribution"] = "lognorm"
    params["movement"]["dist_params"] = {
        "s": particle["speed_sdlog"],  # shape parameter
        "scale": np.exp(particle["speed_meanlog"]),  # scale = exp(meanlog)
    }

    # Other biological parameters
    for k, v in particle.items():
        if ("rate" in k or "prob" in k) and v < 0:
            v = 0.0
        if not k.startswith("speed_"):  # skip speed params
            set_in_params(params, k, v)

    params["init"]["phenotype"] = "proliferative"
    return params

# --- pyabc model closure ---
def make_pyabc_model(obs: np.ndarray, start_step: int, steps: int, base_defaults: Dict, rng: np.random.Generator):
    def model(p: Dict) -> Dict:
        sim_params = make_params_from_particle(base_defaults, p)
        seed = int(rng.integers(0, 2**31 - 1))
        sim = simulate_timeseries(sim_params, steps=steps, seed=seed)
        seg = sim[start_step:, :]
        return {"S0": seg[:, 0], "S1": seg[:, 1], "S2": seg[:, 2]}
    return model

# --- Distance function ---
def make_segment_distance(obs: np.ndarray, start_step: int):
    obs_seg = obs[start_step:, :]
    eps = 1e-12
    obs_std = np.std(obs_seg, axis=0, ddof=1) + eps
    def distance(x: Dict, x0: Dict) -> float:
        sim_mat = np.column_stack([x["S0"], x["S1"], x["S2"]])
        obs_mat = np.column_stack([x0["S0"], x0["S1"], x0["S2"]])
        n = min(sim_mat.shape[0], obs_mat.shape[0])
        sim_mat, obs_mat = sim_mat[:n, :], obs_mat[:n, :]
        err = np.abs(sim_mat - obs_mat) / obs_std
        return float(np.mean(err))
    return pyabc.distance.FunctionDistance(distance)

# --- CLI ---
def main():
    parser = argparse.ArgumentParser(description="ABC–SMC for Mesa ABM with speed distribution.")
    parser.add_argument("--obs_csv", type=str, default="INV_summary_stats.csv")
    parser.add_argument("--start_step", type=int, default=20)
    parser.add_argument("--popsize", type=int, default=200)
    parser.add_argument("--max_pops", type=int, default=8)
    parser.add_argument("--min_eps", type=float, default=0.5)
    parser.add_argument("--db_file", type=str, default="clusters_abm/results/pyabc_runs.db")
    parser.add_argument("--results_dir", type=str, default="clusters_abm/results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    obs_df = pd.read_csv(args.obs_csv)
    obs = obs_df[["S0", "S1", "S2"]].to_numpy(dtype=float)
    steps = obs.shape[0]
    obs_seg = {"S0": obs[args.start_step:, 0], "S1": obs[args.start_step:, 1], "S2": obs[args.start_step:, 2]}

    # Priors for speed distribution (lognormal)
    prior = Distribution(
        speed_meanlog=RV("uniform", 0.5, 2.5 - 0.5),  # log-scale mean
        speed_sdlog=RV("uniform", 0.7, 3.0 - 0.7),     # log-scale std
        prolif_rate=RV("uniform", 5e-4, 2e-2 - 5e-4),
        adhesion=RV("uniform", 0.2, 1.0 - 0.2),
        fragment_rate=RV("uniform", 0.0, 5e-3),
        merge_prob=RV("uniform", 0.2, 1.0 - 0.2),
    )

    rng = np.random.default_rng(args.seed)
    model_func = make_pyabc_model(obs, args.start_step, steps, DEFAULTS, rng)
    dist_func = make_segment_distance(obs, args.start_step)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    db_file = Path(args.db_file)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    db_uri = f"sqlite:///{db_file.resolve()}"

    sampler = MulticoreEvalParallelSampler()
    pop_strategy = ConstantPopulationSize(args.popsize)

    abc = ABCSMC(models=model_func, parameter_priors=prior, distance_function=dist_func,
                 population_size=pop_strategy, sampler=sampler)

    history = abc.new(db_uri, obs_seg)
    print(f"pyabc run started. DB: {db_uri}")
    history = abc.run(minimum_epsilon=args.min_eps, max_nr_populations=args.max_pops)

    print(f"Done. n_populations={history.n_populations}")
    df, w = history.get_distribution(m=0, t=history.max_t)
    print("\nPosterior summary:")
    for name in ["speed_meanlog", "speed_sdlog", "prolif_rate", "adhesion", "fragment_rate", "merge_prob"]:
        vals = df[name].to_numpy()
        def wq(q):
            idx = np.argsort(vals)
            v = vals[idx]
            ww = w[idx] / np.sum(w[idx])
            c = np.cumsum(ww)
            return v[np.searchsorted(c, q)]
        print(f"{name:15s}: median={wq(0.5):.6g}, 5%={wq(0.05):.6g}, 95%={wq(0.95):.6g}")

if __name__ == "__main__":
    main()
