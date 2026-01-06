
# pyabc_fit_mesa.py
# ABC–SMC with pyabc for your Mesa ABM, fitting over the segment t >= start_step.
# British-English comments; supports module mode (recommended) and script mode.

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# --- pyabc imports (ABC–SMC orchestration, priors, population strategies, sampler)
import pyabc
from pyabc import ABCSMC, Distribution, RV
from pyabc.populationstrategy import ConstantPopulationSize  # or AdaptivePopulationSize later
from pyabc.sampler import MulticoreEvalParallelSampler       # multi-core dynamic sampler

# --- import your Mesa model (support both module and script modes)
try:
    # Module mode: python -m clusters_abm.pyabc_fit_mesa ...
    from .clusters_model import ClustersModel  # type: ignore
    from .utils import DEFAULTS                # type: ignore
except Exception:
    # Script mode: python pyabc_fit_mesa.py ...
    from clusters_model import ClustersModel
    from utils import DEFAULTS


# ---------------------------
# Summary statistics (current state)
# ---------------------------
def compute_summary_from_model(model) -> Tuple[float, float, float]:
    """
    Return (S0, S1, S2) for the *current* model state.

      S0 = number of alive clusters
      S1 = mean cluster size
      S2 = mean squared cluster size, E[size^2]
    """
    ids, pos, radii, sizes, speeds = model._log_alive_snapshot()
    n = len(sizes)
    if n == 0:
        return 0.0, 0.0, 0.0
    s0 = float(n)
    s1 = float(np.mean(sizes))
    s2 = float(np.mean(sizes ** 2))
    return s0, s1, s2


def simulate_timeseries(params: Dict, steps: int, seed: int) -> np.ndarray:
    """
    Run the Mesa model for `steps` ticks and collect (S0,S1,S2) each tick.
    Returns shape (steps, 3).
    """
    # Seed via Model.__init__(seed=...) for reproducible stdlib+NumPy RNGs in Mesa.
    # (Mesa best-practices recommend passing the seed into Model.__init__.)  # See refs.
    model = ClustersModel(params=params, seed=seed)

    out = np.zeros((steps, 3), dtype=float)
    # t=0 snapshot
    out[0, :] = compute_summary_from_model(model)
    # advance
    for t in range(1, steps):
        model.step()
        out[t, :] = compute_summary_from_model(model)
    return out


# ---------------------------
# Build params from particle
# ---------------------------
def set_in_params(base: Dict, dotted: str, value: float) -> None:
    keys = dotted.split(".")
    d = base
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = float(value)


def make_params_from_particle(defaults: Dict, particle: Dict[str, float]) -> Dict:
    # shallow copy per level (DEFAULTS is simple nested dicts)
    params = {
        "space": dict(defaults["space"]),
        "time": dict(defaults["time"]),
        "physics": dict(defaults["physics"]),
        "phenotypes": {
            "proliferative": dict(defaults["phenotypes"]["proliferative"]),
            "invasive":      dict(defaults["phenotypes"]["invasive"]),
        },
        "merge": dict(defaults["merge"]),
        "init":  dict(defaults["init"]),
    }
    for k, v in particle.items():
        if ("rate" in k or "prob" in k) and v < 0:
            v = 0.0
        set_in_params(params, k, v)
    # Monoculture initial condition (your workflow)
    params["init"]["phenotype"] = "proliferative"
    return params


# ---------------------------
# pyabc model(): sampled parameters -> simulated segment dict
# ---------------------------
def make_pyabc_model(
    obs: np.ndarray,
    start_step: int,
    steps: int,
    base_defaults: Dict,
    rng: np.random.Generator,
):
    """
    Return a closure 'model(p) -> dict' for pyabc.
    It generates a deterministic per-simulation seed from `rng` and returns dicts
    containing arrays for the segment (t >= start_step).
    """
    def model(p: Dict) -> Dict:
        sim_params = make_params_from_particle(base_defaults, {
            "phenotypes.proliferative.speed_base":     p["speed_base"],
            "phenotypes.proliferative.prolif_rate":    p["prolif_rate"],
            "phenotypes.proliferative.adhesion":       p["adhesion"],
            "phenotypes.proliferative.fragment_rate":  p["fragment_rate"],
            "merge.prob_contact_merge":                p["merge_prob"],
        })
        # draw a per-simulation seed from the local RNG seeded by --seed
        seed = int(rng.integers(0, 2**31 - 1))
        sim = simulate_timeseries(sim_params, steps=steps, seed=seed)
        seg = sim[start_step:, :]  # (T_seg, 3)
        return {"S0": seg[:, 0], "S1": seg[:, 1], "S2": seg[:, 2]}
    return model


# ---------------------------
# Distance on the time-segment t >= start_step
# ---------------------------
def make_segment_distance(obs: np.ndarray, start_step: int):
    """
    Normalised MAE over the segment t>=start_step.
    We normalise each summary by its *observed segment* standard deviation.

    Returns a callable suitable for pyabc's FunctionDistance wrapper.
    """
    obs_seg = obs[start_step:, :]  # (T_seg, 3)
    eps = 1e-12
    obs_std = np.std(obs_seg, axis=0, ddof=1) + eps  # (3,)

    def distance(x: Dict, x0: Dict) -> float:
        sim_mat = np.column_stack([x["S0"], x["S1"], x["S2"]])
        obs_mat = np.column_stack([x0["S0"], x0["S1"], x0["S2"]])
        # Robust length alignment if needed
        if sim_mat.shape != obs_mat.shape:
            n = min(sim_mat.shape[0], obs_mat.shape[0])
            sim_mat = sim_mat[:n, :]
            obs_mat = obs_mat[:n, :]
        err = np.abs(sim_mat - obs_mat) / obs_std  # broadcast (T_seg,3)/(3,)
        return float(np.mean(err))

    return pyabc.distance.FunctionDistance(distance)


# ---------------------------
# CLI + pyabc run
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="pyabc ABC–SMC for Mesa ABM on segment t>=start_step.")
    parser.add_argument("--obs_csv", type=str, default="INV_summary_stats.csv",
                        help="CSV with columns S0,S1,S2 across time points (S2 = mean squared size).")
    parser.add_argument("--start_step", type=int, default=20, help="Segment start: fit to t >= start_step.")
    parser.add_argument("--popsize", type=int, default=200, help="Particles per population.")
    parser.add_argument("--max_pops", type=int, default=8, help="Max ABC–SMC populations.")
    parser.add_argument("--min_eps", type=float, default=0.5, help="Stop if epsilon <= min_eps.")
    parser.add_argument("--db_file", type=str, default="clusters_abm/results/pyabc_runs.db",
                        help="SQLite DB file path (relative or absolute).")
    parser.add_argument("--results_dir", type=str, default="clusters_abm/results",
                        help="Directory for DB and logs (created if missing).")
    parser.add_argument("--seed", type=int, default=42, help="Global seed for deterministic runs.")
    args = parser.parse_args()

    # Load observed summaries
    csv_path = Path(args.obs_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Observed CSV not found: {csv_path.resolve()}")
    obs_df = pd.read_csv(csv_path)
    if not all(c in obs_df.columns for c in ["S0", "S1", "S2"]):
        raise ValueError("Observed CSV must contain columns: S0, S1, S2")
    obs = obs_df[["S0", "S1", "S2"]].to_numpy(dtype=float)
    steps = obs.shape[0]
    if args.start_step < 0 or args.start_step >= steps:
        raise ValueError(f"start_step must be in [0, {steps-1}]")

    # Observed dict for pyabc (time-segment arrays)
    obs_seg = {
        "S0": obs[args.start_step:, 0],
        "S1": obs[args.start_step:, 1],
        "S2": obs[args.start_step:, 2],
    }

    # Priors (uniform); note scipy's uniform uses (loc, scale) where scale = high - low.
    prior = Distribution(
        speed_base   = RV("uniform", 0.5,   20.0 - 0.5),
        prolif_rate  = RV("uniform", 5e-4,  2e-2 - 5e-4),
        adhesion     = RV("uniform", 0.2,   1.0 - 0.2),
        fragment_rate= RV("uniform", 0.0,   5e-3 - 0.0),
        merge_prob   = RV("uniform", 0.2,   1.0 - 0.2),
    )
    base_defaults = DEFAULTS

    # Local RNG seeded by --seed to generate per-simulation seeds in the model closure
    rng = np.random.default_rng(args.seed)

    # Build pyabc model and distance
    model_func = make_pyabc_model(
        obs=obs,
        start_step=args.start_step,
        steps=steps,
        base_defaults=base_defaults,
        rng=rng,
    )
    dist_func = make_segment_distance(obs=obs, start_step=args.start_step)

    # Storage: guarantee directory exists, then build explicit sqlite URI
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    db_file = Path(args.db_file)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    db_uri = f"sqlite:///{db_file.resolve()}"   # explicit SQLAlchemy URI
    # Alternatively: pyabc.create_sqlite_db_id(file_=...) also returns a valid sqlite URI,
    # used widely in pyabc examples. [1](https://zenodo.org/records/14632533)

    # Multi-core dynamic sampler (fastest default on desktop/workstation)  [3](https://mesa.readthedocs.io/latest/migration_guide.html)
    sampler = MulticoreEvalParallelSampler()

    # Constant population size (you can switch to AdaptivePopulationSize later)  [4](https://github.com/ClaireGuerin/mesa-fish-school/)
    pop_strategy = ConstantPopulationSize(args.popsize)

    # Orchestrate ABC–SMC  (model returns dicts; distance can be a callable wrapper) [5](https://mesa.readthedocs.io/stable/_modules/experimental/continuous_space/continuous_space_agents.html)
    abc = ABCSMC(
        models=model_func,
        parameter_priors=prior,
        distance_function=dist_func,
        population_size=pop_strategy,
        sampler=sampler,
    )

    # Start new run
    history = abc.new(db_uri, obs_seg)
    print(f"pyabc run started. DB: {db_uri}")

    # Run until epsilon small or population cap reached
    history = abc.run(minimum_epsilon=args.min_eps, max_nr_populations=args.max_pops)
    last_eps = history.get_all_populations()["epsilon"].iloc[-1]
    print(f"Done. n_populations={history.n_populations}, last_eps={last_eps:.3f}")
    print(f"Results DB: {db_uri}")

    # Quick posterior summary (last population, weighted quantiles)
    df, w = history.get_distribution(m=0, t=history.max_t)
    print("\nPosterior summary (last population, weighted):")
    for name in ["speed_base", "prolif_rate", "adhesion", "fragment_rate", "merge_prob"]:
        vals = df[name].to_numpy()
        # weighted quantiles
        def wq(q):
            idx = np.argsort(vals)
            v = vals[idx]
            ww = w[idx] / np.sum(w[idx])
            c = np.cumsum(ww)
            return v[np.searchsorted(c, q)]
        print(f"  {name:15s}: median={wq(0.5):.6g}, 5%={wq(0.05):.6g}, 95%={wq(0.95):.6g}")


if __name__ == "__main__":
    main()
