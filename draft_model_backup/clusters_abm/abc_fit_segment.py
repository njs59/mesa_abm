
# abc_fit_segment.py
# Fit Mesa ABM parameters by ABC against a time segment t >= start_step (default 20).
# Includes progress bars via tqdm (with a safe fallback).

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# --- Progress bar (tqdm) with safe fallback
try:
    from tqdm.auto import tqdm  # best UX in terminals & notebooks
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        # Minimal fallback: no external dependency needed.
        # Just yields items without a visual bar.
        return iterable if iterable is not None else range(kwargs.get("total", 0))

# --- Import your ABM (adjust package/script imports as needed)
try:
    # When running as a module: python -m clusters_abm.abc_fit_segment
    from .clusters_model import ClustersModel  # type: ignore
    from .utils import DEFAULTS               # type: ignore
except Exception:
    # When running as a standalone script in the same folder
    from clusters_model import ClustersModel
    from utils import DEFAULTS


# ---------------------------
# 1) Summary statistics at current state
# ---------------------------
def compute_summary_from_model(model) -> Tuple[float, float, float]:
    """
    Return (S0, S1, S2) for the *current* model state.

    Definitions:
      S0 = number of alive clusters,
      S1 = mean cluster size,
      S2 = mean squared cluster size (E[size^2]).
    """
    ids, pos, radii, sizes, speeds = model._log_alive_snapshot()
    n = len(sizes)
    if n == 0:
        return 0.0, 0.0, 0.0
    s0 = float(n)
    s1 = float(np.mean(sizes))
    s2 = float(np.mean(sizes ** 2))
    return s0, s1, s2


def simulate_summaries(params: Dict, steps: int, seed: int) -> np.ndarray:
    """
    Build a ClustersModel with 'params' and run for 'steps' time points,
    returning (steps, 3) with [S0, S1, S2] at each time.
    """
    model = ClustersModel(params=params, seed=seed)

    out = np.zeros((steps, 3), dtype=float)
    # t=0 snapshot
    out[0, :] = compute_summary_from_model(model)

    # steps-1 additional updates
    for t in range(1, steps):
        model.step()
        out[t, :] = compute_summary_from_model(model)

    return out


# ---------------------------
# 2) Priors
# ---------------------------
@dataclass
class Prior:
    name: str
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


def build_default_priors() -> List[Prior]:
    """
    Uniform priors on key parameters (monoculture: proliferative phenotype).
    Adjust ranges to your biology.
    """
    return [
        Prior("phenotypes.proliferative.speed_base", 0.5, 6.0),
        Prior("phenotypes.proliferative.prolif_rate", 5e-4, 2e-2),
        Prior("phenotypes.proliferative.adhesion", 0.2, 1.0),
        Prior("phenotypes.proliferative.fragment_rate", 0.0, 5e-3),
        Prior("merge.prob_contact_merge", 0.2, 1.0),
    ]


def set_in_params(base: Dict, dotted: str, value: float) -> None:
    """In-place write 'value' to nested dict 'base' using dotted path."""
    keys = dotted.split(".")
    d = base
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = float(value)


def make_params_from_particle(defaults: Dict, particle: Dict[str, float]) -> Dict:
    """Create a copy of DEFAULTS with the particle values inserted."""
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
    }

    for k, v in particle.items():
        if ("rate" in k or "prob" in k) and v < 0:
            v = 0.0
        set_in_params(params, k, v)

    params["init"]["phenotype"] = "proliferative"
    return params


# ---------------------------
# 3) Distance over time–segment
# ---------------------------
def distance_segment(sim: np.ndarray, obs: np.ndarray, start_step: int, eps: float = 1e-12) -> float:
    """
    Normalised MAE over segment t >= start_step and all stats.
    Each summary is normalised by the observed segment std dev.
    """
    assert sim.shape == obs.shape
    sim_seg = sim[start_step:, :]
    obs_seg = obs[start_step:, :]
    obs_std = np.std(obs_seg, axis=0, ddof=1) + eps
    err = np.abs(sim_seg - obs_seg) / obs_std
    return float(np.mean(err))


def evaluate_particle_segment(
    particle: Dict[str, float],
    obs: np.ndarray,
    start_step: int,
    replicates: int,
    base_defaults: Dict,
    rng: np.random.Generator,
    show_pb_repl: bool = False,
) -> float:
    """
    Run 'replicates' simulations for the particle and return mean distance over t >= start_step.
    """
    steps = obs.shape[0]
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=replicates, dtype=np.int64)

    dists = []
    iterator = tqdm(seeds, desc="Replicates", leave=False) if show_pb_repl else seeds
    for s in iterator:
        params = make_params_from_particle(base_defaults, particle)
        sim = simulate_summaries(params=params, steps=steps, seed=int(s))
        dists.append(distance_segment(sim, obs, start_step=start_step))
    return float(np.mean(dists))


# ---------------------------
# 4) ABC rejection (time–segment) with progress bars
# ---------------------------
def abc_rejection_segment(
    obs_csv: str,
    start_step: int = 20,
    n_particles: int = 400,
    accept_frac: float = 0.10,
    replicates: int = 3,
    seed: int = 42,
    out_csv: str = "results/abc_posterior_segment.csv",
    pb_replicates: bool = False,
) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    # Load observed summaries
    obs_df = pd.read_csv(obs_csv)
    if not all(col in obs_df.columns for col in ["S0", "S1", "S2"]):
        raise ValueError("Observed CSV must contain columns: S0, S1, S2.")

    obs = obs_df[["S0", "S1", "S2"]].to_numpy(dtype=float)
    steps = obs.shape[0]
    if start_step < 0 or start_step >= steps:
        raise ValueError(f"start_step must be in [0, {steps-1}], got {start_step}.")

    priors = build_default_priors()
    rng = np.random.default_rng(seed)

    # Draw prior particles
    particles: List[Dict[str, float]] = []
    for _ in range(n_particles):
        p = {}
        for pr in priors:
            p[pr.name] = pr.sample(rng)
        particles.append(p)

    # Evaluate particles with a progress bar
    base_defaults = DEFAULTS
    distances = np.zeros(n_particles, dtype=float)

    running_mean = 0.0
    for i in tqdm(range(n_particles), desc="Evaluating particles"):
        d = evaluate_particle_segment(
            particle=particles[i],
            obs=obs,
            start_step=start_step,
            replicates=replicates,
            base_defaults=base_defaults,
            rng=rng,
            show_pb_repl=pb_replicates,
        )
        distances[i] = d
        running_mean = distances[: i + 1].mean()
        # Optional: update the bar description with current running mean
        # (tqdm does not allow resetting desc per-iteration easily, so we print occasionally)
        if (i + 1) % max(1, n_particles // 10) == 0:
            tqdm.write(f"[{i+1}/{n_particles}] mean distance so far = {running_mean:.3f}")

    # Accept the best fraction
    n_accept = max(1, int(math.ceil(accept_frac * n_particles)))
    idx = np.argsort(distances)[:n_accept]
    accepted = [particles[i] for i in idx]
    accepted_dist = distances[idx]

    # Save posterior
    post = pd.DataFrame(accepted)
    post["distance"] = accepted_dist
    post.to_csv(out_csv, index=False)

    # Report
    print("\n--- ABC results (time–segment) ---")
    print(f"Segment: t >= {start_step}")
    print(f"Accepted: {n_accept}/{n_particles} (ε = {distances[idx[-1]]:.3f})")
    print(post.describe(percentiles=[0.05, 0.5, 0.95]))


# ---------------------------
# 5) CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ABC rejection to fit Mesa ABM parameters on a time–segment (default: t >= 20)."
    )
    parser.add_argument("--obs_csv", type=str, default="INV_summary_stats.csv",
                        help="CSV with columns S0,S1,S2 across timepoints; S2 = mean squared size.")
    parser.add_argument("--start_step", type=int, default=20, help="Start of the segment (fit to t >= start_step).")
    parser.add_argument("--n_particles", type=int, default=400, help="Number of prior particles.")
    parser.add_argument("--accept_frac", type=float, default=0.10, help="Fraction of best particles to accept.")
    parser.add_argument("--replicates", type=int, default=3, help="Sim replicates per particle (few for quick tests).")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--out_csv", type=str, default="results/abc_posterior_segment.csv",
                        help="Where to write posterior samples.")
    parser.add_argument("--pb_replicates", action="store_true",
                        help="Show an inner progress bar for per-particle replicates (off by default).")
    args = parser.parse_args()

    abc_rejection_segment(
        obs_csv=args.obs_csv,
        start_step=args.start_step,
        n_particles=args.n_particles,
        accept_frac=args.accept_frac,
        replicates=args.replicates,
        seed=args.seed,
        out_csv=args.out_csv,
        pb_replicates=args.pb_replicates,
    )
