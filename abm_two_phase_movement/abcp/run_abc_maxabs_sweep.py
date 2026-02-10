# abcp/run_abc_maxabs_sweep.py
#!/usr/bin/env python3
"""
ABC-SMC using MaxAbsScaler to rescale summary stats and L2 distance
between rescaled observed and simulated stats.

UPDATED FOR NEW ABM:
 - No motion/speed parameters or sweep.
 - Single run: particle→params uses only biological/interaction params.
 - Physics defaults enforced (soft separation; fragment min-sep).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyabc
from sklearn.preprocessing import MaxAbsScaler

from abm.clusters_model import ClustersModel
from abcp.compute_summary import simulate_timeseries
from abcp.priors import load_priors
from abcp.abc_model_wrapper import particle_to_params


# --- Small helpers (coercions consistent with your prior scripts) ---
def _coerce_merge_params(params: dict) -> None:
    """Ensure params['merge'] contains only 'p_merge' in [0,1]."""
    merge = params.setdefault("merge", {})
    if "p_merge" in merge:
        p = float(merge["p_merge"])
        merge["p_merge"] = max(0.0, min(1.0, p))
        merge.pop("prob_contact_merge", None)
        merge.pop("adhesion", None)
        return
    # Legacy fallbacks (kept for backward compat, though you won't use them now)
    pcm = merge.pop("prob_contact_merge", None)
    adh = merge.pop("adhesion", None)
    if pcm is not None or adh is not None:
        pcm_val = float(pcm) if pcm is not None else 1.0
        adh_val = float(adh) if adh is not None else 1.0
        merge["p_merge"] = max(0.0, min(1.0, pcm_val * adh_val))
        return
    merge["p_merge"] = 0.9


def _ensure_physics_defaults(params: dict) -> None:
    """Default soft separation + fragment min-sep as per your ABM settings."""
    phys = params.setdefault("physics", {})
    phys.setdefault("soft_separate", True)
    phys.setdefault("softness", 0.15)
    phys.setdefault("fragment_minsep_factor", 1.1)


# --- Model factory ---
def make_model_factory(seed: int = 42):
    def factory(params_dict):
        return ClustersModel(params=params_dict, seed=seed)
    return factory


# --- ABC runner ---
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
    seed: int,
    workers: int,
    show_progress: bool,
):
    # order observed
    obs_sorted = obs_df.sort_values("timestep").reset_index(drop=True)
    obs_mat = obs_sorted[stats].to_numpy(float)
    T, K = obs_mat.shape

    scaler = MaxAbsScaler()
    scaler.fit(obs_mat)
    obs_scaled = scaler.transform(obs_mat).flatten()

    model_factory = make_model_factory(seed=seed)

    # pyABC model wrapper
    def abm_model(particle):
        params = particle_to_params(particle)   # <- fixed movement parameters (no speed/direction)
        _coerce_merge_params(params)
        _ensure_physics_defaults(params)
        sim_mat = simulate_timeseries(
            model_factory,
            params=params,
            total_steps=total_steps,
            sample_steps=tuple(timesteps),
        )
        full_order = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
        idx = [full_order.index(s) for s in stats]
        sim_sel = sim_mat[:, idx]
        sim_scaled = scaler.transform(sim_sel).flatten()
        return {f"y_{i}": float(v) for i, v in enumerate(sim_scaled)}

    observation = {f"y_{i}": float(v) for i, v in enumerate(obs_scaled)}

    # L2 distance in scaled space
    def l2_distance(sim, obs):
        s = np.array([sim[f"y_{i}"] for i in range(T * K)], float)
        o = np.array([obs[f"y_{i}"] for i in range(T * K)], float)
        return float(np.sqrt(np.sum((s - o) ** 2)))

    # Sampler
    sampler = None
    try:
        if workers and workers > 1:
            from pyabc.sampler import MulticoreEvalParallelSampler
            sampler = MulticoreEvalParallelSampler(n_procs=workers)
        else:
            from pyabc.sampler import SingleCoreSampler
            sampler = SingleCoreSampler()
        sampler.show_progress = bool(show_progress)
    except Exception as e:
        print(f"[warn] Sampler config failed: {e}", file=sys.stderr)
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
    return abc.run(max_nr_populations=maxgen, minimum_epsilon=min_eps)


def main():
    ap = argparse.ArgumentParser(description="ABC-SMC with MaxAbsScaler + L2 (fixed movement)")
    ap.add_argument("--db", type=str, default="results/abc_maxabs.db")
    ap.add_argument("--observed_ts", type=str, required=True)
    ap.add_argument("--priors_yaml", type=str, default="priors.yaml")
    ap.add_argument("--popsize", type=int, default=200)
    ap.add_argument("--maxgen", type=int, default=12)
    ap.add_argument("--min_eps", type=float, default=0.5)
    ap.add_argument("--total_steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--show_progress", action="store_true")
    ap.add_argument("--no_gr", action="store_true")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True)

    obs_df = pd.read_csv(args.observed_ts)
    stats = ["S0", "S1", "S2", "NND_med"] if args.no_gr else ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
    for s in stats:
        if s not in obs_df.columns:
            raise ValueError(f"Missing {s} in observed CSV.")
    timesteps = obs_df["timestep"].astype(int).tolist()

    prior = load_priors(args.priors_yaml if Path(args.priors_yaml).exists() else None)

    db_path = Path(args.db)
    print(f"\n[RUN] DB: {db_path}")
    run_one(
        db_path=db_path,
        obs_df=obs_df,
        stats=stats,
        timesteps=timesteps,
        prior=prior,
        popsize=args.popsize,
        maxgen=args.maxgen,
        min_eps=args.min_eps,
        total_steps=args.total_steps,
        seed=args.seed,
        workers=args.workers,
        show_progress=args.show_progress,
    )


if __name__ == "__main__":
    main()