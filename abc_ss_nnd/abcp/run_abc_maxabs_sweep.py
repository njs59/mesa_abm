
#!/usr/bin/env python3
"""
ABC-SMC with MaxAbs scaling + L2, updated to use SSNND_med (disc approximation)
as the sole spatial statistic alongside S0, S1, S2.
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

# ---- schema helpers (unchanged) ----
def _coerce_merge_params(params: dict) -> None:
    merge = params.setdefault("merge", {})
    if "p_merge" in merge:
        p = float(merge["p_merge"])
        merge["p_merge"] = max(0.0, min(1.0, p))
        merge.pop("prob_contact_merge", None)
        merge.pop("adhesion", None)
        return
    pcm = merge.pop("prob_contact_merge", None)
    adh = merge.pop("adhesion", None)
    if pcm is not None or adh is not None:
        pcm_val = float(pcm) if pcm is not None else 1.0
        adh_val = float(adh) if adh is not None else 1.0
        p = pcm_val * adh_val
        merge["p_merge"] = max(0.0, min(1.0, p))
        return
    merge["p_merge"] = 0.9

def _ensure_physics_defaults(params: dict) -> None:
    phys = params.setdefault("physics", {})
    phys.setdefault("soft_separate", True)
    phys.setdefault("softness", 0.15)
    phys.setdefault("fragment_minsep_factor", 1.1)

# ---- Model factory ----
def make_model_factory(seed: int = 42):
    def factory(params_dict):
        return ClustersModel(params=params_dict, seed=seed)
    return factory

# ---- ABC runner ----
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
    show_progress: bool,
):
    obs_sorted = obs_df.sort_values("timestep").reset_index(drop=True)
    obs_mat = obs_sorted[stats].to_numpy(float)
    T, K = obs_mat.shape

    scaler = MaxAbsScaler()
    scaler.fit(obs_mat)
    obs_scaled = scaler.transform(obs_mat).flatten()

    model_factory = make_model_factory(seed=seed)

    def abm_model(particle):
        params = particle_to_params(particle, motion=motion, speed_dist=speed)
        _coerce_merge_params(params)
        _ensure_physics_defaults(params)
        sim_mat = simulate_timeseries(
            model_factory,
            params=params,
            total_steps=total_steps,
            sample_steps=tuple(timesteps),
        )
        full_order = ["S0", "S1", "S2", "SSNND_med"]
        idx = [full_order.index(s) for s in stats]
        sim_sel = sim_mat[:, idx]
        sim_scaled = scaler.transform(sim_sel).flatten()
        return {f"y_{i}": float(v) for i, v in enumerate(sim_scaled)}

    observation = {f"y_{i}": float(v) for i, v in enumerate(obs_scaled)}

    def l2_distance(sim, obs):
        s = np.array([sim[f"y_{i}"] for i in range(T * K)], float)
        o = np.array([obs[f"y_{i}"] for i in range(T * K)], float)
        return float(np.sqrt(np.sum((s - o) ** 2)))

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
    ap = argparse.ArgumentParser(description="ABC-SMC with SSNND_med (disc) + S0,S1,S2")
    ap.add_argument("--db", type=str, default="results/abc_maxabs.db")
    ap.add_argument("--db_template", type=str, default=None)
    ap.add_argument("--observed_ts", type=str, required=True)
    ap.add_argument("--priors_yaml", type=str, default="priors.yaml")
    ap.add_argument("--motion", type=str, default="isotropic")
    ap.add_argument("--speed", type=str, default="constant")
    ap.add_argument("--popsize", type=int, default=200)
    ap.add_argument("--maxgen", type=int, default=12)
    ap.add_argument("--min_eps", type=float, default=0.5)
    ap.add_argument("--total_steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed_increment", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--show_progress", action="store_true")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True)
    obs_df = pd.read_csv(args.observed_ts)

    # Required stats in observed CSV
    stats = ["S0", "S1", "S2", "SSNND_med"]
    for s in stats:
        if s not in obs_df.columns:
            raise ValueError(f"Missing {s} in observed CSV.")

    timesteps = obs_df["timestep"].astype(int).tolist()
    prior = load_priors(args.priors_yaml if Path(args.priors_yaml).exists() else None)

    def expand(arg, valid):
        a = arg.lower()
        if a == "all":
            return valid
        if "," in arg:
            parts = [x.strip() for x in arg.split(",") if x.strip()]
            return parts
        return [arg]

    valid_motion = ["isotropic", "persistent"]
    valid_speed = ["constant", "lognorm", "gamma", "weibull"]

    motion_list = expand(args.motion, valid_motion)
    speed_list = expand(args.speed, valid_speed)
    do_sweep = (len(motion_list) > 1 or len(speed_list) > 1)

    template = args.db_template if args.db_template else None

    if do_sweep:
        if not template:
            raise ValueError("You must supply --db_template when sweeping.")
        combo_idx = 0
        for motion in motion_list:
            for speed in speed_list:
                seed = args.seed + combo_idx * max(1, args.seed_increment)
                db_path = template.format(motion=motion, speed=speed)
                print(f"[SWEEP] motion={motion} speed={speed} seed={seed}")
                print(f"[DB] {db_path}")
                run_one(
                    db_path=Path(db_path),
                    obs_df=obs_df,
                    stats=stats,
                    timesteps=timesteps,
                    prior=prior,
                    popsize=args.popsize,
                    maxgen=args.maxgen,
                    min_eps=args.min_eps,
                    total_steps=args.total_steps,
                    motion=motion,
                    speed=speed,
                    seed=seed,
                    workers=args.workers,
                    show_progress=args.show_progress,
                )
                combo_idx += 1
        print("Sweep finished.")
        return

    # single run
    db_path = template.format(motion=motion_list[0], speed=speed_list[0]) if template else args.db
    print(f"[SINGLE RUN] DB: {db_path}")
    run_one(
        db_path=Path(db_path),
        obs_df=obs_df,
        stats=stats,
        timesteps=timesteps,
        prior=prior,
        popsize=args.popsize,
        maxgen=args.maxgen,
        min_eps=args.min_eps,
        total_steps=args.total_steps,
        motion=motion_list[0],
        speed=speed_list[0],
        seed=args.seed,
        workers=args.workers,
        show_progress=args.show_progress,
    )

if __name__ == "__main__":
    main()
