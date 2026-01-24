
#!/usr/bin/env python3
"""
ABC-SMC using MaxAbsScaler to rescale summary stats to [-1,1]
and using L2 norm between rescaled observed and simulated stats.

Enhancements:
- Population-level progress bar via pyABC sampler.show_progress.
- Built-in sweep over multiple motion and speed (noise) models from the CLI.
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
    show_progress: bool = True,
):
    # Sort observed
    obs_sorted = obs_df.sort_values("timestep").reset_index(drop=True)
    obs_mat = obs_sorted[stats].to_numpy(float)  # shape (T,K)
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
        sim_sel = sim_mat[:, idx]  # T x K
        sim_scaled = scaler.transform(sim_sel)  # apply same scaler
        sim_scaled_vec = sim_scaled.flatten()
        return {f"y_{i}": float(v) for i, v in enumerate(sim_scaled_vec)}

    # Observation vector (scaled) for pyABC API
    observation = {f"y_{i}": float(v) for i, v in enumerate(obs_scaled_vec)}

    # L2 distance in rescaled space
    def l2_distance(sim, obs):
        sim_v = np.array([sim[f"y_{i}"] for i in range(T * K)], float)
        obs_v = np.array([obs[f"y_{i}"] for i in range(T * K)], float)
        return float(np.sqrt(np.sum((sim_v - obs_v) ** 2)))

    # Sampler with progress bar
    sampler = None
    try:
        # Prefer multicore when workers > 1, otherwise single-core
        if workers and workers > 1:
            from pyabc.sampler import MulticoreEvalParallelSampler
            sampler = MulticoreEvalParallelSampler(n_procs=workers)
        else:
            from pyabc.sampler import SingleCoreSampler
            sampler = SingleCoreSampler()
        # Enable per-population progress bar
        sampler.show_progress = bool(show_progress)
    except Exception as e:
        # Fall back to default sampler without progress if imports failed
        print(f"[warn] Could not configure sampler with progress ({e}). Falling back.", file=sys.stderr)
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


def _expand_list(arg_val: str, *, valid: list[str], arg_name: str) -> list[str]:
    """
    Expand a comma-separated list or 'all' into a validated list.
    """
    if arg_val.lower() == "all":
        return valid
    parts = [p.strip() for p in arg_val.split(",") if p.strip()]
    bad = [p for p in parts if p not in valid]
    if bad:
        raise ValueError(f"Invalid {arg_name}: {bad}. Valid: {valid}")
    return parts


def main():
    ap = argparse.ArgumentParser(description="ABC-SMC with MaxAbsScaler + L2 norm")
    ap.add_argument("--db", type=str, default="results/abc_maxabs.db",
                    help="SQLite DB path (used only for single run).")
    ap.add_argument("--db_template", type=str, default=None,
                    help="DB template for sweeps. You may use {motion} and {speed}. "
                         "Example: results/abc_maxabs_{motion}_{speed}.db")
    ap.add_argument("--observed_ts", type=str, required=True)
    ap.add_argument("--priors_yaml", type=str, default="priors.yaml")
    ap.add_argument("--motion", type=str, default="isotropic",
                    choices=["isotropic", "persistent", "all"],
                    help="Movement model or 'all' (for sweeping).")
    ap.add_argument("--speed", type=str, default="constant",
                    choices=["constant", "lognorm", "gamma", "weibull", "all"],
                    help="Speed/noise model or 'all' (for sweeping).")
    ap.add_argument("--popsize", type=int, default=200)
    ap.add_argument("--maxgen", type=int, default=12)
    ap.add_argument("--min_eps", type=float, default=0.5)
    ap.add_argument("--total_steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed_increment", type=int, default=1,
                    help="Added to seed for each subsequent combo in a sweep.")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--show_progress", action="store_true",
                    help="Show a per-population progress bar.")
    ap.add_argument("--no_gr", action="store_true")

    args = ap.parse_args()
    Path("results").mkdir(exist_ok=True)

    # Load observed timeseries
    obs_df = pd.read_csv(args.observed_ts)

    # Stats to use
    if args.no_gr:
        stats = ["S0", "S1", "S2", "NND_med"]
    else:
        stats = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
    for s in stats:
        if s not in obs_df.columns:
            raise ValueError(f"Missing {s} in observed CSV")

    timesteps = obs_df["timestep"].astype(int).tolist()

    # Priors (allow auto-detection if priors.yaml not found)
    prior = load_priors(args.priors_yaml if Path(args.priors_yaml).exists() else None)

    valid_motion = ["isotropic", "persistent"]
    valid_speed = ["constant", "lognorm", "gamma", "weibull"]

    # Decide whether to sweep or single run
    motion_list = _expand_list(args.motion, valid=valid_motion, arg_name="--motion") \
        if args.motion in ("all",) or "," in args.motion else [args.motion]
    speed_list  = _expand_list(args.speed,  valid=valid_speed,  arg_name="--speed") \
        if args.speed in ("all",) or "," in args.speed else [args.speed]

    do_sweep = len(motion_list) > 1 or len(speed_list) > 1

    # DB template logic
    if do_sweep:
        db_template = args.db_template
        if db_template is None or db_template.strip() == "":
            # Default template if none provided
            suffix = "_noGR" if args.no_gr else ""
            db_template = f"results/abc_maxabs"+"_{motion}"+"_{speed}"+f"{suffix}.db"
            # The f-string above is split to avoid accidental format at parse time
            db_template = db_template.replace("{motion}", "{motion}").replace("{speed}", "{speed}")
    else:
        db_template = None  # single run uses --db

    # Run
    if do_sweep:
        combo_idx = 0
        for motion in motion_list:
            for speed in speed_list:
                seed = args.seed + combo_idx * max(1, args.seed_increment)
                suffix = "_noGR" if args.no_gr else ""
                db_path = db_template.format(motion=motion, speed=speed)
                # if user forgot to encode noGR in template, append here
                if suffix and not db_path.endswith(".db"):
                    # unlikely, but keep safe
                    db_path = f"{db_path}{suffix}"
                elif suffix and db_path.endswith(".db") and suffix not in db_path:
                    stem = db_path[:-3]
                    db_path = f"{stem}{suffix}.db"

                print(f"\n[run] motion={motion}  speed={speed}  seed={seed}")
                print(f"[db ] {db_path}")
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
        print("\nSweep finished.")
    else:
        # Single run
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
            motion=motion_list[0],
            speed=speed_list[0],
            seed=args.seed,
            workers=args.workers,
            show_progress=args.show_progress,
        )
        print(f"\nABC-SMC MaxAbsScaler run complete. DB saved to {args.db}")


if __name__ == "__main__":
    main()
