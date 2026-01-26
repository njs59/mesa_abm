
#!/usr/bin/env python3
"""
ABC-SMC using MaxAbsScaler to rescale summary stats to [-1,1]
and using L2 norm between rescaled observed and simulated stats.

Updates for new ABM API:
- Uses a single merge probability parameter: params["merge"]["p_merge"].
- Backwards compatibility: if particles/priors provide legacy
  'prob_contact_merge' and/or 'adhesion', they are converted to 'p_merge'
  (product, clamped to [0,1]) and legacy keys are removed.
- Ensures physics knobs for Fix 1 (soft separation) and Fix 5 (fragment
  min separation) are present unless explicitly provided.
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


def _coerce_merge_params(params: dict) -> None:
    """
    Ensure params["merge"] contains ONLY 'p_merge' in [0,1].
    Accept legacy inputs and convert:
      - If 'p_merge' already present: clamp to [0,1].
      - Else, if 'prob_contact_merge' and/or 'adhesion' are present:
          p_merge = clamp( (prob_contact_merge if present else 1.0)
                          * (adhesion if present else 1.0) )
        Remove legacy keys.
      - Else, default p_merge=0.9 (kept from previous default behaviour).

    Modifies 'params' in place.
    """
    merge = params.setdefault("merge", {})
    if "p_merge" in merge:
        p = float(merge["p_merge"])
        merge["p_merge"] = max(0.0, min(1.0, p))
        # Drop legacy keys if still present
        merge.pop("prob_contact_merge", None)
        merge.pop("adhesion", None)
        return

    # Legacy keys path
    pcm = merge.pop("prob_contact_merge", None)
    adh = merge.pop("adhesion", None)
    if pcm is not None or adh is not None:
        pcm_val = float(pcm) if pcm is not None else 1.0
        adh_val = float(adh) if adh is not None else 1.0
        p = pcm_val * adh_val
        merge["p_merge"] = max(0.0, min(1.0, float(p)))
        return

    # Nothing provided -> sensible default
    merge["p_merge"] = 0.9


def _ensure_physics_defaults(params: dict) -> None:
    """
    Ensure physics parameters for Fix 1 & 5 exist unless the prior/particle
    already supplied them.
    """
    phys = params.setdefault("physics", {})
    phys.setdefault("soft_separate", True)
    phys.setdefault("softness", 0.15)
    phys.setdefault("fragment_minsep_factor", 1.1)


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
        # Translate particle -> nested params for ABM
        params = particle_to_params(particle, motion=motion, speed_dist=speed)

        # --- Ensure new ABM schema compatibility ---
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
    ap = argparse.ArgumentParser(description="ABC-SMC with MaxAbsScaler + L2 norm (updated for single p_merge)")
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
    speed_list = _expand_list(args.speed, valid=valid_speed, arg_name="--speed") \
        if args.speed in ("all",) or "," in args.speed else [args.speed]

    do_sweep = len(motion_list) > 1 or len(speed_list) > 1

    # DB template logic
    if do_sweep:
        db_template = args.db_template
        if db_template is None or db_template.strip() == "":
            # Default template if none provided
            suffix = "_noGR" if args.no_gr else ""
            db_template = "results/abc_maxabs" + "_{motion}" + "_{speed}" + f"{suffix}.db"
            # make sure the braces survive this format() later
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
                    db_path = f"{db_path}{suffix}"
                elif suffix and db_path.endswith(".db") and suffix not in db_path:
                    stem = db_path[:-3]
                    db_path = f"{stem}{suffix}.db"

                print(f"\n[run] motion={motion} speed={speed} seed={seed}")
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
