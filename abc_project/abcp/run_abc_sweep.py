
#!/usr/bin/env python3
"""
Run ABC‑SMC for the clustering ABM with optional sweep mode.

Key features:
- Sweep over multiple movement models (--motions) and speed distributions (--speeds).
- Sweep over multiple random seeds (--seeds).
- One SQLite DB per run using a configurable template and optional timestamp.
- Avoid overwriting by default; auto-uniquify filenames unless --overwrite is passed.
- Optional --dry_run lists the planned jobs and DB names without executing.

Examples
--------
Single run (backwards compatible):
    python run_abc_sweep.py \
        --observed_ts observed/PRO_ABM_ready_summary.csv \
        --no_gr \
        --popsize 200 --maxgen 12 --min_eps 0.5 \
        --motion persistent \
        --speed constant \
        --seed 42 \
        --workers 8 \
        --db results/abc_single.db

Sweep mode:
    python run_abc_sweep.py \
        --observed_ts observed/PRO_ABM_ready_summary.csv \
        --no_gr \
        --popsize 200 --maxgen 12 --min_eps 0.5 \
        --motions persistent isotropic \
        --speeds constant lognorm gamma weibull \
        --seeds 42 \
        --db_template "results/abc_{obs}_{motion}_{speed}_seed{seed}.db" \
        --timestamp \
        --workers 8
"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Sequence, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

# Third-party: pyabc & your project modules
import pyabc
from abm.clusters_model import ClustersModel
from abcp.compute_summary import simulate_timeseries
from abcp.priors import load_priors
from abcp.abc_model_wrapper import particle_to_params

# Optional multicore sampler (pyabc)
try:
    from pyabc.sampler import MulticoreEvalParallelSampler
except Exception:  # pragma: no cover
    MulticoreEvalParallelSampler = None


# ----------------------------- Utilities ------------------------------------ #
def make_model_factory(seed: int = 42):
    """Return a factory that builds the model with a fixed random seed."""
    def factory(params_dict):
        return ClustersModel(params=params_dict, seed=seed)
    return factory


def unique_path(path: Path) -> Path:
    """If path exists, append _1, _2, ... to make it unique."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    parent = path.parent
    k = 1
    while True:
        p = parent / f"{stem}_{k}{suffix}"
        if not p.exists():
            return p
        k += 1


def append_to_stem(p: Path, suffix: str) -> Path:
    """
    Safely append text to a file stem without relying on Path.with_stem
    (for compatibility with older Python versions).
    """
    return p.with_name(f"{p.stem}{suffix}{p.suffix}")


def validate_observed_df(df: pd.DataFrame, required_stats: Sequence[str]) -> None:
    """Validate that observed DataFrame has expected columns."""
    if "timestep" not in df.columns:
        raise ValueError("Observed CSV must contain a 'timestep' column.")
    missing = [s for s in required_stats if s not in df.columns]
    if missing:
        raise ValueError(
            f"Observed CSV missing required columns: {', '.join(missing)}"
        )


def make_sampler(workers: int):
    """
    Create a pyabc multicore sampler if available and workers > 1.
    Tries the standard `n_procs` parameter first; falls back gracefully.
    """
    if workers and workers > 1 and MulticoreEvalParallelSampler is not None:
        # Standard pyabc signature is `n_procs`
        try:
            return MulticoreEvalParallelSampler(n_procs=workers)
        except TypeError:
            # Some forks/versions might use `processes`; try once
            try:
                return MulticoreEvalParallelSampler(processes=workers)  # type: ignore[call-arg]
            except TypeError:
                print("[warn] Could not configure MulticoreEvalParallelSampler; "
                      "falling back to single-process.")
    return None


# ----------------------------- Core runner ---------------------------------- #
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
) -> Any:
    """
    Run a single ABC‑SMC with the given motion & speed, writing to db_path.
    Returns the pyabc History object.
    """
    # Prepare observed vectors & normalisation
    # Ensure timestep order is ascending and aligned with stats
    obs_df_sorted = obs_df.sort_values("timestep").reset_index(drop=True)
    obs_mat = obs_df_sorted[stats].to_numpy(dtype=float)
    obs_vec = obs_mat.flatten()

    # Per-stat normalisation across timesteps (use per-stat std; avoid zeros)
    stds = obs_mat.std(axis=0)
    stds = np.where(stds < 1e-12, 1.0, stds)  # avoid division by ~0
    stds_mat = np.tile(stds, (len(timesteps), 1))
    stds_rep = stds_mat.flatten()

    # Weights: full for S0,S1,S2,NND; down‑weight g(r) if present
    if set(stats) == {"S0", "S1", "S2", "NND_med"}:
        base_weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    else:
        base_weights = np.array(
            [1.0 if s in {"S0", "S1", "S2", "NND_med"} else 0.3 for s in stats],
            dtype=float,
        )
    weights_rep = np.tile(base_weights, len(timesteps))

    # Model factory with per‑run seed
    model_factory = make_model_factory(seed=seed)

    # Model wrapper for pyABC
    def abm_model(particle):
        params = particle_to_params(particle, motion=motion, speed_dist=speed)
        sim_mat = simulate_timeseries(
            model_factory,
            params=params,
            total_steps=total_steps,
            sample_steps=tuple(timesteps),
        )
        # Columns produced by simulate_timeseries in fixed order:
        full_order = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
        col_idx = [full_order.index(s) for s in stats]
        sim_sel = sim_mat[:, col_idx]
        sim_vec = sim_sel.flatten()
        return {f"y_{i}": float(v) for i, v in enumerate(sim_vec)}

    observation = {f"y_{i}": float(v) for i, v in enumerate(obs_vec)}

    # Weighted Z‑score distance
    def distance(sim: Dict[str, float], obs: Dict[str, float]) -> float:
        sim_v = np.array([sim[f"y_{i}"] for i in range(len(obs_vec))], float)
        obs_v = np.array([obs[f"y_{i}"] for i in range(len(obs_vec))], float)
        z = (sim_v - obs_v) / stds_rep
        z = z * weights_rep
        return float(np.sqrt(np.sum(z * z)))

    # Sampler (optional parallel)
    sampler = make_sampler(workers)

    abc = pyabc.ABCSMC(
        models=abm_model,
        parameter_priors=prior,
        distance_function=distance,
        population_size=popsize,
        sampler=sampler,
    )
    db_url = f"sqlite:///{db_path}"
    abc.new(db_url, observation)
    history = abc.run(max_nr_populations=maxgen, minimum_epsilon=min_eps)
    return history


# ----------------------------- CLI & main ----------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Run ABC‑SMC for the clustering ABM.\n"
            "Supports sweep over multiple motions and speed distributions."
        )
    )
    ap.add_argument(
        "--db", type=str, default="results/abc_run.db",
        help="DB path for single run, or base when not sweeping"
    )
    ap.add_argument(
        "--db_template", type=str,
        default="results/abc_{obs}_{motion}_{speed}_seed{seed}.db",
        help="Template for per‑run DB when sweeping; placeholders: {obs},{motion},{speed},{seed},{ts}"
    )
    ap.add_argument(
        "--timestamp", action="store_true",
        help="Append a timestamp to each DB filename (format yyyymmdd_HHMMSS)"
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite an existing DB if present (default: make unique suffix)"
    )
    ap.add_argument("--popsize", type=int, default=200)
    ap.add_argument("--maxgen", type=int, default=12)
    ap.add_argument("--min_eps", type=float, default=0.5)
    ap.add_argument(
        "--observed_ts", type=str,
        default="observed/INV_ABM_ready_summary.csv",
        help="CSV with columns: timestep, stats..."
    )
    ap.add_argument("--t_start", type=int, default=22, help="(compatibility; not used)")
    ap.add_argument("--total_steps", type=int, default=300)
    ap.add_argument("--priors_yaml", type=str, default="priors.yaml")

    # Single values (backwards compatible)
    ap.add_argument(
        "--motion", type=str, default="isotropic",
        choices=["isotropic", "persistent"],
        help="Movement model for single run (ignored if --motions provided)"
    )
    ap.add_argument(
        "--speed", type=str, default="constant",
        choices=["constant", "lognorm", "gamma", "weibull"],
        help="Speed distribution for single run (ignored if --speeds provided)"
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Seed for single run (ignored if --seeds provided)"
    )

    # Sweep values
    ap.add_argument(
        "--motions", nargs="+", default=None,
        choices=["isotropic", "persistent"],
        help="Run all these motion models (sweep mode)"
    )
    ap.add_argument(
        "--speeds", nargs="+", default=None,
        choices=["constant", "lognorm", "gamma", "weibull"],
        help="Run all these speed distributions (sweep mode)"
    )
    ap.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Seeds to sweep (e.g., 42 43 44)"
    )

    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes for simulation")
    ap.add_argument(
        "--no_gr", action="store_true",
        help="Use only S0, S1, S2, NND_med (drop g_r40, g_r80)"
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="List the runs and DB paths without executing"
    )
    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True, parents=True)

    # Load observed time‑series
    obs_csv = Path(args.observed_ts)
    if not obs_csv.exists():
        raise FileNotFoundError(f"Observed CSV not found: {obs_csv}")
    obs_df = pd.read_csv(obs_csv)

    # Decide stats set based on flag
    stats = (
        ["S0", "S1", "S2", "NND_med"]
        if args.no_gr
        else ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
    )

    # Validate presence of columns
    validate_observed_df(obs_df, stats)
    timesteps = obs_df["timestep"].astype(int).to_list()

    # Priors (falls back to defaults if file missing)
    prior = load_priors(args.priors_yaml if Path(args.priors_yaml).exists() else None)

    # Decide sweep lists
    motions = args.motions if args.motions else [args.motion]
    speeds = args.speeds if args.speeds else [args.speed]
    seeds = args.seeds if args.seeds else [args.seed]

    # For naming
    obs_tag = obs_csv.stem
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S") if args.timestamp else None

    # Derive run list
    run_plan: List[Tuple[str, str, int, Path]] = []
    for motion in motions:
        for speed in speeds:
            for seed in seeds:
                if (len(motions) * len(speeds) * len(seeds)) == 1:
                    # Single run: respect --db exactly
                    db_path = Path(args.db)
                else:
                    # Use template
                    db_name = args.db_template.format(
                        obs=obs_tag, motion=motion, speed=speed, seed=seed, ts=(ts_tag or "")
                    )
                    db_path = Path(db_name)
                    if args.timestamp and "{ts}" not in args.db_template:
                        # Append timestamp if user didn't include {ts}
                        db_path = append_to_stem(db_path, f"_{ts_tag}")
                    if not args.overwrite:
                        db_path = unique_path(db_path)
                run_plan.append((motion, speed, seed, db_path))

    # Dry run preview
    if args.dry_run:
        print("\nPlanned runs (dry-run):")
        for (motion, speed, seed, db_path) in run_plan:
            print(f"  motion={motion:10s} speed={speed:8s} seed={seed:4d} → {db_path}")
        print(f"\nTotal runs: {len(run_plan)} (no execution performed)")
        return

    # Execute runs
    for (motion, speed, seed, db_path) in run_plan:
        print("\n" + "=" * 78)
        print(f"Starting ABC run → motion={motion}  speed={speed}  seed={seed}")
        print(f"DB: {db_path}")
        print("=" * 78)

        _ = run_one(
            db_path=db_path,
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
        )
        print(f"ABC finished. DB saved to {db_path}")

    print("\nAll runs completed.")


if __name__ == "__main__":
    main()
