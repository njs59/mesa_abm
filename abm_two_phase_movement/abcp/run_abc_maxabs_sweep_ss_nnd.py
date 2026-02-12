#!/usr/bin/env python3
"""
ABC‑SMC with MaxAbs scaling and a rich, one-line progress bar, comparing
the model against the 4 stats:
  ["S0", "S1", "S2", "SSNND_med"]

Notes:
- simulate_timeseries(factory, params, ...) must return a (T x 6) array in the
  order ["S0","S1","S2","SSNND_med","g_r40","g_r80"].
- The observed CSV must contain columns ["timestep","S0","S1","S2","SSNND_med"].
- Use --no_gr to enforce the 4-stat distance (this file assumes 4-stat fitting).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import types
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

# ---- Silence tqdm BEFORE importing pyABC (so pyABC can't overwrite our line) ----
class _SilentBar:
    def __init__(self, *a, **k): self.total = k.get("total", 0); self.n = 0
    def update(self, *a, **k): pass
    def close(self): pass
    def refresh(self): pass
    def __enter__(self): return self
    def __exit__(self, *exc): pass

_tqdm = types.ModuleType("tqdm"); _tqdm.tqdm = _SilentBar
sys.modules["tqdm"] = _tqdm
_tqdm_auto = types.ModuleType("tqdm.auto"); _tqdm_auto.tqdm = _SilentBar
sys.modules["tqdm.auto"] = _tqdm_auto

# Now it's safe to import pyABC; its tqdm will be no‑op and won't print
import pyabc  # noqa: E402
from abm.clusters_model import ClustersModel  # noqa: E402
from abcp.compute_summary import simulate_timeseries  # noqa: E402
from abcp.priors import load_priors  # noqa: E402
from abcp.abc_model_wrapper import particle_to_params  # noqa: E402


# ==============================
# GLOBAL STATE FOR PROGRESS BAR
# ==============================
CURRENT_POP = None   # current population index t
CURRENT_ACCEPTED = 0 # accepted so far in this population
CURRENT_REQUIRED = 0 # required popsize for this population
POP_START = None     # start time of current population
PROGRESS_ON = True   # master switch


def _render_bar(t: int, acc: int, req: int, finished: bool) -> None:
    """Draw a 1‑line live progress bar with elapsed + ETA."""
    if POP_START is None:
        elapsed = 0.0
        eta = float("inf")
    else:
        elapsed = time.time() - POP_START
        rate = acc / elapsed if elapsed > 0 else 0.0
        eta = (req - acc) / rate if rate > 0 else float("inf")

    width = 26
    fill = int((acc / max(1, req)) * width)
    bar = "[" + "=" * fill + " " * (width - fill) + "]"

    if not finished:
        sys.stdout.write(
            f"\rPop {t} {bar} {acc}/{req} accepted "
            f"\n elapsed {elapsed:5.1f}s "
            f"\n ETA {eta:5.1f}s"
        )
        sys.stdout.flush()
    else:
        sys.stdout.write(
            f"\rPop {t} {bar} {req}/{req} accepted "
            f"\n total {elapsed:5.1f}s \n Done.\n"
        )
        sys.stdout.flush()


# ==============================
# TOP‑LEVEL, PICKLABLE CALLBACK
# ==============================
def progress_callback(status: dict) -> None:
    """
    Called by pyABC's sampler in the MAIN process (spawn‑safe).
    Keys typically present:
      - 't': population index (int), -1 for calibration
      - 'n_accepted': accepted count so far (int)
      - 'n_required': required in this population (popsize)
      - 'population_finished': bool
    """
    global CURRENT_POP, CURRENT_ACCEPTED, CURRENT_REQUIRED, POP_START, PROGRESS_ON
    if not PROGRESS_ON:
        return

    t = status.get("t", None)
    if t is None or t < 0:
        return  # ignore calibration

    n_acc = status.get("n_accepted", None)
    n_req = status.get("n_required", None)
    finished = bool(status.get("population_finished", False))

    # NEW population start
    if CURRENT_POP != t:
        CURRENT_POP = t
        CURRENT_ACCEPTED = 0
        CURRENT_REQUIRED = int(n_req if n_req is not None else 0)
        POP_START = time.time()
        _render_bar(t, 0, CURRENT_REQUIRED, finished=False)
        return

    # Update counts
    if n_acc is not None:
        CURRENT_ACCEPTED = int(n_acc)
    if n_req is not None:
        CURRENT_REQUIRED = int(n_req)

    # Draw line
    if finished:
        _render_bar(t, CURRENT_REQUIRED, CURRENT_REQUIRED, finished=True)
    else:
        _render_bar(t, CURRENT_ACCEPTED, CURRENT_REQUIRED, finished=False)


# ==============================
# MERGE / PHYSICS HELPERS
# ==============================
def _coerce_merge_params(params: dict) -> None:
    """Ensure `merge.p_merge` ∈ [0,1] and drop legacy keys if present."""
    merge = params.setdefault("merge", {})
    if "p_merge" in merge:
        merge["p_merge"] = float(np.clip(merge["p_merge"], 0.0, 1.0))
    # legacy cleanup
    merge.pop("prob_contact_merge", None)
    merge.pop("adhesion", None)

    # Legacy fallback: derive p_merge if old keys present (unlikely now)
    pcm = merge.pop("prob_contact_merge", None)
    adh = merge.pop("adhesion", None)
    if pcm is not None or adh is not None:
        p = (float(pcm) if pcm is not None else 1.0) * (float(adh) if adh is not None else 1.0)
        merge["p_merge"] = float(np.clip(p, 0.0, 1.0))
        return

    merge["p_merge"] = merge.get("p_merge", 0.9)


def _ensure_physics_defaults(params: dict) -> None:
    """Default soft separation + fragment min‑sep consistent with the ABM."""
    phys = params.setdefault("physics", {})
    phys.setdefault("soft_separate", True)
    phys.setdefault("softness", 0.15)
    phys.setdefault("fragment_minsep_factor", 1.1)


# ==============================
# MODEL FACTORY
# ==============================
def make_model_factory(seed: int):
    def factory(p: dict):
        return ClustersModel(params=p, seed=seed)
    return factory


# ==============================
# ABC RUNNER
# ==============================
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
    global PROGRESS_ON
    PROGRESS_ON = bool(show_progress)

    # ---- observed scaling (MaxAbs) ----
    obs_df = obs_df.sort_values("timestep").reset_index(drop=True)

    # Check presence of required columns
    missing = [s for s in stats if s not in obs_df.columns]
    if missing:
        raise ValueError(
            f"Observed CSV missing required columns for 4-stat fit: {missing}"
        )

    obs_mat = obs_df[stats].to_numpy(float)  # T x K (K=4 for 4-stat fit)
    T, K = obs_mat.shape

    # Fit scaler on observed; flatten target vector
    scaler = MaxAbsScaler().fit(obs_mat)
    obs_scaled = scaler.transform(obs_mat).flatten()
    observation = {f"y_{i}": float(v) for i, v in enumerate(obs_scaled)}

    # ---- model wrapper ----
    model_factory = make_model_factory(seed)

    # IMPORTANT: Column order expected from simulate_timeseries:
    full = ["S0", "S1", "S2", "SSNND_med", "g_r40", "g_r80"]
    col_idx = [full.index(s) for s in stats]

    def abm_model(particle: dict):
        params = particle_to_params(particle)
        _coerce_merge_params(params)
        _ensure_physics_defaults(params)
        pseed = int(particle.get("_seed", seed))

        def factory(p: dict):
            return ClustersModel(params=p, seed=pseed)

        sim = simulate_timeseries(
            factory,
            params=params,
            total_steps=total_steps,
            sample_steps=tuple(timesteps),
        )
        sel = sim[:, col_idx]  # T x K
        sel_scaled = scaler.transform(sel).flatten()
        return {f"y_{i}": float(v) for i, v in enumerate(sel_scaled)}

    # ---- L2 distance in scaled space ----
    def l2_distance(sim: dict, obs: dict) -> float:
        L = T * K
        s = np.array([sim[f"y_{i}"] for i in range(L)], float)
        o = np.array([obs[f"y_{i}"] for i in range(L)], float)
        return float(np.linalg.norm(s - o))

    # ---- sampler ----
    if workers > 1:
        from pyabc.sampler import MulticoreEvalParallelSampler
        sampler = MulticoreEvalParallelSampler(n_procs=int(workers))
    else:
        from pyabc.sampler import SingleCoreSampler
        sampler = SingleCoreSampler()

    # Must be True so the sampler emits events to our callback
    sampler.show_progress = True
    sampler.info_callback = progress_callback

    # ---- ABCSMC ----
    abc = pyabc.ABCSMC(
        models=abm_model,
        parameter_priors=prior,
        distance_function=l2_distance,
        population_size=popsize,
        sampler=sampler,
    )
    abc.new(f"sqlite:///{db_path}", observation)

    # Run true SMC (pyABC handles epsilon, weights, resampling)
    abc.run(max_nr_populations=maxgen, minimum_epsilon=min_eps)


# ==============================
# CLI
# ==============================
def main():
    ap = argparse.ArgumentParser(
        description="ABC‑SMC with per‑population progress (accepted, elapsed, ETA) using SSNND_med."
    )
    ap.add_argument("--db", required=True)
    ap.add_argument("--observed_ts", required=True)
    ap.add_argument("--priors_yaml", default="priors.yaml")
    ap.add_argument("--popsize", type=int, default=200)
    ap.add_argument("--maxgen", type=int, default=8)
    ap.add_argument("--min_eps", type=float, default=0.5)
    ap.add_argument("--total_steps", type=int, default=145)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--show_progress", action="store_true")
    ap.add_argument("--no_gr", action="store_true", help="Fit to 4 stats (S0,S1,S2,SSNND_med)")

    args = ap.parse_args()
    Path("results").mkdir(exist_ok=True)

    # Load observed
    obs_df = pd.read_csv(args.observed_ts)

    # Choose stats (we default to 4‑stat fit)
    if args.no_gr:
        stats = ["S0", "S1", "S2", "SSNND_med"]
    else:
        # If you ever want to include g(r) stats later, extend here.
        stats = ["S0", "S1", "S2", "SSNND_med", "g_r40", "g_r80"]
        # But this file is intended for 4‑stat SSNND use; --no_gr is recommended.

    # Validate that all required observed columns are present
    for s in stats:
        if s not in obs_df.columns:
            raise ValueError(
                f"Observed CSV is missing column '{s}'. "
                f"Got: {list(obs_df.columns)}"
            )

    # Extract all timesteps present in observed
    if "timestep" not in obs_df.columns:
        raise ValueError("Observed CSV must have a 'timestep' column.")
    timesteps = obs_df["timestep"].astype(int).tolist()

    # Build priors
    prior = load_priors(args.priors_yaml if Path(args.priors_yaml).exists() else None)

    print(f"[RUN] DB: {args.db}")
    run_one(
        db_path=Path(args.db),
        obs_df=obs_df,
        stats=stats if args.no_gr else stats[:4],  # ensure distance uses 4‑stat set by default
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