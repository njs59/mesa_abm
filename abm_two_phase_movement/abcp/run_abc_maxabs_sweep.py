#!/usr/bin/env python3
"""
ABC‑SMC with a reliable, live text progress bar per population:
  accepted/total | elapsed | ETA

- True pyABC SMC (epsilon schedule, weights, resampling).
- Parallel across particles via MulticoreEvalParallelSampler.
- NEW ABM mapping: simulate_timeseries(factory, params, ...).
- Progress bar uses a top‑level sampler.info_callback (main process).
- We silence pyABC/tqdm's internal bar *before importing pyABC*, so it
  cannot overwrite our line.
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

# -----------------------------------------------------------------------------
# Silence tqdm globally BEFORE importing pyABC (prevents pyABC's own bar)
# -----------------------------------------------------------------------------
class _SilentBar:
    def __init__(self, *a, **k): self.total = k.get("total", 0); self.n = 0
    def update(self, *a, **k): pass
    def close(self): pass
    def refresh(self): pass
    def __enter__(self): return self
    def __exit__(self, *exc): pass

# Install stubs so any later "import tqdm" uses these
_tqdm = types.ModuleType("tqdm"); _tqdm.tqdm = _SilentBar
sys.modules["tqdm"] = _tqdm
_tqdm_auto = types.ModuleType("tqdm.auto"); _tqdm_auto.tqdm = _SilentBar
sys.modules["tqdm.auto"] = _tqdm_auto

# Now it's safe to import pyABC; its tqdm will be no-op and won't print
import pyabc  # noqa: E402

from abm.clusters_model import ClustersModel  # noqa: E402
from abcp.compute_summary import simulate_timeseries  # noqa: E402
from abcp.priors import load_priors  # noqa: E402
from abcp.abc_model_wrapper import particle_to_params  # noqa: E402


# ==============================
# GLOBAL STATE FOR PROGRESS BAR
# ==============================
CURRENT_POP = None          # current population index t
CURRENT_ACCEPTED = 0        # accepted so far in this population
CURRENT_REQUIRED = 0        # required popsize for this population
POP_START = None            # start time of current population
PROGRESS_ON = True          # master switch


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
            f"| elapsed {elapsed:5.1f}s "
            f"| ETA {eta:5.1f}s"
        )
        sys.stdout.flush()
    else:
        sys.stdout.write(
            f"\rPop {t} {bar} {req}/{req} accepted "
            f"| total {elapsed:5.1f}s | Done.\n"
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
        merge.pop("prob_contact_merge", None)
        merge.pop("adhesion", None)
        return
    # Legacy fallback support
    pcm = merge.pop("prob_contact_merge", None)
    adh = merge.pop("adhesion", None)
    if pcm is not None or adh is not None:
        p = (float(pcm) if pcm is not None else 1.0) * (float(adh) if adh is not None else 1.0)
        merge["p_merge"] = float(np.clip(p, 0.0, 1.0))
        return
    merge["p_merge"] = merge.get("p_merge", 0.9)


def _ensure_physics_defaults(params: dict) -> None:
    """Default soft separation + fragment min‑sep consistent with the new ABM."""
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
    obs_mat = obs_df[stats].to_numpy(float)  # T x K
    T, K = obs_mat.shape
    scaler = MaxAbsScaler().fit(obs_mat)
    obs_scaled = scaler.transform(obs_mat).flatten()
    observation = {f"y_{i}": float(v) for i, v in enumerate(obs_scaled)}

    # ---- model wrapper ----
    model_factory = make_model_factory(seed)
    full = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
    col_idx = [full.index(s) for s in stats]

    def abm_model(particle: dict):
        params = particle_to_params(particle)
        _coerce_merge_params(params)
        _ensure_physics_defaults(params)

        pseed = int(particle.get("_seed", seed))

        def factory(p: dict):
            return ClustersModel(params=p, seed=pseed)

        sim = simulate_timeseries(factory, params=params,
                                  total_steps=total_steps,
                                  sample_steps=tuple(timesteps))
        sel = sim[:, col_idx]           # T x K
        sel_scaled = scaler.transform(sel).flatten()
        return {f"y_{i}": float(v) for i, v in enumerate(sel_scaled)}

    # ---- distance in scaled space ----
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

    # IMPORTANT:
    # - must be True so the sampler emits events to our callback
    # - internal tqdm is already silenced globally, so no clobbering output
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
    ap = argparse.ArgumentParser(description="ABC‑SMC with per‑population progress (accepted, elapsed, ETA)")
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
    ap.add_argument("--no_gr", action="store_true")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True)
    obs_df = pd.read_csv(args.observed_ts)

    stats = ["S0", "S1", "S2", "NND_med"] if args.no_gr else \
            ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
    for s in stats:
        if s not in obs_df.columns:
            raise ValueError(f"Missing '{s}' in observed CSV.")

    timesteps = obs_df["timestep"].astype(int).tolist()
    prior = load_priors(args.priors_yaml if Path(args.priors_yaml).exists() else None)

    print(f"[RUN] DB: {args.db}")
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
        seed=args.seed,
        workers=args.workers,
        show_progress=args.show_progress,
    )


if __name__ == "__main__":
    main()