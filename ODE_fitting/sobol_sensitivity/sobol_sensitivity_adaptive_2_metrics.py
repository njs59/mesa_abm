# sobol_sensitivity/sobol_sensitivity.py
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from scipy.stats import sobol_indices, uniform  # SciPy ≥ 1.13  (n must be 2^m)
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ---- import your ABM (package is a sibling of this folder) ----
from abm_sobol.clusters_model import ClustersModel
from abm_sobol.utils import DEFAULTS as ABM_DEFAULTS


# ============================================================
# Configuration
# ============================================================
@dataclass
class SAConfig:
    steps: int = 300
    force_phase2: bool = True
    replicates: int = 10      # kept for backward-compat; not used when adaptive.enabled=True
    base_seed: int = 42


@dataclass
class AdaptiveR:
    """Adaptive replicate allocation per Sobol column."""
    enable: bool = True           # turn adaptive allocation on/off
    R_min: int = 6                # pilot replicates per column
    R_step: int = 2               # extra reps to add per batch for columns that still need it
    R_max: int = 16               # hard cap per column
    rel_sem_target: float = 0.07  # stop when relative SEM <= 7% on BOTH outputs
    eps_mean: float = 1e-8        # stabiliser for near-zero means


PARAM_NAMES = [
    "merge.p_merge",
    "phenotypes.proliferative.prolif_rate",
    "phenotypes.proliferative.fragment_rate",
    "movement_v2.phase2.speed_dist.params.a",
    "movement_v2.phase2.turning.kappa",
]

# Your updated distributions
DISTS = [
    uniform(loc=0.10, scale=0.90),
    uniform(loc=0.001, scale=0.019),
    uniform(loc=0.0000, scale=0.0020),
    uniform(loc=0.50, scale=2.00),
    uniform(loc=0.0,  scale=0.40),
]


# ============================================================
# Utilities
# ============================================================
def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _apply_params_to_defaults(x):
    params = deepcopy(ABM_DEFAULTS)
    p_merge, prolif, frag, a_shape, kappa = map(float, x)
    params["merge"]["p_merge"] = p_merge
    params["phenotypes"]["proliferative"]["prolif_rate"] = prolif
    params["phenotypes"]["proliferative"]["fragment_rate"] = frag
    params["movement_v2"]["phase2"]["speed_dist"]["params"]["a"] = a_shape
    params["movement_v2"]["phase2"]["turning"]["kappa"] = kappa
    return params


# --- One ABM run (logging disabled during Sobol runs for speed) ---
def _run_once(x, steps, seed, force_phase2):
    params = _apply_params_to_defaults(x)
    model = ClustersModel(params=params, seed=seed)

    # Disable logging entirely during Sobol runs (no physics change)
    try:
        model.enable_logging = False
    except Exception:
        pass
    # model.log_every = steps  # alternative: sparse logging

    if force_phase2:
        for a in list(getattr(model, "agent_set", [])):
            try:
                a.movement_phase = 2
                a.phase_switch_time = float("inf")
            except Exception:
                pass

    for _ in range(steps):
        model.step()

    # Compute outputs directly from live agents (no logs needed)
    alive_sizes = [float(a.size) for a in model.agent_set if getattr(a, "alive", True)]
    total_clusters = float(len(alive_sizes))
    mean_size = float(np.mean(alive_sizes)) if alive_sizes else 0.0
    return total_clusters, mean_size


# --- top-level worker for multiprocessing (must be picklable) ---
def _abm_replicate_worker(args):
    j, x_col, steps, seed, force_phase2 = args
    y1, y2 = _run_once(x_col, steps=steps, seed=seed, force_phase2=force_phase2)
    return j, y1, y2


# Global counter to de-correlate seed streams across SciPy's multiple func calls
_EVAL_COUNTER = 0


def make_abm_vectorised_func_adaptive(cfg: SAConfig,
                                      pool: mp.pool.Pool,
                                      outer_pbar=None,
                                      chunksize: int = 16,
                                      adapt: AdaptiveR = AdaptiveR()):
    """
    Returns a callable f(x) matching SciPy's expectation:
      - x shape: (d, n)
      - returns: (s, n), s=2 outputs
    Adaptive replicate allocation:
      1) Pilot with R_min reps for all columns.
      2) Compute per-column relative SEM on BOTH outputs.
      3) Add R_step reps only to columns that haven't hit rel_sem_target.
      4) Repeat until all pass or R_max reached.
    Also returns a 'state' dict capturing replicate counts used per column.
    """
    s = 2
    d_expected = 5
    state = {"replicate_counts": None, "total_runs": 0}  # captured for reporting later

    def f(x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2 and x.shape[0] == d_expected, f"Expected (5, n), got {x.shape}"
        n = x.shape[1]

        global _EVAL_COUNTER
        call_seed_seq = np.random.SeedSequence(cfg.base_seed + _EVAL_COUNTER)
        _EVAL_COUNTER += 1

        # --- Helper to build tasks for a set of columns, with a given #reps per col ---
        def submit_jobs(cols: np.ndarray, reps_per_col: int):
            tasks = []
            # Stable per-column seeds across the whole call
            col_children = call_seed_seq.spawn(n)
            for j in cols:
                rep_children = col_children[j].spawn(reps_per_col)
                seeds_j = [int(ss.generate_state(1)[0]) for ss in rep_children]
                x_col = x[:, j].copy()
                for seed in seeds_j:
                    tasks.append((j, x_col, cfg.steps, seed, cfg.force_phase2))
            return tasks

        # Accumulators (per output × column)
        sums = np.zeros((s, n), dtype=float)
        sums2 = np.zeros((s, n), dtype=float)  # for variance
        counts = np.zeros(n, dtype=int)

        # Process a batch (list of tasks) and update accumulators
        def process_batch(tasks):
            completed = 0
            # When we schedule tasks, increase the progress bar's total dynamically
            if outer_pbar is not None:
                outer_pbar.total = (outer_pbar.total or 0) + len(tasks)
                outer_pbar.refresh()
            for j, y1, y2 in pool.imap_unordered(_abm_replicate_worker, tasks, chunksize=chunksize):
                y = np.array([y1, y2], dtype=float)
                sums[:, j] += y
                sums2[:, j] += y * y
                counts[j] += 1
                state["total_runs"] += 1
                if outer_pbar is not None:
                    outer_pbar.update(1)
                completed += 1
            return completed

        # --- 1) Pilot for all columns ---
        cols_all = np.arange(n, dtype=int)
        pilot_R = adapt.R_min if adapt.enable else cfg.replicates
        pilot_tasks = submit_jobs(cols_all, pilot_R)
        process_batch(pilot_tasks)

        if not adapt.enable:
            out = sums / counts.clip(min=1)
            state["replicate_counts"] = counts.copy()
            return out

        # --- Function to check which columns have met the rel SEM target on BOTH outputs ---
        def columns_meeting_target():
            means = sums / counts.clip(min=1)                                   # (2, n)
            var = (sums2 - (sums * sums) / counts.clip(min=1)) / np.maximum(counts - 1, 1)
            sem = np.sqrt(var / counts.clip(min=1))                             # (2, n)
            denom = np.maximum(np.abs(means), adapt.eps_mean)
            rsem = sem / denom                                                  # (2, n)
            meet = np.logical_and(rsem[0] <= adapt.rel_sem_target,
                                  rsem[1] <= adapt.rel_sem_target)              # (n,)
            return meet

        done = columns_meeting_target()
        R_done = counts.copy()

        # --- 2) Add replicates only where needed, in small batches ---
        while (not np.all(done)) and (np.max(R_done) < adapt.R_max):
            remaining = np.where(~done & (R_done < adapt.R_max))[0]
            if remaining.size == 0:
                break
            batch_R = adapt.R_step
            batch_tasks = submit_jobs(remaining, batch_R)
            process_batch(batch_tasks)
            R_done = counts.copy()
            done = columns_meeting_target()

        out = sums / counts.clip(min=1)
        state["replicate_counts"] = counts.copy()
        return out

    return f, state


# ============================================================
# Saving + plotting
# ============================================================
def _robust_dist_bounds(d):
    """Robustly determine [low, high] for a SciPy distribution."""
    low = None
    high = None
    kwds = getattr(d, "kwds", None)
    if isinstance(kwds, dict) and ("loc" in kwds) and ("scale" in kwds):
        loc = float(kwds["loc"]); scale = float(kwds["scale"])
        low, high = loc, loc + scale
    if (low is None or high is None) and hasattr(d, "dist"):
        kw2 = getattr(d.dist, "kwds", None)
        if isinstance(kw2, dict) and ("loc" in kw2) and ("scale" in kw2):
            loc = float(kw2["loc"]); scale = float(kw2["scale"])
            low, high = loc, loc + scale
    if (low is None) or (high is None):
        eps = np.finfo(float).eps
        try:
            low = float(d.ppf(0.0))
            high = float(d.ppf(1.0 - eps))
        except Exception:
            xs = d.ppf(np.linspace(0.0, 1.0 - eps, 1001))
            xs = np.asarray(xs, dtype=float)
            low, high = float(np.min(xs)), float(np.max(xs))
    return {"type": getattr(getattr(d, "dist", d), "name", "unknown"),
            "low": low, "high": high}


def _save_results(out_dir: Path, res, boot, cfg: SAConfig, n: int, adapt: AdaptiveR,
                  replicate_counts: np.ndarray | None, total_runs: int | None):
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "config": asdict(cfg),
        "n_power_of_two": n,
        "parameters": PARAM_NAMES,
        "distributions": [_robust_dist_bounds(d) for d in DISTS],
        "adaptive": asdict(adapt),
        "replicate_counts_summary": (
            {
                "n_columns": None if replicate_counts is None else int(replicate_counts.size),
                "min": None if replicate_counts is None else int(np.min(replicate_counts)),
                "max": None if replicate_counts is None else int(np.max(replicate_counts)),
                "mean": None if replicate_counts is None else float(np.mean(replicate_counts)),
                "median": None if replicate_counts is None else float(np.median(replicate_counts)),
                "total_abm_runs": int(total_runs) if total_runs is not None else None,
            }
        ),
        "notes": "SciPy sobol_indices (Saltelli design). n*(d+2) columns; adaptive replicates per column.",
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))

    # CSV helpers
    def _write_matrix(name: str, arr_2d: np.ndarray, row_labels):
        hdr = ",".join(["output"] + PARAM_NAMES)
        with open(out_dir / name, "w") as f:
            f.write(hdr + "\n")
            for i, lab in enumerate(row_labels):
                row = [lab] + [f"{float(v):.8f}" for v in arr_2d[i]]
                f.write(",".join(row) + "\n")

    labels = ["Total clusters", "Mean cluster size"]
    _write_matrix("indices_first_order.csv", res.first_order, labels)
    _write_matrix("indices_total_order.csv", res.total_order, labels)
    _write_matrix("ci_first_order_low.csv",  boot.first_order.confidence_interval.low,  labels)
    _write_matrix("ci_first_order_high.csv", boot.first_order.confidence_interval.high, labels)
    _write_matrix("ci_total_order_low.csv",  boot.total_order.confidence_interval.low,  labels)
    _write_matrix("ci_total_order_high.csv", boot.total_order.confidence_interval.high, labels)

    # Compact NPZ with raw arrays
    np.savez(out_dir / "sobol_arrays.npz",
             first_order=res.first_order,
             total_order=res.total_order,
             S1_CI_low=boot.first_order.confidence_interval.low,
             S1_CI_high=boot.first_order.confidence_interval.high,
             ST_CI_low=boot.total_order.confidence_interval.low,
             ST_CI_high=boot.total_order.confidence_interval.high)

    # Also save per-column replicate counts (if available)
    if replicate_counts is not None:
        np.savetxt(out_dir / "replicate_counts_per_column.csv",
                   replicate_counts.reshape(1, -1), delimiter=",", fmt="%d")


def _save_plot(out_dir: Path, res, boot):
    x = np.arange(len(PARAM_NAMES))
    s_labels = ["Total clusters", "Mean cluster size"]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for s_idx, ax in enumerate(axs):
        s1 = res.first_order[s_idx]
        st = res.total_order[s_idx]
        s1_lo = boot.first_order.confidence_interval.low[s_idx]
        s1_hi = boot.first_order.confidence_interval.high[s_idx]
        st_lo = boot.total_order.confidence_interval.low[s_idx]
        st_hi = boot.total_order.confidence_interval.high[s_idx]

        ax.errorbar(x - 0.12, s1,
                    yerr=[s1 - s1_lo, s1_hi - s1], fmt='o',
                    capsize=3, label="S1")
        ax.errorbar(x + 0.12, st,
                    yerr=[st - st_lo, st_hi - st], fmt='o',
                    capsize=3, label="ST")

        ax.set_xticks(x, PARAM_NAMES, rotation=30, ha="right")
        ax.set_title(s_labels[s_idx])
        ax.set_ylabel("Sobol index")
        ax.legend()

    fig.savefig(out_dir / "quickplot.png", dpi=200)
    plt.close(fig)


# ============================================================
# Main entry
# ============================================================
def run_sobol(
    n_power_of_two: int = 256,
    cfg: SAConfig = SAConfig(),
    ci_resamples: int = 500,
    confidence: float = 0.95,
    output_dir: str | Path = "sobol_results",
    n_workers: int | None = None,
    chunksize: int = 16,             # larger chunks reduce scheduling overhead
    adaptive: AdaptiveR = AdaptiveR()  # <— turn adaptive on/off & set its knobs
):
    """
    Parallel Sobol' run with adaptive replicate allocation.
    SciPy builds the Saltelli design (A, B, A_B^i) of n*(d+2) columns.
    We return column-averaged outputs with per-column replicate counts chosen adaptively.
    """
    d = 5

    # Results directory
    out_dir = Path(output_dir) / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Progress bar: dynamic total (we increase total as we schedule batches)
    outer_pbar = tqdm(total=0, unit="run", desc="Sobol ABM runs",
                      mininterval=0.5, dynamic_ncols=True) if tqdm else None

    # Sensible default for workers
    if n_workers is None:
        try:
            n_workers = max(1, (os.cpu_count() or 2) - 1)
        except Exception:
            n_workers = 1

    print(f"[info] Using {n_workers} worker processes for parallel ABM runs.")
    if adaptive.enable:
        print(f"[info] Adaptive replicates: R_min={adaptive.R_min}, R_step={adaptive.R_step}, "
              f"R_max={adaptive.R_max}, rel_sem_target={adaptive.rel_sem_target:.2%}")

    # On macOS, prefer "spawn" context for safety
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        # Build vectorised callable (adaptive) and get a state container for reporting
        f, state = make_abm_vectorised_func_adaptive(cfg, pool=pool,
                                                     outer_pbar=outer_pbar,
                                                     chunksize=chunksize,
                                                     adapt=adaptive)

        # SciPy computes A, B, and A_B^i, and calls f(x) several times
        res = sobol_indices(func=f, n=n_power_of_two, dists=DISTS)
        # Bootstrap confidence intervals (on both S1 and ST)
        boot = res.bootstrap(confidence_level=confidence, n_resamples=ci_resamples)

    if outer_pbar is not None:
        outer_pbar.close()

    # Save & plot
    replicate_counts = state.get("replicate_counts", None)
    total_runs = state.get("total_runs", None)
    _save_results(out_dir, res, boot, cfg, n_power_of_two, adaptive, replicate_counts, total_runs)
    _save_plot(out_dir, res, boot)

    # Summary
    print("\nRun summary")
    print("-----------")
    print(f"Outputs           : Total clusters, Mean cluster size")
    print(f"Parameters (d)    : {d}")
    print(f"n (power of two)  : {n_power_of_two}")
    if replicate_counts is not None:
        print(f"Replicates (per col): min={int(np.min(replicate_counts))}, "
              f"median={int(np.median(replicate_counts))}, mean={np.mean(replicate_counts):.2f}, "
              f"max={int(np.max(replicate_counts))}")
        print(f"Total ABM runs    : {int(total_runs)} (adaptive)")
    else:
        # Fallback if adaptive disabled: n*(d+2)*R
        total_param_points = n_power_of_two * (d + 2)
        print(f"Replicates (R)    : {cfg.replicates}")
        print(f"Total ABM runs    : {total_param_points * cfg.replicates}")
    print(f"Workers           : {n_workers}")
    print(f"Saved to          : {out_dir.resolve()}")

    return res, boot, out_dir


# ============================================================
# CLI example
# ============================================================
if __name__ == "__main__":
    # Avoid BLAS oversubscription with multiprocessing
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    cfg = SAConfig(steps=300, force_phase2=True, replicates=8)  # 'replicates' unused when adaptive.enable=True
    adaptive = AdaptiveR(enable=True, R_min=6, R_step=2, R_max=50, rel_sem_target=0.07)

    res, boot, out = run_sobol(
        n_power_of_two=64,     # Must be a power of two (64, 128, 256, ...)
        cfg=cfg,
        ci_resamples=300,
        output_dir="sobol_results",
        n_workers=None,        # auto: CPU-1
        chunksize=16,
        adaptive=adaptive,
    )