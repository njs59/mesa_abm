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

from scipy.stats import sobol_indices, uniform  # SciPy ≥ 1.13  (n must be 2^m)  [SciPy docs]
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
    replicates: int = 10
    base_seed: int = 42


PARAM_NAMES = [
    "merge.p_merge",
    "phenotypes.proliferative.prolif_rate",
    "phenotypes.proliferative.fragment_rate",
    "movement_v2.phase2.speed_dist.params.a",
    "movement_v2.phase2.turning.kappa",
]

# ===== Your updated distributions =====
DISTS = [
    uniform(loc=0.10, scale=0.90),     # p_merge ∈ [0.10, 1.00]
    uniform(loc=0.001, scale=0.019),   # prolif_rate ∈ [0.001, 0.020]
    uniform(loc=0.0000, scale=0.0020), # frag_rate ∈ [0.0000, 0.0020]
    uniform(loc=0.50, scale=2.00),     # a (gamma shape) ∈ [0.50, 2.50]
    uniform(loc=0.0,  scale=0.40),     # kappa ∈ [0.00, 0.40]
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


# --- ABM single run (logging disabled during Sobol runs for speed) ---
def _run_once(x, steps, seed, force_phase2):
    params = _apply_params_to_defaults(x)
    model = ClustersModel(params=params, seed=seed)

    # Disable logging entirely during Sobol runs (no model behaviour change)
    model.enable_logging = False
    # If you want sparse logging instead:
    # model.log_every = steps

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


def make_abm_vectorised_func(cfg: SAConfig, pool: mp.pool.Pool,
                             outer_pbar=None, chunksize: int = 8):
    """
    f(x): x shape (d, n) → returns (s, n) with s=2 outputs.
    Submits all (column, replicate) tasks to a shared pool and aggregates.
    """
    s = 2
    d_expected = 5

    def f(x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2 and x.shape[0] == d_expected, f"Expected (5, n), got {x.shape}"
        n = x.shape[1]

        global _EVAL_COUNTER
        call_seed_seq = np.random.SeedSequence(cfg.base_seed + _EVAL_COUNTER)
        _EVAL_COUNTER += 1

        # Build tasks across all columns and replicates
        col_children = call_seed_seq.spawn(n)
        tasks = []
        for j in range(n):
            rep_children = col_children[j].spawn(cfg.replicates)
            seeds_j = [int(ss.generate_state(1)[0]) for ss in rep_children]
            x_col = x[:, j].copy()
            for seed in seeds_j:
                tasks.append((j, x_col, cfg.steps, seed, cfg.force_phase2))

        sums = np.zeros((s, n), dtype=float)
        counts = np.zeros(n, dtype=int)

        # Larger chunksize reduces scheduling overhead for long tasks
        for j, y1, y2 in pool.imap_unordered(_abm_replicate_worker, tasks, chunksize=chunksize):
            sums[0, j] += y1
            sums[1, j] += y2
            counts[j] += 1
            if outer_pbar is not None:
                outer_pbar.update(1)

        out = np.zeros((s, n), dtype=float)
        for j in range(n):
            if counts[j] > 0:
                out[:, j] = sums[:, j] / counts[j]
        return out

    return f


# ============================================================
# Saving + plotting
# ============================================================
def _robust_dist_bounds(d):
    """
    Robustly determine [low, high] for a SciPy distribution:
    - Try frozen rv_frozen.kwds (preferred).
    - Try attributes on d.dist if present.
    - Fallback to PPF to avoid relying on private attributes.
    """
    low = None
    high = None

    # Preferred: frozen with .kwds
    kwds = getattr(d, "kwds", None)
    if isinstance(kwds, dict) and ("loc" in kwds) and ("scale" in kwds):
        loc = float(kwds["loc"])
        scale = float(kwds["scale"])
        low = loc
        high = loc + scale

    # Try d.dist.kwds (some versions expose it there)
    if (low is None or high is None) and hasattr(d, "dist"):
        kw2 = getattr(d.dist, "kwds", None)
        if isinstance(kw2, dict) and ("loc" in kw2) and ("scale" in kw2):
            loc = float(kw2["loc"])
            scale = float(kw2["scale"])
            low = loc
            high = loc + scale

    # Fallback: PPF
    if (low is None) or (high is None):
        eps = np.finfo(float).eps
        try:
            low = float(d.ppf(0.0))
            high = float(d.ppf(1.0 - eps))
        except Exception:
            xs = d.ppf(np.linspace(0.0, 1.0 - eps, 1001))
            xs = np.asarray(xs, dtype=float)
            low = float(np.min(xs))
            high = float(np.max(xs))

    return {"type": getattr(getattr(d, "dist", d), "name", "unknown"),
            "low": low, "high": high}


def _save_results(out_dir: Path, res, boot, cfg: SAConfig, n: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "config": asdict(cfg),
        "n_power_of_two": n,
        "parameters": PARAM_NAMES,
        "distributions": [_robust_dist_bounds(d) for d in DISTS],
        "notes": "SciPy sobol_indices (Saltelli 2010). n*(d+2) parameter points; n must be 2^m.",
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

    # Indices (shape s x d) with s=2 outputs
    labels = ["Total clusters", "Mean cluster size"]
    _write_matrix("indices_first_order.csv", res.first_order, labels)
    _write_matrix("indices_total_order.csv", res.total_order, labels)

    # CIs: save as separate "low" and "high" files for clarity
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
    chunksize: int = 8,  # larger chunks reduce scheduling overhead
):
    """
    Parallel Sobol' run.

    SciPy's sobol_indices uses a Saltelli-type design with A, B and A_B^i matrices;
    total parameter points (function columns) = n * (d + 2); n must be a power of two.  # [SciPy docs]
    """
    d = 5
    total_param_points = n_power_of_two * (d + 2)
    total_abm_runs = total_param_points * cfg.replicates

    # Results directory
    out_dir = Path(output_dir) / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Progress bar (counts real ABM runs)
    outer_pbar = tqdm(total=total_abm_runs, unit="run",
                      desc="Sobol ABM runs", mininterval=0.5) if tqdm else None

    # Sensible default for workers
    if n_workers is None:
        try:
            n_workers = max(1, (os.cpu_count() or 2) - 1)
        except Exception:
            n_workers = 1

    print(f"[info] Using {n_workers} worker processes for parallel ABM runs.")

    # On macOS, prefer "spawn" context for safety
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        # Build vectorised callable that uses the shared pool
        f = make_abm_vectorised_func(cfg, pool=pool, outer_pbar=outer_pbar, chunksize=chunksize)

        # SciPy computes A, B, and A_B^i, and calls f(x) several times  [SciPy docs]
        res = sobol_indices(func=f, n=n_power_of_two, dists=DISTS)
        # Bootstrap confidence intervals (on both S1 and ST)
        boot = res.bootstrap(confidence_level=confidence, n_resamples=ci_resamples)

    if outer_pbar is not None:
        outer_pbar.close()

    # Save & plot
    _save_results(out_dir, res, boot, cfg, n_power_of_two)
    _save_plot(out_dir, res, boot)

    # Summary
    print("\nRun summary")
    print("-----------")
    print(f"Outputs           : Total clusters, Mean cluster size")
    print(f"Parameters (d)    : {d}")
    print(f"n (power of two)  : {n_power_of_two}")
    print(f"Param points      : {total_param_points} (= n*(d+2))")
    print(f"Replicates (R)    : {cfg.replicates}")
    print(f"Total ABM runs    : {total_abm_runs}")
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

    cfg = SAConfig(steps=300, force_phase2=True, replicates=8)
    res, boot, out = run_sobol(
        n_power_of_two=64,     # Must be a power of two (64, 128, 256, ...)  [SciPy docs]
        cfg=cfg,
        ci_resamples=300,
        output_dir="sobol_results",
        n_workers=None,        # auto: CPU-1
        chunksize=8,
    )


    # # in __main__
    # res, boot, out = run_sobol(
    #     n_power_of_two=8,        # tiny test (7*8=56 points)
    #     cfg=SAConfig(steps=100, replicates=2),
    #     ci_resamples=50,
    # )
