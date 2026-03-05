# sobol_sensitivity/sobol_sensitivity.py
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import Optional

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from scipy.stats import sobol_indices, uniform
from scipy.spatial import cKDTree          # for median NN + Clark–Evans
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ---- import your ABM ----
from abm_sobol.clusters_model import ClustersModel
from abm_sobol.utils import DEFAULTS as ABM_DEFAULTS


# ============================================================
# Configuration
# ============================================================
@dataclass
class SAConfig:
    steps: int = 145
    force_phase2: bool = True
    replicates: int = 10     # ignored if adaptive.enable=True
    base_seed: int = 42


@dataclass
class AdaptiveR:
    enable: bool = True
    R_min: int = 6
    R_step: int = 2
    R_max: int = 16
    rel_sem_target: float = 0.07
    eps_mean: float = 1e-8


PARAM_NAMES = [
    "merge.p_merge",
    "phenotypes.proliferative.prolif_rate",
    "phenotypes.proliferative.fragment_rate",
    "movement_v2.phase2.speed_dist.params.a",
    "movement_v2.phase2.turning.kappa",
]

# ---- Outputs we compute ----
OUTPUT_LABELS = [
    "Number of clusters",
    "Mean cluster size",
    "Variance of cluster size",
    "Median NN distance",
    "Clark-Evans R",
]

DISTS = [
    uniform(loc=0.10, scale=0.90),
    uniform(loc=0.001, scale=0.019),
    # uniform(loc=0.001, scale=0.049),
    uniform(loc=0.0000, scale=0.0020),
    # uniform(loc=0.0000, scale=0.010),
    # uniform(loc=0.50, scale=2.00),
    uniform(loc=1.90, scale=0.40),
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


# ---- Metric helpers ----
def _median_nn_distance(xy: np.ndarray) -> float:
    """Median nearest-neighbour distance via cKDTree."""
    xy = np.asarray(xy)
    if xy.shape[0] < 2:
        return 0.0
    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=2, workers=-1)  # self + nearest
    return float(np.median(d[:, 1]))


def _clark_evans_R(xy: np.ndarray, width: float, height: float) -> float:
    """
    Clark–Evans R index (no edge correction):
        R = mean_NN_obs / (0.5 / sqrt(lambda))
        lambda = N / A
    """
    xy = np.asarray(xy)
    n = xy.shape[0]
    if n < 2:
        return 0.0

    # observed mean NN
    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=2, workers=-1)
    d_obs = float(np.mean(d[:, 1]))

    A = width * height
    lam = n / A if A > 0 else np.inf
    d_csr = 0.5 / np.sqrt(lam) if lam > 0 else np.inf
    if d_csr == 0:
        return 0.0

    return float(d_obs / d_csr)


# ============================================================
# Single ABM simulation
# ============================================================
def _run_once(x, steps, seed, force_phase2):
    params = _apply_params_to_defaults(x)
    model = ClustersModel(params=params, seed=seed)

    # Disable logging for speed
    try:
        model.enable_logging = False
    except Exception:
        pass

    if force_phase2:
        for a in list(getattr(model, "agent_set", [])):
            try:
                a.movement_phase = 2
                a.phase_switch_time = float("inf")
            except Exception:
                pass

    for _ in range(steps):
        model.step()

    # Gather alive agents
    alive = [a for a in model.agent_set if getattr(a, "alive", True)]
    sizes = np.array([float(a.size) for a in alive], float)
    xy = np.array([a.pos for a in alive if a.pos is not None], float)

    n_clusters = float(sizes.size)
    mean_size  = float(np.mean(sizes)) if sizes.size else 0.0
    var_size   = float(np.var(sizes, ddof=1)) if sizes.size > 1 else 0.0
    mnn        = _median_nn_distance(xy) if xy.size else 0.0

    W = float(model.params["space"]["width"])
    H = float(model.params["space"]["height"])
    R = _clark_evans_R(xy, W, H) if xy.size else 0.0

    return (
        n_clusters,
        mean_size,
        var_size,
        mnn,
        R,
    )


def _abm_replicate_worker(args):
    j, x_col, steps, seed, force_phase2 = args
    return j, _run_once(x_col, steps, seed, force_phase2)


# Global seed offset counter
_EVAL_COUNTER = 0


# ============================================================
# Adaptive vectorised function
# ============================================================
def make_abm_vectorised_func_adaptive(cfg, pool, outer_pbar, chunksize, adapt):
    s = len(OUTPUT_LABELS)
    d_expected = 5

    state = {"replicate_counts": None,
             "total_runs": 0,
             "rsem_at_stop": None}

    def f(x: np.ndarray) -> np.ndarray:
        assert x.shape[0] == d_expected
        n = x.shape[1]

        global _EVAL_COUNTER
        call_seed_seq = np.random.SeedSequence(cfg.base_seed + _EVAL_COUNTER)
        _EVAL_COUNTER += 1

        # Pre-allocate
        sums  = np.zeros((s, n), float)
        sums2 = np.zeros((s, n), float)
        counts = np.zeros(n, int)

        def rsem_snapshot():
            means = sums / counts.clip(min=1)
            var   = (sums2 - (sums * sums) / counts.clip(min=1)) \
                    / np.maximum(counts - 1, 1)
            sem   = np.sqrt(var / counts.clip(min=1))
            denom = np.maximum(np.abs(means), adapt.eps_mean)
            return sem / denom   # (s, n)

        def submit_jobs(cols, reps):
            tasks = []
            col_children = call_seed_seq.spawn(n)
            for j in cols:
                rep_children = col_children[j].spawn(reps)
                seeds = [int(ss.generate_state(1)[0]) for ss in rep_children]
                xj = x[:, j].copy()
                for sd in seeds:
                    tasks.append((j, xj, cfg.steps, sd, cfg.force_phase2))
            return tasks

        def process(tasks):
            if outer_pbar is not None:
                outer_pbar.total += len(tasks)
                outer_pbar.refresh()

            for j, out_tuple in pool.imap_unordered(_abm_replicate_worker, tasks,
                                                    chunksize=chunksize):
                y = np.asarray(out_tuple, float)
                sums[:, j]  += y
                sums2[:, j] += y * y
                counts[j]   += 1
                state["total_runs"] += 1
                if outer_pbar is not None:
                    outer_pbar.update(1)

            rsem = rsem_snapshot()
            return np.all(rsem <= adapt.rel_sem_target, axis=0)

        # ---- Pilot ----
        cols = np.arange(n)
        done = process(submit_jobs(cols, adapt.R_min if adapt.enable else cfg.replicates))

        if not adapt.enable:
            state["replicate_counts"] = counts.copy()
            state["rsem_at_stop"] = rsem_snapshot()
            return sums / counts.clip(min=1)

        # ---- Adaptive phase ----
        while not np.all(done) and np.max(counts) < adapt.R_max:
            remaining = np.where(~done & (counts < adapt.R_max))[0]
            if remaining.size == 0:
                break
            done = process(submit_jobs(remaining, adapt.R_step))

        state["replicate_counts"] = counts.copy()
        state["rsem_at_stop"] = rsem_snapshot()
        return sums / counts.clip(min=1)

    return f, state


# ============================================================
# Saving + plotting
# ============================================================
def _robust_dist_bounds(d):
    low = None
    high = None
    kw = getattr(d, "kwds", None)
    if isinstance(kw, dict) and "loc" in kw and "scale" in kw:
        low = kw["loc"]; high = kw["loc"] + kw["scale"]
    return {"type": getattr(getattr(d, "dist", d), "name", "unknown"),
            "low": float(low), "high": float(high)}


def _save_results(out_dir, res, boot, cfg, n, adapt,
                  rep_counts, total_runs, rsem_at_stop):

    out_dir.mkdir(parents=True, exist_ok=True)

    # RSEM info
    if rsem_at_stop is not None:
        rsem_max = np.max(rsem_at_stop, axis=0)
        rsem_ratio = rsem_at_stop / max(adapt.rel_sem_target, 1e-12)
        rsem_ratio_max = np.max(rsem_ratio, axis=0)

    # Config
    meta = {
        "config": asdict(cfg),
        "n_power_of_two": n,
        "parameters": PARAM_NAMES,
        "outputs": OUTPUT_LABELS,
        "adaptive": asdict(adapt),
        "replicate_counts_summary": (
            None if rep_counts is None else {
                "n_columns": int(rep_counts.size),
                "min": int(np.min(rep_counts)),
                "max": int(np.max(rep_counts)),
                "mean": float(np.mean(rep_counts)),
                "median": float(np.median(rep_counts)),
                "total_abm_runs": int(total_runs),
            }
        ),
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))

    # CSV writers
    def write_matrix(name, arr, labels):
        hdr = ",".join(["output"] + PARAM_NAMES)
        with open(out_dir / name, "w") as f:
            f.write(hdr + "\n")
            for i, lab in enumerate(labels):
                row = [lab] + [f"{float(v):.8f}" for v in arr[i]]
                f.write(",".join(row) + "\n")

    write_matrix("indices_first_order.csv", res.first_order, OUTPUT_LABELS)
    write_matrix("indices_total_order.csv", res.total_order, OUTPUT_LABELS)
    write_matrix("ci_first_order_low.csv",  boot.first_order.confidence_interval.low, OUTPUT_LABELS)
    write_matrix("ci_first_order_high.csv", boot.first_order.confidence_interval.high, OUTPUT_LABELS)
    write_matrix("ci_total_order_low.csv",  boot.total_order.confidence_interval.low, OUTPUT_LABELS)
    write_matrix("ci_total_order_high.csv", boot.total_order.confidence_interval.high, OUTPUT_LABELS)

    # RSEM
    if rsem_at_stop is not None:
        # per-output RSEM
        with open(out_dir / "rsem_per_column.csv", "w") as f:
            hdr = ["output"] + [f"col_{j:04d}" for j in range(rsem_at_stop.shape[1])]
            f.write(",".join(hdr) + "\n")
            for i, lab in enumerate(OUTPUT_LABELS):
                row = [lab] + [f"{float(v):.8g}" for v in rsem_at_stop[i]]
                f.write(",".join(row) + "\n")

        np.savetxt(out_dir / "rsem_max_per_column.csv",
                   rsem_max.reshape(1, -1), delimiter=",", fmt="%.8g")

    # Save NPZ bundle
    np.savez(
        out_dir / "sobol_arrays.npz",
        first_order=res.first_order,
        total_order=res.total_order,
        S1_CI_low=boot.first_order.confidence_interval.low,
        S1_CI_high=boot.first_order.confidence_interval.high,
        ST_CI_low=boot.total_order.confidence_interval.low,
        ST_CI_high=boot.total_order.confidence_interval.high,
        replicate_counts=np.array([]) if rep_counts is None else rep_counts,
        rsem_at_stop=np.array([]) if rsem_at_stop is None else rsem_at_stop,
    )


def _save_plot(out_dir, res, boot):
    """Quick plot for the first two outputs."""
    if len(OUTPUT_LABELS) < 2:
        return
    x = np.arange(len(PARAM_NAMES))
    labels = OUTPUT_LABELS[:2]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for i, ax in enumerate(axs):
        s1 = res.first_order[i]
        st = res.total_order[i]
        s1_lo = boot.first_order.confidence_interval.low[i]
        s1_hi = boot.first_order.confidence_interval.high[i]
        st_lo = boot.total_order.confidence_interval.low[i]
        st_hi = boot.total_order.confidence_interval.high[i]

        ax.errorbar(x - 0.12, s1,
                    yerr=[s1 - s1_lo, s1_hi - s1], fmt='o', capsize=3, label="S1")
        ax.errorbar(x + 0.12, st,
                    yerr=[st - st_lo, st_hi - st], fmt='o', capsize=3, label="ST")

        ax.set_xticks(x, PARAM_NAMES, rotation=30, ha="right")
        ax.set_title(labels[i])
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
    chunksize: int = 16,
    adaptive: AdaptiveR = AdaptiveR(),
):
    d = 5
    out_dir = Path(output_dir) / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    outer_pbar = tqdm(total=0, unit="run",
                      desc="Sobol ABM runs",
                      mininterval=0.5,
                      dynamic_ncols=True) if tqdm else None

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2))

    print(f"[info] Using {n_workers} worker processes")
    if adaptive.enable:
        print(f"[info] Adaptive: R_min={adaptive.R_min}, R_step={adaptive.R_step}, "
              f"R_max={adaptive.R_max}, rel_sem_target={adaptive.rel_sem_target:.2%}")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        f, state = make_abm_vectorised_func_adaptive(
            cfg, pool, outer_pbar, chunksize, adaptive
        )

        res = sobol_indices(func=f, n=n_power_of_two, dists=DISTS)
        boot = res.bootstrap(confidence_level=confidence, n_resamples=ci_resamples)

    if outer_pbar:
        outer_pbar.close()

    rep_counts = state["replicate_counts"]
    total_runs = state["total_runs"]
    rsem_at_stop = state["rsem_at_stop"]

    _save_results(out_dir, res, boot, cfg, n_power_of_two,
                  adaptive, rep_counts, total_runs, rsem_at_stop)
    _save_plot(out_dir, res, boot)

    # Summary
    print("\nRun summary")
    print("-----------")
    print(f"Outputs        : {', '.join(OUTPUT_LABELS)}")
    print(f"n (power of 2) : {n_power_of_two}")
    print(f"Workers        : {n_workers}")
    print(f"Saved to       : {out_dir.resolve()}")

    if rep_counts is not None:
        rsem_max = np.max(rsem_at_stop, axis=0)
        frac_bad = np.mean(rsem_max > adaptive.rel_sem_target) * 100.0
        print(f"Columns > threshold: {frac_bad:.1f}%")

    print(f"Total ABM runs : {total_runs}")
    return res, boot, out_dir


# ============================================================
# CLI entry
# ============================================================
if __name__ == "__main__":
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    cfg = SAConfig(steps=145, force_phase2=True, replicates=8)
    adaptive = AdaptiveR(enable=True, R_min=6, R_step=2, R_max=200, rel_sem_target=0.05)

    run_sobol(
        n_power_of_two=128,
        cfg=cfg,
        ci_resamples=300,
        output_dir="sobol_results",
        n_workers=None,
        chunksize=16,
        adaptive=adaptive,
    )