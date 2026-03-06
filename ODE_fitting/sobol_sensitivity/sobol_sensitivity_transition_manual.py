# sobol_sensitivity/sobol_sensitivity_transition_full.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from copy import deepcopy
from pathlib import Path
from datetime import datetime
import numpy as np
import multiprocessing as mp
import json
import os
import matplotlib.pyplot as plt

from scipy.stats import sobol_indices, uniform
from scipy.spatial import cKDTree

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ---- ABM imports ----
from abm_sobol.clusters_model import ClustersModel
from abm_sobol.utils import DEFAULTS as ABM_DEFAULTS


# ============================================================
# CONFIG
# ============================================================
@dataclass
class SAConfig:
    steps: int = 145
    force_phase2: bool = False
    replicates: int = 10
    base_seed: int = 42

@dataclass
class AdaptiveR:
    enable: bool = True
    R_min: int = 6
    R_step: int = 2
    R_max: int = 200
    rel_sem_target: float = 0.05
    eps_mean: float = 1e-8

OUTPUT_LABELS = [
    "Number of clusters",
    "Mean cluster size",
    "Variance of cluster size",
    "Median NN distance",
    "Clark-Evans R",
]


# ============================================================
# PARAMETER DEFINITIONS (YOUR EXACT VALUES)
# ============================================================

PARAM_INFO = [
    # ---- Core biological parameters ----
    ("merge.p_merge",                              ABM_DEFAULTS["merge"]["p_merge"],                                        0.10,   1.00),
    ("phenotypes.proliferative.prolif_rate",        ABM_DEFAULTS["phenotypes"]["proliferative"]["prolif_rate"],            0.001,  0.020),
    ("phenotypes.proliferative.fragment_rate",      ABM_DEFAULTS["phenotypes"]["proliferative"]["fragment_rate"],          0.000,  0.002),

    # ---- Phase‑1 movement ----
    ("movement_v2.phase1.speed_dist.params.s",      0.970295,   0.485148,   1.94059),
    ("movement_v2.phase1.speed_dist.params.scale",  4.51726,    2.25863,    11.2931),
    ("movement_v2.phase1.turning.kappa",            0.241876,   0.0,        0.5),

    # ---- Phase‑2 movement ----
    ("movement_v2.phase2.speed_dist.params.a",      2.08196,    1.04098,    5.20489),
    ("movement_v2.phase2.speed_dist.params.scale",  3.54876,    1.77438,    8.87191),
    ("movement_v2.phase2.turning.kappa",            0.146987,   0.0,        0.5),

    # ---- Transition parameters ----
    ("movement_v2.transition.p_max",                1.0,        0.0,        1.0),
    ("movement_v2.transition.shift",                13.1826,    6.59129,    26.3652),
    ("movement_v2.transition.b",                    0.0278436,  0.0139218,  0.0556872),
    ("movement_v2.transition.c",                    0.0308548,  0.0154274,  0.0617096),
    ("movement_v2.transition.t_max",                400.0,      200.0,      800.0),
]

PARAM_NAMES  = [p[0] for p in PARAM_INFO]
DEFAULTS     = [p[1] for p in PARAM_INFO]
LOWER        = [p[2] for p in PARAM_INFO]
UPPER        = [p[3] for p in PARAM_INFO]
DISTS        = [uniform(loc=low, scale=(high - low)) for low, high in zip(LOWER, UPPER)]


# ============================================================
# Helpers
# ============================================================

def _timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def _set_by_path(pdict, path, value):
    keys = path.split(".")
    cur = pdict
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = float(value)


def _median_nn_distance(xy):
    xy = np.asarray(xy)
    if xy.shape[0] < 2:
        return 0.0
    tree = cKDTree(xy)
    try:
        d, _ = tree.query(xy, k=2, workers=-1)
    except TypeError:
        d, _ = tree.query(xy, k=2)
    return float(np.median(d[:,1]))


def _clark_evans_R(xy, width, height):
    xy = np.asarray(xy)
    n = xy.shape[0]
    if n < 2:
        return 0.0
    tree = cKDTree(xy)
    try:
        d, _ = tree.query(xy, k=2, workers=-1)
    except TypeError:
        d, _ = tree.query(xy, k=2)
    d_obs = float(np.mean(d[:,1]))
    A = width * height
    lam = n / A
    d_csr = 0.5 / np.sqrt(lam)
    return float(d_obs / d_csr)


def _run_once(x_col, steps, seed, force_phase2):
    params = deepcopy(ABM_DEFAULTS)
    for name, val in zip(PARAM_NAMES, x_col):
        _set_by_path(params, name, float(val))

    model = ClustersModel(params=params, seed=seed)
    try: model.enable_logging = False
    except: pass

    if force_phase2:
        for a in model.agent_set:
            a.movement_phase = 2
            a.phase_switch_time = float("inf")

    for _ in range(steps):
        model.step()

    alive = [a for a in model.agent_set if getattr(a, "alive", True)]
    sizes = np.array([float(a.size) for a in alive])
    xy = np.array([a.pos for a in alive if a.pos is not None])

    n_clusters = float(len(sizes))
    mean_size  = float(np.mean(sizes)) if sizes.size else 0.0
    var_size   = float(np.var(sizes, ddof=1)) if sizes.size > 1 else 0.0
    mnn        = _median_nn_distance(xy) if xy.size else 0.0

    W = params["space"]["width"]
    H = params["space"]["height"]
    R = _clark_evans_R(xy, W, H) if xy.size else 0.0
    return (n_clusters, mean_size, var_size, mnn, R)


def _worker(args):
    j, x_col, steps, seed, force_phase2 = args
    return j, _run_once(x_col, steps, seed, force_phase2)


# ============================================================
# Adaptive vectorised wrapper
# ============================================================

def make_vectorised(cfg, pool, outer_pbar, chunksize, adapt):
    s = len(OUTPUT_LABELS)
    d = len(PARAM_NAMES)

    state = {
        "replicate_counts": None,
        "total_runs": 0,
        "rsem_at_stop": None,
    }

    def f(x):
        assert x.shape[0] == d
        n = x.shape[1]

        sums  = np.zeros((s, n))
        sums2 = np.zeros((s, n))
        counts = np.zeros(n, int)

        def rsem_snapshot():
            means = sums / counts.clip(min=1)
            var = (sums2 - (sums*sums)/counts.clip(min=1)) / np.maximum(counts-1, 1)
            sem = np.sqrt(var / counts.clip(min=1))
            denom = np.maximum(np.abs(means), adapt.eps_mean)
            return sem / denom

        def submit(cols, reps, seedseq):
            tasks = []
            children = seedseq.spawn(n)
            for j in cols:
                repkids = children[j].spawn(reps)
                seeds = [int(ss.generate_state(1)[0]) for ss in repkids]
                xj = x[:,j].copy()
                for sd in seeds:
                    tasks.append((j, xj, cfg.steps, sd, cfg.force_phase2))
            return tasks

        seedseq = np.random.SeedSequence(cfg.base_seed)
        cols = np.arange(n)
        tasks = submit(cols, adapt.R_min, seedseq)
        done = _process_tasks(tasks)

        if not adapt.enable:
            state["replicate_counts"] = counts.copy()
            state["rsem_at_stop"] = rsem_snapshot()
            return sums / counts.clip(min=1)

        while not np.all(done) and np.max(counts) < adapt.R_max:
            remaining = np.where(~done & (counts < adapt.R_max))[0]
            if remaining.size == 0:
                break
            tasks = submit(remaining, adapt.R_step, seedseq)
            done = _process_tasks(tasks)

        state["replicate_counts"] = counts.copy()
        state["rsem_at_stop"] = rsem_snapshot()
        return sums / counts.clip(min=1)

        # local
        def _process_tasks(tasks):
            if outer_pbar:
                outer_pbar.total += len(tasks)
                outer_pbar.refresh()
            for j, outtuple in pool.imap_unordered(_worker, tasks, chunksize=chunksize):
                y = np.asarray(outtuple)
                sums[:,j] += y
                sums2[:,j] += y*y
                counts[j] += 1
                state["total_runs"] += 1
                if outer_pbar:
                    outer_pbar.update(1)
            return np.all(rsem_snapshot() <= adapt.rel_sem_target, axis=0)

    return f, state


# ============================================================
# Save results + simple plot
# ============================================================

def _save(outdir, res, boot, cfg, n, adapt, rep_counts, total_runs, rsem):
    outdir.mkdir(parents=True, exist_ok=True)

    meta = {
        "config": asdict(cfg),
        "n_power_of_two": n,
        "parameters": PARAM_NAMES,
        "outputs": OUTPUT_LABELS,
        "adaptive": asdict(adapt),
    }
    (outdir/"config.json").write_text(json.dumps(meta, indent=2))

    np.savez(outdir/"sobol_arrays.npz",
             first_order=res.first_order,
             total_order=res.total_order,
             S1_CI_low=boot.first_order.confidence_interval.low,
             S1_CI_high=boot.first_order.confidence_interval.high,
             ST_CI_low=boot.total_order.confidence_interval.low,
             ST_CI_high=boot.total_order.confidence_interval.high,
             replicate_counts=rep_counts,
             rsem_at_stop=rsem)


# ============================================================
# MAIN
# ============================================================

def run_sobol(n_power_of_two=128,
              cfg=SAConfig(),
              ci_resamples=300,
              confidence=0.95,
              output_dir="sobol_results",
              n_workers=None,
              chunksize=16,
              adaptive=AdaptiveR()):

    # PRINT PARAMETERS BEFORE RUN
    print("\n[info] PARAMETERS USED:")
    w = max(len(n) for n in PARAM_NAMES)
    print(f"  {'parameter'.ljust(w)}    default        low         high")
    print("  " + "-"*(w+40))
    for name, d, lo, hi in zip(PARAM_NAMES, DEFAULTS, LOWER, UPPER):
        print(f"  {name.ljust(w)}    {d:>10.5g}   {lo:>10.5g}   {hi:>10.5g}")
    print()

    outdir = Path(output_dir) / _timestamp()

    outer_pbar = tqdm(total=0, unit="run", desc="Sobol ABM runs",
                      mininterval=0.5, dynamic_ncols=True) if tqdm else None

    if n_workers is None:
        n_workers = max(1, os.cpu_count() or 2)

    print(f"[info] Using {n_workers} worker processes")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:

        f_vec, state = make_vectorised(cfg, pool, outer_pbar, chunksize, adaptive)

        res = sobol_indices(func=f_vec, n=n_power_of_two, dists=DISTS)
        boot = res.bootstrap(confidence_level=confidence, n_resamples=ci_resamples)

    if outer_pbar:
        outer_pbar.close()

    _save(outdir, res, boot, cfg, n_power_of_two, adaptive,
          state["replicate_counts"], state["total_runs"], state["rsem_at_stop"])

    print("\nRun complete.")
    print("Saved to:", outdir.resolve())
    return res, boot, outdir


if __name__ == "__main__":
    for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    run_sobol()