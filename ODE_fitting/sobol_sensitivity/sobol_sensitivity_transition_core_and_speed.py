#!/usr/bin/env python3
"""
Consolidated Sobol script:
- Core parameters
- Phase 1 speed params (lognormal)
- Phase 2 speed params (gamma)
- Robust warm-up to detect worker crashes
- Auto project-root insertion for macOS spawn
- Fixed scoping of inner functions
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from copy import deepcopy
from pathlib import Path
from datetime import datetime
import numpy as np
import multiprocessing as mp
import json
import os
import sys

from scipy.stats import sobol_indices, uniform
from scipy.spatial import cKDTree
try:
    from tqdm import tqdm
except:
    tqdm = None

# ======================================================================
# FIX: ensure project root is visible to worker processes
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import ABM now that path is safe
from abm_sobol.clusters_model import ClustersModel
from abm_sobol.utils import DEFAULTS as ABM_DEFAULTS

# ======================================================================
# OUTPUT LABELS
# ======================================================================
OUTPUT_LABELS = [
    "Number of clusters",
    "Mean cluster size",
    "Variance of cluster size",
    "Median NN distance",
    "Clark-Evans R",
]

# ======================================================================
# PARAMETER DEFINITIONS
# ======================================================================
PARAM_INFO = [
    # Core
    ("merge.p_merge", ABM_DEFAULTS["merge"]["p_merge"], 0.10, 1.00),
    ("phenotypes.proliferative.prolif_rate",
        ABM_DEFAULTS["phenotypes"]["proliferative"]["prolif_rate"],
        0.001, 0.020),
    ("phenotypes.proliferative.fragment_rate",
        ABM_DEFAULTS["phenotypes"]["proliferative"]["fragment_rate"],
        0.000, 0.002),

    # Phase‑1 speed distribution (lognormal style)
    ("movement_v2.phase1.speed_dist.params.s",
        ABM_DEFAULTS["movement_v2"]["phase1"]["speed_dist"]["params"]["s"],
        0.485148, 1.94059),

    ("movement_v2.phase1.speed_dist.params.scale",
        ABM_DEFAULTS["movement_v2"]["phase1"]["speed_dist"]["params"]["scale"],
        2.25863, 7.2931),   # you used 7.2931 last run; restore to 11.2931 if desired

    # Phase‑2 gamma distribution
    ("movement_v2.phase2.speed_dist.params.a",
        ABM_DEFAULTS["movement_v2"]["phase2"]["speed_dist"]["params"]["a"],
        0.50, 2.50),

    ("movement_v2.phase2.speed_dist.params.scale",
        ABM_DEFAULTS["movement_v2"]["phase2"]["speed_dist"]["params"]["scale"],
        1.01, 3.00),
]

PARAM_NAMES = [p[0] for p in PARAM_INFO]
DEFAULTS    = [float(p[1]) for p in PARAM_INFO]
LOWER       = [float(p[2]) for p in PARAM_INFO]
UPPER       = [float(p[3]) for p in PARAM_INFO]
DISTS       = [uniform(loc=lo, scale=(hi-lo)) for lo,hi in zip(LOWER,UPPER)]

# ======================================================================
# CONFIG
# ======================================================================
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

def _timestamp(): return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ======================================================================
# ABM RUN HELPERS
# ======================================================================
def _set_by_path(pdict, path, value):
    keys = path.split(".")
    cur = pdict
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = float(value)

def _median_nn_distance(xy):
    xy = np.asarray(xy)
    if xy.shape[0] < 2: return 0.0
    try: d,_ = cKDTree(xy).query(xy, k=2, workers=-1)
    except: d,_ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:,1]))

def _clark_evans_R(xy, W, H):
    xy = np.asarray(xy)
    if xy.shape[0] < 2: return 0.0
    try: d,_ = cKDTree(xy).query(xy, k=2, workers=-1)
    except: d,_ = cKDTree(xy).query(xy, k=2)
    d_obs = float(np.mean(d[:,1]))
    lam   = xy.shape[0] / (W*H)
    d_csr = 0.5 / np.sqrt(lam)
    return float(d_obs / d_csr)

def _run_once(x_col, steps, seed, force_phase2):
    params = deepcopy(ABM_DEFAULTS)

    for name,val in zip(PARAM_NAMES, x_col):
        _set_by_path(params, name, val)

    model = ClustersModel(params=params, seed=seed)
    try: model.enable_logging = False
    except: pass

    if force_phase2:
        for a in model.agent_set:
            a.movement_phase = 2
            a.phase_switch_time = float("inf")

    for _ in range(steps):
        model.step()

    alive = [a for a in model.agent_set if getattr(a,"alive",True)]
    sizes = np.array([a.size for a in alive], float)
    xy    = np.array([a.pos for a in alive if a.pos is not None], float)

    n    = float(len(sizes))
    mean = float(np.mean(sizes)) if sizes.size else 0.0
    var  = float(np.var(sizes, ddof=1)) if sizes.size>1 else 0.0
    mnn  = _median_nn_distance(xy) if xy.size else 0.0

    W = float(params["space"]["width"])
    H = float(params["space"]["height"])
    R = _clark_evans_R(xy, W, H) if xy.size else 0.0

    return (n, mean, var, mnn, R)

def _worker(args):
    j,x_col,steps,seed,f2 = args
    return j,_run_once(x_col,steps,seed,f2)

# ======================================================================
# VECTORISED ADAPTIVE FUNCTION
# ======================================================================
def make_vectorised(cfg, pool, outer_pbar, chunksize, adapt):
    s = len(OUTPUT_LABELS)
    d = len(PARAM_NAMES)

    state = {"replicate_counts":None, "total_runs":0, "rsem_at_stop":None}

    def f(x):
        assert x.shape[0] == d
        n = x.shape[1]

        sums  = np.zeros((s,n))
        sums2 = np.zeros((s,n))
        counts = np.zeros(n,int)

        def rsem():
            means = sums / counts.clip(min=1)
            var   = (sums2 - (sums*sums)/counts.clip(min=1)) / np.maximum(counts-1,1)
            sem   = np.sqrt(var / counts.clip(min=1))
            denom = np.maximum(np.abs(means), adapt.eps_mean)
            return sem/denom

        seedseq = np.random.SeedSequence(cfg.base_seed)

        def submit(cols, reps):
            tasks = []
            kids = seedseq.spawn(n)
            for j in cols:
                repkids = kids[j].spawn(reps)
                seeds = [int(ss.generate_state(1)[0]) for ss in repkids]
                xj = x[:,j].copy()
                for sd in seeds:
                    tasks.append((j,xj,cfg.steps,sd,cfg.force_phase2))
            return tasks

        def _process(tasks):
            if outer_pbar:
                outer_pbar.total += len(tasks)
                outer_pbar.refresh()
            for j,out in pool.imap_unordered(_worker, tasks, chunksize=chunksize):
                y = np.asarray(out)
                sums[:,j]  += y
                sums2[:,j] += y*y
                counts[j]  += 1
                state["total_runs"] += 1
                if outer_pbar: outer_pbar.update(1)
            return np.all(rsem() <= adapt.rel_sem_target, axis=0)

        # Pilot round
        cols = np.arange(n)
        done = _process(submit(cols, adapt.R_min if adapt.enable else cfg.replicates))

        # Adaptive loop
        if adapt.enable:
            while not np.all(done) and np.max(counts) < adapt.R_max:
                remain = np.where(~done & (counts<adapt.R_max))[0]
                if remain.size == 0:
                    break
                done = _process(submit(remain, adapt.R_step))

        state["replicate_counts"]=counts.copy()
        state["rsem_at_stop"]=rsem()
        return sums / counts.clip(min=1)

    return f, state

# ======================================================================
# WARM‑UP: Catch worker failures before Sobol starts
# ======================================================================
def warmup(pool, cfg):
    print("[debug] Warm‑up: verifying worker can import ABM + run step() ...")
    mid = np.array([(lo+hi)/2 for lo,hi in zip(LOWER,UPPER)], float)
    task = (0, mid, 10, cfg.base_seed, cfg.force_phase2)
    try:
        out = pool.apply_async(_worker, (task,)).get(timeout=30)
        print("[debug] Warm‑up OK:", out)
    except Exception as e:
        print("\n[ERROR] Worker crashed during warm‑up.\n"
              "Most common cause: worker cannot import abm_sobol.\n"
              "Run `pip install -e .` in project root or fix sys.path.")
        raise

# ======================================================================
# MAIN
# ======================================================================
def run_sobol(n_power_of_two=128,
              cfg=SAConfig(),
              ci_resamples=300,
              confidence=0.95,
              output_dir="sobol_results",
              n_workers=None,
              chunksize=16,
              adaptive=AdaptiveR()):

    print("\n[info] PARAMETERS USED:")
    w = max(len(n) for n in PARAM_NAMES)
    print(f"  {'parameter'.ljust(w)}    default      low       high")
    print("  "+"-"*(w+34))
    for n,d,lo,hi in zip(PARAM_NAMES,DEFAULTS,LOWER,UPPER):
        print(f"  {n.ljust(w)}   {d:>8.4g}   {lo:>8.4g}   {hi:>8.4g}")
    print()

    if n_workers is None:
        n_workers = max(1, os.cpu_count() or 2)
    print(f"[info] Using {n_workers} worker processes")

    outdir = Path(output_dir)/_timestamp()
    outer = tqdm(total=0, unit="run", desc="Sobol ABM runs",
                 mininterval=0.5, dynamic_ncols=True) if tqdm else None

    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, maxtasksperchild=1) as pool:

        # ---- warm‑up (crashes immediately if workers fail)
        warmup(pool, cfg)

        # ---- build vec func + run sobol
        f_vec, state = make_vectorised(cfg, pool, outer, chunksize, adaptive)

        res  = sobol_indices(func=f_vec, n=n_power_of_two, dists=DISTS)
        boot = res.bootstrap(confidence_level=confidence, n_resamples=ci_resamples)

    if outer: outer.close()

    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir/"sobol_arrays.npz",
             first_order=res.first_order,
             total_order=res.total_order,
             S1_CI_low=boot.first_order.confidence_interval.low,
             S1_CI_high=boot.first_order.confidence_interval.high,
             ST_CI_low=boot.total_order.confidence_interval.low,
             ST_CI_high=boot.total_order.confidence_interval.high)

    (outdir/"config.json").write_text(json.dumps({
        "parameters":PARAM_NAMES,
        "defaults":DEFAULTS,
        "lower":LOWER,
        "upper":UPPER,
        "config":asdict(cfg)
    }, indent=2))

    print("\nSaved:", outdir.resolve())


if __name__ == "__main__":
    for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    run_sobol()