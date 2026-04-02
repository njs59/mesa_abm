# Sensitivity_analysis/sobol_runner.py
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.stats import sobol_indices, uniform
from scipy.spatial import cKDTree
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ---- ABM imports (current codebase) ----
from abm.clusters_model import ClustersModel
from abm.utils import DEFAULTS as ABM_DEFAULTS

# ============================================================
# Configuration
# ============================================================
@dataclass
class SAConfig:
    steps: int = 145
    force_phase2: bool = True
    replicates: int = 10  # ignored if adaptive.enable=True
    base_seed: int = 42

@dataclass
class AdaptiveR:
    enable: bool = True
    R_min: int = 6
    R_step: int = 2
    R_max: int = 16
    rel_sem_target: float = 0.07
    eps_mean: float = 1e-8

# Parameters (d=6) – keep dummy as an inert dimension
# Proliferative baseline 
PARAM_NAMES = [
    "merge.p_merge",
    "phenotypes.proliferative.prolif_rate",
    "phenotypes.proliferative.fragment_rate",
    "movement_v2.proliferative.phase2.speed_dist.params.a",
    "movement_v2.proliferative.phase2.turning.kappa",
    "dummy_param",  # sampled but intentionally UNUSED
]

# Invasive baseline
# PARAM_NAMES = [
#     "merge.p_merge",
#     "phenotypes.invasive.prolif_rate",
#     "phenotypes.invasive.fragment_rate",
#     "movement_v2.invasive.phase2.speed_dist.params.a",
#     "movement_v2.invasive.phase2.turning.kappa",
#     "dummy_param",  # sampled but intentionally UNUSED
# ]

# ---- Outputs (define 5-core and 4-core sets) ----  # NEW
OUTPUT_KEYS_CORE5 = ["n_clusters", "mean_size", "var_size", "median_nnd", "clark_evans_R"]
OUTPUT_KEYS_CORE4 = ["n_clusters", "mean_size", "var_size", "median_nnd"]

KEY_TO_LABEL = {
    "n_clusters":     "Number of clusters",
    "mean_size":      "Mean cluster size",
    "var_size":       "Variance of cluster size",
    "median_nnd":     "Median NN distance",
    "clark_evans_R":  "Clark-Evans R",
}

# Default Distributions DO NOT CHANGE
# DISTS = [
#     uniform(loc=0.10, scale=0.90),   # p_merge: 0.10–1.00
#     uniform(loc=0.001, scale=0.019), # prolif: 0.001–0.020
#     uniform(loc=0.0000, scale=0.0020),# fragment: 0.0000–0.0020
#     uniform(loc=0.70, scale=1.40),   # a (shape): 0.70–2.10
#     uniform(loc=0.0, scale=0.40),    # kappa: 0.0–0.4
#     uniform(loc=0.0, scale=1.0),     # dummy_param: 0–1 (ignored)
# ]


# Param changes

# Tighten prolif 1:
# prolif 0.003 - 0.013
# uniform(loc=0.003, scale=0.010),  # prolif:    0.003–0.013

# Tighten prolif 1, relax fragment:
# prolif 0.003 - 0.013
# uniform(loc=0.003, scale=0.010),  # prolif:    0.003–0.013
# fragment 0.0000 - 0.01
# uniform(loc=0.0000, scale=0.010),# fragment:  0.0000–0.010


# Tighten prolif 1, relax kappa:
# prolif 0.003 - 0.013
# uniform(loc=0.003, scale=0.010),  # prolif:    0.003–0.013
# kappa 0.0000 - 2.0
# uniform(loc=0.0000, scale=2.0),   # kappa:  0.0-2.0

# Tighten prolif 1, tighter merge:
# prolif 0.003 - 0.013
# uniform(loc=0.003, scale=0.010),  # prolif:    0.003–0.013
# merge 0.4-1.0
# uniform(loc=0.40,  scale=0.60),   # p_merge:   0.40–1.00

# Tighten prolif 2, tighter merge:
# prolif 0.009 - 0.013
# uniform(loc=0.009, scale=0.004),  # prolif:    0.009–0.013
# merge 0.4-1.0
# uniform(loc=0.40,  scale=0.60),   # p_merge:   0.40–1.00

# Tighten prolif 2:
# prolif 0.009 - 0.013
# uniform(loc=0.009, scale=0.004),  # prolif:    0.009–0.013


# Distributions
DISTS = [
    uniform(loc=0.10,  scale=0.90),   # p_merge:   0.10–1.00
    uniform(loc=0.009, scale=0.004),  # prolif:    0.009–0.013
    uniform(loc=0.0000, scale=0.0020),# fragment: 0.0000–0.0020
    uniform(loc=0.70,  scale=1.40),   # a (shape): 0.70–2.10
    uniform(loc=0.0, scale=0.4),      # kappa:  0.0-0.4
    uniform(loc=0.0,   scale=1.0),    # dummy_param: 0–1 (ignored)
]

# ============================================================
# Utilities
# ============================================================
def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def _apply_params_to_defaults(x):
    """Apply vector x (len=6) to deep-copied DEFAULTS; 6th dummy is ignored."""
    params = deepcopy(ABM_DEFAULTS)
    p_merge, prolif, frag, a_shape, kappa, dummy = map(float, x)
    params["merge"]["p_merge"] = p_merge
    params["phenotypes"]["proliferative"]["prolif_rate"] = prolif
    params["phenotypes"]["proliferative"]["fragment_rate"] = frag
    params["movement_v2"]["proliferative"]["phase2"]["speed_dist"]["params"]["a"] = a_shape
    params["movement_v2"]["proliferative"]["phase2"]["turning"]["kappa"] = kappa

    # params["phenotypes"]["invasive"]["prolif_rate"] = prolif
    # params["phenotypes"]["invasive"]["fragment_rate"] = frag
    # params["movement_v2"]["invasive"]["phase2"]["speed_dist"]["params"]["a"] = a_shape
    # params["movement_v2"]["invasive"]["phase2"]["turning"]["kappa"] = kappa
    return params  # dummy intentionally ignored

def _median_nn_distance(xy: np.ndarray) -> float:
    xy = np.asarray(xy)
    if xy.shape[0] < 2:
        return 0.0
    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=2, workers=-1)
    return float(np.median(d[:, 1]))

def _clark_evans_R(xy: np.ndarray, width: float, height: float) -> float:
    xy = np.asarray(xy)
    n = xy.shape[0]
    if n < 2:
        return 0.0
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
# Single ABM replicate -> full 5-tuple in fixed order
# ============================================================
def _run_once(x, steps, seed, force_phase2):
    params = _apply_params_to_defaults(x)
    model = ClustersModel(params=params, seed=seed)
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

    alive = [a for a in model.agent_set if getattr(a, "alive", True)]
    sizes = np.array([float(a.size) for a in alive], float)
    xy = np.array([a.pos for a in alive if a.pos is not None], float)
    n_clusters = float(sizes.size)
    mean_size = float(np.mean(sizes)) if sizes.size else 0.0
    var_size = float(np.var(sizes, ddof=1)) if sizes.size > 1 else 0.0
    mnn = _median_nn_distance(xy) if xy.size else 0.0
    W = float(model.params["space"]["width"])
    H = float(model.params["space"]["height"])
    R = _clark_evans_R(xy, W, H) if xy.size else 0.0

    # Fixed 5‑tuple in the canonical order
    return (n_clusters, mean_size, var_size, mnn, R)

def _abm_replicate_worker(args):
    j, x_col, steps, seed, force_phase2, selected_idx = args  # CHANGED: pass indices in
    full = _run_once(x_col, steps, seed, force_phase2)
    y = np.asarray([full[k] for k in selected_idx], float)     # slice to requested metrics
    return j, y

_EVAL_COUNTER = 0

# ============================================================
# Adaptive vectorised wrapper
# ============================================================
def make_abm_vectorised_func_adaptive(cfg, pool, outer_pbar, chunksize, adapt,
                                      selected_idx: List[int], selected_labels: List[str]):
    s = len(selected_idx)                   # number of outputs kept   # CHANGED
    d_expected = len(PARAM_NAMES)
    state = {"replicate_counts": None, "total_runs": 0, "rsem_at_stop": None}

    def f(x: np.ndarray) -> np.ndarray:
        assert x.shape[0] == d_expected
        n = x.shape[1]
        global _EVAL_COUNTER
        call_seed_seq = np.random.SeedSequence(cfg.base_seed + _EVAL_COUNTER)
        _EVAL_COUNTER += 1

        sums = np.zeros((s, n), float)
        sums2 = np.zeros((s, n), float)
        counts = np.zeros(n, int)

        def rsem_snapshot():
            means = sums / counts.clip(min=1)
            var = (sums2 - (sums * sums) / counts.clip(min=1)) / np.maximum(counts - 1, 1)
            sem = np.sqrt(var / counts.clip(min=1))
            denom = np.maximum(np.abs(means), adapt.eps_mean)
            return sem / denom

        def submit_jobs(cols, reps):
            tasks = []
            col_children = call_seed_seq.spawn(n)
            for j in cols:
                rep_children = col_children[j].spawn(reps)
                seeds = [int(ss.generate_state(1)[0]) for ss in rep_children]
                xj = x[:, j].copy()
                for sd in seeds:
                    tasks.append((j, xj, cfg.steps, sd, cfg.force_phase2, selected_idx))  # pass indices
            return tasks

        def process(tasks):
            if outer_pbar is not None:
                outer_pbar.total += len(tasks)
                outer_pbar.refresh()
            for j, y in pool.imap_unordered(_abm_replicate_worker, tasks, chunksize=chunksize):
                sums[:, j] += y
                sums2[:, j] += y * y
                counts[j] += 1
                state["total_runs"] += 1
                if outer_pbar is not None:
                    outer_pbar.update(1)
            return np.all(rsem_snapshot() <= adapt.rel_sem_target, axis=0)

        # pilot
        cols = np.arange(n)
        done = process(submit_jobs(cols, adapt.R_min if adapt.enable else cfg.replicates))
        if not adapt.enable:
            state["replicate_counts"] = counts.copy()
            state["rsem_at_stop"] = rsem_snapshot()
            return sums / counts.clip(min=1)

        # adaptive loop
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
# Save results (CSV + NPZ) and quick plot
# ============================================================
def _save_results(out_dir, res, boot, cfg, n, adapt, rep_counts, total_runs, rsem_at_stop,
                  selected_labels: List[str], metrics_set: str):  # CHANGED
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": asdict(cfg),
        "n_power_of_two": n,
        "parameters": PARAM_NAMES,
        "outputs": selected_labels,             # CHANGED: only kept labels
        "metrics_set": metrics_set,             # NEW: record variant
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

    def write_matrix(name, arr, labels):
        hdr = ",".join(["output"] + PARAM_NAMES)
        with open(out_dir / name, "w") as f:
            f.write(hdr + "\n")
            for i, lab in enumerate(labels):
                row = [lab] + [f"{float(v):.8f}" for v in arr[i]]
                f.write(",".join(row) + "\n")

    write_matrix("indices_first_order.csv", res.first_order, selected_labels)
    write_matrix("indices_total_order.csv", res.total_order, selected_labels)
    write_matrix("ci_first_order_low.csv",  boot.first_order.confidence_interval.low,  selected_labels)
    write_matrix("ci_first_order_high.csv", boot.first_order.confidence_interval.high, selected_labels)
    write_matrix("ci_total_order_low.csv",  boot.total_order.confidence_interval.low,  selected_labels)
    write_matrix("ci_total_order_high.csv", boot.total_order.confidence_interval.high, selected_labels)

    # NPZ bundle for plotting scripts
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

def _save_plot(out_dir, res, boot, selected_labels: List[str]):  # CHANGED
    if len(selected_labels) < 1:
        return
    x = np.arange(len(PARAM_NAMES))
    nfig = min(2, len(selected_labels))
    labels = selected_labels[:nfig]
    fig, axs = plt.subplots(1, nfig, figsize=(6 * nfig, 4), constrained_layout=True)
    if nfig == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        s1 = res.first_order[i]; st = res.total_order[i]
        s1_lo = boot.first_order.confidence_interval.low[i]
        s1_hi = boot.first_order.confidence_interval.high[i]
        st_lo = boot.total_order.confidence_interval.low[i]
        st_hi = boot.total_order.confidence_interval.high[i]
        ax.errorbar(x - 0.12, s1, yerr=[s1 - s1_lo, s1_hi - s1], fmt='o', capsize=3, label="S1")
        ax.errorbar(x + 0.12, st, yerr=[st - st_lo, st_hi - st], fmt='o', capsize=3, label="ST")
        ax.set_xticks(x, PARAM_NAMES, rotation=30, ha="right")
        ax.set_title(labels[i]); ax.set_ylabel("Sobol index"); ax.legend()
    fig.savefig(out_dir / "quickplot.png", dpi=200); plt.close(fig)

# ============================================================
# Main API
# ============================================================
def run_sobol(
    n_power_of_two: int = 256,
    cfg: SAConfig = SAConfig(),
    ci_resamples: int = 500,
    confidence: float = 0.95,
    output_dir: str | Path = "Sensitivity_analysis_results",
    n_workers: int | None = None,
    chunksize: int = 16,
    adaptive: AdaptiveR = AdaptiveR(),
    metrics_set: str = "core5",  # NEW: 'core4' (no Clark-Evans) or 'core5'
):
    # Resolve output set
    if metrics_set not in {"core4", "core5"}:
        raise ValueError("metrics_set must be 'core4' or 'core5'")

    output_keys = OUTPUT_KEYS_CORE4 if metrics_set == "core4" else OUTPUT_KEYS_CORE5
    selected_labels = [KEY_TO_LABEL[k] for k in output_keys]
    # Indices into the canonical 5‑tuple
    all_keys = OUTPUT_KEYS_CORE5
    selected_idx = [all_keys.index(k) for k in output_keys]

    # Name the run directory with the set for clarity
    out_dir = Path(output_dir) / f"{_timestamp()}__{metrics_set}"   # CHANGED
    out_dir.mkdir(parents=True, exist_ok=True)

    outer_pbar = tqdm(total=0, unit="run", desc="Sobol ABM runs",
                      mininterval=0.5, dynamic_ncols=True) if tqdm else None

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2))
    print(f"[info] Using {n_workers} worker processes")
    if adaptive.enable:
        print(f"[info] Adaptive: R_min={adaptive.R_min}, R_step={adaptive.R_step}, "
              f"R_max={adaptive.R_max}, rel_sem_target={adaptive.rel_sem_target:.2%}")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        f, state = make_abm_vectorised_func_adaptive(
            cfg, pool, outer_pbar, chunksize, adaptive,
            selected_idx=selected_idx, selected_labels=selected_labels
        )
        res = sobol_indices(func=f, n=n_power_of_two, dists=DISTS)
        boot = res.bootstrap(confidence_level=confidence, n_resamples=ci_resamples)

    if outer_pbar:
        outer_pbar.close()

    rep_counts = state["replicate_counts"]
    total_runs = state["total_runs"]
    rsem_at_stop = state["rsem_at_stop"]

    _save_results(out_dir, res, boot, cfg, n_power_of_two, adaptive, rep_counts, total_runs,
                  rsem_at_stop, selected_labels=selected_labels, metrics_set=metrics_set)
    _save_plot(out_dir, res, boot, selected_labels=selected_labels)

    print("\nRun summary")
    print("-----------")
    print(f"Outputs          : {', '.join(selected_labels)}")
    print(f"n (power of 2)   : {n_power_of_two}")
    print(f"Workers          : {n_workers}")
    print(f"Saved to         : {out_dir.resolve()}")
    if rep_counts is not None and rsem_at_stop is not None:
        rsem_max = np.max(rsem_at_stop, axis=0)
        frac_bad = np.mean(rsem_max > adaptive.rel_sem_target) * 100.0
        print(f"Columns > threshold: {frac_bad:.1f}%")
        print(f"Total ABM runs  : {total_runs}")

    return res, boot, out_dir