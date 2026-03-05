# sobol_sensitivity/sobol_sensitivity_6_param_5_metrics_fixed.py
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

# SciPy >= 1.13 provides sobol_indices
from scipy.stats import sobol_indices, uniform
from scipy.spatial import cKDTree  # for median NN + Clark–Evans

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ---- import ABM ----
from abm_sobol.clusters_model import ClustersModel
from abm_sobol.utils import DEFAULTS as ABM_DEFAULTS

# ============================================================
# Config
# ============================================================
@dataclass
class SAConfig:
    steps: int = 145
    force_phase2: bool = True
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

# ============================================================
# Parameters (6 total): include both gamma 'a' and 'scale'
# ============================================================
PARAM_NAMES = [
    "merge.p_merge",
    "phenotypes.proliferative.prolif_rate",
    "phenotypes.proliferative.fragment_rate",
    "movement_v2.phase2.speed_dist.params.a",      # gamma shape
    "movement_v2.phase2.speed_dist.params.scale",  # gamma scale (6th param)
    "movement_v2.phase2.turning.kappa",
]

# Priors
DISTS = [
    uniform(loc=0.10, scale=0.90),  # p_merge
    uniform(loc=0.001, scale=0.019),  # prolif
    uniform(loc=0.0000, scale=0.0020),  # fragment
    uniform(loc=0.50, scale=2.00),  # a (shape)
    uniform(loc=1.00, scale=5.00),  # scale (~1..6)
    uniform(loc=0.0, scale=0.40),  # kappa
]

# Outputs (5)
OUTPUT_LABELS = [
    "Number of clusters",
    "Mean cluster size",
    "Variance of cluster size",
    "Median NN distance",
    "Clark-Evans R",
]

DEBUG = False  # set True for extra prints

# ============================================================
# Helpers
# ============================================================
def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def _apply_params_to_defaults(x):
    """Inject 6 parameters in the order: p_merge, prolif, frag, a, scale, kappa"""
    params = deepcopy(ABM_DEFAULTS)
    p_merge, prolif, frag, a_shape, a_scale, kappa = map(float, x)
    params["merge"]["p_merge"] = p_merge
    params["phenotypes"]["proliferative"]["prolif_rate"] = prolif
    params["phenotypes"]["proliferative"]["fragment_rate"] = frag
    params["movement_v2"]["phase2"]["speed_dist"]["params"]["a"] = a_shape
    params["movement_v2"]["phase2"]["speed_dist"]["params"]["scale"] = a_scale
    params["movement_v2"]["phase2"]["turning"]["kappa"] = kappa
    return params

def _median_nn_distance(xy: np.ndarray) -> float:
    xy = np.asarray(xy, float)
    if xy.shape[0] < 2:
        return 0.0
    tree = cKDTree(xy)
    # Older SciPy might not have workers kwarg
    try:
        d, _ = tree.query(xy, k=2, workers=-1)
    except TypeError:
        d, _ = tree.query(xy, k=2)
    return float(np.median(d[:, 1]))

def _clark_evans_R(xy: np.ndarray, width: float, height: float) -> float:
    xy = np.asarray(xy, float)
    n = xy.shape[0]
    if n < 2:
        return 0.0
    tree = cKDTree(xy)
    try:
        d, _ = tree.query(xy, k=2, workers=-1)
    except TypeError:
        d, _ = tree.query(xy, k=2)
    d_obs = float(np.mean(d[:, 1]))
    A = width * height
    lam = n / A if A > 0 else np.inf
    d_csr = 0.5 / np.sqrt(lam) if lam > 0 else np.inf
    return float(d_obs / d_csr) if d_csr > 0 else 0.0

# ============================================================
# One ABM simulation
# ============================================================
def _run_once(x, steps, seed, force_phase2):
    params = _apply_params_to_defaults(x)
    model = ClustersModel(params=params, seed=seed)
    try:
        model.enable_logging = False
    except Exception:
        pass
    if force_phase2:
        for a in list(model.agent_set):
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
    return (n_clusters, mean_size, var_size, mnn, R)

def _abm_replicate_worker(args):
    j, x_col, steps, seed, force_phase2 = args
    try:
        return j, _run_once(x_col, steps, seed, force_phase2)
    except Exception:
        import traceback; traceback.print_exc()
        raise

_EVAL_COUNTER = 0

# ============================================================
# Adaptive Vectorised Function (d = 6 expected)
# ============================================================
def make_abm_vectorised_func_adaptive(cfg, pool, outer_pbar, chunksize, adapt):
    s = len(OUTPUT_LABELS)
    d_expected = 6  # must match d in run_sobol

    state = {"replicate_counts": None,
             "total_runs": 0,
             "rsem_at_stop": None}

    def f(x: np.ndarray) -> np.ndarray:
        if DEBUG:
            print(f"[debug] f() called with x shape {x.shape}")
        assert x.shape[0] == d_expected, f"Expected (6, n) but got {x.shape}"
        n = x.shape[1]

        global _EVAL_COUNTER
        call_seed_seq = np.random.SeedSequence(cfg.base_seed + _EVAL_COUNTER)
        _EVAL_COUNTER += 1

        # Accumulators
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
                    tasks.append((j, xj, cfg.steps, sd, cfg.force_phase2))
            return tasks

        def process(tasks):
            if outer_pbar is not None:
                outer_pbar.total += len(tasks)
                outer_pbar.refresh()
            if not tasks:
                return np.ones(n, dtype=bool)
            for j, out_tuple in pool.imap_unordered(_abm_replicate_worker, tasks, chunksize=chunksize):
                y = np.asarray(out_tuple, float)
                sums[:, j] += y
                sums2[:, j] += y * y
                counts[j] += 1
                state["total_runs"] += 1
                if outer_pbar is not None:
                    outer_pbar.update(1)
            rsem = rsem_snapshot()
            return np.all(rsem <= adapt.rel_sem_target, axis=0)

        # Pilot
        cols = np.arange(n)
        done = process(submit_jobs(cols, adapt.R_min if adapt.enable else cfg.replicates))
        if not adapt.enable:
            state["replicate_counts"] = counts.copy()
            state["rsem_at_stop"] = rsem_snapshot()
            return sums / counts.clip(min=1)

        # Adaptive
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
def _save_results(out_dir, res, boot, cfg, n, adapt,
                  rep_counts, total_runs, rsem_at_stop):
    out_dir.mkdir(parents=True, exist_ok=True)

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
                "median": float(np.median(rep_counts)),
                "mean": float(np.mean(rep_counts)),
                "total_abm_runs": int(total_runs),
            }
        ),
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))

    def write_matrix(name, arr):
        hdr = ["output"] + PARAM_NAMES
        with open(out_dir / name, "w") as f:
            f.write(",".join(hdr) + "\n")
            for lab, row in zip(OUTPUT_LABELS, arr):
                f.write(",".join([lab] + [f"{float(v):.8f}" for v in row]) + "\n")

    write_matrix("indices_first_order.csv", res.first_order)
    write_matrix("indices_total_order.csv", res.total_order)
    write_matrix("ci_first_order_low.csv", boot.first_order.confidence_interval.low)
    write_matrix("ci_first_order_high.csv", boot.first_order.confidence_interval.high)
    write_matrix("ci_total_order_low.csv", boot.total_order.confidence_interval.low)
    write_matrix("ci_total_order_high.csv", boot.total_order.confidence_interval.high)

    if rsem_at_stop is not None:
        with open(out_dir / "rsem_per_column.csv", "w") as f:
            hdr = ["output"] + [f"col_{j:04d}" for j in range(rsem_at_stop.shape[1])]
            f.write(",".join(hdr) + "\n")
            for lab, row in zip(OUTPUT_LABELS, rsem_at_stop):
                f.write(",".join([lab] + [f"{float(v):.8g}" for v in row]) + "\n")

    np.savez(out_dir / "sobol_arrays.npz",
             first_order=res.first_order,
             total_order=res.total_order,
             S1_CI_low=boot.first_order.confidence_interval.low,
             S1_CI_high=boot.first_order.confidence_interval.high,
             ST_CI_low=boot.total_order.confidence_interval.low,
             ST_CI_high=boot.total_order.confidence_interval.high,
             replicate_counts=rep_counts,
             rsem_at_stop=rsem_at_stop)


def _save_plot_all_outputs(out_dir: Path, res, boot):
    s = len(OUTPUT_LABELS)
    d = len(PARAM_NAMES)
    x = np.arange(d)
    ncols = 2
    nrows = int(np.ceil(s / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4*nrows), constrained_layout=True)
    axes = axes.flatten()
    for i in range(s):
        ax = axes[i]
        s1 = res.first_order[i]
        st = res.total_order[i]
        s1_lo = boot.first_order.confidence_interval.low[i]
        s1_hi = boot.first_order.confidence_interval.high[i]
        st_lo = boot.total_order.confidence_interval.low[i]
        st_hi = boot.total_order.confidence_interval.high[i]
        ax.errorbar(x - 0.12, s1, yerr=[s1 - s1_lo, s1_hi - s1], fmt='o', capsize=3, label="S1")
        ax.errorbar(x + 0.12, st, yerr=[st - st_lo, st_hi - st], fmt='o', capsize=3, label="ST")
        ax.set_xticks(x, PARAM_NAMES, rotation=30, ha="right")
        ax.set_title(OUTPUT_LABELS[i])
        ax.set_ylabel("Sobol index")
        ax.legend()
    for j in range(s, len(axes)):
        axes[j].axis("off")
    fig.savefig(out_dir / "sobol_all_outputs.png", dpi=200)
    plt.close(fig)

# ============================================================
# Preflight checks + main run
# ============================================================

def _preflight_or_raise(d: int):
    # Dimensionality consistency
    if len(PARAM_NAMES) != d or len(DISTS) != d:
        raise ValueError(
            f"Dimensionality mismatch: d={d}, "
            f"len(PARAM_NAMES)={len(PARAM_NAMES)}, len(DISTS)={len(DISTS)}"
        )
    # Defaults must include gamma params 'a' and 'scale' and turning 'kappa'
    try:
        params = ABM_DEFAULTS["movement_v2"]["phase2"]["speed_dist"]["params"]
        _ = params["a"]; _ = params["scale"]
        _ = ABM_DEFAULTS["movement_v2"]["phase2"]["turning"]["kappa"]
    except Exception as e:
        raise KeyError("Defaults missing one of: params.a / params.scale / turning.kappa") from e


def _pilot_warmup(pool, cfg: SAConfig, adapt: AdaptiveR, d: int):
    """Run exactly one tiny replicate through the pool to surface import/pickling issues quickly.
    Raises RuntimeError with a helpful message on timeout.
    """
    import time
    # Mid-points of the prior for a single column
    x = np.zeros((d, 1), float)
    for i, dist in enumerate(DISTS):
        loc = getattr(dist, "kwds", {}).get("loc", 0.0)
        sca = getattr(dist, "kwds", {}).get("scale", 1.0)
        x[i, 0] = float(loc + 0.5 * sca)
    task = (0, x[:, 0].copy(), max(5, min(20, cfg.steps)),  # keep warm-up fast
            int(np.random.SeedSequence(cfg.base_seed).generate_state(1)[0]),
            cfg.force_phase2)
    # Use apply_async to be able to timeout
    async_res = pool.apply_async(_abm_replicate_worker, (task,))
    try:
        _ = async_res.get(timeout=120)  # 2 minutes should be plenty
    except mp.context.TimeoutError:
        raise RuntimeError(
            "Multiprocessing warm-up timed out. Likely the workers failed to import the module or "
            "'abm_sobol' package under macOS 'spawn'. Try running with `python -m ...,` ensure the "
            "project root is on PYTHONPATH, or install your package with `pip install -e .`."
        )


def run_sobol(
    n_power_of_two: int = 128,  # must be power of two
    cfg: SAConfig = SAConfig(),
    ci_resamples: int = 300,
    confidence: float = 0.95,
    output_dir: str | Path = "sobol_results",
    n_workers: int | None = None,
    chunksize: int = 16,
    adaptive: AdaptiveR = AdaptiveR(),
):
    d = 6  # must match d_expected
    s = len(OUTPUT_LABELS)

    _preflight_or_raise(d)

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
        print(f"[info] Adaptive replicates: R_min={adaptive.R_min}, R_step={adaptive.R_step}, "
              f"R_max={adaptive.R_max}, rel_sem_target={adaptive.rel_sem_target:.2%}")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        # Quick warm-up to surface worker import/pickling problems with a clear error
        _pilot_warmup(pool, cfg, adaptive, d)

        f_vec, state = make_abm_vectorised_func_adaptive(cfg, pool, outer_pbar, chunksize, adaptive)

        # Real Sobol
        res = sobol_indices(func=f_vec, n=n_power_of_two, dists=DISTS)
        boot = res.bootstrap(confidence_level=confidence, n_resamples=ci_resamples)

    if outer_pbar:
        outer_pbar.close()

    rep_counts = state["replicate_counts"]
    total_runs = state["total_runs"]
    rsem_at_stop = state["rsem_at_stop"]

    _save_results(out_dir, res, boot, cfg, n_power_of_two, adaptive,
                  rep_counts, total_runs, rsem_at_stop)
    _save_plot_all_outputs(out_dir, res, boot)

    print("\nRun summary")
    print("-----------")
    print(f"Outputs : {', '.join(OUTPUT_LABELS)}")
    print(f"Parameters (d) : {d}")
    print(f"n (power of 2) : {n_power_of_two}")
    if rep_counts is not None:
        rsem_max = np.max(rsem_at_stop, axis=0)
        frac_bad = np.mean(rsem_max > adaptive.rel_sem_target) * 100.0
        print(f"Columns above rel_sem_target at stop: {frac_bad:.1f}%")
        print(f"Total ABM runs : {total_runs}")
    print(f"Saved to : {out_dir.resolve()}")

    return res, boot, out_dir


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    # Reduce BLAS threading for stability/perf with multiprocessing
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    # Be explicit about start method on macOS
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # already set
        pass

    cfg = SAConfig(steps=145, force_phase2=True)
    adaptive = AdaptiveR(enable=True, R_min=6, R_step=2, R_max=200, rel_sem_target=0.05)

    run_sobol(
        n_power_of_two=1024,
        cfg=cfg,
        ci_resamples=300,
        output_dir="sobol_results",
        n_workers=None,  # auto
        chunksize=16,
        adaptive=adaptive,
    )
