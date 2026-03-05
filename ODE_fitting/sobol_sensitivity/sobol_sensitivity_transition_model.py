# sobol_sensitivity/sobol_sensitivity_movement_transition_many_params.py
"""
Sobol sensitivity for the movement Phase1→Phase2 transition model with MANY parameters.

Key features
------------
- Automatically discovers *numeric* parameters under `movement_v2` for **phase1**, **phase2**,
  and any **transition/switch** subtrees (e.g., `transition`, `phase_transition`, `phase_switch`).
- Builds *uniform* priors around each discovered default value with sensible heuristics per-name.
- Runs adaptive-replicate Sobol (SciPy ≥ 1.13 `sobol_indices`) with multiprocessing (spawn).
- Saves full arrays (NPZ), CSVs, and a quick multi‑panel plot per output.

Usage
-----
Run from the project root (parent of `sobol_sensitivity/`):

    python -m sobol_sensitivity.sobol_sensitivity_movement_transition_many_params

Or import `run_sobol(...)` and call programmatically.

Notes
-----
- By default `force_phase2=False` (we *want* Phase1→Phase2 transitions to occur).
- The prior ranges are heuristics; review the saved `auto_parameter_ranges.csv` in the output folder
  and adjust the logic below if needed.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
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

# ---- ABM imports ----
from abm_sobol.clusters_model import ClustersModel
from abm_sobol.utils import DEFAULTS as ABM_DEFAULTS

# ============================================================
# Config
# ============================================================
@dataclass
class SAConfig:
    steps: int = 145
    force_phase2: bool = False   # <<< transitions need this off by default
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

# Outputs (5)
OUTPUT_LABELS = [
    "Number of clusters",
    "Mean cluster size",
    "Variance of cluster size",
    "Median NN distance",
    "Clark-Evans R",
]

DEBUG = False

# ============================================================
# Helpers — parameters discovery and priors
# ============================================================

def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# param-path helpers
PathStr = str


def _iter_numeric_leaves(d: Dict[str, Any], path_prefix: List[str]) -> List[Tuple[PathStr, float]]:
    out: List[Tuple[PathStr, float]] = []
    for k, v in d.items():
        newp = path_prefix + [k]
        if isinstance(v, dict):
            out.extend(_iter_numeric_leaves(v, newp))
        else:
            if isinstance(v, (int, float)) and np.isfinite(v):
                out.append((".".join(newp), float(v)))
    return out


def _get_subdict(root: Dict[str, Any], path: List[str]) -> Optional[Dict[str, Any]]:
    cur: Any = root
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur if isinstance(cur, dict) else None


TRANSITION_KEYS = {"transition", "phase_transition", "phase_switch", "switch", "switching"}


def _movement_numeric_params(defaults: Dict[str, Any]) -> List[Tuple[PathStr, float]]:
    """Collect numeric leaf params under movement_v2.phase1, movement_v2.phase2,
    and any transition-related subtree names.
    """
    mv2 = _get_subdict(defaults, ["movement_v2"]) or {}
    params: List[Tuple[PathStr, float]] = []

    # phase1
    p1 = mv2.get("phase1")
    if isinstance(p1, dict):
        params.extend(_iter_numeric_leaves(p1, ["movement_v2", "phase1"]))

    # phase2
    p2 = mv2.get("phase2")
    if isinstance(p2, dict):
        params.extend(_iter_numeric_leaves(p2, ["movement_v2", "phase2"]))

    # Any transition-like subtree directly under movement_v2
    for k, v in mv2.items():
        if isinstance(v, dict) and any(t in k.lower() for t in TRANSITION_KEYS):
            params.extend(_iter_numeric_leaves(v, ["movement_v2", k]))

    # Deduplicate in stable order (just in case)
    seen = set()
    uniq: List[Tuple[PathStr, float]] = []
    for path, val in params:
        if path not in seen:
            uniq.append((path, val))
            seen.add(path)
    return uniq


# Heuristic prior builder ------------------------------------------------------

def _uniform_from_default(path: str, default: float) -> Tuple[float, float]:
    """Return (loc, scale) for scipy.stats.uniform given a default value.
    Heuristics by common parameter names.
    """
    p = path.lower()
    # Safety for nan/inf
    if not np.isfinite(default):
        default = 1.0

    def clamp_pos(a: float, b: float) -> Tuple[float, float]:
        lo = max(0.0, a)
        hi = max(lo + 1e-8, b)
        return lo, hi - lo

    # Named heuristics
    if "kappa" in p:
        # Turning concentration, often small-ish; if default is 0 use [0, 1]
        if default <= 0:
            return 0.0, 1.0
        lo, hi = 0.0, max(1.5 * default, 0.5)
        return clamp_pos(lo, hi)

    if p.endswith(".params.a") or "shape" in p:
        # Gamma shape; restrict to positive
        base = max(default, 0.5)
        lo, hi = 0.5 * base, 2.5 * base
        return clamp_pos(lo, hi)

    if p.endswith(".params.scale") or "scale" in p:
        base = max(default, 0.1)
        lo, hi = 0.5 * base, 2.5 * base
        return clamp_pos(lo, hi)

    if any(tok in p for tok in ("rate", "hazard", "lambda")):
        base = max(default, 1e-3)
        lo, hi = 0.25 * base, 4.0 * base
        return clamp_pos(lo, hi)

    if any(tok in p for tok in ("prob", "p_", "p.", "probability")):
        # Probability in [0,1]
        lo, hi = 0.0, 1.0
        return clamp_pos(lo, hi)

    # Generic positive ranges around default
    if default > 0:
        lo, hi = 0.5 * default, 2.0 * default
        return clamp_pos(lo, hi)
    else:
        # Unknown, non-positive default → [0, 1]
        return 0.0, 1.0


# Build parameter list and distributions --------------------------------------

def build_parameters_and_dists(defaults: Dict[str, Any]) -> Tuple[List[str], List[Any], List[Tuple[float, float, float]]]:
    """Return (PARAM_NAMES, DISTS, RANGES_INFO) where RANGES_INFO has (default, low, high)."""
    discovered = _movement_numeric_params(defaults)
    if not discovered:
        raise RuntimeError("No movement parameters found under movement_v2.* for phase1/phase2/transition.")

    names: List[str] = []
    dists: List[Any] = []
    ranges: List[Tuple[float, float, float]] = []  # default, low, high

    for path, default in discovered:
        loc, scale = _uniform_from_default(path, default)
        names.append(path)
        dists.append(uniform(loc=loc, scale=scale))
        ranges.append((default, loc, loc + scale))
    return names, dists, ranges

# ============================================================
# Metrics helpers
# ============================================================

def _median_nn_distance(xy: np.ndarray) -> float:
    xy = np.asarray(xy, float)
    if xy.shape[0] < 2:
        return 0.0
    tree = cKDTree(xy)
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
# ABM glue
# ============================================================

def _set_by_path(d: Dict[str, Any], path: str, value: float):
    keys = path.split('.')
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = float(value)


def _run_once(param_paths: List[str], x_col: np.ndarray, steps: int, seed: int, force_phase2: bool):
    params = deepcopy(ABM_DEFAULTS)
    for i, pth in enumerate(param_paths):
        _set_by_path(params, pth, float(x_col[i]))

    model = ClustersModel(params=params, seed=seed)
    try:
        model.enable_logging = False
    except Exception:
        pass

    if force_phase2:
        for a in list(model.agent_set):
            try:
                a.movement_phase = 2
                a.phase_switch_time = float('inf')
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
    j, param_paths, x_col, steps, seed, force_phase2 = args
    try:
        return j, _run_once(param_paths, x_col, steps, seed, force_phase2)
    except Exception:
        import traceback; traceback.print_exc()
        raise

_EVAL_COUNTER = 0

# ============================================================
# Adaptive vectorised function
# ============================================================

def make_abm_vectorised_func_adaptive(param_paths: List[str], cfg: SAConfig, pool, outer_pbar, chunksize, adapt):
    s = len(OUTPUT_LABELS)
    d_expected = len(param_paths)

    state = {"replicate_counts": None,
             "total_runs": 0,
             "rsem_at_stop": None}

    def f(x: np.ndarray) -> np.ndarray:
        if DEBUG:
            print(f"[debug] f() shape {x.shape}, expecting d={d_expected}")
        assert x.shape[0] == d_expected, f"Expected ({d_expected}, n) but got {x.shape}"
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
                    tasks.append((j, param_paths, xj, cfg.steps, sd, cfg.force_phase2))
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

        # Adaptive phase
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

def _save_results(out_dir: Path, res, boot, cfg: SAConfig, n_power2: int, adapt: AdaptiveR,
                  param_names: List[str], ranges_info: List[Tuple[float, float, float]],
                  rep_counts, total_runs, rsem_at_stop):
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "config": asdict(cfg),
        "n_power_of_two": n_power2,
        "parameters": param_names,
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

    # Save Sobol arrays (same field names as before)
    np.savez(out_dir / "sobol_arrays.npz",
             first_order=res.first_order,
             total_order=res.total_order,
             S1_CI_low=boot.first_order.confidence_interval.low,
             S1_CI_high=boot.first_order.confidence_interval.high,
             ST_CI_low=boot.total_order.confidence_interval.low,
             ST_CI_high=boot.total_order.confidence_interval.high,
             replicate_counts=np.array([]) if rep_counts is None else rep_counts,
             rsem_at_stop=np.array([]) if rsem_at_stop is None else rsem_at_stop)

    # Save human-readable CSVs
    def write_matrix(name, arr):
        hdr = ["output"] + param_names
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

    # Save the auto-derived ranges for later review/tweaks
    with open(out_dir / "auto_parameter_ranges.csv", "w") as f:
        f.write("parameter,default,low,high\n")
        for name, (default, low, high) in zip(param_names, ranges_info):
            f.write(f"{name},{default:.8g},{low:.8g},{high:.8g}\n")


def _save_plot_all_outputs(out_dir: Path, res, boot, param_names: List[str]):
    d = len(param_names)
    x = np.arange(d)

    # Short labels (last 1–2 path segments)
    def short_label(p: str) -> str:
        parts = p.split('.')
        if len(parts) <= 2:
            return p
        last2 = ".".join(parts[-2:])
        return last2 if len(last2) <= 18 else parts[-1]

    short = [short_label(p) for p in param_names]

    s = len(OUTPUT_LABELS)
    ncols = 2
    nrows = int(np.ceil(s / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(12, 0.6*d), 3.8*nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for i in range(s):
        ax = axes[i]
        s1 = res.first_order[i]
        st = res.total_order[i]
        s1_lo = boot.first_order.confidence_interval.low[i]
        s1_hi = boot.first_order.confidence_interval.high[i]
        st_lo = boot.total_order.confidence_interval.low[i]
        st_hi = boot.total_order.confidence_interval.high[i]
        ax.errorbar(x - 0.12, s1, yerr=[s1 - s1_lo, s1_hi - s1], fmt='o', capsize=2, label='S1')
        ax.errorbar(x + 0.12, st, yerr=[st - st_lo, st_hi - st], fmt='o', capsize=2, label='ST')
        ax.set_xticks(x, short, rotation=25, ha='right')
        ax.set_ylabel('Sobol index')
        ax.set_title(OUTPUT_LABELS[i])
        ax.set_ylim(0, max(1.0, float(np.nanmax([s1, st]))*1.2))
        ax.grid(True, alpha=0.25)
        ax.legend()

    for j in range(s, len(axes)):
        axes[j].axis('off')

    fig.savefig(out_dir / 'sobol_all_outputs.png', dpi=200)
    plt.close(fig)

# ============================================================
# Main
# ============================================================

def _warmup(pool, param_paths: List[str], cfg: SAConfig):
    """One quick replicate through the pool with small steps to flush import/pickling issues."""
    import time
    names, d, steps = param_paths, len(param_paths), max(10, min(30, cfg.steps))

    # Build a mid-point vector for 1 column using the inferred priors
    # Rebuild dists to match the warm-up context
    names2, dists2, ranges = build_parameters_and_dists(ABM_DEFAULTS)
    assert names2 == names, "Parameter list changed between warm-up and run; abort."

    x = np.zeros((d, 1), float)
    for i, dist in enumerate(dists2):
        loc = getattr(dist, 'kwds', {}).get('loc', 0.0)
        sca = getattr(dist, 'kwds', {}).get('scale', 1.0)
        x[i, 0] = float(loc + 0.5 * sca)

    task = (0, param_paths, x[:, 0].copy(), steps,
            int(np.random.SeedSequence(cfg.base_seed).generate_state(1)[0]),
            cfg.force_phase2)
    async_res = pool.apply_async(_abm_replicate_worker, (task,))
    try:
        _ = async_res.get(timeout=180)
    except mp.context.TimeoutError:
        raise RuntimeError(
            "Warm-up timed out. Ensure the project is importable under macOS 'spawn'. "
            "Run as 'python -m sobol_sensitivity.sobol_sensitivity_movement_transition_many_params' "
            "from the project root, or install your package with 'pip install -e .'"
        )


def run_sobol(
    n_power_of_two: int = 64,  # large d → start smaller by default
    cfg: SAConfig = SAConfig(),
    ci_resamples: int = 300,
    confidence: float = 0.95,
    output_dir: str | Path = 'sobol_results',
    n_workers: Optional[int] = None,
    chunksize: int = 16,
    adaptive: AdaptiveR = AdaptiveR(),
):
    # Discover parameters & priors from defaults
    PARAM_NAMES, DISTS, RANGES = build_parameters_and_dists(ABM_DEFAULTS)
    d = len(PARAM_NAMES)
    s = len(OUTPUT_LABELS)

    out_dir = Path(output_dir) / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    outer_pbar = tqdm(total=0, unit='run', desc='Sobol ABM runs', mininterval=0.5, dynamic_ncols=True) if tqdm else None

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2))
    print(f"[info] Using {n_workers} worker processes")
    if adaptive.enable:
        print(f"[info] Adaptive replicates: R_min={adaptive.R_min}, R_step={adaptive.R_step}, "
              f"R_max={adaptive.R_max}, rel_sem_target={adaptive.rel_sem_target:.2%}")
    print(f"[info] Discovered movement params: d = {d}")

    # Save the discovered parameter list + ranges eagerly for review
    with open(out_dir / 'auto_parameter_ranges.csv', 'w') as f:
        f.write('parameter,default,low,high\n')
        for name, (default, low, high) in zip(PARAM_NAMES, RANGES):
            f.write(f"{name},{default:.8g},{low:.8g},{high:.8g}\n")

    
    # Pretty-print the parameter list and ranges before starting Sobol
    print("[info] Parameter ranges (before Sobol):")
    max_name = max(len(n) for n in PARAM_NAMES)
    header = f"  {'parameter'.ljust(max_name)}    default        low            high"
    print(header)
    print("  " + "-" * (len(header)-2))
    for name, (default, low, high) in zip(PARAM_NAMES, RANGES):
        print(f"  {name.ljust(max_name)}    {default:>10.6g}    {low:>12.6g}    {high:>12.6g}")
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        # Warm-up one quick task to surface import issues early
        _warmup(pool, PARAM_NAMES, cfg)

        f_vec, state = make_abm_vectorised_func_adaptive(PARAM_NAMES, cfg, pool, outer_pbar, chunksize, adaptive)

        res = sobol_indices(func=f_vec, n=n_power_of_two, dists=DISTS)
        boot = res.bootstrap(confidence_level=confidence, n_resamples=ci_resamples)

    if outer_pbar:
        outer_pbar.close()

    rep_counts = state["replicate_counts"]
    total_runs = state["total_runs"]
    rsem_at_stop = state["rsem_at_stop"]

    _save_results(out_dir, res, boot, cfg, n_power_of_two, adaptive, PARAM_NAMES, RANGES,
                  rep_counts, total_runs, rsem_at_stop)
    _save_plot_all_outputs(out_dir, res, boot, PARAM_NAMES)

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
if __name__ == '__main__':
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    cfg = SAConfig(steps=145, force_phase2=False)
    adaptive = AdaptiveR(enable=True, R_min=6, R_step=2, R_max=10, rel_sem_target=0.05)

    run_sobol(
        n_power_of_two=16,     # start smaller when d is large
        cfg=cfg,
        ci_resamples=300,
        output_dir='sobol_results',
        n_workers=None,
        chunksize=16,
        adaptive=adaptive,
    )
