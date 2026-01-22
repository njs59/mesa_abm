
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Systematic **grid scan** over parameter combinations with **equal-time relative RMSE**
for all four outputs (S0,S1,S2,NND_med). Saves:
- `grid_all_results.csv`: every combination with rRMSE per stat and SCORE (mean of 4 rRMSEs)
- For each parameter, two plots:
  * average SCORE by parameter value
  * best (min) SCORE by parameter value
  saved as `param_{name}_avg.png` and `param_{name}_best.png`

Conventions:
- **Lower SCORE is better** (0.0 would be a perfect fit).
- `--replicates` controls how many ABM runs you perform per parameter set; the
  median over replicates is used for scoring.

Parallel options:
- `--workers N` > 1 enables process-based parallelism.
- `--parallel_scope combos|replicates` chooses the level to parallelise:
  - `combos` (default): parameter combinations in parallel (recommended).
  - `replicates`: combinations sequential, but replicate runs in parallel.

Progress bars:
- Uses `tqdm` if available; otherwise prints a simple single-line progress indicator.
"""

from __future__ import annotations
import argparse, json, itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Progress bar support (optional)
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # Fallback to simple textual progress if tqdm is unavailable

# ---- ABM imports ------------------------------------------------------------
try:
    from abm.clusters_model import ClustersModel  # package style
except Exception:
    try:
        from clusters_model import ClustersModel  # local file style
    except Exception:
        raise ImportError("Could not import ClustersModel. Ensure clusters_model.py is importable and mesa is installed.")

# ---- Stats ------------------------------------------------------------------
def _plain_pairwise_dists(pos: np.ndarray) -> np.ndarray:
    if pos.size == 0:
        return np.empty((0, 0), dtype=float)
    diffs = pos[:, None, :] - pos[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2))

def compute_snapshot_summaries(model) -> Dict[str, float]:
    ids, pos, radii, sizes, speeds = model._log_alive_snapshot()  # noqa: SLF001
    n = len(sizes)
    if n == 0:
        return {"S0": 0.0, "S1": 0.0, "S2": 0.0, "NND_med": 0.0}
    S0 = float(n)
    S1 = float(np.mean(sizes))
    S2 = float(np.mean(np.square(sizes)))
    D = _plain_pairwise_dists(pos)
    if D.size == 0:
        nnd_med = 0.0
    else:
        D_no_self = D + np.eye(n) * 1e9
        nnd = D_no_self.min(axis=1)
        nnd_med = float(np.median(nnd))
    return {"S0": S0, "S1": S1, "S2": S2, "NND_med": nnd_med}

def simulate_timeseries(params: dict, total_steps: int, sample_steps: Tuple[int, ...], seed: int = 777) -> np.ndarray:
    m = ClustersModel(params=params, seed=seed)
    K = 4  # S0,S1,S2,NND_med
    T = len(sample_steps)
    out = np.zeros((T, K), dtype=float)
    current = 0
    steps_set = set(sample_steps)
    for step in range(total_steps + 1):
        if step in steps_set:
            s = compute_snapshot_summaries(m)
            out[current, :] = [s["S0"], s["S1"], s["S2"], s["NND_med"]]
            current += 1
            if current >= T:
                break
        if step < total_steps:
            m.step()
    return out

# ---- Parameter registry -----------------------------------------------------
@dataclass
class PInfo:
    path: Tuple[str, ...]
    lo: float
    hi: float
    scale: str = "linear"  # "linear" or "log"
    integer: bool = False

REGISTRY: Dict[str, PInfo] = {
    "merge_prob":      PInfo(("merge","prob_contact_merge"), 0.60, 1.00, "linear", False),
    "adhesion":        PInfo(("phenotypes","proliferative","adhesion"), 0.55, 1.00, "linear", False),
    "fragment_rate":   PInfo(("phenotypes","proliferative","fragment_rate"), 2e-4, 1.5e-2, "log", False),
    "heading_sigma":   PInfo(("movement","heading_sigma"), 0.10, 0.90, "linear", False),
    "dist_s":          PInfo(("movement","dist_params","s"), 0.40, 0.95, "linear", False),
    "dist_scale":      PInfo(("movement","dist_params","scale"), 1.50, 3.80, "linear", False),
    "prolif_rate":     PInfo(("phenotypes","proliferative","prolif_rate"), 0.002, 0.012, "linear", False),
    "init_n_clusters": PInfo(("init","n_clusters"), 400, 1400, "linear", True),
}

# ---- Helpers ----------------------------------------------------------------
def deep_set(d: dict, path: Tuple[str, ...], value):
    x = d
    for k in path[:-1]:
        v = x.get(k, None)
        if not isinstance(v, dict):
            v = {}
        x[k] = v
        x = v
    x[path[-1]] = value

def load_defaults(direction: str, speed: str) -> dict:
    try:
        from abm.utils import DEFAULTS as D
    except Exception:
        try:
            from utils import DEFAULTS as D
        except Exception:
            raise ImportError("Could not import utils. Place utils.py on PYTHONPATH.")
    base = json.loads(json.dumps(D))
    base.setdefault("movement", {}).setdefault("dist_params", {})
    base["movement"]["mode"] = "distribution"
    base["movement"]["direction"] = direction
    base["movement"]["distribution"] = speed
    base.setdefault("init", {}).setdefault("phenotype", "proliferative")
    return base

def params_from_vector(vec: dict, direction: str, speed: str) -> dict:
    p = load_defaults(direction, speed)
    for name, val in vec.items():
        if name not in REGISTRY:
            continue
        path = REGISTRY[name].path
        if REGISTRY[name].integer:
            val = int(round(val))
        deep_set(p, path, float(val) if not REGISTRY[name].integer else int(val))
    return p

# ---- Scoring (equal-time relative RMSE) -------------------------------------
def rel_rmse_equal_time(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(1e-9, np.abs(y_true))
    re = (y_pred - y_true) / denom
    return float(np.sqrt(np.mean(np.square(re))))

# ---- Parallel workers (top-level) -------------------------------------------
def _replicate_worker(payload: dict) -> np.ndarray:
    """
    Run a single ABM replicate for a parameter vector.
    payload keys: vec, direction, speed, total_steps, timesteps, seed
    """
    vec = payload["vec"]
    direction = payload["direction"]
    speed = payload["speed"]
    total_steps = payload["total_steps"]
    timesteps = tuple(payload["timesteps"])
    seed = payload["seed"]
    params = params_from_vector(vec, direction, speed)
    sim = simulate_timeseries(params, total_steps=total_steps, sample_steps=timesteps, seed=seed)
    return sim

def _evaluate_combo_worker(payload: dict) -> dict:
    """
    Evaluate one parameter combination (sequential replicates), used when parallelising combos.
    payload keys: vec, base_seed, direction, speed, total_steps, timesteps, replicates,
                  observed_ts, stats(list[str])
    Returns: row dict with vec entries + r_s0, r_s1, r_s2, r_nnd, score
    """
    vec = payload["vec"]
    base_seed = payload["base_seed"]
    direction = payload["direction"]
    speed = payload["speed"]
    total_steps = payload["total_steps"]
    timesteps = tuple(payload["timesteps"])
    replicates = int(payload["replicates"])
    observed_ts = payload["observed_ts"]
    stats = payload["stats"]

    # Load observed locally in the worker
    obs_df = pd.read_csv(observed_ts)
    obs_mat = obs_df[stats].to_numpy(float)
    stats_idx = {s: i for i, s in enumerate(stats)}

    sims = []
    for r in range(replicates):
        seed = base_seed + r
        params = params_from_vector(vec, direction, speed)
        sim = simulate_timeseries(params, total_steps=total_steps, sample_steps=timesteps, seed=seed)
        sims.append(sim)

    sims = np.asarray(sims)
    med = np.median(sims, axis=0)
    r_s0 = rel_rmse_equal_time(obs_mat[:, stats_idx['S0']],      med[:, stats_idx['S0']])
    r_s1 = rel_rmse_equal_time(obs_mat[:, stats_idx['S1']],      med[:, stats_idx['S1']])
    r_s2 = rel_rmse_equal_time(obs_mat[:, stats_idx['S2']],      med[:, stats_idx['S2']])
    r_nnd= rel_rmse_equal_time(obs_mat[:, stats_idx['NND_med']], med[:, stats_idx['NND_med']])
    score = float(np.mean([r_s0, r_s1, r_s2, r_nnd]))
    return {**vec, 'r_s0': r_s0, 'r_s1': r_s1, 'r_s2': r_s2, 'r_nnd': r_nnd, 'score': score}

# ---- CLI & main -------------------------------------------------------------
def parse_cli():
    ap = argparse.ArgumentParser(description="Grid scan with equal-time relative RMSE over all 4 outputs")
    ap.add_argument('--observed_ts', type=str, required=True)
    ap.add_argument('--direction', type=str, choices=['isotropic','persistent'], default='persistent')
    ap.add_argument('--speed', type=str, choices=['lognorm','gamma','weibull','expon','rayleigh','invgauss','constant'], default='lognorm')
    ap.add_argument('--total_steps', type=int, default=300)
    ap.add_argument('--params', nargs='+', default=['merge_prob','adhesion','fragment_rate','heading_sigma','dist_s','dist_scale','init_n_clusters'])
    # Repeated grids: NAME START STOP STEP
    ap.add_argument('--grid', nargs=4, action='append', metavar=('NAME','START','STOP','STEP'), default=None,
                    help='Define a grid for a parameter (inclusive). Repeat per parameter.')

    # Replicates per parameter set
    ap.add_argument('--replicates', type=int, default=3, help='Number of independent ABM runs per parameter set (median is used).')

    # Parallel options
    ap.add_argument('--workers', type=int, default=0, help='Number of worker processes (0 or 1 = no parallel).')
    ap.add_argument('--parallel_scope', type=str, choices=['combos','replicates'], default='combos',
                    help="What to parallelise: 'combos' (default) or 'replicates'.")

    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--outdir', type=str, default='results/grid_scan_relrmse')
    return ap.parse_args()

def main():
    args = parse_cli()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    obs_df = pd.read_csv(args.observed_ts)
    needed = ['timestep','S0','S1','S2','NND_med']
    for c in needed:
        if c not in obs_df.columns:
            raise ValueError(f'Missing column {c} in observed_ts')
    timesteps = obs_df['timestep'].astype(int).to_list()
    stats = ['S0','S1','S2','NND_med']
    stats_idx = {s:i for i,s in enumerate(stats)}
    obs_mat = obs_df[stats].to_numpy(float)

    # Build grids
    grid_map: Dict[str, List[float]] = {}

    def _grid_vals(name, lo_f, hi_f, st_f, integer, eps=1e-12):
        if integer:
            lo_i, hi_i, st_i = int(round(lo_f)), int(round(hi_f)), max(1, int(round(st_f)))
            return list(range(lo_i, hi_i + 1, st_i))
        # float grids
        if abs(hi_f - lo_f) <= eps:
            return [float(lo_f)]
        n = max(1, int(round((hi_f - lo_f) / st_f)))  # robust against floating error
        vals = [lo_f + i * st_f for i in range(n + 1)]
        vals[0] = lo_f
        vals[-1] = hi_f
        return vals

    if args.grid:
        for name, lo, hi, step in args.grid:
            if name not in REGISTRY:
                raise KeyError(f'Unknown parameter for --grid: {name}')
            pi = REGISTRY[name]
            lo_f, hi_f, st_f = float(lo), float(hi), float(step)
            grid_map[name] = _grid_vals(name, lo_f, hi_f, st_f, integer=pi.integer)

    # For any chosen param without an explicit grid, use a single midpoint value
    chosen = []
    for n in args.params:
        if n not in REGISTRY:
            raise KeyError(f'Unknown parameter: {n}')
        if n == 'heading_sigma' and args.direction != 'persistent':
            continue
        chosen.append(n)
        if n not in grid_map:
            pi = REGISTRY[n]
            mid = (pi.lo + pi.hi)/2.0
            grid_map[n] = [int(round(mid))] if pi.integer else [mid]

    # Cartesian product
    names = list(chosen)
    grids = [grid_map[n] for n in names]
    combos = list(itertools.product(*grids))
    n_combos = len(combos)
    print(f"Grid combinations: {n_combos} over params: {', '.join(names)}")

    # Simple fallback progress if tqdm isn't available
    def _print_progress(i: int, total: int, prefix: str = "Progress"):
        width = 30
        done = int((i / total) * width)
        bar = "#" * done + "-" * (width - done)
        print(f"\r{prefix} [{bar}] {i}/{total}", end="", flush=True)
        if i >= total:
            print("")  # newline at completion

    # ---------- Evaluate vector (sequential or replicate-parallel) ----------
    def evaluate_vector(vec: dict, base_seed: int, parallel_reps: bool, rep_workers: int):
        # Sequential replicates (default)
        if not parallel_reps or rep_workers <= 1:
            sims = []
            # Optional nested progress for replicates (only if tqdm is available and not using parallel)
            if tqdm is not None:
                rep_iter = tqdm(range(args.replicates), desc="Replicates", unit="rep", leave=False)
            else:
                rep_iter = range(args.replicates)

            for r in rep_iter:
                seed = base_seed + r
                params = params_from_vector(vec, args.direction, args.speed)
                sim = simulate_timeseries(params, total_steps=args.total_steps, sample_steps=tuple(timesteps), seed=seed)
                sims.append(sim)
        else:
            # Parallelise replicates in a local process pool
            payloads = [{
                "vec": vec,
                "direction": args.direction,
                "speed": args.speed,
                "total_steps": args.total_steps,
                "timesteps": timesteps,
                "seed": base_seed + r,
            } for r in range(args.replicates)]
            sims = []
            if tqdm is not None:
                with ProcessPoolExecutor(max_workers=rep_workers) as ex, tqdm(total=args.replicates, desc="Replicates", unit="rep", leave=False) as t:
                    futures = [ex.submit(_replicate_worker, p) for p in payloads]
                    for fut in as_completed(futures):
                        sims.append(fut.result())
                        t.update(1)
            else:
                with ProcessPoolExecutor(max_workers=rep_workers) as ex:
                    futures = [ex.submit(_replicate_worker, p) for p in payloads]
                    for fut in as_completed(futures):
                        sims.append(fut.result())

        sims = np.asarray(sims)
        med = np.median(sims, axis=0)
        r_s0 = rel_rmse_equal_time(obs_mat[:, stats_idx['S0']],      med[:, stats_idx['S0']])
        r_s1 = rel_rmse_equal_time(obs_mat[:, stats_idx['S1']],      med[:, stats_idx['S1']])
        r_s2 = rel_rmse_equal_time(obs_mat[:, stats_idx['S2']],      med[:, stats_idx['S2']])
        r_nnd= rel_rmse_equal_time(obs_mat[:, stats_idx['NND_med']], med[:, stats_idx['NND_med']])
        score = float(np.mean([r_s0, r_s1, r_s2, r_nnd]))
        return r_s0, r_s1, r_s2, r_nnd, score

    rows: List[dict] = []
    base_seed0 = args.seed * 100000

    # ---------- Parallel policy ----------
    workers = int(args.workers)
    parallel_combos = (args.parallel_scope == 'combos' and workers > 1)
    parallel_reps   = (args.parallel_scope == 'replicates' and workers > 1)

    if parallel_combos and parallel_reps:
        print("[warning] Both levels requested for parallelism; defaulting to 'combos' only.")
        parallel_reps = False

    # ---------- Run scan ----------
    if parallel_combos:
        # Parallelise over combinations (replicates handled sequentially inside worker)
        if tqdm is not None:
            pbar = tqdm(total=n_combos, desc="Parameter combos", unit="combo")
        else:
            pbar = None

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = []
            for i, combo in enumerate(combos):
                vec = {name: float(val) for name, val in zip(names, combo)}
                # cast ints
                for n in names:
                    if REGISTRY[n].integer:
                        vec[n] = int(round(vec[n]))
                payload = {
                    "vec": vec,
                    "base_seed": base_seed0 + i,
                    "direction": args.direction,
                    "speed": args.speed,
                    "total_steps": args.total_steps,
                    "timesteps": timesteps,
                    "replicates": args.replicates,
                    "observed_ts": args.observed_ts,
                    "stats": ['S0','S1','S2','NND_med'],
                }
                futures.append(ex.submit(_evaluate_combo_worker, payload))

            for fut in as_completed(futures):
                rows.append(fut.result())
                if pbar is not None:
                    pbar.update(1)
                else:
                    _print_progress(len(rows), n_combos, prefix="Parameter combos")
        if pbar is not None:
            pbar.close()

    else:
        # Sequential combinations (optionally parallelise replicates)
        if tqdm is not None:
            combo_iter = tqdm(enumerate(combos), total=n_combos, desc="Parameter combos", unit="combo")
        else:
            combo_iter = enumerate(combos)

        for i, combo in combo_iter:
            vec = {name: float(val) for name, val in zip(names, combo)}
            # cast ints
            for n in names:
                if REGISTRY[n].integer:
                    vec[n] = int(round(vec[n]))

            r_s0, r_s1, r_s2, r_nnd, score = evaluate_vector(
                vec, base_seed0 + i,
                parallel_reps=parallel_reps,
                rep_workers=workers if parallel_reps else 1
            )
            rows.append({**vec, 'r_s0': r_s0, 'r_s1': r_s1, 'r_s2': r_s2, 'r_nnd': r_nnd, 'score': score})

            if tqdm is None:
                _print_progress(i + 1, n_combos, prefix="Parameter combos")

    # ---------- Save results ----------
    df = pd.DataFrame(rows)
    df.to_csv(outdir / 'grid_all_results.csv', index=False)

    # ---------- Aggregations and plots ----------
    for n in names:
        if len(set(df[n].tolist())) == 1:
            # Only a single value; skip plotting
            continue
        grp = df.groupby(n)['score']
        avg = grp.mean()
        best = grp.min()
        # Ensure sorted by parameter value
        x = np.array(sorted(avg.index))
        avg_y = np.array([avg.loc[v] for v in x])
        best_y = np.array([best.loc[v] for v in x])

        plt.figure(figsize=(10,4))
        plt.plot(x, avg_y, marker='o', lw=1.5)
        plt.xlabel(n); plt.ylabel('Average SCORE')
        plt.title(f'Average SCORE vs {n}')
        plt.tight_layout(); plt.savefig(outdir / f'param_{n}_avg.png', dpi=160); plt.close()

        plt.figure(figsize=(10,4))
        plt.plot(x, best_y, marker='o', lw=1.5, color='tab:green')
        plt.xlabel(n); plt.ylabel('Best (min) SCORE')
        plt.title(f'Best SCORE vs {n}')
        plt.tight_layout(); plt.savefig(outdir / f'param_{n}_best.png', dpi=160); plt.close()

    print("All done. Results in:", outdir.resolve())

if __name__ == '__main__':
    main()
