#!/usr/bin/env python3
"""
Run 4 ABM movement conditions and export trajectories + summary statistics.
Parallel-enabled (process-based).

Folder layout expected:
  project_root/
    abm/
      clusters_model.py
      utils.py
      cluster_agent.py
    scripts/
      run_movement_conditions.py   <-- this script
    results/                       <-- created if missing

Outputs:
  results/trajectories/{condition}/repeat_###.csv
  results/summary_repeats.csv
  results/summary_conditions.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Condition definition
# -----------------------------
@dataclass(frozen=True)
class Condition:
    name: str
    phenotype: str
    phase2_only: bool  # if True: force all agents into movement_phase=2 immediately


def project_root() -> Path:
    # scripts/ is alongside abm/; root is parent of scripts/
    return Path(__file__).resolve().parents[1]


# -----------------------------
# Geometry / distances
# -----------------------------
def pairwise_min_image_deltas(x: np.ndarray, y: np.ndarray, W: float, H: float, torus: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return dx_ij, dy_ij matrices using minimum-image convention if torus=True.
    dx_ij = x_i - x_j (wrapped to [-W/2, W/2]) if torus, else plain difference.
    """
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]

    if torus:
        dx = (dx + W / 2.0) % W - W / 2.0
        dy = (dy + H / 2.0) % H - H / 2.0

    return dx, dy


def nearest_neighbour_distances(x: np.ndarray, y: np.ndarray, W: float, H: float, torus: bool) -> np.ndarray:
    """Compute nearest-neighbour distance for each point (O(N^2))."""
    n = len(x)
    if n < 2:
        return np.array([], dtype=float)

    dx, dy = pairwise_min_image_deltas(x, y, W, H, torus)
    d2 = dx * dx + dy * dy
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))
    return nn


# -----------------------------
# Clark–Evans stats (Normal approximation)
# -----------------------------
def clark_evans_stats(x: np.ndarray, y: np.ndarray, area: float, W: float, H: float, torus: bool) -> Dict[str, float]:
    """
    Compute Clark–Evans R, z, p (two-sided).
      rE = 1 / (2 * sqrt(lambda)), lambda = n/area
      SE(rO) ≈ 0.26136 / sqrt(n*lambda)
      z = (rO - rE)/SE
      p(two-sided) = erfc(|z|/sqrt(2))
    """
    n = len(x)
    if n < 2 or area <= 0:
        return {"r_obs": np.nan, "r_exp": np.nan, "R": np.nan, "z": np.nan, "p_two_sided": np.nan}

    nn = nearest_neighbour_distances(x, y, W=W, H=H, torus=torus)
    if len(nn) == 0:
        return {"r_obs": np.nan, "r_exp": np.nan, "R": np.nan, "z": np.nan, "p_two_sided": np.nan}

    r_obs = float(np.mean(nn))
    lam = n / area
    r_exp = float(1.0 / (2.0 * math.sqrt(lam)))

    se = float(0.26136 / math.sqrt(n * lam)) if (n * lam) > 0 else np.nan
    z = float((r_obs - r_exp) / se) if (se and np.isfinite(se) and se > 0) else np.nan
    p = float(math.erfc(abs(z) / math.sqrt(2.0))) if np.isfinite(z) else np.nan

    R = float(r_obs / r_exp) if (r_exp and np.isfinite(r_exp) and r_exp > 0) else np.nan
    return {"r_obs": r_obs, "r_exp": r_exp, "R": R, "z": z, "p_two_sided": p}


# -----------------------------
# Summary stats for a repeat
# -----------------------------
def summarise_repeat(model) -> Dict[str, float]:
    """Summary at final timepoint."""
    alive = [a for a in list(model.agent_set) if getattr(a, "alive", True) and getattr(a, "pos", None) is not None]
    n = len(alive)

    if n == 0:
        return {
            "n_clusters": 0,
            "mean_size": np.nan,
            "var_size": np.nan,
            "median_nnd": np.nan,
            "r_obs": np.nan,
            "r_exp": np.nan,
            "clark_evans_R": np.nan,
            "clark_evans_z": np.nan,
            "clark_evans_p": np.nan,
        }

    sizes = np.array([float(a.size) for a in alive], dtype=float)
    x = np.array([float(a.pos[0]) for a in alive], dtype=float)
    y = np.array([float(a.pos[1]) for a in alive], dtype=float)

    W = float(model.params["space"]["width"])
    H = float(model.params["space"]["height"])
    torus = bool(model.params["space"].get("torus", True))
    area = W * H

    nn = nearest_neighbour_distances(x, y, W=W, H=H, torus=torus)
    median_nnd = float(np.median(nn)) if len(nn) else np.nan

    ce = clark_evans_stats(x, y, area=area, W=W, H=H, torus=torus)

    return {
        "n_clusters": int(n),
        "mean_size": float(np.mean(sizes)),
        "var_size": float(np.var(sizes, ddof=1)) if n > 1 else 0.0,
        "median_nnd": median_nnd,
        "r_obs": ce["r_obs"],
        "r_exp": ce["r_exp"],
        "clark_evans_R": ce["R"],
        "clark_evans_z": ce["z"],
        "clark_evans_p": ce["p_two_sided"],
    }


# -----------------------------
# CI utilities
# -----------------------------
def t_critical_975(df: int) -> float:
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except Exception:
        return 1.96


def mean_ci_95(values: np.ndarray) -> Tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    m = float(np.mean(v))
    if n == 1:
        return (m, np.nan, np.nan)
    se = float(np.std(v, ddof=1) / math.sqrt(n))
    tcrit = t_critical_975(n - 1)
    lo = m - tcrit * se
    hi = m + tcrit * se
    return (m, lo, hi)


# -----------------------------
# Trajectory export
# -----------------------------
TRAJ_COLUMNS = [
    "condition",
    "repeat",
    "step",
    "time_min",
    "agent_id",
    "phenotype",
    "movement_phase",
    "x",
    "y",
    "size",
    "radius",
    "speed",
]


def write_snapshot_rows(writer: csv.DictWriter, model, condition: str, rep: int, step: int):
    dt = float(model.dt)
    time_min = step * dt

    for a in list(model.agent_set):
        if not getattr(a, "alive", True):
            continue
        p = getattr(a, "pos", None)
        if p is None:
            continue

        vx = getattr(a, "vel", np.array([np.nan, np.nan], dtype=float))
        speed = float(np.linalg.norm(vx)) if vx is not None else np.nan

        writer.writerow({
            "condition": condition,
            "repeat": rep,
            "step": step,
            "time_min": float(time_min),
            "agent_id": int(getattr(a, "unique_id", -1)),
            "phenotype": str(getattr(a, "phenotype", "")),
            "movement_phase": int(getattr(a, "movement_phase", -1)),
            "x": float(p[0]),
            "y": float(p[1]),
            "size": float(getattr(a, "size", np.nan)),
            "radius": float(getattr(a, "radius", np.nan)),
            "speed": speed,
        })


# -----------------------------
# Worker task (top-level for multiprocessing)
# -----------------------------
def run_one_repeat_task(
    cond_name: str,
    phenotype: str,
    phase2_only: bool,
    rep: int,
    steps: int,
    seed: int,
    out_csv_str: str,
    base_params: dict,
    n_init: int,
    init_size: int,
) -> Dict[str, float]:
    """
    This runs inside a separate process.
    IMPORTANT: imports of your ABM modules occur inside the worker to play nicely with 'spawn' on macOS.
    """
    # (optional) prevent oversubscription if numpy/scipy use BLAS threads
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    from abm.clusters_model import ClustersModel  # import inside worker

    params = deepcopy(base_params)

    params["init"] = {"n_clusters": int(n_init), "size": int(init_size), "phenotype": phenotype}
    params["time"]["steps"] = int(steps)

    model = ClustersModel(params=params, seed=seed)

    if phase2_only:
        for a in list(model.agent_set):
            if getattr(a, "alive", True):
                a.movement_phase = 2
                a.phase_switch_time = np.inf

    out_csv = Path(out_csv_str)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAJ_COLUMNS)
        writer.writeheader()

        write_snapshot_rows(writer, model, cond_name, rep, step=0)

        for s in range(1, steps + 1):
            model.step()
            write_snapshot_rows(writer, model, cond_name, rep, step=s)

    summ = summarise_repeat(model)
    summ.update({
        "condition": cond_name,
        "phenotype": phenotype,
        "phase2_only": bool(phase2_only),
        "repeat": int(rep),
        "seed": int(seed),
        "steps": int(steps),
    })
    return summ


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run ABM movement conditions and export results (parallel).")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeats per condition.")
    parser.add_argument("--steps", type=int, default=None, help="Number of model steps (overrides DEFAULTS['time']['steps']).")
    parser.add_argument("--seed0", type=int, default=123, help="Base seed; repeat i uses seed0+i plus condition offset.")
    parser.add_argument("--n_init", type=int, default=None, help="Initial number of clusters (overrides DEFAULTS['init']['n_clusters']).")
    parser.add_argument("--init_size", type=int, default=None, help="Initial size per cluster (overrides DEFAULTS['init']['size']).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel worker processes. Default: max(1, cpu_count-1).")
    parser.add_argument("--serial", action="store_true", help="Force serial execution (debugging).")
    args = parser.parse_args()

    root = project_root()
    results_dir = root / "results"
    traj_dir = results_dir / "trajectories"
    results_dir.mkdir(exist_ok=True)
    traj_dir.mkdir(exist_ok=True)

    # Import DEFAULTS here (main process only)
    from abm.utils import DEFAULTS
    base_params = deepcopy(DEFAULTS)

    steps = int(args.steps) if args.steps is not None else int(base_params["time"]["steps"])
    n_init = int(args.n_init) if args.n_init is not None else int(base_params.get("init", {}).get("n_clusters", 800))
    init_size = int(args.init_size) if args.init_size is not None else int(base_params.get("init", {}).get("size", 1))

    conditions: List[Condition] = [
        Condition(name="prolif_phase2_only", phenotype="proliferative", phase2_only=True),
        Condition(name="invasive_phase2_only", phenotype="invasive", phase2_only=True),
        Condition(name="prolif_two_phase", phenotype="proliferative", phase2_only=False),
        Condition(name="invasive_two_phase", phenotype="invasive", phase2_only=False),
    ]

    # Build task list
    tasks = []
    for c_idx, cond in enumerate(conditions):
        for r in range(args.repeats):
            seed = int(args.seed0 + 10000 * c_idx + r)
            out_csv = traj_dir / cond.name / f"repeat_{r:03d}.csv"
            tasks.append((c_idx, cond, r, seed, str(out_csv)))

    # Decide worker count
    if args.serial:
        workers = 1
    else:
        if args.workers is not None:
            workers = int(args.workers)
        else:
            workers = max(1, (os.cpu_count() or 2) - 1)

    print(f"Running {len(tasks)} simulations ({len(conditions)} conditions × {args.repeats} repeats)")
    print(f"Steps per run: {steps}; workers: {workers} ({'serial' if workers == 1 else 'parallel'})")

    all_repeat_rows: List[Dict[str, float]] = []

    if workers == 1:
        # Serial execution
        for c_idx, cond, r, seed, out_csv_str in tasks:
            print(f"[{cond.name}] repeat {r+1}/{args.repeats} -> {out_csv_str}")
            row = run_one_repeat_task(
                cond_name=cond.name,
                phenotype=cond.phenotype,
                phase2_only=cond.phase2_only,
                rep=r,
                steps=steps,
                seed=seed,
                out_csv_str=out_csv_str,
                base_params=base_params,
                n_init=n_init,
                init_size=init_size,
            )
            all_repeat_rows.append(row)

    else:
        # Parallel execution (processes)
        import concurrent.futures as cf

        # Submit jobs
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futures = []
            for c_idx, cond, r, seed, out_csv_str in tasks:
                futures.append(
                    ex.submit(
                        run_one_repeat_task,
                        cond.name,
                        cond.phenotype,
                        cond.phase2_only,
                        r,
                        steps,
                        seed,
                        out_csv_str,
                        base_params,
                        n_init,
                        init_size,
                    )
                )

            # Collect as they finish
            completed = 0
            total = len(futures)
            for fut in cf.as_completed(futures):
                completed += 1
                try:
                    row = fut.result()
                    all_repeat_rows.append(row)
                except Exception as e:
                    # Fail fast with context
                    raise RuntimeError(f"A worker failed (completed {completed}/{total}). Error: {e}") from e

                if completed % max(1, total // 20) == 0 or completed == total:
                    print(f"Completed {completed}/{total} runs...")

    # Save per-repeat summary table
    df_rep = pd.DataFrame(all_repeat_rows)
    df_rep = df_rep.sort_values(["condition", "repeat"]).reset_index(drop=True)

    rep_path = results_dir / "summary_repeats.csv"
    df_rep.to_csv(rep_path, index=False)
    print(f"Saved per-repeat summaries: {rep_path}")

    # Aggregate per condition: mean and 95% CI
    metrics = [
        "n_clusters",
        "mean_size",
        "var_size",
        "median_nnd",
        "clark_evans_R",
        "clark_evans_z",
        "clark_evans_p",
    ]

    agg_rows = []
    for cond_name, g in df_rep.groupby("condition"):
        out = {
            "condition": cond_name,
            "phenotype": str(g["phenotype"].iloc[0]),
            "phase2_only": bool(g["phase2_only"].iloc[0]),
            "n_repeats": int(len(g)),
        }
        for m in metrics:
            mean, lo, hi = mean_ci_95(g[m].to_numpy())
            out[f"{m}_mean"] = mean
            out[f"{m}_ci95_lo"] = lo
            out[f"{m}_ci95_hi"] = hi
        agg_rows.append(out)

    df_agg = pd.DataFrame(agg_rows).sort_values("condition").reset_index(drop=True)
    agg_path = results_dir / "summary_conditions.csv"
    df_agg.to_csv(agg_path, index=False)
    print(f"Saved condition aggregates: {agg_path}")


if __name__ == "__main__":
    main()