#!/usr/bin/env python3
"""
Run ABM movement conditions ONLY for the invasive phenotype.
Parallel-enabled (process-based).
"""

from __future__ import annotations
import argparse, csv, math, os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from math import erfc, sqrt

# -----------------------------
# Condition definition
# -----------------------------
@dataclass(frozen=True)
class Condition:
    name: str
    phenotype: str
    phase2_only: bool

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

# -----------------------------
# Geometry helpers (same as original)
# -----------------------------
def pairwise_min_image_deltas(x, y, W, H, torus):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    if torus:
        dx = (dx + W/2) % W - W/2
        dy = (dy + H/2) % H - H/2
    return dx, dy

def nearest_neighbour_distances(x, y, W, H, torus):
    n = len(x)
    if n < 2:
        return np.array([], dtype=float)
    dx, dy = pairwise_min_image_deltas(x, y, W, H, torus)
    d2 = dx*dx + dy*dy
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(np.min(d2, axis=1))

# -----------------------------
# Clark–Evans stats
# -----------------------------
def clark_evans_stats(x, y, area, W, H, torus):
    n = len(x)
    if n < 2 or area <= 0:
        return dict(r_obs=np.nan, r_exp=np.nan, R=np.nan, z=np.nan, p_two_sided=np.nan)
    nn = nearest_neighbour_distances(x, y, W, H, torus)
    if len(nn) == 0:
        return dict(r_obs=np.nan, r_exp=np.nan, R=np.nan, z=np.nan, p_two_sided=np.nan)

    r_obs = float(np.mean(nn))
    lam = n / area
    r_exp = float(1.0 / (2.0 * math.sqrt(lam)))
    se = float(0.26136 / math.sqrt(n * lam)) if (n * lam) > 0 else np.nan
    z = float((r_obs - r_exp) / se) if (se and np.isfinite(se) and se > 0) else np.nan
    p = float(erfc(abs(z) / sqrt(2))) if np.isfinite(z) else np.nan
    R = float(r_obs / r_exp) if (r_exp and np.isfinite(r_exp) and r_exp > 0) else np.nan
    return dict(r_obs=r_obs, r_exp=r_exp, R=R, z=z, p_two_sided=p)

# -----------------------------
# Summary of a single repeat
# -----------------------------
def summarise_repeat(model):
    alive = [a for a in model.agent_set if getattr(a, "alive", True) and getattr(a, "pos", None) is not None]
    n = len(alive)
    if n == 0:
        return dict(
            n_clusters=0, mean_size=np.nan, var_size=np.nan,
            median_nnd=np.nan, r_obs=np.nan, r_exp=np.nan,
            clark_evans_R=np.nan, clark_evans_z=np.nan, clark_evans_p=np.nan
        )

    sizes = np.array([float(a.size) for a in alive], dtype=float)
    x = np.array([float(a.pos[0]) for a in alive])
    y = np.array([float(a.pos[1]) for a in alive])
    W = float(model.params["space"]["width"])
    H = float(model.params["space"]["height"])
    torus = bool(model.params["space"].get("torus", True))
    area = W * H

    nn = nearest_neighbour_distances(x, y, W, H, torus)
    median_nnd = float(np.median(nn)) if len(nn) else np.nan

    ce = clark_evans_stats(x, y, area, W, H, torus)

    return dict(
        n_clusters=n,
        mean_size=float(np.mean(sizes)),
        var_size=float(np.var(sizes, ddof=1)) if n > 1 else 0.0,
        median_nnd=median_nnd,
        r_obs=ce["r_obs"], r_exp=ce["r_exp"],
        clark_evans_R=ce["R"],
        clark_evans_z=ce["z"],
        clark_evans_p=ce["p_two_sided"],
    )

# -----------------------------
# Trajectory export
# -----------------------------
TRAJ_COLUMNS = [
    "condition","repeat","step","time_min",
    "agent_id","phenotype","movement_phase",
    "x","y","size","radius","speed"
]

def write_snapshot_rows(writer, model, condition, rep, step):
    dt = float(model.dt)
    time_min = step * dt
    for a in list(model.agent_set):
        if not getattr(a, "alive", True): continue
        p = getattr(a, "pos", None)
        if p is None: continue
        vx = getattr(a, "vel", np.array([np.nan, np.nan]))
        speed = float(np.linalg.norm(vx))

        writer.writerow(dict(
            condition=condition, repeat=rep, step=step,
            time_min=time_min, agent_id=int(a.unique_id),
            phenotype=str(a.phenotype),
            movement_phase=int(a.movement_phase),
            x=float(p[0]), y=float(p[1]),
            size=float(a.size), radius=float(a.radius),
            speed=speed
        ))

# -----------------------------
# Worker task
# -----------------------------
def run_one_repeat_task(cond_name, phenotype, phase2_only, rep, steps, seed, out_csv_str, base_params, n_init, init_size):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    from abm.clusters_model import ClustersModel

    params = deepcopy(base_params)
    params["init"] = dict(n_clusters=n_init, size=init_size, phenotype=phenotype)
    params["time"]["steps"] = int(steps)

    model = ClustersModel(params=params, seed=seed)

    if phase2_only:
        for a in model.agent_set:
            if getattr(a, "alive", True):
                a.movement_phase = 2
                a.phase_switch_time = np.inf

    out_csv = Path(out_csv_str)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAJ_COLUMNS)
        writer.writeheader()
        write_snapshot_rows(writer, model, cond_name, rep, step=0)

        for s in range(1, steps+1):
            model.step()
            write_snapshot_rows(writer, model, cond_name, rep, s)

    summary = summarise_repeat(model)
    summary.update(dict(
        condition=cond_name, phenotype=phenotype,
        phase2_only=phase2_only,
        repeat=rep, seed=seed, steps=steps
    ))
    return summary

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed0", type=int, default=123)
    parser.add_argument("--n_init", type=int, default=None)
    parser.add_argument("--init_size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--serial", action="store_true")
    args = parser.parse_args()

    root = project_root()
    results_dir = root / "results/results_invasive"
    traj_dir = results_dir / "trajectories"
    results_dir.mkdir(exist_ok=True)
    traj_dir.mkdir(exist_ok=True)

    from abm.utils import DEFAULTS
    base_params = deepcopy(DEFAULTS)

    steps = args.steps or int(base_params["time"]["steps"])
    n_init = args.n_init or int(base_params["init"]["n_clusters"])
    init_size = args.init_size or int(base_params["init"]["size"])

    # ONLY invasive conditions
    conditions = [
        Condition("invasive_phase2_only", "invasive", True),
        Condition("invasive_two_phase", "invasive", False),
    ]

    # Build all tasks
    tasks = []
    for c_idx, cond in enumerate(conditions):
        for r in range(args.repeats):
            seed = args.seed0 + 10000*c_idx + r
            out_csv = traj_dir / cond.name / f"repeat_{r:03d}.csv"
            tasks.append((c_idx, cond, r, seed, str(out_csv)))

    workers = 1 if args.serial else (args.workers or max(1, (os.cpu_count() or 2)-1))

    print(f"Running {len(tasks)} simulations ({len(conditions)} conditions × {args.repeats} repeats)")
    print(f"Steps per run: {steps}; workers={workers}")

    all_rows = []

    if workers == 1:
        for c_idx, cond, r, seed, out_csv_str in tasks:
            row = run_one_repeat_task(
                cond.name, cond.phenotype, cond.phase2_only,
                r, steps, seed, out_csv_str,
                base_params, n_init, init_size
            )
            all_rows.append(row)
    else:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    run_one_repeat_task,
                    cond.name, cond.phenotype, cond.phase2_only,
                    r, steps, seed, out_csv_str,
                    base_params, n_init, init_size
                )
                for (c_idx, cond, r, seed, out_csv_str) in tasks
            ]

            for i, fut in enumerate(cf.as_completed(futures), 1):
                row = fut.result()
                all_rows.append(row)
                if i % max(1, len(futures)//10) == 0:
                    print(f"Completed {i}/{len(futures)}...")

    # Save summaries
    df_rep = pd.DataFrame(all_rows).sort_values(["condition","repeat"]).reset_index(drop=True)
    df_rep.to_csv(results_dir/"summary_repeats.csv", index=False)

    print("Saved:", results_dir/"summary_repeats.csv")

if __name__ == "__main__":
    main()