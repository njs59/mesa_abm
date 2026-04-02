#!/usr/bin/env python3
"""
Run 15 ABM conditions:
  3 phase-behaviour conditions:
      phase2_only
      two_phase_interact
      two_phase_no_interact

  5 cell-type motility variants:
      baseline
      decP1   (phase 1 speed = 0.5 × baseline)
      incP1   (phase 1 speed = 2.0 × baseline)
      decP2   (phase 2 speed = 0.5 × baseline)
      incP2   (phase 2 speed = 2.0 × baseline)

Total = 15 conditions.

OUTPUT LOCATION:
    results-Invasive_3/trajectories/<celltype>__<phasecondition>/repeat_XXX.csv
"""

from __future__ import annotations
import argparse, csv, os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd


# ============================================================
# Condition structures
# ============================================================

phase_conditions = [
    ("phase2_only",          dict(phase2_only=True,  allow_cross=True)),
    ("two_phase_interact",   dict(phase2_only=False, allow_cross=True)),
    ("two_phase_no_interact",dict(phase2_only=False, allow_cross=False)),
]

cell_types = [
    ("baseline", dict(scale_P1=1.0, scale_P2=1.0)),
    ("decP1",    dict(scale_P1=0.5, scale_P2=1.0)),
    ("incP1",    dict(scale_P1=2.0, scale_P2=1.0)),
    ("decP2",    dict(scale_P1=1.0, scale_P2=0.5)),
    ("incP2",    dict(scale_P1=1.0, scale_P2=2.0)),
]


# ============================================================
# Utility for modifying movement speeds
# ============================================================

def apply_speed_scaling(params, scale_P1, scale_P2):
    """
    Multiply mean speeds of phase 1 or phase 2 by scale factors.
    For lognorm/gamma distributions, multiply scale parameter.
    """
    new_params = deepcopy(params)

    mv = new_params["movement_v2"]["invasive"]

    # Phase 1 speed distribution
    sp1 = mv["phase1"]["speed_dist"]
    if "scale" in sp1["params"]:
        sp1["params"]["scale"] *= scale_P1

    # Phase 2 speed distribution
    sp2 = mv["phase2"]["speed_dist"]
    if "scale" in sp2["params"]:
        sp2["params"]["scale"] *= scale_P2

    return new_params


# ============================================================
# Trajectory exporting
# ============================================================

TRAJ_COLUMNS = [
    "condition","repeat","step","time_min",
    "agent_id","phenotype","movement_phase",
    "x","y","size","radius","speed"
]

def write_snapshot(writer, model, condition, rep, step):
    dt = float(model.dt)
    time_min = step * dt

    for a in list(model.agent_set):
        if not getattr(a, "alive", True):
            continue
        p = getattr(a, "pos", None)
        if p is None:
            continue
        v = getattr(a, "vel", np.zeros(2))
        speed = float(np.linalg.norm(v))

        writer.writerow(dict(
            condition=condition,
            repeat=rep,
            step=step,
            time_min=time_min,
            agent_id=int(a.unique_id),
            phenotype=str(a.phenotype),
            movement_phase=int(a.movement_phase),
            x=float(p[0]),
            y=float(p[1]),
            size=float(a.size),
            radius=float(a.radius),
            speed=speed,
        ))


# ============================================================
# Single repeat execution
# ============================================================

def run_repeat(condition_name, celltype_params, phase_flags, rep, seed,
               out_csv_str, base_params, steps, n_init, init_size):

    from abm.clusters_model import ClustersModel

    params = apply_speed_scaling(base_params, 
                                 celltype_params["scale_P1"],
                                 celltype_params["scale_P2"])

    params["init"] = dict(n_clusters=n_init, size=init_size, phenotype="invasive")
    params["interactions"]["allow_cross_phase_interactions"] = phase_flags["allow_cross"]
    params["time"]["steps"] = int(steps)

    model = ClustersModel(params=params, seed=seed)

    # Phase2-only override
    if phase_flags["phase2_only"]:
        for a in model.agent_set:
            a.movement_phase = 2
            a.phase_switch_time = np.inf

    # Write trajectory
    out_csv = Path(out_csv_str)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRAJ_COLUMNS)
        w.writeheader()

        write_snapshot(w, model, condition_name, rep, 0)
        for t in range(1, steps+1):
            model.step()
            write_snapshot(w, model, condition_name, rep, t)

    # Compute summary
    alive = [a for a in model.agent_set if getattr(a, "alive", True)]
    if len(alive) == 0:
        return dict(condition=condition_name, repeat=rep, n_clusters=0)

    sizes = np.array([a.size for a in alive], float)

    return dict(
        condition=condition_name,
        repeat=rep,
        n_clusters=len(sizes),
        mean_size=float(sizes.mean()),
        var_size=float(sizes.var(ddof=1)) if len(sizes) > 1 else 0.0,
    )


# ============================================================
# MAIN
# ============================================================

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

    from abm.utils import DEFAULTS
    base_params = deepcopy(DEFAULTS)

    root = project_root = Path(__file__).resolve().parents[1]
    results_dir = root / "results-Invasive_3"
    traj_dir = results_dir / "trajectories"
    results_dir.mkdir(exist_ok=True)

    steps = args.steps or int(base_params["time"]["steps"])
    n_init = args.n_init or int(base_params["init"]["n_clusters"])
    init_size = args.init_size or int(base_params["init"]["size"])

    # Build 15 conditions
    full_conditions = []
    for cell_name, cell_info in cell_types:
        for phase_name, phase_info in phase_conditions:
            cname = f"{cell_name}__{phase_name}"
            full_conditions.append((cname, cell_info, phase_info))

    tasks = []
    for idx, (cname, cinfo, pinfo) in enumerate(full_conditions):
        for rep in range(args.repeats):
            seed = args.seed0 + 10000*idx + rep
            out_csv = traj_dir / cname / f"repeat_{rep:03d}.csv"
            tasks.append((cname, cinfo, pinfo, rep, seed, str(out_csv)))

    # Run
    workers = 1 if args.serial else (args.workers or max(1, (os.cpu_count() or 2)-1))
    print(f"Running {len(tasks)} runs ({len(full_conditions)} conditions × {args.repeats} repeats)")

    all_rows = []

    if workers == 1:
        for (cname, cinfo, pinfo, rep, seed, out_csv) in tasks:
            row = run_repeat(cname, cinfo, pinfo, rep, seed,
                             out_csv, base_params, steps, n_init, init_size)
            all_rows.append(row)
    else:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(run_repeat, cname, cinfo, pinfo, rep, seed,
                          out_csv, base_params, steps, n_init, init_size)
                for (cname, cinfo, pinfo, rep, seed, out_csv) in tasks
            ]
            for fut in cf.as_completed(futs):
                all_rows.append(fut.result())

    # Write summary
    df = pd.DataFrame(all_rows)
    df.to_csv(results_dir/"summary_repeats.csv", index=False)
    print("Saved:", results_dir/"summary_repeats.csv")


if __name__ == "__main__":
    main()