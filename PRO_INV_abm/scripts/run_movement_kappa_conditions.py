#!/usr/bin/env python3
"""
Parallel runner for 10 kappa conditions × 3 movement models = 30 conditions.
Each condition is repeated N times (default: 10).

This version:
 - Runs in parallel using ProcessPoolExecutor
 - Prints how many workers are used
 - Prints progress updates: "<done> / <total> simulations complete"
 - Produces both summary_repeats.csv and summary_timeseries.csv
 - Stores full trajectories in trajectories/<condition>/repeat_XXX.csv
"""

from __future__ import annotations
import argparse, csv, os
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import concurrent.futures as cf


# ---------------------------------------------------------------------
# 10 KAPPA CONDITIONS (None = use default value from movement_v2)
# ---------------------------------------------------------------------
kappa_conditions = [
    ("Default", dict(k1=None, k2=None)),
    ("P1_0",    dict(k1=0,    k2=None)),
    ("P2_0",    dict(k1=None, k2=0)),
    ("P1P2_0",  dict(k1=0,    k2=0)),
    ("P1_1",    dict(k1=1,    k2=None)),
    ("P2_1",    dict(k1=None, k2=1)),
    ("P1P2_1",  dict(k1=1,    k2=1)),
    ("P1_2",    dict(k1=2,    k2=None)),
    ("P2_2",    dict(k1=None, k2=2)),
    ("P1P2_2",  dict(k1=2,    k2=2)),
]

movement_models = [
    ("phase2_only",        dict(phase2_only=True,  allow_cross=True)),
    ("two_phase_interact", dict(phase2_only=False, allow_cross=True)),
    ("two_phase_no_interact", dict(phase2_only=False, allow_cross=False)),
]


# ---------------------------------------------------------------------
# Apply kappa overrides
# ---------------------------------------------------------------------
def apply_kappas(base_params, k1_override, k2_override):
    p = deepcopy(base_params)
    mv = p["movement_v2"]["invasive"]

    if k1_override is not None:
        mv["phase1"]["turning"]["kappa"] = float(k1_override)

    if k2_override is not None:
        mv["phase2"]["turning"]["kappa"] = float(k2_override)

    return p


# ---------------------------------------------------------------------
# Spatial summary routine
# ---------------------------------------------------------------------
def compute_spatial(df_t, W, H, torus=True):
    if df_t.empty or len(df_t) < 2:
        return dict(median_nnd=np.nan, R=np.nan, z=np.nan)

    x = df_t["x"].to_numpy(float)
    y = df_t["y"].to_numpy(float)
    n = len(x)
    area = W * H

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]

    if torus:
        dx = (dx + W/2) % W - W/2
        dy = (dy + H/2) % H - H/2

    d2 = dx * dx + dy * dy
    np.fill_diagonal(d2, np.inf)

    nn = np.sqrt(np.min(d2, axis=1))
    r_obs = float(nn.mean())
    lam = n / area
    if lam <= 0:
        return dict(median_nnd=np.nan, R=np.nan, z=np.nan)

    r_exp = 1/(2*np.sqrt(lam))
    se = 0.26136 / np.sqrt(n*lam)
    if se <= 0:
        return dict(median_nnd=np.nan, R=np.nan, z=np.nan)

    z = (r_obs - r_exp) / se
    R = r_obs / r_exp

    return dict(
        median_nnd=float(np.median(nn)),
        R=R,
        z=z
    )


# ---------------------------------------------------------------------
# Worker: run one (condition, repeat)
# ---------------------------------------------------------------------
def run_single_sim(args):
    (
        cname,                      # condition name, e.g. "P1_0__phase2_only"
        kinf,                       # dict(k1=?, k2=?)
        minf,                       # dict(phase2_only=?, allow_cross=?)
        rep,
        seed,
        out_csv_str,
        base_params,
        steps,
        n_init,
        init_size,
        W, H, torus
    ) = args

    from abm.clusters_model import ClustersModel

    params = apply_kappas(base_params, kinf["k1"], kinf["k2"])
    params["init"] = dict(n_clusters=n_init, size=init_size, phenotype="invasive")
    params["interactions"]["allow_cross_phase_interactions"] = minf["allow_cross"]
    params["time"]["steps"] = int(steps)

    model = ClustersModel(params=params, seed=seed)

    # phase2 override
    if minf["phase2_only"]:
        for a in model.agent_set:
            a.movement_phase = 2
            a.phase_switch_time = np.inf

    out_csv = Path(out_csv_str)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    TRAJ_COLUMNS = [
        "condition","repeat","step","time_min",
        "agent_id","phenotype","movement_phase",
        "x","y","size","radius","speed"
    ]

    # Per-repeat time-series accumulation
    ts_rows = []

    # write trajectories + summary
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRAJ_COLUMNS)
        w.writeheader()

        # t=0
        write_snapshot(w, model, cname, rep, 0)
        sizes0 = [a.size for a in model.agent_set]
        ts_rows.append(dict(
            condition=cname, repeat=rep, step=0, time_min=0.0,
            n_clusters=len(sizes0),
            mean_size=float(np.mean(sizes0)) if len(sizes0) else np.nan,
            var_size=float(np.var(sizes0, ddof=1)) if len(sizes0)>1 else 0.0,
            median_nnd=np.nan, R=np.nan, z=np.nan
        ))

        # steps 1..T
        for t in range(1, steps+1):
            model.step()
            write_snapshot(w, model, cname, rep, t)

            alive = [
                a for a in model.agent_set
                if getattr(a, "alive", True) and getattr(a, "pos", None) is not None
            ]

            if len(alive) == 0:
                ts_rows.append(dict(
                    condition=cname, repeat=rep, step=t, time_min=t*model.dt,
                    n_clusters=0, mean_size=np.nan, var_size=np.nan,
                    median_nnd=np.nan, R=np.nan, z=np.nan
                ))
                continue

            df_t = pd.DataFrame({
                "x": [a.pos[0] for a in alive],
                "y": [a.pos[1] for a in alive],
                "size": [a.size for a in alive],
            })

            spatial = compute_spatial(df_t, W, H, torus)
            ts_rows.append(dict(
                condition=cname, repeat=rep, step=t, time_min=t*model.dt,
                n_clusters=len(df_t),
                mean_size=float(df_t["size"].mean()),
                var_size=float(df_t["size"].var(ddof=1)) if len(df_t)>1 else 0.0,
                median_nnd=spatial["median_nnd"],
                R=spatial["R"],
                z=spatial["z"]
            ))

    # Final state summary
    last = ts_rows[-1]
    summary_row = dict(
        condition=cname,
        repeat=rep,
        n_clusters=last["n_clusters"],
        mean_size=last["mean_size"],
        var_size=last["var_size"]
    )

    return ts_rows, summary_row


# ---------------------------------------------------------------------
# Snapshot writer (helper)
# ---------------------------------------------------------------------
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
            condition=condition, repeat=rep, step=step, time_min=time_min,
            agent_id=int(a.unique_id), phenotype=str(a.phenotype),
            movement_phase=int(a.movement_phase),
            x=float(p[0]), y=float(p[1]),
            size=float(a.size), radius=float(a.radius), speed=speed
        ))


# ---------------------------------------------------------------------
# MAIN (parallel execution)
# ---------------------------------------------------------------------
def main():
    from abm.utils import DEFAULTS

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed0", type=int, default=123)
    parser.add_argument("--n_init", type=int, default=None)
    parser.add_argument("--init_size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    base_params = deepcopy(DEFAULTS)

    # geometry for spatial stats
    W = float(base_params["space"]["width"])
    H = float(base_params["space"]["height"])
    torus = bool(base_params["space"]["torus"])

    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results/results_kappa_conditions"
    traj_dir = results_dir / "trajectories"
    results_dir.mkdir(exist_ok=True, parents=True)

    steps = args.steps or int(base_params["time"]["steps"])
    n_init = args.n_init or int(base_params["init"]["n_clusters"])
    init_size = args.init_size or int(base_params["init"]["size"])

    # Build all 30 conditions
    full_conditions = []
    for kname, kinf in kappa_conditions:
        for mname, minf in movement_models:
            cname = f"{kname}__{mname}"
            full_conditions.append((cname, kinf, minf))

    # Build task list
    tasks = []
    for idx, (cname, kinf, minf) in enumerate(full_conditions):
        for rep in range(args.repeats):
            seed = args.seed0 + 10000*idx + rep
            out_csv = traj_dir / cname / f"repeat_{rep:03d}.csv"
            tasks.append((
                cname, kinf, minf, rep, seed, str(out_csv),
                base_params, steps, n_init, init_size,
                W, H, torus
            ))

    total = len(tasks)
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    print(f"Running {total} simulations across {workers} workers\n")

    # Parallel execution with progress updates
    summary_all = []
    ts_all = []
    completed = 0

    with cf.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_single_sim, t) for t in tasks]
        for fut in cf.as_completed(futures):
            ts_rows, summary_row = fut.result()
            ts_all.extend(ts_rows)
            summary_all.append(summary_row)

            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"{completed} / {total} simulations complete")

    # Save outputs
    pd.DataFrame(summary_all).to_csv(results_dir/"summary_repeats.csv", index=False)
    pd.DataFrame(ts_all).to_csv(results_dir/"summary_timeseries.csv", index=False)

    print("\nSaved:")
    print(" -", results_dir/"summary_repeats.csv")
    print(" -", results_dir/"summary_timeseries.csv")


if __name__ == "__main__":
    main()