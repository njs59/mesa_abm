#!/usr/bin/env python3
"""
run_extreme_conditions_forward.py

Reruns exactly 3 conditions (baseline + pos extreme + neg extreme) using
Multi_phase_analysis.analysis_utilssame proven entrypoint as run_fifteen_conditions.py). [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/__main__.py)

Writes under:
  results/fifteen_conditions/extreme_phase2only/rerun/
"""

# --- ensure project root is on sys.path (so imports work robustly) ---
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = None
for p in [THIS_DIR] + list(THIS_DIR.parents):
    if (p / "abm").is_dir():
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    raise RuntimeError(f"Could not locate project root containing 'abm/' from {THIS_DIR}")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---------------------------------------------------------------

import argparse
import os
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from abm.utils import DEFAULTS
from Multi_phase_analysis.analysis_utils import run_single_sim  # same call signature used in run_fifteen_conditions.py [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/__main__.py)

# ONLY the 4 metrics requested
METRICS = ["n_clusters", "mean_size", "var_size", "median_nnd"]

# same condition structure / phase flags as your 15-condition driver [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/__main__.py)
PHASE_MAP = {
    "phase2_only": dict(phase2_only=True, allow_cross=True),
    "two_phase_interact": dict(phase2_only=False, allow_cross=True),
    "two_phase_no_interact": dict(phase2_only=False, allow_cross=False),
}

SPEED_MAP = {
    "baseline": dict(scale_P1=1.0, scale_P2=1.0),
    "decP1": dict(scale_P1=0.5, scale_P2=1.0),
    "incP1": dict(scale_P1=2.0, scale_P2=1.0),
    "decP2": dict(scale_P1=1.0, scale_P2=0.5),
    "incP2": dict(scale_P1=1.0, scale_P2=2.0),
}


def override_speeds(base_params: dict, *, scale_P1: float, scale_P2: float) -> dict:
    """Matches your run_fifteen_conditions.override_speeds() intent. [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/__main__.py)"""
    p = deepcopy(base_params)
    mv = p["movement_v2"]["invasive"]
    if "scale" in mv["phase1"]["speed_dist"]["params"]:
        mv["phase1"]["speed_dist"]["params"]["scale"] *= float(scale_P1)
    if "scale" in mv["phase2"]["speed_dist"]["params"]:
        mv["phase2"]["speed_dist"]["params"]["scale"] *= float(scale_P2)
    return p


def _run_one(task):
    (condition, rep, seed, steps, base_params, overrides, flags, n_init, init_size, traj_csv) = task
    ts_rows, summary = run_single_sim(
        condition=condition,
        rep=rep,
        seed=seed,
        steps=steps,
        base_params=base_params,
        overrides=overrides,
        phase2_only=flags["phase2_only"],
        allow_cross=flags["allow_cross"],
        n_init=n_init,
        init_size=init_size,
        traj_csv=traj_csv,
    )
    # force these fields to exist
    summary = dict(summary)
    summary["condition"] = condition
    summary["rep"] = rep
    return ts_rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=123)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--serial", action="store_true")
    args = ap.parse_args()

    res_dir = THIS_DIR / "results" / "fifteen_conditions"
    sel_dir = res_dir / "extreme_phase2only"
    selected_csv = sel_dir / "selected_conditions.csv"
    extreme_csv = sel_dir / "extreme_events.csv"

    if not selected_csv.exists():
        raise FileNotFoundError(f"Missing {selected_csv}. Run find_extreme_conditions.py first.")
    if not extreme_csv.exists():
        raise FileNotFoundError(f"Missing {extreme_csv}. Run find_extreme_conditions.py first.")

    selected = pd.read_csv(selected_csv)
    conditions = selected["condition"].tolist()

    extreme_events = pd.read_csv(extreme_csv)
    max_step_needed = int(extreme_events["step"].max())

    base_params = deepcopy(DEFAULTS)
    steps_default = int(base_params["time"]["steps"])
    steps = int(args.steps) if args.steps is not None else steps_default
    steps = max(steps, max_step_needed)

    n_init = int(base_params["init"]["n_clusters"])
    init_size = int(base_params["init"]["size"])

    out_root = sel_dir / "rerun"
    traj_root = out_root / "trajectories"
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for cond in conditions:
        variant, phase_model = cond.split("__")
        flags = PHASE_MAP[phase_model]
        overrides = override_speeds(base_params, **SPEED_MAP[variant])

        for rep in range(int(args.repeats)):
            seed = int(args.seed0 + (hash(cond) % 100000) + rep)
            traj_csv = traj_root / cond / f"repeat_{rep:03d}.csv"
            traj_csv.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((cond, rep, seed, steps, base_params, overrides, flags, n_init, init_size, traj_csv))

    workers = 1 if args.serial else (args.workers or max(1, (os.cpu_count() or 2) - 1))
    print(f"Running {len(tasks)} sims (3 conditions × {args.repeats} repeats) with workers={workers}")

    ts_all, rep_all = [], []

    if workers == 1:
        for t in tasks:
            ts_rows, summary = _run_one(t)
            ts_all.extend(ts_rows)
            rep_all.append(summary)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_one, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                ts_rows, summary = fut.result()
                ts_all.extend(ts_rows)
                rep_all.append(summary)
                if i % max(1, len(futs) // 10) == 0:
                    print(f"  completed {i}/{len(futs)}")

    ts_df = pd.DataFrame(ts_all)
    rep_df = pd.DataFrame(rep_all)

    ts_df.to_csv(out_root / "summary_timeseries.csv", index=False)
    rep_df.to_csv(out_root / "summary_repeats.csv", index=False)

    ts_mean = ts_df.groupby(["condition", "step"], as_index=False)[METRICS + ["time_min"]].mean()
    ts_mean.to_csv(out_root / "summary_timeseries_mean.csv", index=False)

    print("Saved:")
    print(" -", out_root / "summary_timeseries.csv")
    print(" -", out_root / "summary_timeseries_mean.csv")
    print(" -", out_root / "summary_repeats.csv")
    print(" - trajectories under", traj_root)


if __name__ == "__main__":
    main()