#!/usr/bin/env python3
"""
rerun_movement_model_extremes.py

Reruns ONLY the 4 conditions identified by
find_movement_model_extremes.py:

  - phenotype__phase2_only      (baseline for max increase)
  - phenotype__two_phase_*      (model for max increase)
  - phenotype__phase2_only      (baseline for max decrease)
  - phenotype__two_phase_*      (model for max decrease)

All simulations are run via:
  Multi_phase_analysis.analysis_utils.run_single_sim

This is the SAME execution path as run_fifteen_conditions.py.
No direct ClustersModel calls. No custom init logic.

Outputs:
  results/fifteen_conditions/movement_effects/rerun/
    ├── trajectories/<condition>/repeat_XXX.csv
    ├── summary_timeseries.csv
    ├── summary_timeseries_mean.csv
    └── summary_repeats.csv
"""

# ---------------------------------------------------------------------
# Ensure project root is importable (same pattern as your other drivers)
# ---------------------------------------------------------------------
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = None
for p in [THIS_DIR] + list(THIS_DIR.parents):
    if (p / "abm").is_dir():
        PROJECT_ROOT = p
        break

if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate project root containing 'abm/'")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---------------------------------------------------------------------

import argparse
import os
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from abm.utils import DEFAULTS
from Multi_phase_analysis.analysis_utils import run_single_sim

# ---------------------------------------------------------------------
# Condition definitions (IDENTICAL to run_fifteen_conditions.py)
# ---------------------------------------------------------------------
PHASE_FLAGS = {
    "phase2_only": dict(phase2_only=True, allow_cross=True),
    "two_phase_interact": dict(phase2_only=False, allow_cross=True),
    "two_phase_no_interact": dict(phase2_only=False, allow_cross=False),
}

SPEED_SCALES = {
    "baseline": dict(scale_P1=1.0, scale_P2=1.0),
    "decP1": dict(scale_P1=0.5, scale_P2=1.0),
    "incP1": dict(scale_P1=2.0, scale_P2=1.0),
    "decP2": dict(scale_P1=1.0, scale_P2=0.5),
    "incP2": dict(scale_P1=1.0, scale_P2=2.0),
}


def override_speeds(base_params: dict, *, scale_P1: float, scale_P2: float) -> dict:
    """
    Same logic as run_fifteen_conditions.override_speeds
    """
    p = deepcopy(base_params)
    mv = p["movement_v2"]["invasive"]

    if "scale" in mv["phase1"]["speed_dist"]["params"]:
        mv["phase1"]["speed_dist"]["params"]["scale"] *= float(scale_P1)
    if "scale" in mv["phase2"]["speed_dist"]["params"]:
        mv["phase2"]["speed_dist"]["params"]["scale"] *= float(scale_P2)

    return p


def _run_one(task):
    (
        condition, rep, seed, steps,
        base_params, overrides, flags,
        n_init, init_size, traj_csv
    ) = task

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

    # -----------------------------------------------------------------
    # Input files from finder script
    # -----------------------------------------------------------------
    RES_DIR = THIS_DIR / "results" / "fifteen_conditions"
    EFFECT_DIR = RES_DIR / "movement_effects"

    rerun_list = EFFECT_DIR / "rerun_conditions.csv"
    extremes_csv = EFFECT_DIR / "extremes.csv"

    if not rerun_list.exists():
        raise FileNotFoundError(f"Missing {rerun_list}")
    if not extremes_csv.exists():
        raise FileNotFoundError(f"Missing {extremes_csv}")

    conditions = pd.read_csv(rerun_list)["condition"].tolist()
    extremes = pd.read_csv(extremes_csv)

    # ensure simulation runs long enough to include both extreme times
    max_step_needed = int(extremes["step"].max())

    base_params = deepcopy(DEFAULTS)
    steps_default = int(base_params["time"]["steps"])
    steps = int(args.steps) if args.steps is not None else steps_default
    steps = max(steps, max_step_needed)

    n_init = int(base_params["init"]["n_clusters"])
    init_size = int(base_params["init"]["size"])

    # -----------------------------------------------------------------
    # Output structure
    # -----------------------------------------------------------------
    out_root = EFFECT_DIR / "rerun"
    traj_root = out_root / "trajectories"
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for cond in conditions:
        variant, model = cond.split("__")
        flags = PHASE_FLAGS[model]
        overrides = override_speeds(base_params, **SPEED_SCALES[variant])

        for rep in range(args.repeats):
            seed = int(args.seed0 + (hash(cond) % 100000) + rep)
            traj_csv = traj_root / cond / f"repeat_{rep:03d}.csv"
            traj_csv.parent.mkdir(parents=True, exist_ok=True)

            tasks.append((
                cond, rep, seed, steps,
                base_params, overrides, flags,
                n_init, init_size, traj_csv
            ))

    workers = 1 if args.serial else (args.workers or max(1, (os.cpu_count() or 2) - 1))
    print(f"Running {len(tasks)} simulations with workers={workers}")

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

    METRICS = ["n_clusters", "mean_size", "var_size", "median_nnd"]
    ts_mean = ts_df.groupby(
        ["condition", "step"], as_index=False
    )[METRICS + ["time_min"]].mean()

    ts_mean.to_csv(out_root / "summary_timeseries_mean.csv", index=False)

    print("Saved:")
    print(" -", out_root / "summary_timeseries.csv")
    print(" -", out_root / "summary_timeseries_mean.csv")
    print(" -", out_root / "summary_repeats.csv")
    print(" - trajectories under", traj_root)


if __name__ == "__main__":
    main()
