#!/usr/bin/env python3
"""
rerun_kappa_movement_model_extremes.py

Reruns ONLY the 4 κ conditions listed in:
  results/kappa_conditions/movement_effects/rerun_conditions.csv

All simulations run via Multi_phase_analysis.analysis_utils.run_single_sim
(the same proven pipeline used by run_kappa_conditions.py). [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/run_kappa_conditions.py)

Outputs:
  results/kappa_conditions/movement_effects/rerun/
    ├── trajectories/<condition>/repeat_XXX.csv
    ├── summary_timeseries.csv
    ├── summary_timeseries_mean.csv
    └── summary_repeats.csv
"""

# ---------------------------------------------------------------------
# Ensure project root is importable (consistent with your drivers) [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/run_kappa_conditions.py)[1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/run_kappa_conditions.py)
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

# Only these 4 metrics are relevant downstream [2](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/plot_kappa_conditions.py)
METRICS = ["n_clusters", "mean_size", "var_size", "median_nnd"]

# Same phase flags as run_kappa_conditions.py [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/run_kappa_conditions.py)
PHASE_FLAGS = {
    "phase2_only": dict(phase2_only=True, allow_cross=True),
    "two_phase_interact": dict(phase2_only=False, allow_cross=True),
    "two_phase_no_interact": dict(phase2_only=False, allow_cross=False),
}

# Same κ condition definitions as run_kappa_conditions.py [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/run_kappa_conditions.py)
KAPPA_CONDS = {
    "Default": (None, None),
    "P1_0": (0, None),
    "P2_0": (None, 0),
    "P1P2_0": (0, 0),
    "P1_1": (1, None),
    "P2_1": (None, 1),
    "P1P2_1": (1, 1),
    "P1_2": (2, None),
    "P2_2": (None, 2),
    "P1P2_2": (2, 2),
}


def override_kappas(base_params: dict, *, k1, k2) -> dict:
    """Matches run_kappa_conditions.override_kappas() exactly. [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/run_kappa_conditions.py)"""
    p = deepcopy(base_params)
    mv = p["movement_v2"]["invasive"]
    if k1 is not None:
        mv["phase1"]["turning"]["kappa"] = float(k1)
    if k2 is not None:
        mv["phase2"]["turning"]["kappa"] = float(k2)
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

    eff_dir = THIS_DIR / "results" / "kappa_conditions" / "movement_effects"
    rerun_list = eff_dir / "rerun_conditions.csv"
    extremes_csv = eff_dir / "extremes.csv"

    if not rerun_list.exists():
        raise FileNotFoundError(f"Missing {rerun_list}. Run find_kappa_movement_model_extremes.py first.")
    if not extremes_csv.exists():
        raise FileNotFoundError(f"Missing {extremes_csv}. Run find_kappa_movement_model_extremes.py first.")

    conditions = pd.read_csv(rerun_list)["condition"].tolist()
    extremes = pd.read_csv(extremes_csv)

    max_step_needed = int(extremes["step"].max())

    base_params = deepcopy(DEFAULTS)
    steps_default = int(base_params["time"]["steps"])
    steps = int(args.steps) if args.steps is not None else steps_default
    steps = max(steps, max_step_needed)

    n_init = int(base_params["init"]["n_clusters"])
    init_size = int(base_params["init"]["size"])

    out_root = eff_dir / "rerun"
    traj_root = out_root / "trajectories"
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for cond in conditions:
        kappa_name, model = cond.split("__")

        if kappa_name not in KAPPA_CONDS:
            raise KeyError(f"Unknown kappa label '{kappa_name}'. Expected one of: {list(KAPPA_CONDS.keys())}")

        k1, k2 = KAPPA_CONDS[kappa_name]
        overrides = override_kappas(base_params, k1=k1, k2=k2)
        flags = PHASE_FLAGS[model]

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

    ts_mean = ts_df.groupby(["condition", "step"], as_index=False)[METRICS + ["time_min"]].mean()
    ts_mean.to_csv(out_root / "summary_timeseries_mean.csv", index=False)

    print("Saved:")
    print(" -", out_root / "summary_timeseries.csv")
    print(" -", out_root / "summary_timeseries_mean.csv")
    print(" -", out_root / "summary_repeats.csv")
    print(" - trajectories under", traj_root)


if __name__ == "__main__":
    main()