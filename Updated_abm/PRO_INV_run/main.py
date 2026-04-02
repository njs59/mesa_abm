#!/usr/bin/env python3
from __future__ import annotations
"""
Runs PRO vs INV scenarios (Phase 2 ONLY) and generates plots, with timestamped outputs
and CLI control of initial cluster counts for PRO/INV.

Scenarios:
  - S_PRO_ONLY:                  PRO baseline only (global baseline for plots)
  - S1_INV_BASE:                 INV baseline
  - S2_INV_MORE_IC:              INV with more initial clusters
  - S3_INV_MORE_IC_LOWER_PROLIF: INV with more IC + lower proliferation

Outputs (example):
  PRO_INV_results/<RUN_ID>/
    config.json
    trajectories/<scenario>/<PRO|INV>/repeat_XXX.csv
    summary/<scenario>__timeseries_PRObaseline_vs_INV.csv
    plots/<scenario>/*.png and pctdiff/*.png
    plots/_combined/pctdiff/*.png
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from .run_experiments import run_all_scenarios
from .plot_timeseries import plot_all_scenarios


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(
        description="PRO vs INV (Phase 2 only) scenarios with timestamped outputs and custom IC."
    )
    # General run control
    p.add_argument("--repeats", type=int, default=10, help="Repeats per condition (per scenario).")
    p.add_argument("--steps", type=int, default=None, help='Override DEFAULTS["time"]["steps"].')
    p.add_argument("--seed0", type=int, default=123, help="Base seed.")
    p.add_argument("--workers", type=int, default=None, help="Parallel workers (default: cpu_count-1).")
    p.add_argument("--serial", action="store_true", help="Force serial execution (debug).")

    # Initial cluster counts + INV scaling
    p.add_argument("--pro_n_init_base", type=int, default=850, help="Baseline PRO initial clusters (S_PRO_ONLY).")
    p.add_argument("--inv_n_init_base", type=int, default=850, help="Baseline INV initial clusters (S1_INV_BASE).")
    p.add_argument("--inv_n_init_more", type=int, default=1100, help="INV initial clusters for S2/S3.")
    p.add_argument("--inv_prolif_scale", type=float, default=0.9,
                   help="INV proliferation scale relative to PRO for S3 (e.g., 0.9 means INV = 0.9 × PRO).")

    args = p.parse_args()
    root = project_root()

    # Timestamped results folder
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = root / "PRO_INV_results" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: run PRO only once (S_PRO_ONLY). Set pro_n_init=None in INV scenarios.
    scenarios = [
        ("S_PRO_ONLY", args.pro_n_init_base, None, 1.0),
        ("S1_INV_BASE", None, args.inv_n_init_base, 1.0),
        ("S2_INV_MORE_IC", None, args.inv_n_init_more, 1.0),
        ("S3_INV_MORE_IC_LOWER_PROLIF", None, args.inv_n_init_more, args.inv_prolif_scale),
    ]

    # Save run metadata
    meta = {
        "run_id": run_id,
        "repeats": args.repeats,
        "steps": args.steps,
        "seed0": args.seed0,
        "workers": None if args.serial else (args.workers or max(1, (os.cpu_count() or 2) - 1)),
        "phase2_only": True,
        "pro_n_init_base": args.pro_n_init_base,
        "inv_n_init_base": args.inv_n_init_base,
        "inv_n_init_more": args.inv_n_init_more,
        "inv_prolif_scale": args.inv_prolif_scale,
        "scenarios": [{"name": s[0], "pro_n_init": s[1], "inv_n_init": s[2], "inv_scale": s[3]} for s in scenarios],
    }
    with (results_dir / "config.json").open("w") as f:
        json.dump(meta, f, indent=2)

    # 1) Run simulations (writes trajectories inside results_dir)
    run_all_scenarios(
        scenarios=scenarios,
        repeats=args.repeats,
        steps=args.steps,
        seed0=args.seed0,
        workers=args.workers,
        serial=args.serial,
        results_dir=results_dir,
    )

    # 2) Plot: each INV scenario vs the PRO baseline scenario
    pro_scenario = "S_PRO_ONLY"
    inv_scenarios = [s[0] for s in scenarios if s[2] is not None]  # scenarios that have INV configured
    plot_all_scenarios(
        inv_scenarios=inv_scenarios,
        pro_scenario=pro_scenario,
        results_dir=results_dir,
        plots_dir=results_dir / "plots",
    )

    print(f"[DONE] Results saved under: {results_dir}")


if __name__ == "__main__":
    main()