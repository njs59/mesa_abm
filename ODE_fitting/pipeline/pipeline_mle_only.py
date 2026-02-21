#!/usr/bin/env python3
"""
MLE-only pipeline with HPC job-array orchestration.

CLI:
  prepare        -> expands sweep, validates (Option A), writes scenarios.json
  run-scenario   -> runs ONE scenario (ABM + MLE) by index (for job arrays)
  finalize       -> consolidates all results and generates plots
  run-all        -> sequential local run (no HPC), convenience wrapper
"""

from __future__ import annotations
import os
import json
import argparse
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from .abm_runner import run_forward_sims
from .mle_runner import run_mle
from .model_registry import get_model_meta
from .utils_logging import (
    make_run_dir, save_manifest, save_run_input,
    timer, utcnow
)

from .sweep_utils import (
    load_cfg, slice_data, write_abm_sweep_summary, _flatten,
    make_scenario_name, expand_param_sweep, extract_overrides,
    deduce_swept_param_keys, ensure_dir,
)
from .mle_collectors import MLEAggregator
from .mle_plots import plot_1d_sweep, plot_2d_heatmaps


# ----------------------------- helpers -----------------------------

def _require_one_or_two(keys: List[str]):
    n = len(keys)
    if n == 0:
        raise ValueError(
            "No swept ABM parameters detected. Configure 1 or 2 overrides in `abm_param_sweep`."
        )
    if n > 2:
        raise ValueError(
            f"Found {n} swept ABM parameters ({keys}). Option A selected: only 1 or 2 swept parameters are supported."
        )


def _write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _scenario_manifest(sweep_list: List[Dict[str, Any]], run_root: str) -> List[Dict[str, Any]]:
    """
    Build a simple manifest for all scenarios with stable names and essential fields.
    """
    manifest = []
    for i, sc in enumerate(sweep_list):
        mode = sc["mode"]
        abm_params = sc.get("abm_params", {})
        overrides = abm_params.get("overrides", {})
        scenario_name = make_scenario_name(i, mode, overrides)
        scenario_dir = os.path.join(run_root, scenario_name)
        manifest.append({
            "index": i,
            "mode": mode,
            "scenario_name": scenario_name,
            "scenario_dir": scenario_dir,
            "abm_params": abm_params,     # includes overrides
        })
    return manifest


def _run_single_scenario(*, scenario: Dict[str, Any], cfg: Dict[str, Any]):
    """
    Run ABM + MLE for exactly one scenario (no shared global files).
    Writes outputs under scenario_dir only.
    """
    mode = scenario["mode"]
    scenario_name = scenario["scenario_name"]
    scenario_dir = scenario["scenario_dir"]
    abm_params = scenario.get("abm_params", {})
    overrides = abm_params.get("overrides", {}) or {}

    ensure_dir(scenario_dir)
    fsim_dir = os.path.join(scenario_dir, "forward_sims")
    ensure_dir(fsim_dir)

    # ABM
    abm_cfg = {**cfg["abm"], "mode": mode, "abm_params": abm_params}
    with timer() as t_abm:
        mean_csv = run_forward_sims(abm_cfg, fsim_dir)
    abm_seconds = round(t_abm.seconds, 3)   # uses .seconds from your utils_logging.timer

    # Slice for MLE
    _, times, data_values = slice_data(mean_csv, mode)

    # MLE for each model
    model_timings = {}
    for model_key in cfg["models"]:
        model_dir = os.path.join(scenario_dir, f"model_{model_key}")
        ensure_dir(model_dir)
        with timer() as t_mle:
            run_mle(
                model_key, times, data_values,
                out_dir=os.path.join(model_dir, "mle"),
                model_meta_overrides=cfg.get("models_meta", {}).get(model_key, {})
            )
        model_timings[model_key] = {"mle_seconds": round(t_mle.seconds, 3)}

    # Per-scenario status (local only, safe for HPC)
    status = {
        "scenario": scenario_name,
        "mode": mode,
        "finished": utcnow(),
        "timings": {
            "abm_seconds": abm_seconds,
            "models": model_timings,
        },
        "mean_csv": os.path.relpath(mean_csv, scenario_dir),
        "overrides": overrides,
    }
    _write_json(os.path.join(scenario_dir, "status.json"), status)


# ----------------------------- subcommands -----------------------------

def cmd_prepare(args):
    cfg = load_cfg(args.config)
    results_root = os.path.abspath(cfg.get("results_root", "pipeline_results"))
    run_root = make_run_dir(results_root)

    # Save config snapshot for traceability
    save_run_input(run_root, cfg)
    save_manifest(run_root, cfg)

    # Expand sweep and validate
    base_scenarios = cfg.get("abm_sweep", []) or []
    sweep_list = expand_param_sweep(cfg, base_scenarios)

    swept_keys = deduce_swept_param_keys(sweep_list)
    _require_one_or_two(swept_keys)

    # Write scenarios manifest
    manifest = _scenario_manifest(sweep_list, run_root)
    _write_json(os.path.join(run_root, "scenarios.json"), manifest)

    # Also write a terse, user-friendly mapping (index -> scenario_name)
    short = {m["index"]: m["scenario_name"] for m in manifest}
    _write_json(os.path.join(run_root, "scenario_index_map.json"), short)

    print(f"\nPrepared {len(manifest)} scenarios.")
    print(f"RUN_ROOT={run_root}")
    print("Next:")
    print("  # SLURM example")
    print("  export RUN_ROOT=\"<THE_PATH_ABOVE>\"")
    print("  sbatch --array=0-{N} your_job.slurm   # where N = total_scenarios-1")
    print("  # Each job calls:")
    print("  python -m your_package.pipeline_mle_only run-scenario --config pipeline_config.yaml --run-root \"$RUN_ROOT\" --scenario-index $SLURM_ARRAY_TASK_ID")
    print("\nWhen all jobs are finished, finalise:")
    print("  python -m your_package.pipeline_mle_only finalize --config pipeline_config.yaml --run-root \"$RUN_ROOT\"")


def cmd_run_scenario(args):
    # Load manifest and pick scenario
    manifest = _read_json(os.path.join(args.run_root, "scenarios.json"))
    idx = int(args.scenario_index)
    scenario = None
    for m in manifest:
        if m["index"] == idx:
            scenario = m
            break
    if scenario is None:
        raise IndexError(f"Scenario index {idx} not found in scenarios.json")

    cfg = load_cfg(args.config)
    _run_single_scenario(scenario=scenario, cfg=cfg)
    print(f"Finished scenario index={idx} name={scenario['scenario_name']}")


def cmd_finalize(args):
    """
    Consolidate all per-scenario outputs into a single CSV and generate plots.
    Does not re-run anything; it only reads results.
    """
    run_root = args.run_root
    cfg = load_cfg(args.config)

    manifest = _read_json(os.path.join(run_root, "scenarios.json"))
    swept_keys = sorted(list({
        k for m in manifest
        for k in (m.get("abm_params", {}).get("overrides", {}) or {}).keys()
    }))
    _require_one_or_two(swept_keys)

    # Build sweep summary + aggregator from existing outputs
    summary_rows = []
    aggregator = MLEAggregator(run_dir=run_root, swept_keys=swept_keys)

    for m in manifest:
        scenario_dir = m["scenario_dir"]
        scenario_name = m["scenario_name"]
        mode = m["mode"]
        overrides = (m.get("abm_params", {}) or {}).get("overrides", {}) or {}

        # Count scenario if any model produced an MLE output
        found_any = False
        for model_key in cfg["models"]:
            import yaml
            mle_yaml = os.path.join(scenario_dir, f"model_{model_key}", "mle", "mle_results.yaml")
            if not os.path.isfile(mle_yaml):
                continue
            with open(mle_yaml, "r") as f:
                mle_out = yaml.safe_load(f) or {}
            meta = get_model_meta(model_key, cfg.get("models_meta", {}).get(model_key, {}))
            ode_param_names = meta["param_names"]

            aggregator.add_record(
                scenario_name=scenario_name,
                mode=mode,
                model_key=model_key,
                overrides=overrides,
                mle_out=mle_out,
                ode_param_names=ode_param_names,
            )
            found_any = True

        if found_any:
            aggregator.add_abm_run()

        # Sweep summary row
        row = {
            "scenario": scenario_name,
            "mode": mode,
        }
        for k, v in _flatten(overrides).items():
            row[f"override.{k}"] = v
        summary_rows.append(row)

    # Write summary + consolidated results
    write_abm_sweep_summary(run_root, summary_rows)
    aggregator.save_counters()
    results_csv = os.path.join(run_root, "mle_results.csv")
    aggregator.save_results_csv(results_csv)

    # Plots per model
    plots_root = os.path.join(run_root, "plots")
    ensure_dir(plots_root)

    df_all = aggregator.to_dataframe()
    if df_all.empty:
        print("No scenario outputs found; nothing to plot.")
        return

    if len(swept_keys) == 1:
        x_key = f"abm::{swept_keys[0]}"
        for model_key in cfg["models"]:
            meta = get_model_meta(model_key, cfg.get("models_meta", {}).get(model_key, {}))
            model_dir = os.path.join(plots_root, model_key)
            ensure_dir(model_dir)
            df_m = df_all[df_all["model"] == model_key].copy()
            plot_1d_sweep(
                df=df_m, x_key=x_key,
                ode_param_names=meta["param_names"],
                out_dir=model_dir, model_key=model_key,
            )
    else:  # len == 2
        x_key = f"abm::{swept_keys[0]}"
        y_key = f"abm::{swept_keys[1]}"
        for model_key in cfg["models"]:
            meta = get_model_meta(model_key, cfg.get("models_meta", {}).get(model_key, {}))
            model_dir = os.path.join(plots_root, model_key)
            ensure_dir(model_dir)
            df_m = df_all[df_all["model"] == model_key].copy()
            plot_2d_heatmaps(
                df=df_m, x_key=x_key, y_key=y_key,
                ode_param_names=meta["param_names"],
                out_dir=model_dir, model_key=model_key,
            )

    print(f"\nFinalised. Run folder: {run_root}")
    print(f" - Results table: {results_csv}")
    print(f" - Plots:         {plots_root}")


def cmd_run_all(args):
    """
    Convenience local mode: runs everything sequentially (no HPC).
    Equivalent to: prepare -> run all -> finalize (but within one process).
    """
    cfg = load_cfg(args.config)
    results_root = os.path.abspath(cfg.get("results_root", "pipeline_results"))
    run_root = make_run_dir(results_root)

    # Expand and validate
    base_scenarios = cfg.get("abm_sweep", []) or []
    sweep_list = expand_param_sweep(cfg, base_scenarios)
    swept_keys = deduce_swept_param_keys(sweep_list)
    _require_one_or_two(swept_keys)

    manifest = _scenario_manifest(sweep_list, run_root)
    _write_json(os.path.join(run_root, "scenarios.json"), manifest)

    # Run all
    for m in manifest:
        _run_single_scenario(scenario=m, cfg=cfg)

    # Finalise
    class A:
        pass
    a = A()
    a.config = args.config
    a.run_root = run_root
    cmd_finalize(a)


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="MLE-only pipeline (HPC-ready).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Expand sweep and write scenarios.json")
    p_prep.add_argument("--config", type=str, required=True)
    p_prep.set_defaults(func=cmd_prepare)

    p_run = sub.add_parser("run-scenario", help="Run ONE scenario by index (HPC array).")
    p_run.add_argument("--config", type=str, required=True)
    p_run.add_argument("--run-root", type=str, required=True)
    p_run.add_argument("--scenario-index", type=int, required=True)
    p_run.set_defaults(func=cmd_run_scenario)

    p_fin = sub.add_parser("finalize", help="Merge results and plot.")
    p_fin.add_argument("--config", type=str, required=True)
    p_fin.add_argument("--run-root", type=str, required=True)
    p_fin.set_defaults(func=cmd_finalize)

    p_all = sub.add_parser("run-all", help="Sequential local run (no HPC).")
    p_all.add_argument("--config", type=str, required=True)
    p_all.set_defaults(func=cmd_run_all)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()