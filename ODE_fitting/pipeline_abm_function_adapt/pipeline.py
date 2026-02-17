#!/usr/bin/env python3
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import yaml
import argparse
import pandas as pd

from .abm_runner import run_forward_sims
from .mle_runner import run_mle
from .mcmc_runner import run_adaptive_mcmc
from .diagnostics_runner import run_diagnostics
from .aic_tools import write_aic_table

from .utils_logging import (
    make_run_dir, save_manifest, save_run_input,
    timer, write_timings,
    init_run_status, update_run_status, utcnow
)

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def slice_data(mean_csv, mode):
    df = pd.read_csv(mean_csv)
    if mode in ["data_t71_phase2", "singletons_phase2_fit71plus", "singletons_phase1_to_2_fit71plus"]:
        df = df[df["step"] >= 71].reset_index(drop=True)
    times = df["step"].to_numpy()
    data_values = df[["num_clusters", "mean_cluster_size", "mean_squared_cluster_size"]].to_numpy()
    return df, times, data_values

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                  default=os.path.join(os.path.dirname(__file__), "pipeline_config.yaml"))
    args = p.parse_args()

    cfg = load_cfg(args.config)
    results_root = os.path.abspath(cfg.get("results_root", "pipeline_results"))
    run_dir = make_run_dir(results_root)

    init_run_status(run_dir)
    update_run_status(run_dir, {"status": "running"})
    save_run_input(run_dir, cfg)

    timings = {"scenarios": {}}
    sweep_list = cfg["abm_sweep"]

    with timer() as t_all:
        for i, scenario in enumerate(sweep_list):
            mode = scenario["mode"]
            abm_params = scenario.get("abm_params", {})
            abm_cfg = {**cfg["abm"], "mode": mode, "abm_params": abm_params}

            scenario_name = f"abm_{i:02d}_{mode}"
            scenario_dir  = os.path.join(run_dir, scenario_name)
            fsim_dir      = os.path.join(scenario_dir, "forward_sims")
            os.makedirs(scenario_dir, exist_ok=True)

            update_run_status(run_dir, {"sections": {scenario_name: {"abm": {"started": utcnow()}}}})
            with timer() as t_abm:
                mean_csv = run_forward_sims(abm_cfg, fsim_dir)
            timings["scenarios"].setdefault(scenario_name, {})
            timings["scenarios"][scenario_name]["abm_seconds"] = round(t_abm.seconds, 3)
            update_run_status(run_dir, {"sections": {scenario_name: {"abm": {"finished": utcnow(), "duration_seconds": round(t_abm.seconds, 3)}}}})

            df, times, data_values = slice_data(mean_csv, mode)

            # Fit all ODE models for this scenario
            model_results = []
            for model_key in cfg["models"]:
                model_dir = os.path.join(scenario_dir, f"model_{model_key}")
                os.makedirs(model_dir, exist_ok=True)

                # MLE
                update_run_status(run_dir, {"sections": {scenario_name: {model_key: {"mle": {"started": utcnow()}}}}})
                with timer() as t_mle:
                    mle_out = run_mle(model_key, times, data_values,
                                      out_dir=os.path.join(model_dir, "mle"),
                                      model_meta_overrides=cfg.get("models_meta", {}).get(model_key, {}))
                timings["scenarios"][scenario_name].setdefault(model_key, {})
                timings["scenarios"][scenario_name][model_key]["mle_seconds"] = round(t_mle.seconds, 3)
                update_run_status(run_dir, {"sections": {scenario_name: {model_key: {"mle": {
                    "finished": utcnow(), "duration_seconds": round(t_mle.seconds, 3), "AIC": mle_out["AIC"]
                }}}}})

                # MCMC (diagnostics)
                def _progress(b, total, rhat):
                    update_run_status(run_dir, {"sections": {scenario_name: {model_key: {
                        "mcmc": {"current_block": b, "total_iters": total, "last_rhat": rhat}
                    }}}})

                update_run_status(run_dir, {"sections": {scenario_name: {model_key: {"mcmc": {"started": utcnow()}}}}})
                with timer() as t_mcmc:
                    mcmc_out = run_adaptive_mcmc(
                        model_key, times, data_values,
                        mcmc_cfg=cfg["mcmc"], out_dir=model_dir,
                        model_meta_overrides=cfg.get("models_meta", {}).get(model_key, {}),
                        progress_callback=_progress
                    )
                timings["scenarios"][scenario_name][model_key]["mcmc_seconds"] = round(t_mcmc.seconds, 3)

                # Diagnostics
                diag_dir = os.path.join(model_dir, "diagnostics")
                update_run_status(run_dir, {"sections": {scenario_name: {model_key: {"diagnostics": {"started": utcnow()}}}}})
                with timer() as t_diag:
                    run_diagnostics(model_key, mcmc_out["chains"], mcmc_out["post_burn_flat"],
                                    times, data_values, diag_dir,
                                    nsamples_ppc=cfg["mcmc"].get("nsamples_ppc", 300),
                                    seed=cfg["mcmc"].get("random_seed", 12345))
                timings["scenarios"][scenario_name][model_key]["diagnostics_seconds"] = round(t_diag.seconds, 3)
                update_run_status(run_dir, {"sections": {scenario_name: {model_key: {"diagnostics": {
                    "finished": utcnow(), "duration_seconds": round(t_diag.seconds, 3)
                }}}}})

                model_results.append({
                    "model_key": model_key,
                    "AIC": mle_out["AIC"],
                    "max_loglik": mle_out["max_loglik"],
                    "n_params": mle_out["k_params"],
                })

            write_aic_table(model_results, os.path.join(scenario_dir, "model_comparison.csv"))

    timings["pipeline_total_seconds"] = round(t_all.seconds, 3)
    write_timings(run_dir, timings)
    save_manifest(run_dir, cfg)
    update_run_status(run_dir, {"status": "finished", "pipeline_total": {
        "finished": utcnow(), "duration_seconds": timings["pipeline_total_seconds"]
    }})

    print(f"\nPipeline complete. Run folder: {run_dir}")

if __name__ == "__main__":
    main()