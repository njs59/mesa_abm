#!/usr/bin/env python3
import os
import yaml
import argparse
import pandas as pd

from .abm_runner import run_forward_sims
from .mcmc_runner import run_adaptive_mcmc
from .diagnostics_runner import run_diagnostics
from .aic_tools import write_aic_table
from .mle_runner import run_mle

from .utils_logging import (
    make_run_dir, save_manifest,
    timer, write_timings, save_run_input,
    init_run_status, update_run_status, utcnow
)

def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    p = argparse.ArgumentParser(description='Automated ABM→ODE pipeline')
    p.add_argument('--config', type=str,
                   default=os.path.join(os.path.dirname(__file__), 'pipeline_config.yaml'))
    args = p.parse_args()

    cfg = load_cfg(args.config)

    results_root = os.path.abspath(cfg.get('results_root', 'pipeline_results'))
    run_dir = make_run_dir(results_root)
    forward_dir = os.path.join(run_dir, 'forward_sims')

    # Live status initialisation
    init_run_status(run_dir)
    update_run_status(run_dir, {"status": "running"})

    timings = {}

    # 1. Forward ABM sims
    update_run_status(run_dir, {
        "sections": {"abm_forward": {"started": utcnow(), "finished": None}}
    })
    with timer() as t_abm:
        mean_csv = run_forward_sims(
            abm_cfg=cfg.get('abm', {}),
            defaults_yaml=cfg['abm'].get('defaults_yaml', 'scripts/scripts_defaults.yaml'),
            background_dir=cfg['abm'].get('background_dir', 'background'),
            run_forward_dir=forward_dir,
        )
    timings["abm_forward_seconds"] = round(t_abm.seconds, 3)
    update_run_status(run_dir, {
        "sections": {"abm_forward": {
            "finished": utcnow(),
            "duration_seconds": timings["abm_forward_seconds"]
        }}
    })

    df = pd.read_csv(mean_csv)
    times = df['step'].to_numpy()
    data_values = df[['num_clusters', 'mean_cluster_size', 'mean_squared_cluster_size']].to_numpy()

    mcmc_cfg = cfg.get('mcmc', {})
    models_meta_cfg = cfg.get('models_meta', {})
    model_results = []
    timings["models"] = {}

    # Main loop timer
    with timer() as t_total:

        for model_key in cfg.get('models', []):
            model_out_dir = os.path.join(run_dir, f"model_{model_key}")
            os.makedirs(model_out_dir, exist_ok=True)

            # ---------------------
            # MLE STAGE
            # ---------------------
            update_run_status(run_dir, {
                "sections": {"models": {model_key: {"mle": {"started": utcnow()}}}}
            })

            mle_dir = os.path.join(model_out_dir, "mle")
            with timer() as t_mle:
                mle_res = run_mle(
                    model_key=model_key,
                    times=times,
                    data_values=data_values,
                    out_dir=mle_dir,
                    model_meta_overrides=models_meta_cfg.get(model_key, {})
                )
            timings["models"].setdefault(model_key, {})
            timings["models"][model_key]["mle_seconds"] = round(t_mle.seconds, 3)

            update_run_status(run_dir, {
                "sections": {"models": {model_key: {"mle": {
                    "finished": utcnow(),
                    "duration_seconds": round(t_mle.seconds, 3),
                    "AIC": mle_res["AIC"],
                    "max_loglik": mle_res["max_loglik"],
                    "n_params": mle_res["k_params"],
                }}}}
            })

            # ---------------------
            # MCMC (diagnostics only, AIC from MLE)
            # ---------------------
            update_run_status(run_dir, {
                "sections": {"models": {
                    model_key: {"mcmc": {"started": utcnow(), "current_block": 0}}
                }}
            })

            def _progress(block_idx, total_iters, rhat_vec):
                update_run_status(run_dir, {
                    "sections": {"models": {
                        model_key: {"mcmc": {
                            "current_block": block_idx,
                            "total_iters_so_far": total_iters,
                            "last_rhat": rhat_vec,
                        }}
                    }}
                })

            with timer() as t_mcmc:
                mcmc_out = run_adaptive_mcmc(
                    model_key=model_key,
                    times=times,
                    data_values=data_values,
                    mcmc_cfg=mcmc_cfg,
                    out_dir=model_out_dir,
                    model_meta_overrides=models_meta_cfg.get(model_key, {}),
                    progress_callback=_progress,
                )
            timings["models"][model_key]["mcmc_seconds"] = round(t_mcmc.seconds, 3)

            update_run_status(run_dir, {
                "sections": {"models": {
                    model_key: {"mcmc": {
                        "finished": utcnow(),
                        "duration_seconds": round(t_mcmc.seconds, 3)
                    }}
                }}
            })

            # ---------------------
            # Diagnostics
            # ---------------------
            diag_dir = os.path.join(model_out_dir, "diagnostics")
            update_run_status(run_dir, {
                "sections": {"models": {
                    model_key: {"diagnostics": {"started": utcnow()}}
                }}
            })
            with timer() as t_diag:
                run_diagnostics(
                    model_key=model_key,
                    chains=mcmc_out['chains'],
                    flat_post=mcmc_out['post_burn_flat'],
                    times=times,
                    data=data_values,
                    out_dir=diag_dir,
                    nsamples_ppc=int(mcmc_cfg.get('nsamples_ppc', 300)),
                    seed=int(mcmc_cfg.get('random_seed', 12345)),
                )
            timings["models"][model_key]["diagnostics_seconds"] = round(t_diag.seconds, 3)

            update_run_status(run_dir, {
                "sections": {"models": {
                    model_key: {"diagnostics": {
                        "finished": utcnow(),
                        "duration_seconds": round(t_diag.seconds, 3)
                    }}
                }}
            })

            # Add model summary (MLE-only AIC)
            model_results.append({
                "model_key": model_key,
                "AIC": mle_res["AIC"],
                "max_loglik": mle_res["max_loglik"],
                "n_params": mle_res["k_params"],
            })

        # AIC table
        with timer() as t_aic:
            aic_csv = os.path.join(run_dir, "model_comparison.csv")
            write_aic_table(model_results, aic_csv)
        timings["aic_seconds"] = round(t_aic.seconds, 3)

    timings["pipeline_total_seconds"] = round(t_total.seconds, 3)

    save_run_input(run_dir, cfg)
    write_timings(run_dir, timings)
    save_manifest(run_dir, cfg, extra={
        "forward_means_stats": os.path.relpath(mean_csv, run_dir),
        "models_run": [m["model_key"] for m in model_results],
        "aic_csv": os.path.relpath(aic_csv, run_dir),
        "run_status_yaml": "run_status.yaml",
        "timings_yaml": "timings.yaml",
    })

    update_run_status(run_dir, {
        "status": "finished",
        "pipeline_total": {
            "finished": utcnow(),
            "duration_seconds": timings["pipeline_total_seconds"]
        }
    })

    print(f"\nPipeline complete. Run folder: {run_dir}")

if __name__ == "__main__":
    main()