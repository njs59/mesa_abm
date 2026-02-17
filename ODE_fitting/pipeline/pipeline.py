#!/usr/bin/env python3
"""
High-level orchestration pipeline for:
  ABM forward sims → summary stats → ODE MCMC → diagnostics → AIC comparison.

Usage (from repo root):
  python -m ODE_fitting.pipeline.pipeline \
    --config ODE_fitting/pipeline/pipeline_config.yaml
"""
import os
import yaml
import argparse
import pandas as pd

from .abm_runner import run_forward_sims
from .mcmc_runner import run_adaptive_mcmc
from .diagnostics_runner import run_diagnostics
from .aic_tools import write_aic_table

# NEW: timing + live status helpers
from .utils_logging import (
    make_run_dir, save_manifest,  # existing
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

    # Resolve roots
    results_root = os.path.abspath(cfg.get('results_root', 'ODE_fitting/results'))
    run_dir = make_run_dir(results_root)
    forward_dir = os.path.join(run_dir, 'forward_sims')

    # Initialise live status file and set running state
    init_run_status(run_dir)
    update_run_status(run_dir, {"status": "running"})

    timings = {}

    # 1) Forward ABM sims (timed + status)
    update_run_status(run_dir, {
        "sections": {"abm_forward": {"started": utcnow(), "finished": None}}
    })
    with timer() as t_abm:
        mean_csv = run_forward_sims(
            abm_cfg=cfg.get('abm', {}),
            defaults_yaml=cfg['abm'].get('defaults_yaml', 'ODE_fitting/scripts/scripts_defaults.yaml'),
            background_dir=cfg['abm'].get('background_dir', 'ODE_fitting/background'),
            run_forward_dir=forward_dir,
        )
    timings["abm_forward_seconds"] = round(t_abm.seconds, 3)
    update_run_status(run_dir, {
        "sections": {"abm_forward": {
            "finished": utcnow(),
            "duration_seconds": timings["abm_forward_seconds"]
        }}
    })

    # Load data for fitting
    df = pd.read_csv(mean_csv)
    times = df['step'].to_numpy()
    data_values = df[['num_clusters', 'mean_cluster_size', 'mean_squared_cluster_size']].to_numpy()

    # 2) Loop over selected models, run MCMC, run diagnostics
    mcmc_cfg = cfg.get('mcmc', {})
    models_meta_cfg = cfg.get('models_meta', {})
    model_results = []
    timings["models"] = {}

    with timer() as t_total:
        for model_key in cfg.get('models', []):
            model_out_dir = os.path.join(run_dir, f'model_{model_key}')
            os.makedirs(model_out_dir, exist_ok=True)

            # Status: model MCMC started
            update_run_status(run_dir, {
                "sections": {
                    "models": {
                        model_key: {
                            "mcmc": {"started": utcnow(), "finished": None, "current_block": 0},
                            "diagnostics": {"started": None, "finished": None}
                        }
                    }
                }
            })

            # Optional per-block progress reporter (block index, iters so far, last R-hat vector)
            def _progress(block_idx, total_iters_so_far, rhat_vec):
                update_run_status(run_dir, {
                    "sections": {
                        "models": {
                            model_key: {
                                "mcmc": {
                                    "current_block": int(block_idx),
                                    "total_iters_so_far": int(total_iters_so_far),
                                    "last_rhat": [float(x) for x in (rhat_vec or [])],
                                }
                            }
                        }
                    }
                })

            # MCMC timing
            with timer() as t_mcmc:
                res = run_adaptive_mcmc(
                    model_key=model_key,
                    times=times,
                    data_values=data_values,
                    mcmc_cfg=mcmc_cfg,
                    out_dir=model_out_dir,
                    model_meta_overrides=models_meta_cfg.get(model_key, {}),
                    random_seed=int(mcmc_cfg.get('random_seed', 12345)),
                    progress_callback=_progress,   # NEW: live updates each block
                )
            update_run_status(run_dir, {
                "sections": {
                    "models": {
                        model_key: {"mcmc": {
                            "finished": utcnow(),
                            "duration_seconds": round(t_mcmc.seconds, 3),
                            "n_params": int(res["n_params"]),
                            # "AIC": float(res["AIC"]),
                            # "max_loglik": float(res["max_loglik"]),
                        }}
                    }
                }
            })

            # Diagnostics timing + status
            diag_dir = os.path.join(model_out_dir, 'diagnostics')
            update_run_status(run_dir, {
                "sections": {"models": {model_key: {"diagnostics": {"started": utcnow()}}}}
            })
            with timer() as t_diag:
                run_diagnostics(
                    model_key=model_key,
                    chains=res['chains'],
                    flat_post=res['post_burn_flat'],
                    times=times,
                    data=data_values,
                    out_dir=diag_dir,
                    nsamples_ppc=int(mcmc_cfg.get('nsamples_ppc', 300)),
                    seed=int(mcmc_cfg.get('random_seed', 12345)),
                )
            update_run_status(run_dir, {
                "sections": {"models": {model_key: {"diagnostics": {
                    "finished": utcnow(),
                    "duration_seconds": round(t_diag.seconds, 3)
                }}}}
            })

            # Record timings per model
            model_results.append(res)
            timings["models"][model_key] = {
                "mcmc_seconds": round(t_mcmc.seconds, 3),
                "diagnostics_seconds": round(t_diag.seconds, 3),
                "total_seconds": round(t_mcmc.seconds + t_diag.seconds, 3),
                "n_params": res["n_params"],
                # "max_loglik": res["max_loglik"],
                # "AIC": res["AIC"],
            }

        # 3) AIC table (timed + status)
        # update_run_status(run_dir, {"sections": {"aic": {"started": utcnow(), "finished": None}}})
        # with timer() as t_aic:
        #     aic_csv = os.path.join(run_dir, 'model_comparison.csv')
        #     write_aic_table(model_results, aic_csv)
        # timings["aic_seconds"] = round(t_aic.seconds, 3)
        # update_run_status(run_dir, {"sections": {"aic": {
        #     "finished": utcnow(),
        #     "duration_seconds": timings["aic_seconds"]
        # }}})

    timings["pipeline_total_seconds"] = round(t_total.seconds, 3)

    # Input snapshot + timings + manifest
    save_run_input(run_dir, cfg)
    write_timings(run_dir, timings)
    save_manifest(run_dir, cfg, extra={
        'forward_means_stats': os.path.relpath(mean_csv, run_dir),
        'models_run': [m['model_key'] for m in model_results],
        # 'aic_csv': os.path.relpath(aic_csv, run_dir),
        'timings_yaml': 'timings.yaml',
        'abm_forward_seconds': timings.get('abm_forward_seconds'),
        'pipeline_total_seconds': timings.get('pipeline_total_seconds'),
        'run_status_yaml': 'run_status.yaml',
    })

    # Mark finished in live status
    update_run_status(run_dir, {
        "status": "finished",
        "pipeline_total": {
            "finished": utcnow(),
            "duration_seconds": timings["pipeline_total_seconds"]
        }
    })

    print(f"\nPipeline complete. Run folder: {run_dir}")


if __name__ == '__main__':
    main()