#!/usr/bin/env python3
"""
High-level orchestration pipeline for:
  ABM forward sims → summary stats → ODE MCMC → diagnostics → AIC comparison.

Usage (from repo root):
  python -m ODE_fitting.pipeline.pipeline \
    --config ODE_fitting/pipeline/pipeline_config.yaml

Notes:
- We keep per-model parallelism exactly as in your scripts:
  * ABM uses multiprocessing.Pool via your forward-sims script
  * MCMC uses PINTS AdaptiveCovarianceMCMC with set_parallel(True)
- No extra cross-model parallelism is added (as requested).
"""
import os
import yaml
import argparse
import pandas as pd

from .abm_runner import run_forward_sims
from .mcmc_runner import run_adaptive_mcmc
from .diagnostics_runner import run_diagnostics
from .aic_tools import write_aic_table
from .utils_logging import make_run_dir, save_manifest


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

    # 1) Forward ABM sims (Phase 2 only script)
    mean_csv = run_forward_sims(
        abm_cfg=cfg.get('abm', {}),
        defaults_yaml=cfg['abm'].get('defaults_yaml', 'ODE_fitting/scripts/scripts_defaults.yaml'),
        background_dir=cfg['abm'].get('background_dir', 'ODE_fitting/background'),
        run_forward_dir=forward_dir,
    )

    # Load data for fitting
    df = pd.read_csv(mean_csv)
    times = df['step'].to_numpy()
    data_values = df[['num_clusters', 'mean_cluster_size', 'mean_squared_cluster_size']].to_numpy()

    # 2) Loop over selected models, run MCMC, run diagnostics
    mcmc_cfg = cfg.get('mcmc', {})
    # Optional per-model meta overrides from config (priors / initials / names)
    models_meta_cfg = cfg.get('models_meta', {})
    model_results = []

    for model_key in cfg.get('models', []):
        model_out_dir = os.path.join(run_dir, f'model_{model_key}')
        os.makedirs(model_out_dir, exist_ok=True)

        res = run_adaptive_mcmc(
            model_key=model_key,
            times=times,
            data_values=data_values,
            mcmc_cfg=mcmc_cfg,
            out_dir=model_out_dir,
            model_meta_overrides=models_meta_cfg.get(model_key, {}),
        )
        model_results.append(res)

        # Diagnostics and PPC
        diag_dir = os.path.join(model_out_dir, 'diagnostics')
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

    # 3) AIC table
    aic_csv = os.path.join(run_dir, 'model_comparison.csv')
    write_aic_table(model_results, aic_csv)

    # Save manifest
    save_manifest(run_dir, cfg, extra={
        'forward_means_stats': os.path.relpath(mean_csv, run_dir),
        'models_run': [m['model_key'] for m in model_results],
        'aic_csv': os.path.relpath(aic_csv, run_dir),
    })

    print(f"\nPipeline complete. Run folder: {run_dir}")


if __name__ == '__main__':
    main()