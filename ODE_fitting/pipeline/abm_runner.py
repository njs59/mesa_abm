import os
import sys

# Import your existing ABM forward sims script (as a module, not a subprocess)
from ODE_fitting.scripts import run_forward_sims_phase_2 as fwd


def _as_argv(args_dict):
    argv = [fwd.__file__]
    for k, v in args_dict.items():
        if isinstance(v, bool):
            if v:
                argv.append(f"--{k}")
            continue
        argv.extend([f"--{k.replace('_','-')}", str(v)])
    return argv


def run_forward_sims(abm_cfg: dict, defaults_yaml: str, background_dir: str, run_forward_dir: str) -> str:
    """
    Runs forward ABM simulations using your existing script, saving outputs
    into ``run_forward_dir``. Returns the path to the mean stats CSV.
    """
    os.makedirs(run_forward_dir, exist_ok=True)

    best_model_json = os.path.join(background_dir, 'best_model.json')
    mean_init_json = os.path.join(background_dir, 'mean_initial_clusters.json')

    arg_map = {
        'start-step': abm_cfg.get('start_step', 71),
        'n-runs': abm_cfg.get('n_runs', 100),
        'n-workers': abm_cfg.get('n_workers', 8),
        'seed': abm_cfg.get('seed', 42),
        'defaults': os.path.abspath(defaults_yaml),
        'best-model': best_model_json,
        'mean-clusters': mean_init_json,
        'results-dir': run_forward_dir,
    }

    # Drive the module as if it were called via CLI
    _bak = sys.argv[:]
    try:
        sys.argv = _as_argv(arg_map)
        fwd.main()
    finally:
        sys.argv = _bak

    mean_csv = os.path.join(run_forward_dir, 'forward_means_stats.csv')
    if not os.path.exists(mean_csv):
        raise FileNotFoundError(f"Expected forward means at {mean_csv} not found")
    return mean_csv