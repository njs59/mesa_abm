
# sweeps/run.py
import copy
import numpy as np
from typing import Optional, Dict, Any

from clusters_abm.clusters_model import ClustersModel
from clusters_abm.utils import DEFAULTS


def run_once(
    steps: int,
    seed: int,
    base_params: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
):
    """
    Run a single simulation and return a dictionary of summary metrics.
    - steps: number of model steps to run
    - seed: RNG seed
    - base_params: params dict to start from (DEFAULTS if None)
    - overrides: nested values to override (e.g., {"merge.prob_contact_merge": 0.6})
    """
    params = copy.deepcopy(base_params or DEFAULTS)

    # Apply dot-path overrides such as "merge.prob_contact_merge": 0.6
    if overrides:
        for k, v in overrides.items():
            parts = k.split('.')
            d = params
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = v

    model = ClustersModel(params=params, seed=seed)

    # Run
    for _ in range(int(steps)):
        model.step()

    # Final snapshot metrics from logs (robust to empty frames)
    final_sizes = model.size_log[-1] if len(model.size_log) > 0 else []
    final_speeds = model.speed_log[-1] if len(model.speed_log) > 0 else []

    n_final = int(len(final_sizes))
    mean_size_final = float(np.mean(final_sizes)) if n_final > 0 else 0.0
    mean_speed_final = float(np.mean(final_speeds)) if len(final_speeds) > 0 else 0.0
    total_cells_final = float(np.sum(final_sizes)) if n_final > 0 else 0.0

    # Time-series summaries (means over all frames)
    if model.size_log:
        n_over_time = np.asarray([len(s) for s in model.size_log], dtype=int)
        mean_size_over_time = np.asarray([
            float(np.mean(s)) if len(s) > 0 else 0.0 for s in model.size_log
        ], dtype=float)
    else:
        n_over_time = np.array([n_final])
        mean_size_over_time = np.array([mean_size_final])

    if model.speed_log:
        mean_speed_over_time = np.asarray([
            float(np.mean(sp)) if len(sp) > 0 else 0.0 for sp in model.speed_log
        ], dtype=float)
    else:
        mean_speed_over_time = np.array([mean_speed_final])

    metrics = {
        "n_final": n_final,
        "mean_size_final": mean_size_final,
        "mean_speed_final": mean_speed_final,
        "total_cells_final": total_cells_final,
        "n_mean_over_time": float(np.mean(n_over_time)),
        "mean_size_over_time": float(np.mean(mean_size_over_time)),
        "mean_speed_over_time": float(np.mean(mean_speed_over_time)),
    }
    return metrics
