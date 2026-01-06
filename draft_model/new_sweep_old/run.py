
# new_sweep/run.py
import copy
import numpy as np
from typing import Optional, Dict, Any
from clusters_abm.clusters_model import ClustersModel
from clusters_abm.utils import DEFAULTS

def run_once(steps: int, seed: int, base_params: Optional[Dict[str, Any]] = None, overrides: Optional[Dict[str, Any]] = None):
    params = copy.deepcopy(base_params or DEFAULTS)
    if overrides:
        for k, v in overrides.items():
            parts = k.split('.')
            d = params
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = v
    model = ClustersModel(params=params, seed=seed)
    for _ in range(int(steps)):
        model.step()
    final_sizes = model.size_log[-1] if len(model.size_log) > 0 else []
    final_speeds = model.speed_log[-1] if len(model.speed_log) > 0 else []
    n_final = int(len(final_sizes))
    mean_size_final = float(np.mean(final_sizes)) if n_final > 0 else 0.0
    mean_speed_final = float(np.mean(final_speeds)) if len(final_speeds) > 0 else 0.0
    total_cells_final = float(np.sum(final_sizes)) if n_final > 0 else 0.0
    if model.size_log:
        n_over_time = np.asarray([len(s) for s in model.size_log], dtype=int)
        mean_size_over_time = np.asarray([float(np.mean(s)) if len(s) > 0 else 0.0 for s in model.size_log], dtype=float)
    else:
        n_over_time = np.array([n_final])
        mean_size_over_time = np.array([mean_size_final])
    if model.speed_log:
        mean_speed_over_time = np.asarray([float(np.mean(sp)) if len(sp) > 0 else 0.0 for sp in model.speed_log], dtype=float)
    else:
        mean_speed_over_time = np.array([mean_speed_final])
    return {
        'n_final': n_final,
        'mean_size_final': mean_size_final,
        'mean_speed_final': mean_speed_final,
        'total_cells_final': total_cells_final,
        'n_mean_over_time': float(np.mean(n_over_time)),
        'mean_size_over_time': float(np.mean(mean_size_over_time)),
        'mean_speed_over_time': float(np.mean(mean_speed_over_time)),
    }
