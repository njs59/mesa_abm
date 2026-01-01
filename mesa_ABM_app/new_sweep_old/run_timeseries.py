
# new_sweep/run_timeseries.py
import copy
import os
import pandas as pd
from typing import Optional, Dict, Any
from clusters_abm.clusters_model import ClustersModel
from clusters_abm.utils import DEFAULTS
from new_sweep_old.metrics import summary_time_series, speed_size_slope

def apply_overrides(params: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = copy.deepcopy(params)
    if overrides:
        for k, v in overrides.items():
            parts = k.split('.')
            d = params
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = v
    return params

def run_timeseries(steps: int, seed: int, base_params: Optional[Dict[str, Any]] = None, overrides: Optional[Dict[str, Any]] = None,
                   out_csv: Optional[str] = None) -> pd.DataFrame:
    params = apply_overrides(base_params or DEFAULTS, overrides)
    m = ClustersModel(params=params, seed=seed)
    for _ in range(int(steps)):
        m.step()
    summ = summary_time_series(m)
    df = pd.DataFrame({
        'time_min': summ['time_min'],
        'S0': summ['S0'],
        'S1': summ['S1'],
        'S2': summ['S2'],
        'total_cells': summ['total_cells'],
        'mean_speed': summ['mean_speed'],
    })
    df.attrs['speed_size_slope'] = speed_size_slope(m)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df

if __name__ == '__main__':
    df = run_timeseries(steps=200, seed=2026, overrides={
        'merge.prob_contact_merge': 0.8,
        'phenotypes.invasive.speed_base': 3.0,
        'phenotypes.proliferative.speed_base': 1.2,
        'time.dt': 0.5,
    }, out_csv='results/timeseries_demo.csv')
    print(df.head())
