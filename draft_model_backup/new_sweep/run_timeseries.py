
# new_sweep/run_timeseries.py
import argparse
import os
import pandas as pd
from typing import Optional, Dict, Any
from new_sweep_old.metrics import summary_time_series, speed_size_slope
from clusters_abm.clusters_model import ClustersModel
from clusters_abm.utils import DEFAULTS


def apply_overrides(params: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import copy
    params = copy.deepcopy(params)
    if overrides:
        for k, v in overrides.items():
            parts = k.split('.')
            d = params
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = v
    return params


def run_timeseries(steps: int, seed: int, overrides: Optional[Dict[str, Any]] = None, out_csv: Optional[str] = None) -> pd.DataFrame:
    params = apply_overrides(DEFAULTS, overrides)
    params.setdefault('init', {})
    params['init'].setdefault('n_clusters', 1000)
    params['init'].setdefault('size', 1)
    params['init'].setdefault('phenotype', 'proliferative')

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
    parser = argparse.ArgumentParser(description='Run one time-series scenario with new_results defaults')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--init_phenotype', type=str, choices=['proliferative','invasive'], default='proliferative')
    parser.add_argument('--dt', type=float, default=0.5)
    parser.add_argument('--out_csv', type=str, default='new_results/timeseries/timeseries_demo.csv')
    args = parser.parse_args()

    overrides = {
        'merge.prob_contact_merge': 0.8,
        'phenotypes.invasive.speed_base': 3.0,
        'phenotypes.proliferative.speed_base': 1.2,
        'time.dt': float(args.dt),
        'init.phenotype': str(args.init_phenotype),
        'init.n_clusters': 1000,
        'init.size': 1,
    }
    os.makedirs('new_results/timeseries', exist_ok=True)
    df = run_timeseries(steps=args.steps, seed=args.seed, overrides=overrides, out_csv=args.out_csv)
    print(df.head())
