#!/usr/bin/env python3
"""
Run 3 movement conditions for invasive phenotype and save trajectories + time series summaries.
- phase2_only
- two_phase_interact
- two_phase_no_interact
Results under: Multi_phase_analysis/results/three_conditions/
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from abm.utils import DEFAULTS
from Multi_phase_analysis.analysis_utils import run_single_sim

COND_DEF = [
    ("phase2_only", dict(phase2_only=True,  allow_cross=True)),
    ("two_phase_interact", dict(phase2_only=False, allow_cross=True)),
    ("two_phase_no_interact", dict(phase2_only=False, allow_cross=False)),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=100)
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--seed0', type=int, default=123)
    ap.add_argument('--n_init', type=int, default=None)
    ap.add_argument('--init_size', type=int, default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--serial', action='store_true')
    args = ap.parse_args()

    base_params = deepcopy(DEFAULTS)
    steps = args.steps or int(base_params['time']['steps'])
    n_init = args.n_init or int(base_params['init']['n_clusters'])
    init_size = args.init_size or int(base_params['init']['size'])

    out_root = Path(__file__).resolve().parent / 'results' / 'three_conditions'
    traj_root = out_root / 'trajectories'
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for cidx, (cname, flags) in enumerate(COND_DEF):
        for rep in range(args.repeats):
            seed = args.seed0 + 10000*cidx + rep
            traj_csv = traj_root / cname / f'repeat_{rep:03d}.csv'
            tasks.append((cname, flags, rep, seed, traj_csv))

    workers = 1 if args.serial else (args.workers or max(1, (os.cpu_count() or 2)-1))
    print(f'Running {len(tasks)} runs ({len(COND_DEF)} conditions × {args.repeats} repeats) with workers={workers}')

    ts_all, rep_all = [], []
    if workers == 1:
        for cname, flags, rep, seed, traj_csv in tasks:
            ts_rows, summary = run_single_sim(
                condition=cname, rep=rep, seed=seed, steps=steps,
                base_params=base_params, overrides={},
                phase2_only=flags['phase2_only'], allow_cross=flags['allow_cross'],
                n_init=n_init, init_size=init_size, traj_csv=traj_csv,
            )
            ts_all.extend(ts_rows)
            rep_all.append(summary)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_single_sim,
                              condition=cname, rep=rep, seed=seed, steps=steps,
                              base_params=base_params, overrides={},
                              phase2_only=flags['phase2_only'], allow_cross=flags['allow_cross'],
                              n_init=n_init, init_size=init_size, traj_csv=traj_csv)
                    for (cname, flags, rep, seed, traj_csv) in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                ts_rows, summary = fut.result()
                ts_all.extend(ts_rows)
                rep_all.append(summary)
                if i % max(1,len(futs)//10) == 0:
                    print(f'Completed {i}/{len(futs)}')

    pd.DataFrame(ts_all).to_csv(out_root/'summary_timeseries.csv', index=False)
    pd.DataFrame(rep_all).to_csv(out_root/'summary_repeats.csv', index=False)
    print('Saved:')
    print(' -', out_root/'summary_timeseries.csv')
    print(' -', out_root/'summary_repeats.csv')

if __name__ == '__main__':
    main()