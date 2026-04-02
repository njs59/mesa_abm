#!/usr/bin/env python3
"""
Run kappa turning conditions × 3 movement models.
Results under: Multi_phase_analysis/results/kappa_conditions/
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

KAPPA_CONDS = [
    ("Default",   dict(k1=None, k2=None)),
    ("P1_0",      dict(k1=0,   k2=None)),
    ("P2_0",      dict(k1=None,k2=0)),
    ("P1P2_0",    dict(k1=0,   k2=0)),
    ("P1_1",      dict(k1=1,   k2=None)),
    ("P2_1",      dict(k1=None,k2=1)),
    ("P1P2_1",    dict(k1=1,   k2=1)),
    ("P1_2",      dict(k1=2,   k2=None)),
    ("P2_2",      dict(k1=None,k2=2)),
    ("P1P2_2",    dict(k1=2,   k2=2)),
]

PHASE_FLAGS = [
    ("phase2_only", dict(phase2_only=True,  allow_cross=True)),
    ("two_phase_interact", dict(phase2_only=False, allow_cross=True)),
    ("two_phase_no_interact", dict(phase2_only=False, allow_cross=False)),
]

def override_kappas(base_params: dict, *, k1, k2) -> dict:
    p = deepcopy(base_params)
    mv = p['movement_v2']['invasive']
    if k1 is not None:
        mv['phase1']['turning']['kappa'] = float(k1)
    if k2 is not None:
        mv['phase2']['turning']['kappa'] = float(k2)
    return p

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

    out_root = Path(__file__).resolve().parent / 'results' / 'kappa_conditions'
    traj_root = out_root / 'trajectories'
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for (kname, kinf) in KAPPA_CONDS:
        for (pname, pflags) in PHASE_FLAGS:
            cname = f'{kname}__{pname}'
            for rep in range(args.repeats):
                seed = args.seed0 + hash(cname) % 100000 + rep
                traj_csv = traj_root / cname / f'repeat_{rep:03d}.csv'
                overrides = override_kappas(base_params, **kinf)
                tasks.append((cname, pflags, overrides, rep, seed, traj_csv))

    workers = 1 if args.serial else (args.workers or max(1, (os.cpu_count() or 2)-1))
    print(f'Running {len(tasks)} simulations with workers={workers}')

    ts_all, rep_all = [], []
    if workers == 1:
        for cname, flags, overrides, rep, seed, traj_csv in tasks:
            ts_rows, summary = run_single_sim(
                condition=cname, rep=rep, seed=seed, steps=steps,
                base_params=base_params, overrides=overrides,
                phase2_only=flags['phase2_only'], allow_cross=flags['allow_cross'],
                n_init=n_init, init_size=init_size, traj_csv=traj_csv,
            )
            ts_all.extend(ts_rows)
            rep_all.append(summary)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_single_sim,
                              condition=cname, rep=rep, seed=seed, steps=steps,
                              base_params=base_params, overrides=overrides,
                              phase2_only=flags['phase2_only'], allow_cross=flags['allow_cross'],
                              n_init=n_init, init_size=init_size, traj_csv=traj_csv)
                    for (cname, flags, overrides, rep, seed, traj_csv) in tasks]
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