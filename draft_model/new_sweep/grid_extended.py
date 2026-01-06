
# new_sweep/grid_extended.py
import argparse
import itertools
import os
import multiprocessing as mp
import pandas as pd
from new_sweep_old.run import run_once
from clusters_abm.utils import DEFAULTS


def _worker(args):
    steps, base_seed, idx, overrides = args
    seed = int(base_seed + idx)
    return run_once(steps=steps, seed=seed, base_params=DEFAULTS, overrides=overrides)


def main():
    parser = argparse.ArgumentParser(description='Extended parameter sweeps for clusters ABM')
    parser.add_argument('--steps', type=int, default=int(DEFAULTS['time']['steps']))
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--merge_prob', type=float, nargs='+', default=[DEFAULTS['merge']['prob_contact_merge']])
    parser.add_argument('--dt', type=float, nargs='+', default=[DEFAULTS['time']['dt']])
    parser.add_argument('--inv_speed', type=float, nargs='+', default=[DEFAULTS['phenotypes']['invasive']['speed_base']])
    parser.add_argument('--pro_speed', type=float, nargs='+', default=[DEFAULTS['phenotypes']['proliferative']['speed_base']])
    parser.add_argument('--inv_exp', type=float, nargs='+', default=[DEFAULTS['phenotypes']['invasive'].get('speed_size_exp', 0.0)])
    parser.add_argument('--pro_exp', type=float, nargs='+', default=[DEFAULTS['phenotypes']['proliferative'].get('speed_size_exp', 0.0)])
    parser.add_argument('--inv_adh', type=float, nargs='+', default=[DEFAULTS['phenotypes']['invasive']['adhesion']])
    parser.add_argument('--pro_adh', type=float, nargs='+', default=[DEFAULTS['phenotypes']['proliferative']['adhesion']])
    parser.add_argument('--inv_prolif', type=float, nargs='+', default=[DEFAULTS['phenotypes']['invasive']['prolif_rate']])
    parser.add_argument('--pro_prolif', type=float, nargs='+', default=[DEFAULTS['phenotypes']['proliferative']['prolif_rate']])
    parser.add_argument('--inv_frag', type=float, nargs='+', default=[DEFAULTS['phenotypes']['invasive']['fragment_rate']])
    parser.add_argument('--pro_frag', type=float, nargs='+', default=[DEFAULTS['phenotypes']['proliferative']['fragment_rate']])
    parser.add_argument('--inv_frag_exp', type=float, nargs='+', default=[DEFAULTS['phenotypes']['invasive'].get('frag_size_exp', 0.0)])
    parser.add_argument('--pro_frag_exp', type=float, nargs='+', default=[DEFAULTS['phenotypes']['proliferative'].get('frag_size_exp', 0.0)])
    # parser.add_argument('--init_phenotype', type=str, choices=['proliferative','invasive'], default='proliferative')

    parser.add_argument('--init_phenotype', type=str, nargs='+', choices=['proliferative', 'invasive'],
            default=['proliferative'],   # can pass one or both
            help='Initial phenotype for the 1,000 singletons; supports multiple values to sweep'
    )

    parser.add_argument('--out_csv', type=str, default='new_results/sweeps_extended.csv')
    args = parser.parse_args()

    combos = list(itertools.product(
        args.merge_prob, args.dt,
        args.inv_speed, args.pro_speed,
        args.inv_exp, args.pro_exp,
        args.inv_adh, args.pro_adh,
        args.inv_prolif, args.pro_prolif,
        args.inv_frag, args.pro_frag,
        args.inv_frag_exp, args.pro_frag_exp,
        args.init_phenotype
    ))

    work = []
    base_seed = 101
    for (merge_prob, dt, inv_speed, pro_speed, inv_exp, pro_exp, inv_adh, pro_adh,
         inv_prolif, pro_prolif, inv_frag, pro_frag, inv_frag_exp, pro_frag_exp, init_phenotype) in combos:
        for rep in range(args.seeds):
            overrides = {
                'merge.prob_contact_merge': float(merge_prob),
                'time.dt': float(dt),
                'phenotypes.invasive.speed_base': float(inv_speed),
                'phenotypes.proliferative.speed_base': float(pro_speed),
                'phenotypes.invasive.speed_size_exp': float(inv_exp),
                'phenotypes.proliferative.speed_size_exp': float(pro_exp),
                'phenotypes.invasive.adhesion': float(inv_adh),
                'phenotypes.proliferative.adhesion': float(pro_adh),
                'phenotypes.invasive.prolif_rate': float(inv_prolif),
                'phenotypes.proliferative.prolif_rate': float(pro_prolif),
                'phenotypes.invasive.fragment_rate': float(inv_frag),
                'phenotypes.proliferative.fragment_rate': float(pro_frag),
                'phenotypes.invasive.frag_size_exp': float(inv_frag_exp),
                'phenotypes.proliferative.frag_size_exp': float(pro_frag_exp),
                'init.phenotype': str(init_phenotype),
                'init.n_clusters': 1000,
                'init.size': 1,
            }
            work.append((args.steps, base_seed, rep, overrides))

    if args.jobs > 1:
        jobs = max(1, int(args.jobs))
        with mp.get_context('spawn').Pool(processes=jobs) as pool:
            results = pool.map(_worker, work)
    else:
        results = [_worker(w) for w in work]

    rows = []
    i = 0
    for (merge_prob, dt, inv_speed, pro_speed, inv_exp, pro_exp, inv_adh, pro_adh,
         inv_prolif, pro_prolif, inv_frag, pro_frag, inv_frag_exp, pro_frag_exp, init_phenotype) in combos:
        for rep in range(args.seeds):
            metrics = results[i]; i += 1
            rows.append({
                'merge_prob': float(merge_prob), 'dt': float(dt),
                'inv_speed': float(inv_speed), 'pro_speed': float(pro_speed),
                'inv_exp': float(inv_exp), 'pro_exp': float(pro_exp),
                'inv_adh': float(inv_adh), 'pro_adh': float(pro_adh),
                'inv_prolif': float(inv_prolif), 'pro_prolif': float(pro_prolif),
                'inv_frag': float(inv_frag), 'pro_frag': float(pro_frag),
                'inv_frag_exp': float(inv_frag_exp), 'pro_frag_exp': float(pro_frag_exp),
                'init_phenotype': str(init_phenotype),
                'rep': int(rep),
                **metrics,
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote extended sweep results -> {args.out_csv}")
    print(f"Rows: {len(df)}; unique settings: {len(combos)}; replicates: {args.seeds}")

if __name__ == '__main__':
    main()
