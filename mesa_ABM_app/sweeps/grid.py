
# sweeps/grid.py
import argparse
import itertools
import os
import multiprocessing as mp
import pandas as pd

from sweeps.run import run_once
from clusters_abm.utils import DEFAULTS


def _worker(args):
    steps, base_seed, idx, overrides = args
    seed = int(base_seed + idx)
    return run_once(steps=steps, seed=seed, base_params=DEFAULTS, overrides=overrides)


def main():
    parser = argparse.ArgumentParser(description="Parameter sweeps for clusters ABM")
    parser.add_argument("--steps", type=int, default=int(DEFAULTS["time"]["steps"]), help="Simulation steps per run")
    parser.add_argument("--seeds", type=int, default=5, help="Replicates per setting")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers (<= CPU count)")

    # Grids (use nargs='+' to accept many values)
    parser.add_argument("--merge_prob", type=float, nargs="+", default=[DEFAULTS["merge"]["prob_contact_merge"]])
    parser.add_argument("--invasive_speed", type=float, nargs="+", default=[DEFAULTS["phenotypes"]["invasive"]["speed_base"]])
    parser.add_argument("--proliferative_speed", type=float, nargs="+", default=[DEFAULTS["phenotypes"]["proliferative"]["speed_base"]])
    parser.add_argument("--dt", type=float, nargs="+", default=[DEFAULTS["time"]["dt"]])

    parser.add_argument("--out_csv", type=str, default="results/sweeps.csv")

    args = parser.parse_args()

    # Build parameter combinations
    combos = list(itertools.product(args.merge_prob, args.invasive_speed, args.proliferative_speed, args.dt))

    rows = []
    work = []
    base_seed = 42

    for (merge_prob, inv_speed, prolif_speed, dt) in combos:
        for rep in range(args.seeds):
            overrides = {
                "merge.prob_contact_merge": float(merge_prob),
                "phenotypes.invasive.speed_base": float(inv_speed),
                "phenotypes.proliferative.speed_base": float(prolif_speed),
                "time.dt": float(dt),
            }
            work.append((args.steps, base_seed, rep, overrides))

    # Run (parallel or serial)
    if args.jobs > 1:
        jobs = max(1, int(args.jobs))
        with mp.get_context("spawn").Pool(processes=jobs) as pool:
            results = pool.map(_worker, work)
    else:
        results = [_worker(w) for w in work]

    # Collect rows with parameters re-attached
    i = 0
    for (merge_prob, inv_speed, prolif_speed, dt) in combos:
        for rep in range(args.seeds):
            metrics = results[i]
            i += 1
            rows.append({
                "merge_prob": float(merge_prob),
                "invasive_speed": float(inv_speed),
                "proliferative_speed": float(prolif_speed),
                "dt": float(dt),
                "rep": int(rep),
                **metrics,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(f"Wrote sweep results -> {args.out_csv}")
    print(f"Rows: {len(df)}; unique settings: {len(combos)}; replicates per setting: {args.seeds}")


if __name__ == "__main__":
    main()
