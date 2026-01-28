
"""
Run a single simulation with the updated ABM and save CSV logs.
Usage:
  python scripts/run_sim.py --out results/run1 --steps 300 \
     --movement constant --direction isotropic --speed 1.0 \
     --merge_strength 0.6 --sigma 0.35 --preserve area
"""
import argparse, os, json
from pathlib import Path
import numpy as np
from abm.clusters_model import ClustersModel
from abm import utils as U

def build_params(args):
    p = dict(U.DEFAULTS)
    p["time"]["steps"] = int(args.steps)
    p["movement"]["mode"] = "constant" if args.movement == "constant" else "distribution"
    p["movement"]["direction"] = args.direction
    p["phenotypes"]["proliferative"]["speed_base"] = float(args.speed)
    if args.movement == "distribution":
        if args.dist == "lognorm":
            p["movement"]["distribution"] = "lognorm"
            p["movement"]["dist_params"] = {"s": args.s, "scale": args.scale}
        elif args.dist == "gamma":
            p["movement"]["distribution"] = "gamma"
            p["movement"]["dist_params"] = {"a": args.a, "scale": args.scale}
    p["merge"]["strength"] = float(args.merge_strength)
    p["radius_noise"]["enable"] = True
    p["radius_noise"]["sigma"] = float(args.sigma)
    p["radius_noise"]["preserve"] = args.preserve
    p["radius_noise"]["merge_combine"] = args.merge_combine
    p["radius_noise"]["apply_after_merge"] = (not args.keep_volume_conserving)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--movement', choices=['constant','distribution'], default='constant')
    ap.add_argument('--direction', choices=['isotropic','persistent'], default='isotropic')
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--dist', choices=['lognorm','gamma'], default='lognorm')
    ap.add_argument('--s', type=float, default=0.6)
    ap.add_argument('--a', type=float, default=2.0)
    ap.add_argument('--scale', type=float, default=2.0)
    ap.add_argument('--merge_strength', type=float, default=0.6)
    ap.add_argument('--sigma', type=float, default=0.35)
    ap.add_argument('--preserve', choices=['radius','area'], default='area')
    ap.add_argument('--merge_combine', choices=['max','weighted','self'], default='max')
    ap.add_argument('--keep_volume_conserving', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    p = build_params(args)
    model = ClustersModel(params=p, seed=args.seed)
    for _ in range(args.steps):
        model.step()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    U.export_timeseries_state(model, out_csv=str(out_dir / 'state_timeseries.csv'))
    with open(out_dir / 'params.json', 'w') as f:
        json.dump(p, f, indent=2)
    print(f"Saved results to {out_dir}")

if __name__ == '__main__':
    main()
