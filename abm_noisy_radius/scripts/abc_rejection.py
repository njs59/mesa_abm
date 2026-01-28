
"""
Simple ABC-rejection for movement parameters across multiple movement models.
- Supports 'constant' speed and 'distribution' with 'lognorm' or 'gamma'.
- Uses summary stats: S0 (#clusters at final), S1 (mean size final), S2 (var size final),
  and mean NND at final (no wrap) + time-averaged mean NND.
Usage:
  python scripts/abc_rejection.py --observed data/observed_summary.json --out results/abc --n 200 --keep 40
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from abm.clusters_model import ClustersModel
from abm import utils as U


def summaries_from_model(model) -> dict:
    # compute from logs directly
    sizes = model.size_log[-1]
    S0 = len(sizes)
    S1 = float(np.mean(sizes)) if S0>0 else np.nan
    S2 = float(np.var(sizes)) if S0>0 else np.nan
    # NND final and time-avg (no wrap)
    def nnd_no_wrap(P):
        n = P.shape[0]
        if n<=1:
            return np.array([])
        dmin = []
        for i in range(n):
            dx = P[i,0]-P[:,0]; dy = P[i,1]-P[:,1]
            d = np.hypot(dx,dy); d[i]=np.inf
            dmin.append(d.min())
        return np.array(dmin)
    nnd_final = nnd_no_wrap(model.pos_log[-1])
    mNND_final = float(np.nanmean(nnd_final)) if nnd_final.size>0 else np.nan
    mNND_time = []
    for P in model.pos_log:
        mNND_time.append(np.nanmean(nnd_no_wrap(P)) if P.size>0 else np.nan)
    mNND_mean = float(np.nanmean(mNND_time))
    return {"S0":S0, "S1":S1, "S2":S2, "NND_final":mNND_final, "NND_mean":mNND_mean}


def distance(sim: dict, obs: dict, weights=None) -> float:
    keys = ["S0","S1","S2","NND_final","NND_mean"]
    if weights is None:
        weights = {k:1.0 for k in keys}
    d2 = 0.0
    for k in keys:
        if np.isnan(sim[k]) or np.isnan(obs.get(k, np.nan)):
            continue
        d2 += weights[k] * ( (sim[k]-obs[k])**2 )
    return float(np.sqrt(d2))


def sample_params(rng: np.random.Generator, model_name: str) -> dict:
    p = dict(U.DEFAULTS)
    # movement model choice
    if model_name == 'constant':
        p['movement']['mode'] = 'constant'
        p['phenotypes']['proliferative']['speed_base'] = float(rng.uniform(0.3, 3.0))
    elif model_name == 'lognorm':
        p['movement']['mode'] = 'distribution'
        p['movement']['distribution'] = 'lognorm'
        p['movement']['dist_params'] = { 's': float(rng.uniform(0.2, 1.2)), 'scale': float(rng.uniform(0.5, 4.0)) }
    elif model_name == 'gamma':
        p['movement']['mode'] = 'distribution'
        p['movement']['distribution'] = 'gamma'
        p['movement']['dist_params'] = { 'a': float(rng.uniform(1.0, 5.0)), 'scale': float(rng.uniform(0.2, 3.0)) }
    # direction
    p['movement']['direction'] = rng.choice(['isotropic','persistent'])
    if p['movement']['direction']=='persistent':
        p['movement']['heading_sigma'] = float(rng.uniform(0.05, 0.4))
    # merge strength prior
    p['merge']['strength'] = float(rng.uniform(0.2, 0.9))
    # radius noise prior (persistent per agent)
    p['radius_noise']['enable'] = True
    p['radius_noise']['sigma'] = float(rng.uniform(0.1, 0.6))
    p['radius_noise']['preserve'] = 'area'
    p['radius_noise']['merge_combine'] = rng.choice(['max','weighted'])
    p['radius_noise']['apply_after_merge'] = True
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--observed', required=True, help='JSON with observed summaries (S0,S1,S2,NND_final,NND_mean)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=200, help='number of prior samples')
    ap.add_argument('--keep', type=int, default=40, help='accept top K by distance')
    ap.add_argument('--model', choices=['constant','lognorm','gamma','all'], default='all')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()

    with open(args.observed,'r') as f:
        obs = json.load(f)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    models = ['constant','lognorm','gamma'] if args.model=='all' else [args.model]
    results = {}
    for mdl in models:
        sims = []
        dists = []
        params_list = []
        for i in range(args.n):
            p = sample_params(rng, mdl)
            M = ClustersModel(params=p, seed=int(rng.integers(1,2**31-1)))
            for _ in range(args.steps):
                M.step()
            sim_sum = summaries_from_model(M)
            dist = distance(sim_sum, obs)
            sims.append(sim_sum); dists.append(dist); params_list.append(p)
        order = np.argsort(np.array(dists))
        keep_idx = order[:args.keep]
        results[mdl] = {
            'accepted_params': [params_list[i] for i in keep_idx],
            'accepted_summaries': [sims[i] for i in keep_idx],
            'accepted_distances': [float(dists[i]) for i in keep_idx]
        }
        # Save CSV of accepted summaries
        import pandas as pd
        df = pd.DataFrame(results[mdl]['accepted_summaries'])
        df['distance'] = results[mdl]['accepted_distances']
        df.to_csv(out_dir / f'accepted_{mdl}.csv', index=False)

    with open(out_dir / 'abc_results.json','w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved ABC results in {out_dir}")

if __name__ == '__main__':
    main()
