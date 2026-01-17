#!/usr/bin/env python3
"""
ABC Posterior Analysis (fixed initial n via CLI --n0)
- Same as with_init version but overrides initial cluster count.
- Time-series PPC aligns row0 to t_start and simulates exactly t_start + N steps.
Mesa target: 2.2.4
"""
import argparse
from pathlib import Path
import glob as globlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pyabc

from clusters_abm.clusters_model import ClustersModel
from clusters_abm.utils import DEFAULTS

# ---------------------- summaries ----------------------
def compute_summary(model):
    ids, pos, radii, sizes, speeds = model._log_alive_snapshot()
    n = len(sizes)
    if n == 0:
        return {"S0":0.0, "S1":0.0, "S2":0.0}
    return {"S0": float(n), "S1": float(np.mean(sizes)), "S2": float(np.mean(sizes**2))}

# ---------------------- mapping ------------------------
def _set_nested(base: dict, dotted: str, value):
    keys = dotted.split(".")
    d = base
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value

def build_speed_params(speed_dist: str, particle: dict):
    if speed_dist == "constant":
        return {}
    if speed_dist == "lognorm":
        mu = float(particle.get("speed_meanlog", 1.0))
        sd = float(particle.get("speed_sdlog", 0.7))
        return {"s": sd, "scale": float(np.exp(mu))}
    if speed_dist == "gamma":
        shape = float(particle.get("speed_shape", 2.0))
        scale = float(particle.get("speed_scale", 1.0))
        return {"a": shape, "scale": scale}
    if speed_dist == "weibull":
        shape = float(particle.get("speed_shape", 2.0))
        scale = float(particle.get("speed_scale", 2.0))
        return {"c": shape, "scale": scale}
    raise ValueError(f"Unknown speed_dist {speed_dist}")

def particle_to_params(defaults, particle, motion, speed_dist, n0: int):
    params = {
        "space": dict(defaults["space"]),
        "time": dict(defaults["time"]),
        "physics": dict(defaults["physics"]),
        "phenotypes": {
            "proliferative": dict(defaults["phenotypes"]["proliferative"]),
            "invasive": dict(defaults["phenotypes"]["invasive"]),
        },
        "merge": dict(defaults["merge"]),
        "init": dict(defaults["init"]),
        "movement": dict(defaults["movement"]),
    }
    params["movement"]["direction"] = motion
    if speed_dist == "constant":
        params["movement"]["mode"] = "constant"
        params["movement"].pop("distribution", None)
        params["movement"].pop("dist_params", None)
    else:
        params["movement"]["mode"] = "distribution"
        params["movement"]["distribution"] = speed_dist
        params["movement"]["dist_params"] = build_speed_params(speed_dist, particle)
    if motion == "persistent":
        hs = float(particle.get("heading_sigma", params["movement"].get("heading_sigma", 0.25)))
        params["movement"]["heading_sigma"] = max(0.0, hs)
    else:
        params["movement"].pop("heading_sigma", None)

    mapping = {
        "prolif_rate": "phenotypes.proliferative.prolif_rate",
        "adhesion": "phenotypes.proliferative.adhesion",
        "fragment_rate": "phenotypes.proliferative.fragment_rate",
        "merge_prob": "merge.prob_contact_merge",
    }
    for k, v in particle.items():
        if k.startswith("speed_") or k == "heading_sigma":
            continue
        if ("rate" in k or "prob" in k) and v < 0:
            v = 0.0
        if k in mapping:
            _set_nested(params, mapping[k], float(v))
        else:
            try:
                params[k] = float(v)
            except Exception:
                pass

    # override initial clusters from CLI
    params["init"]["n_clusters"] = int(max(1, round(n0)))
    params["init"]["phenotype"] = "proliferative"
    return params

# ---------------------- pyABC I/O ----------------------
def load_posterior(db_path: Path):
    h = pyabc.History(f"sqlite:///{db_path}")
    pops = h.get_all_populations()
    eps_df = pd.DataFrame({"t": pops["t"].to_numpy(), "epsilon": pops["epsilon"].to_numpy()})
    t = h.max_t
    while t >= 0:
        df, w = h.get_distribution(m=0, t=t)
        if len(df) > 0:
            w = np.asarray(w, float)
            w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
            return df, w, int(t), eps_df
        t -= 1
    raise RuntimeError(f"No posterior populations found in {db_path}")

# ---------------------- ABM run ------------------------
def run_abm(params_dict):
    m = ClustersModel(params=params_dict, seed=42)
    steps = int(params_dict["time"]["steps"])
    for _ in range(steps):
        m.step()
    return compute_summary(m)

def simulate_timeseries(params_dict, total_steps: int, start_step: int = 0, seed: int = 123):
    m = ClustersModel(params=params_dict, seed=seed)
    out = np.zeros((total_steps, 3), dtype=float)
    out[0, :] = [compute_summary(m)[k] for k in ("S0","S1","S2")]
    for t in range(1, total_steps):
        m.step(); s = compute_summary(m)
        out[t, :] = [s["S0"], s["S1"], s["S2"]]
    if start_step > 0:
        out = out[start_step:]
    return out

# ---------------------- helpers ------------------------
def hpd(arr, prob=0.9):
    arr = np.sort(np.asarray(arr))
    n = len(arr)
    if n == 0:
        return (np.nan, np.nan)
    k = max(1, int(prob * n))
    widths = arr[k:] - arr[:n - k]
    i = int(np.argmin(widths))
    return float(arr[i]), float(arr[i + k - 1])

# ---------------------- analysis -----------------------
def analyze_file(db_path: Path,
                 observed_final: dict,
                 obs_ts_df: pd.DataFrame | None,
                 t_start: int,
                 outdir: Path,
                 motion: str,
                 speed_dist: str,
                 n0: int,
                 pp_samples: int = 80,
                 sweep_pts: int = 15,
                 local_delta: float = 0.05):
    print(f"\n=== Analyzing: {db_path.name} ===")
    params, weights, t_used, eps_df = load_posterior(db_path)

    # PPC (final)
    samples = params.sample(pp_samples, replace=True, weights=weights, random_state=42)
    ppc_out = []
    for _, row in samples.iterrows():
        pmap = particle_to_params(DEFAULTS, row.to_dict(), motion, speed_dist, n0)
        ppc_out.append(run_abm(pmap))
    ppc_df = pd.DataFrame(ppc_out); ppc_df.to_csv(outdir / f"{db_path.stem}_PPC.csv", index=False)
    for s in ["S0","S1","S2"]:
        plt.figure(); sns.histplot(ppc_df[s], kde=True); plt.axvline(observed_final[s], color='red'); plt.title(f"PPC {s}"); plt.savefig(outdir / f"{db_path.stem}_PPC_{s}.png"); plt.close()

    # Time-series PPC
    if obs_ts_df is not None and len(obs_ts_df) > 0:
        obs_ts = obs_ts_df[["S0","S1","S2"]].to_numpy(float)
        T = obs_ts.shape[0]; total_steps = t_start + T
        rng = np.random.default_rng(123)
        idxs = rng.choice(np.arange(len(params)), size=min(pp_samples, len(params)), replace=False, p=weights/weights.sum())
        sims = np.zeros((len(idxs), T, 3), dtype=float)
        for j, i in enumerate(idxs):
            particle = params.iloc[i].to_dict()
            pmap = particle_to_params(DEFAULTS, particle, motion, speed_dist, n0)
            seg = simulate_timeseries(pmap, total_steps=total_steps, start_step=t_start, seed=int(rng.integers(1, 2**31-1)))
            sims[j,:,:] = seg[:T,:]
        med = np.median(sims, axis=0)
        q05 = np.quantile(sims, 0.05, axis=0)
        q95 = np.quantile(sims, 0.95, axis=0)
        pd.DataFrame(med, columns=["S0","S1","S2"]).to_csv(outdir / f"{db_path.stem}_PPC_ts_median.csv", index=False)
        pd.DataFrame(q05, columns=["S0","S1","S2"]).to_csv(outdir / f"{db_path.stem}_PPC_ts_q05.csv", index=False)
        pd.DataFrame(q95, columns=["S0","S1","S2"]).to_csv(outdir / f"{db_path.stem}_PPC_ts_q95.csv", index=False)
        t = np.arange(T) + t_start
        for k, s in enumerate(["S0","S1","S2"]):
            plt.figure(figsize=(8,4)); plt.fill_between(t, q05[:,k], q95[:,k], color="#cfe8ff", alpha=0.8, label="5–95% band"); plt.plot(t, med[:,k], color="#1f77b4", lw=1.8, label="median"); plt.plot(t, obs_ts[:,k], color="black", lw=1.5, label="observed"); plt.xlabel("step"); plt.ylabel(s); plt.title(f"Time-series PPC — {s}"); plt.legend(); plt.tight_layout(); plt.savefig(outdir / f"{db_path.stem}_PPC_ts_{s}.png"); plt.close()
        disc_per_draw = []
        for j in range(sims.shape[0]):
            d = np.sqrt(np.sum((sims[j,:,:] - obs_ts)**2, axis=1))
            disc_per_draw.append(d)
        disc_per_draw = np.asarray(disc_per_draw)
        disc_med = np.median(disc_per_draw, axis=0)
        disc_q05 = np.quantile(disc_per_draw, 0.05, axis=0)
        disc_q95 = np.quantile(disc_per_draw, 0.95, axis=0)
        pd.DataFrame({"step": t, "disc_med": disc_med, "disc_q05": disc_q05, "disc_q95": disc_q95}).to_csv(outdir / f"{db_path.stem}_PPC_ts_discrepancy.csv", index=False)
        plt.figure(figsize=(8,4)); plt.fill_between(t, disc_q05, disc_q95, alpha=0.3, color="#ffd2cc", label="5–95% band"); plt.plot(t, disc_med, color="#d62728", lw=1.6, label="median discrepancy"); plt.title("Discrepancy vs time"); plt.xlabel("step"); plt.ylabel("Euclidean discrepancy"); plt.legend(); plt.tight_layout(); plt.savefig(outdir / f"{db_path.stem}_PPC_ts_discrepancy.png"); plt.close()

    # Identifiability
    corr = params.corr(); plt.figure(figsize=(10,8)); sns.heatmap(corr, cmap="vlag", center=0); plt.title("Posterior correlation"); plt.tight_layout(); plt.savefig(outdir / f"{db_path.stem}_correlation.png"); plt.close()
    for c in params.columns: plt.figure(); sns.kdeplot(params[c], fill=True); plt.title(f"{c} marginal"); plt.tight_layout(); plt.savefig(outdir / f"{db_path.stem}_marginal_{c}.png"); plt.close()
    X = StandardScaler().fit_transform(params.values)
    pca = PCA().fit(X)
    plt.figure(); plt.plot(np.cumsum(pca.explained_variance_ratio_)); plt.title("PCA variance explained"); plt.tight_layout(); plt.savefig(outdir / f"{db_path.stem}_PCA.png"); plt.close()
    loadings = pd.DataFrame(pca.components_.T, index=params.columns, columns=[f"PC{i}" for i in range(len(params.columns))])
    loadings.to_csv(outdir / f"{db_path.stem}_PCA_loadings.csv")
    for i in range(min(3, loadings.shape[1])):
        pc = f"PC{i}"; plt.figure(figsize=(10,4)); loadings[pc].plot(kind="bar"); plt.title(f"PCA loadings: {pc}"); plt.tight_layout(); plt.savefig(outdir / f"{db_path.stem}_PCA_loadings_{pc}.png"); plt.close()
    top = {pc: list(loadings[pc].abs().sort_values(ascending=False).index[:5]) for pc in loadings.columns}
    pd.DataFrame(top).to_csv(outdir / f"{db_path.stem}_PCA_top_parameters.csv", index=False)

    # Posterior summary
    w = np.asarray(weights); ess = float(1.0/np.sum(w**2))
    rows = []
    for c in params.columns:
        arr = params[c].to_numpy(float)
        m = float(np.sum(w*arr)/w.sum())
        sd = float(np.sqrt(max(0.0, np.sum(w*(arr-m)**2)/w.sum())))
        lo, hi = hpd(arr, 0.9)
        rows.append({"param": c, "mean": m, "sd": sd, "HPD_low": lo, "HPD_high": hi})
    pd.DataFrame(rows).to_csv(outdir / f"{db_path.stem}_posterior_summary.csv", index=False)

    # epsilon trajectory
    plt.figure(); plt.plot(eps_df["t"], eps_df["epsilon"], marker='o'); plt.xlabel("population t"); plt.ylabel("epsilon"); plt.title(f"Epsilon trajectory (t_used={t_used})"); plt.tight_layout(); plt.savefig(outdir / f"{db_path.stem}_epsilon_trajectory.png"); plt.close()

    # IC sweeps
    mean_part = params.mean(); ic_dict = {}
    for p in params.columns:
        centre = mean_part[p]; sweep = np.linspace(0.8*centre, 1.2*centre, sweep_pts)
        d_list = []
        for val in sweep:
            mp = mean_part.copy(); mp[p] = val
            pmap = particle_to_params(DEFAULTS, mp.to_dict(), motion, speed_dist, n0)
            sim = run_abm(pmap)
            d = np.sqrt((sim["S0"]-observed_final["S0"])**2 + (sim["S1"]-observed_final["S1"])**2 + (sim["S2"]-observed_final["S2"])**2)
            d_list.append(d)
        IC = float(np.min(d_list)) + len(params.columns)*np.log(3)
        ic_dict[p] = {"IC": IC, "min_disc": float(np.min(d_list))}
        plt.figure(); plt.plot(sweep, d_list, marker='o'); plt.xlabel(p); plt.ylabel("discrepancy"); plt.title(f"IC sweep for {p}"); plt.tight_layout(); plt.savefig(outdir / f"{db_path.stem}_ICsweep_{p}.png"); plt.close()
    pd.DataFrame(ic_dict).T.sort_values("min_disc").to_csv(outdir / f"{db_path.stem}_IC_ranked.csv")

    # Local sensitivity
    sens = []
    for p in params.columns:
        c = float(mean_part[p])
        if c == 0:
            sens.append({"param": p, "elasticity": np.nan}); continue
        vm, vp = c*(1.0-local_delta), c*(1.0+local_delta)
        for v, tag in [(vm, 'minus'), (vp, 'plus')]:
            mp = mean_part.copy(); mp[p] = v
            sim = run_abm(particle_to_params(DEFAULTS, mp.to_dict(), motion, speed_dist, n0))
            d = np.sqrt((sim["S0"]-observed_final["S0"])**2 + (sim["S1"]-observed_final["S1"])**2 + (sim["S2"]-observed_final["S2"])**2)
            if tag=='minus': d_minus = d
            else: d_plus = d
        elasticity = (d_plus - d_minus)/(2.0*local_delta*c)
        sens.append({"param": p, "elasticity": float(elasticity)})
    pd.DataFrame(sens).to_csv(outdir / f"{db_path.stem}_local_sensitivity.csv", index=False)

    print(f"Finished {db_path.name} | ESS≈{ess:.1f} | t_used={t_used}")

# ---------------------- CLI ----------------------------
def main():
    ap = argparse.ArgumentParser(description="ABC posterior analysis (fixed init via --n0)")
    ap.add_argument("--glob", nargs="*", default=None, help="Glob patterns for .db files")
    ap.add_argument("--db", nargs="*", default=None, help="Explicit .db files")
    ap.add_argument("--obs", type=str, required=True, help="Observed FINAL stats CSV with one row: S0,S1,S2")
    ap.add_argument("--obs_ts", type=str, required=True, help="Observed time-series CSV with columns S0,S1,S2 (row0 is timestep t_start)")
    ap.add_argument("--t_start", type=int, default=22, help="Observed time-series starting step (default 22)")
    ap.add_argument("--n0", type=int, required=True, help="Initial cluster count to use in simulations")
    ap.add_argument("--outdir", type=str, default="abc_analysis_results")
    ap.add_argument("--pp", type=int, default=80)
    ap.add_argument("--sweep", type=int, default=15)
    ap.add_argument("--local_delta", type=float, default=0.05)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    obs_final_df = pd.read_csv(args.obs)
    observed_final = {"S0": float(obs_final_df["S0"][0]), "S1": float(obs_final_df["S1"][0]), "S2": float(obs_final_df["S2"][0])}
    obs_ts_df = pd.read_csv(args.obs_ts)

    paths = []
    if args.glob:
        for g in args.glob:
            paths.extend(Path(p) for p in globlib.glob(g))
    if args.db:
        paths.extend(Path(d) for d in args.db)
    if not paths:
        raise RuntimeError("No .db files found")

    for p in paths:
        name = p.stem
        motion = "isotropic" if "_isotropic_" in name else ("persistent" if "_persistent_" in name else "isotropic")
        if "_lognorm_" in name: speed = "lognorm"
        elif "_gamma_" in name: speed = "gamma"
        elif "_weibull_" in name: speed = "weibull"
        else: speed = "constant"
        analyze_file(
            db_path=p,
            observed_final=observed_final,
            obs_ts_df=obs_ts_df,
            t_start=int(args.t_start),
            outdir=outdir,
            motion=motion,
            speed_dist=speed,
            n0=int(args.n0),
            pp_samples=args.pp,
            sweep_pts=args.sweep,
            local_delta=args.local_delta,
        )

if __name__ == "__main__":
    main()
