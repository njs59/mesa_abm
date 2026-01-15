
#!/usr/bin/env python3
"""
ABC Posterior Analysis CLI (enhanced)
- Compatible with pyABC History API (get_distribution, get_all_populations)
- Compatible with your Mesa ABM
- Supports: PPC, IC sweeps, identifiability diagnostics
- Added:
    * PCA loadings CSV
    * PCA top-parameter CSV
    * PCA loadings barplots (PNG)
    * Full posterior summary (weighted)
    * Epsilon trajectory
    * Local sensitivity (elasticity)
    * PPC percentiles, residuals, discrepancy histogram
"""

import argparse
import re
from pathlib import Path
import glob as globlib
import json
import numpy as np
import pandas as pd
import pyabc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- ABM imports ---
from clusters_abm.clusters_model import ClustersModel
from clusters_abm.utils import DEFAULTS


# ============================================================
# SUMMARY FUNCTION
# ============================================================
def compute_summary(model):
    ids, pos, radii, sizes, speeds = model._log_alive_snapshot()
    n = len(sizes)
    if n == 0:
        return {"S0": 0.0, "S1": 0.0, "S2": 0.0}
    return {
        "S0": float(n),
        "S1": float(np.mean(sizes)),
        "S2": float(np.mean(sizes ** 2)),
    }


# ============================================================
# PARAMETER MAPPING
# ============================================================
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


def particle_to_params(defaults, particle, motion, speed_dist, n0):
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

    params["init"]["phenotype"] = "proliferative"
    params["init"]["n_clusters"] = int(max(1, round(n0)))
    return params


# ============================================================
# POSTERIOR LOADER (pyABC)
# ============================================================
def load_posterior(db_path):
    """
    Load latest non-empty posterior population from pyABC database.
    Uses History.get_distribution exactly as recommended. [1](https://pyabc.readthedocs.io/en/latest/api/pyabc.storage.html)
    """
    h = pyabc.History(f"sqlite:///{db_path}")
    pops = h.get_all_populations()
    eps_df = pd.DataFrame({"t": pops["t"].to_numpy(), "epsilon": pops["epsilon"].to_numpy()})

    t = h.max_t
    while t >= 0:
        df, w = h.get_distribution(m=0, t=t)
        if len(df) > 0:
            w = np.asarray(w, float)
            w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
            return df, w, t, eps_df
        t -= 1
    raise RuntimeError(f"No posterior populations found in {db_path}")


# ============================================================
# ABM RUNNER
# ============================================================
def run_abm(params_dict):
    m = ClustersModel(params=params_dict, seed=42)
    steps = int(params_dict["time"]["steps"])
    for _ in range(steps):
        m.step()
    return compute_summary(m)


# ============================================================
# MAIN ANALYSIS PER FILE
# ============================================================
def analyze_file(db_path, observed, outdir, motion, speed_dist, seed, n0,
                 pp_samples=80, sweep_pts=15, pairplot_max=12, local_delta=0.05):

    print(f"\n=== Analyzing: {db_path.name} ===")

    # Load posterior + population epsilons
    params, weights, t_used, eps_df = load_posterior(db_path)

    # ============================================================
    # PPC
    # ============================================================
    samples = params.sample(pp_samples, replace=True, weights=weights, random_state=42)
    ppc_out = []
    for _, row in samples.iterrows():
        pmap = particle_to_params(DEFAULTS, row.to_dict(), motion, speed_dist, n0)
        ppc_out.append(run_abm(pmap))

    ppc_df = pd.DataFrame(ppc_out)
    ppc_df.to_csv(outdir / f"{db_path.stem}_PPC.csv", index=False)

    # PPC histograms
    for s in ["S0", "S1", "S2"]:
        plt.figure()
        sns.histplot(ppc_df[s], kde=True)
        plt.axvline(observed[s], color="red", label="Observed")
        plt.legend()
        plt.title(f"PPC {s}")
        plt.savefig(outdir / f"{db_path.stem}_PPC_{s}.png")
        plt.close()

    # PPC discrepancy histogram
    disc = np.sqrt(
        (ppc_df["S0"] - observed["S0"])**2 +
        (ppc_df["S1"] - observed["S1"])**2 +
        (ppc_df["S2"] - observed["S2"])**2
    )
    plt.figure()
    sns.histplot(disc, bins=25, kde=True)
    plt.title("PPC discrepancy")
    plt.savefig(outdir / f"{db_path.stem}_PPC_discrepancy.png")
    plt.close()

    # PPC percentiles (PIT)
    pit = {k: float((ppc_df[k] <= observed[k]).mean()) for k in ["S0", "S1", "S2"]}
    pd.DataFrame([pit]).to_csv(outdir / f"{db_path.stem}_PPC_percentiles.csv", index=False)

    # ============================================================
    # IDENTIFIABILITY
    # ============================================================

    # Correlation heatmap
    corr = params.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0)
    plt.title("Posterior correlation")
    plt.savefig(outdir / f"{db_path.stem}_correlation.png")
    plt.close()

    # Marginals
    for col in params.columns:
        plt.figure()
        sns.kdeplot(params[col], fill=True)
        plt.title(f"{col} marginal")
        plt.savefig(outdir / f"{db_path.stem}_marginal_{col}.png")
        plt.close()

    # PCA
    X = StandardScaler().fit_transform(params.values)
    pca = PCA().fit(X)

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title("PCA variance explained")
    plt.savefig(outdir / f"{db_path.stem}_PCA.png")
    plt.close()

    # ============================================================
    # NEW: PCA LOADINGS + PNG OUTPUT
    # ============================================================
    loadings = pd.DataFrame(
        pca.components_.T,
        index=params.columns,
        columns=[f"PC{i}" for i in range(len(params.columns))]
    )
    loadings.to_csv(outdir / f"{db_path.stem}_PCA_loadings.csv")

    # Barplots for PC0–PC2
    n_pcs_to_plot = min(3, loadings.shape[1])
    for i in range(n_pcs_to_plot):
        pc = f"PC{i}"
        plt.figure(figsize=(10, 4))
        loadings[pc].plot(kind="bar")
        plt.title(f"PCA loadings for {pc}")
        plt.tight_layout()
        plt.savefig(outdir / f"{db_path.stem}_PCA_loadings_{pc}.png")
        plt.close()

    # Top parameters per PC
    top_params = {}
    for pc in loadings.columns:
        sorted_params = loadings[pc].abs().sort_values(ascending=False)
        top_params[pc] = list(sorted_params.index[:5])
    pd.DataFrame(top_params).to_csv(outdir / f"{db_path.stem}_PCA_top_parameters.csv", index=False)

    # ============================================================
    # POSTERIOR SUMMARY (weighted)
    # ============================================================
    ess = float(1.0 / np.sum(np.asarray(weights) ** 2))
    summary_rows = []
    for c in params.columns:
        col = params[c].to_numpy()
        w = np.asarray(weights)
        wsum = w.sum()
        if wsum <= 0:
            mean_w = np.nan
            var_w = np.nan
        else:
            mean_w = float(np.sum(w * col) / wsum)
            var_w = float(np.sum(w * (col - mean_w) ** 2) / wsum)
        h_low, h_high = hpd(col, 0.9)
        summary_rows.append({
            "param": c,
            "mean": mean_w,
            "sd": np.sqrt(var_w),
            "HPD_low": h_low,
            "HPD_high": h_high,
        })
    pd.DataFrame(summary_rows).to_csv(outdir / f"{db_path.stem}_posterior_summary.csv", index=False)

    # ============================================================
    # EPSILON TRAJECTORY
    # ============================================================
    plt.figure()
    plt.plot(eps_df["t"], eps_df["epsilon"], marker="o")
    plt.xlabel("Population t")
    plt.ylabel("epsilon")
    plt.title(f"Epsilon trajectory (t_used={t_used})")
    plt.grid(alpha=0.3)
    plt.savefig(outdir / f"{db_path.stem}_epsilon_trajectory.png")
    plt.close()

    # ============================================================
    # IC SWEEPS
    # ============================================================
    mean_part = params.mean()
    ic_json = {}

    for p in params.columns:
        centre = mean_part[p]
        sweep = np.linspace(0.8 * centre, 1.2 * centre, sweep_pts)
        d_list = []
        for v in sweep:
            mp = mean_part.copy()
            mp[p] = v
            pmap = particle_to_params(DEFAULTS, mp.to_dict(), motion, speed_dist, n0)
            sim = run_abm(pmap)
            d = np.sqrt(
                (sim["S0"] - observed["S0"])**2 +
                (sim["S1"] - observed["S1"])**2 +
                (sim["S2"] - observed["S2"])**2
            )
            d_list.append(d)

        IC = float(np.min(d_list)) + len(params.columns) * np.log(3)
        ic_json[p] = {"IC": IC, "min_disc": float(np.min(d_list))}

        plt.figure()
        plt.plot(sweep, d_list, marker="o")
        plt.xlabel(p)
        plt.ylabel("Discrepancy")
        plt.title(f"IC sweep for {p}")
        plt.grid(alpha=0.3)
        plt.savefig(outdir / f"{db_path.stem}_ICsweep_{p}.png")
        plt.close()

    ic_df = pd.DataFrame(ic_json).T
    ic_df.to_csv(outdir / f"{db_path.stem}_IC.csv")
    ic_df.sort_values("min_disc").to_csv(outdir / f"{db_path.stem}_IC_ranked.csv")

    # ============================================================
    # LOCAL SENSITIVITY
    # ============================================================
    sens_rows = []
    for p in params.columns:
        c = float(mean_part[p])
        if c == 0:
            sens_rows.append({"param": p, "elasticity": np.nan})
            continue

        v_minus = c * (1.0 - local_delta)
        v_plus = c * (1.0 + local_delta)

        for val, tag in [(v_minus, "minus"), (v_plus, "plus")]:
            mp = mean_part.copy()
            mp[p] = val
            sim = run_abm(particle_to_params(DEFAULTS, mp.to_dict(), motion, speed_dist, n0))
            d = np.sqrt(
                (sim["S0"] - observed["S0"])**2 +
                (sim["S1"] - observed["S1"])**2 +
                (sim["S2"] - observed["S2"])**2
            )
            if tag == "minus":
                d_minus = d
            else:
                d_plus = d

        elasticity = (d_plus - d_minus) / (2.0 * local_delta * c)
        sens_rows.append({"param": p, "elasticity": float(elasticity)})

    pd.DataFrame(sens_rows).to_csv(outdir / f"{db_path.stem}_local_sensitivity.csv", index=False)

    print(f"Finished {db_path.name} | ESS={ess:.1f} | t_used={t_used}")


# ============================================================
# HPD FUNCTION
# ============================================================
def hpd(arr, prob=0.9):
    arr = np.sort(np.asarray(arr))
    n = len(arr)
    if n == 0:
        return (np.nan, np.nan)
    k = max(1, int(prob * n))
    widths = arr[k:] - arr[:n - k]
    i = int(np.argmin(widths))
    return float(arr[i]), float(arr[i + k - 1])


# ============================================================
# FILENAME PARSER
# ============================================================
NAME_PATTERN = re.compile(
    r"(?P<dataset>[A-Za-z0-9]+)_(?P<motion>isotropic|persistent)_(?P<speed_dist>[A-Za-z0-9]+)_s(?P<seed>[0-9]+)\.db"
)

def parse_filename(path: Path):
    m = NAME_PATTERN.match(path.name)
    return m.groupdict() if m else {}


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="ABC posterior analysis (enhanced)")
    ap.add_argument("--glob", nargs="*", default=None)
    ap.add_argument("--db", nargs="*", default=None)
    ap.add_argument("--obs", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="abc_analysis_results")
    ap.add_argument("--pp", type=int, default=80)
    ap.add_argument("--sweep", type=int, default=15)
    ap.add_argument("--n0", type=int, default=800)
    ap.add_argument("--pairplot-max", type=int, default=12)
    ap.add_argument("--local-delta", type=float, default=0.05)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    obsdf = pd.read_csv(args.obs)
    observed = {"S0": float(obsdf["S0"][0]),
                "S1": float(obsdf["S1"][0]),
                "S2": float(obsdf["S2"][0])}

    paths = []
    if args.glob:
        for g in args.glob:
            paths.extend(Path(p) for p in globlib.glob(g))
    if args.db:
        paths.extend(Path(d) for d in args.db)
    if not paths:
        raise RuntimeError("No .db files found")

    for p in paths:
        meta = parse_filename(p)
        if not meta:
            print(f"Warning: could not parse {p.name}, skipping.")
            continue
        analyze_file(
            db_path=p,
            observed=observed,
            outdir=outdir,
            motion=meta["motion"],
            speed_dist=meta["speed_dist"],
            seed=meta["seed"],
            n0=args.n0,
            pp_samples=args.pp,
            sweep_pts=args.sweep,
            pairplot_max=args.pairplot_max,
            local_delta=args.local_delta,
        )

if __name__ == "__main__":
    main()
