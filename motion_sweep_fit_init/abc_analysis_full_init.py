
#!/usr/bin/env python3
"""
ABC Posterior Analysis CLI (enhanced, with init_n as ABC parameter)

CHANGES:
- init_n no longer provided by CLI; now inferred from ABC posterior.
- All sweeping, PPC, posterior summaries now use the particle's init_n.
- PCA loadings CSV + PNG exported.
- IC, sensitivity, PPC unchanged intellectually but now depend on init_n.
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


def particle_to_params(defaults, particle, motion, speed_dist):
    """
    Convert flat ABC particle ➜ nested Mesa model parameters.
    Now includes init_n as an inferred parameter.
    """
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

    # movement
    params["movement"]["direction"] = motion
    if speed_dist == "constant":
        params["movement"]["mode"] = "constant"
        params["movement"].pop("distribution", None)
        params["movement"].pop("dist_params", None)
    else:
        params["movement"]["mode"] = "distribution"
        params["movement"]["distribution"] = speed_dist
        params["movement"]["dist_params"] = build_speed_params(speed_dist, particle)

    # heading sigma
    if motion == "persistent":
        hs = float(particle.get("heading_sigma", params["movement"].get("heading_sigma", 0.25)))
        params["movement"]["heading_sigma"] = max(0.0, hs)
    else:
        params["movement"].pop("heading_sigma", None)

    # biological parameters:
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

    # --- KEY UPDATE: init_n is now a posterior parameter ---
    if "init_n_clusters" not in particle:
        raise KeyError("ABC posterior is missing init_n_clusters, cannot set initial cluster count.")
    params["init"]["n_clusters"] = int(max(1, round(particle["init_n_clusters"])))

    params["init"]["phenotype"] = "proliferative"
    return params


# ============================================================
# POSTERIOR LOADER (pyABC)
# ============================================================
def load_posterior(db_path):
    """
    Retrieve last non-empty population from ABC database using pyABC History API.
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
# MAIN PER-FILE ANALYSIS
# ============================================================
def analyze_file(db_path, observed, outdir, motion, speed_dist, seed,
                 pp_samples=80, sweep_pts=15, pairplot_max=12, local_delta=0.05):

    print(f"\n=== Analyzing: {db_path.name} ===")

    params, weights, t_used, eps_df = load_posterior(db_path)

    # ============================================================
    # PPC
    # ============================================================
    samples = params.sample(pp_samples, replace=True, weights=weights, random_state=42)
    ppc_out = []
    for _, row in samples.iterrows():
        pmap = particle_to_params(DEFAULTS, row.to_dict(), motion, speed_dist)
        ppc_out.append(run_abm(pmap))
    ppc_df = pd.DataFrame(ppc_out)
    ppc_df.to_csv(outdir / f"{db_path.stem}_PPC.csv", index=False)

    for s in ["S0", "S1", "S2"]:
        plt.figure()
        sns.histplot(ppc_df[s], kde=True)
        plt.axvline(observed[s], color="red")
        plt.title(f"PPC {s}")
        plt.savefig(outdir / f"{db_path.stem}_PPC_{s}.png")
        plt.close()

    disc = np.sqrt(
        (ppc_df["S0"] - observed["S0"])**2 +
        (ppc_df["S1"] - observed["S1"])**2 +
        (ppc_df["S2"] - observed["S2"])**2
    )
    plt.figure()
    sns.histplot(disc, kde=True)
    plt.title("PPC discrepancy")
    plt.savefig(outdir / f"{db_path.stem}_PPC_discrepancy.png")
    plt.close()

    pit = {k: float((ppc_df[k] <= observed[k]).mean()) for k in ["S0", "S1", "S2"]}
    pd.DataFrame([pit]).to_csv(outdir / f"{db_path.stem}_PPC_percentiles.csv", index=False)

    # ============================================================
    # IDENTIFIABILITY
    # ============================================================

    corr = params.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0)
    plt.title("Posterior correlation")
    plt.savefig(outdir / f"{db_path.stem}_correlation.png")
    plt.close()

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

    # PCA loadings CSV
    loadings = pd.DataFrame(
        pca.components_.T,
        index=params.columns,
        columns=[f"PC{i}" for i in range(len(params.columns))]
    )
    loadings.to_csv(outdir / f"{db_path.stem}_PCA_loadings.csv")

    # PCA loadings PNG (PC0–PC2)
    for i in range(min(3, len(loadings.columns))):
        pc = f"PC{i}"
        plt.figure(figsize=(10, 4))
        loadings[pc].plot(kind="bar")
        plt.title(f"PCA loadings: {pc}")
        plt.tight_layout()
        plt.savefig(outdir / f"{db_path.stem}_PCA_loadings_{pc}.png")
        plt.close()

    # top parameters per PC
    top_params = {
        pc: list(loadings[pc].abs().sort_values(ascending=False).index[:5])
        for pc in loadings.columns
    }
    pd.DataFrame(top_params).to_csv(
        outdir / f"{db_path.stem}_PCA_top_parameters.csv", index=False
    )

    # ============================================================
    # POSTERIOR SUMMARY
    # ============================================================
    weights_arr = np.asarray(weights)
    ess = float(1.0 / np.sum(weights_arr ** 2))

    rows = []
    for c in params.columns:
        arr = params[c].to_numpy()
        w = weights_arr
        m = float(np.sum(w * arr) / w.sum())
        sd = float(np.sqrt(np.sum(w * (arr - m) ** 2) / w.sum()))
        h_low, h_high = hpd(arr, 0.9)
        rows.append({
            "param": c,
            "mean": m,
            "sd": sd,
            "HPD_low": h_low,
            "HPD_high": h_high,
        })
    pd.DataFrame(rows).to_csv(outdir / f"{db_path.stem}_posterior_summary.csv", index=False)

    # ============================================================
    # EPSILON TRAJECTORY
    # ============================================================
    plt.figure()
    plt.plot(eps_df["t"], eps_df["epsilon"], marker="o")
    plt.title(f"Epsilon trajectory (t_used={t_used})")
    plt.xlabel("Population")
    plt.ylabel("epsilon")
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
            pmap = particle_to_params(DEFAULTS, mp.to_dict(), motion, speed_dist)
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
        plt.title(f"IC sweep for {p}")
        plt.xlabel(p)
        plt.ylabel("discrepancy")
        plt.savefig(outdir / f"{db_path.stem}_ICsweep_{p}.png")
        plt.close()

    pd.DataFrame(ic_json).T.sort_values("min_disc").to_csv(
        outdir / f"{db_path.stem}_IC_ranked.csv"
    )

    print(f"Finished {db_path.name} | ESS={ess:.1f} | t_used={t_used}")


# ============================================================
# HPD
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
    ap = argparse.ArgumentParser(description="ABC posterior analysis (init_n inferred)")
    ap.add_argument("--glob", nargs="*", default=None)
    ap.add_argument("--db", nargs="*", default=None)
    ap.add_argument("--obs", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="abc_analysis_results")
    ap.add_argument("--pp", type=int, default=80)
    ap.add_argument("--sweep", type=int, default=15)
    ap.add_argument("--pairplot-max", type=int, default=12)
    ap.add_argument("--local-delta", type=float, default=0.05)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    obsdf = pd.read_csv(args.obs)
    observed = {
        "S0": float(obsdf["S0"][0]),
        "S1": float(obsdf["S1"][0]),
        "S2": float(obsdf["S2"][0]),
    }

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
            print(f"WARNING: could not parse filename, skipping: {p}")
            continue
        analyze_file(
            db_path=p,
            observed=observed,
            outdir=outdir,
            motion=meta["motion"],
            speed_dist=meta["speed_dist"],
            seed=meta["seed"],
            pp_samples=args.pp,
            sweep_pts=args.sweep,
            pairplot_max=args.pairplot_max,
            local_delta=args.local_delta,
        )

if __name__ == "__main__":
    main()
