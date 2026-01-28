
#!/usr/bin/env python3
"""
Analyse ABC posterior and generate plots/CSVs for stats [S0, S1, S2, SSNND_med].
This version drops g(r) entirely and expects observed CSVs with columns:
  timestep, S0, S1, S2, SSNND_med
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyabc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from abm.clusters_model import ClustersModel
from abcp.abc_model_wrapper import particle_to_params
from abcp.compute_summary import simulate_timeseries

plt.style.use("seaborn-v0_8")


def make_model_factory(seed: int = 123):
    def factory(params_dict):
        return ClustersModel(params=params_dict, seed=seed)
    return factory


def _weighted_resample(df: pd.DataFrame, weights: np.ndarray, n: int, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = np.asarray(weights, dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
    idx = rng.choice(np.arange(len(df)), size=n, replace=True, p=p)
    return df.iloc[idx].reset_index(drop=True)


def _safe_kdeplot_1d(x: np.ndarray, weights: np.ndarray | None, color="#1f77b4", label=None):
    import pandas as pd
    ok = False
    try:
        d = pd.DataFrame({"x": np.asarray(x, float)})
        if weights is not None:
            w = np.asarray(weights, float)
            w = w / (w.sum() if w.sum() > 0 else len(w))
            d["w"] = w
            sns.kdeplot(data=d, x="x", weights="w", fill=True, color=color, alpha=0.8, label=label)
        else:
            sns.kdeplot(data=d, x="x", fill=True, color=color, alpha=0.8, label=label)
        ok = True
    except Exception:
        ok = False
    if not ok:
        try:
            rng = np.random.default_rng(2026)
            n = min(5000, len(x))
            if weights is not None:
                p = np.asarray(weights, float)
                p = p / (p.sum() if p.sum() > 0 else len(p))
                idx = rng.choice(np.arange(len(x)), size=n, replace=True, p=p)
            else:
                idx = rng.choice(np.arange(len(x)), size=n, replace=True)
            xs = np.asarray(x, float)[idx]
            sns.kdeplot(xs, fill=True, color=color, alpha=0.8, label=label)
            ok = True
        except Exception:
            ok = False
    if not ok:
        plt.hist(np.asarray(x, float), bins="auto", density=True, color=color, alpha=0.55, label=label)


def plot_pca_final_biplot(params_df: pd.DataFrame, weights: np.ndarray, outdir: Path, top_n: int = 8):
    if params_df.shape[1] < 2:
        print("PCA biplot skipped: <2 parameters in posterior.")
        return
    scaler = StandardScaler().fit(params_df.values)
    X = scaler.transform(params_df.values)
    pca = PCA().fit(X)
    scores = pca.transform(X)[:, :2]
    w = np.asarray(weights, float)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    sizes = 20 + 2800 * (w - w.min()) / (w.max() - w.min() + 1e-12)
    rad = np.percentile(np.sqrt(scores[:, 0] ** 2 + scores[:, 1] ** 2), 85)
    loadings2 = pca.components_[:2, :].T
    lnorm = np.linalg.norm(loadings2, axis=1)
    top_idx = np.argsort(lnorm)[-min(top_n, len(lnorm)) :]
    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    hb = ax.scatter(scores[:, 0], scores[:, 1], s=sizes, c=w, cmap="viridis", alpha=0.65, edgecolor="none")
    cb = fig.colorbar(hb, ax=ax, pad=0.01)
    cb.set_label("posterior weight")
    for j in top_idx:
        vec = loadings2[j]
        ax.arrow(0, 0, rad * vec[0], rad * vec[1], color="#d62728", width=0.001, head_width=0.08 * rad,
                 length_includes_head=True, alpha=0.9)
        ax.text(rad * vec[0] * 1.07, rad * vec[1] * 1.07, params_df.columns[j], color="#d62728",
                fontsize=9, ha="center", va="center")
    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    ax.set_title("Final-population PCA biplot (PC1 vs PC2)")
    ax.axhline(0, color="#999999", lw=0.6)
    ax.axvline(0, color="#999999", lw=0.6)
    fig.tight_layout()
    fig.savefig(outdir / "pca_final_biplot.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Analyse ABC posterior & PPC for S0,S1,S2,SSNND_med")
    ap.add_argument("--db", type=str, required=True, help="SQLite DB produced by run_abc_maxabs_sweep.py")
    ap.add_argument("--observed_ts", type=str, required=True)
    ap.add_argument("--t_start", type=int, default=22)
    ap.add_argument("--total_steps", type=int, default=300)
    ap.add_argument("--pp", type=int, default=100, help="Posterior predictive draws")
    ap.add_argument("--motion", type=str, default="isotropic", choices=["isotropic", "persistent"])
    ap.add_argument("--speed", type=str, default="constant", choices=["constant", "lognorm", "gamma", "weibull"])
    ap.add_argument("--pca_top_k", type=int, default=5)
    ap.add_argument("--save_all_loadings", action="store_true")
    ap.add_argument("--speed_cols", nargs="+", default=None, help="Explicit speed parameter column names")
    args = ap.parse_args()

    outdir = Path("results")
    outdir.mkdir(exist_ok=True, parents=True)

    # Observed
    obs_df = pd.read_csv(args.observed_ts)
    timesteps = obs_df["timestep"].astype(int).to_list()
    stats = ["S0", "S1", "S2", "SSNND_med"]
    for s in stats:
        if s not in obs_df.columns:
            raise ValueError(f"Missing {s} in observed CSV.")
    obs_mat = obs_df[stats].to_numpy(float)

    # Posterior at final population
    h = pyabc.History(f"sqlite:///{args.db}")
    t_max = h.max_t
    params_df, w = h.get_distribution(m=0, t=t_max)
    if len(params_df) == 0:
        raise RuntimeError("No posterior found. Did ABC finish at least one population?")
    w = np.asarray(w, float)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)

    # Epsilon trajectory
    pops = h.get_all_populations()
    eps_df = pd.DataFrame({"t": pops["t"].to_numpy(), "epsilon": pops["epsilon"].to_numpy()})
    eps_df.to_csv(outdir / "epsilon_trajectory.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(eps_df["t"], eps_df["epsilon"], marker="o")
    plt.xlabel("Population")
    plt.ylabel("Epsilon")
    plt.title("Epsilon trajectory")
    plt.tight_layout()
    plt.savefig(outdir / "epsilon_trajectory.png")
    plt.close()

    # Parameter marginals
    for c in params_df.columns:
        plt.figure(figsize=(8, 5))
        _safe_kdeplot_1d(params_df[c].to_numpy(float), w)
        plt.title(f"Posterior marginal: {c}")
        plt.xlabel(c)
        plt.ylabel("density")
        plt.tight_layout()
        plt.savefig(outdir / f"marginal_{c}.png")
        plt.close()

    # Correlation heatmap
    corr = params_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="vlag", center=0)
    plt.title("Posterior correlation")
    plt.tight_layout()
    plt.savefig(outdir / "posterior_correlation.png")
    plt.close()

    # Posterior predictive checks (re-sim)
    rng = np.random.default_rng(2025)
    idxs = rng.choice(np.arange(len(params_df)), size=min(args.pp, len(params_df)), replace=False, p=w)
    sims = []
    factory = make_model_factory(seed=777)
    full_order = ["S0", "S1", "S2", "SSNND_med"]
    col_idx = [full_order.index(s) for s in stats]
    for i in idxs:
        p = params_df.iloc[i].to_dict()
        params = particle_to_params(p, motion=args.motion, speed_dist=args.speed)
        sim_full = simulate_timeseries(factory, params, total_steps=args.total_steps, sample_steps=tuple(timesteps))
        sims.append(sim_full[:, col_idx])
    sims = np.asarray(sims)  # N x T x K
    med = np.median(sims, axis=0)
    q05 = np.quantile(sims, 0.05, axis=0)
    q95 = np.quantile(sims, 0.95, axis=0)
    t = np.array(timesteps)

    for k, s in enumerate(stats):
        pd.DataFrame(med[:, k], columns=[s]).to_csv(outdir / f"ppc_ts_median_{s}.csv", index=False)
        pd.DataFrame(q05[:, k], columns=[s]).to_csv(outdir / f"ppc_ts_q05_{s}.csv", index=False)
        pd.DataFrame(q95[:, k], columns=[s]).to_csv(outdir / f"ppc_ts_q95_{s}.csv", index=False)
        plt.figure(figsize=(10, 4))
        plt.fill_between(t, q05[:, k], q95[:, k], color="#cfe8ff", alpha=0.8, label="5–95% band")
        plt.plot(t, med[:, k], color="#1f77b4", lw=1.8, label="posterior median")
        plt.plot(t, obs_mat[:, k], color="black", lw=1.2, label="observed")
        plt.xlabel("timestep")
        plt.ylabel(s)
        plt.title(f"Time-series PPC — {s}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"ppc_ts_{s}.png")
        plt.close()

    # Discrepancy over time
    disc_per_draw = [np.sqrt(np.sum((sim - obs_mat) ** 2, axis=1)) for sim in sims]
    disc_per_draw = np.asarray(disc_per_draw)
    disc_med = np.median(disc_per_draw, axis=0)
    disc_q05 = np.quantile(disc_per_draw, 0.05, axis=0)
    disc_q95 = np.quantile(disc_per_draw, 0.95, axis=0)
    pd.DataFrame({"timestep": t, "disc_med": disc_med, "disc_q05": disc_q05, "disc_q95": disc_q95}).to_csv(
        outdir / "ppc_discrepancy.csv", index=False
    )
    plt.figure(figsize=(10, 4))
    plt.fill_between(t, disc_q05, disc_q95, alpha=0.3, color="#ffd2cc", label="5–95% band")
    plt.plot(t, disc_med, color="#d62728", lw=1.6, label="median discrepancy")
    plt.title("Discrepancy vs time")
    plt.xlabel("timestep")
    plt.ylabel("Euclidean discrepancy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "ppc_discrepancy.png")
    plt.close()

    # Final-population PCA biplot
    plot_pca_final_biplot(params_df, w, outdir, top_n=8)
    print("Analysis complete. Outputs saved to results/ .")

if __name__ == "__main__":
    main()
