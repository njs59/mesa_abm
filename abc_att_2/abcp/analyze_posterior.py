
#!/usr/bin/env python3
"""
Analyse ABC posterior and generate plots/CSVs.

This version includes:
- Robust posterior plots for **speed parameters** (auto-detected, with CLI override).
- A single **PCA biplot** for the final population (PC1 vs PC2) with loadings arrows.
- Existing outputs: PPCs, discrepancy vs time, epsilon trajectory, marginals,
  correlation heatmap, PCA over populations (EVR and loadings).

Usage (examples):
  # Full set of stats
  python analyze_posterior.py \
    --db results/abc_lognorm.db \
    --observed_ts observed/INV_ABM_ready_summary.csv \
    --t_start 22 --total_steps 300 --pp 100 \
    --motion isotropic --speed lognorm

  # Drop g(r)
  python analyze_posterior.py \
    --db results/abc_no_gr.db \
    --observed_ts observed/INV_ABM_ready_summary.csv \
    --t_start 22 --total_steps 300 --pp 100 \
    --motion isotropic --speed lognorm --no_gr

  # Force specific speed columns if auto-detect fails
  python analyze_posterior.py ... --speed_cols speed_mu speed_sigma
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


# ---------------------- helpers ----------------------
def make_model_factory(seed: int = 123):
    """Return a callable that builds the ABM with a fixed seed."""
    def factory(params_dict):
        return ClustersModel(params=params_dict, seed=seed)
    return factory


def _weighted_resample(df: pd.DataFrame, weights: np.ndarray, n: int, seed: int = 2026) -> pd.DataFrame:
    """Return a weighted bootstrap sample (with replacement) of n rows from df."""
    rng = np.random.default_rng(seed)
    p = np.asarray(weights, dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
    idx = rng.choice(np.arange(len(df)), size=n, replace=True, p=p)
    return df.iloc[idx].reset_index(drop=True)



def _safe_kdeplot_1d(x: np.ndarray, weights: np.ndarray | None, color="#1f77b4", label=None):
    """
    Version-robust 1D KDE:
      1) Try seaborn with tidy data + weights column.
      2) Fall back to KDE on a weighted resample.
      3) Fall back to a density-normalised histogram.
    """
    import pandas as pd
    ok = False

    # 1) Tidy-data call (most robust across seaborn versions)
    try:
        df = pd.DataFrame({"x": np.asarray(x, dtype=float)})
        if weights is not None:
            df["w"] = np.asarray(weights, dtype=float)
            df["w"] = df["w"] / (df["w"].sum() if df["w"].sum() > 0 else len(df["w"]))
            sns.kdeplot(data=df, x="x", weights="w", fill=True, color=color, alpha=0.8, label=label)
        else:
            sns.kdeplot(data=df, x="x", fill=True, color=color, alpha=0.8, label=label)
        ok = True
    except Exception:
        ok = False

    # 2) Weighted resample + KDE (fallback)
    if not ok:
        try:
            rng = np.random.default_rng(2026)
            n = min(5000, len(x))
            if weights is not None:
                p = np.asarray(weights, dtype=float)
                p = p / (p.sum() if p.sum() > 0 else len(p))
                idx = rng.choice(np.arange(len(x)), size=n, replace=True, p=p)
            else:
                idx = rng.choice(np.arange(len(x)), size=n, replace=True)
            xs = np.asarray(x, dtype=float)[idx]
            sns.kdeplot(xs, fill=True, color=color, alpha=0.8, label=label)
            ok = True
        except Exception:
            ok = False

    # 3) Final fallback: density histogram
    if not ok:
        plt.hist(np.asarray(x, dtype=float), bins="auto", density=True, color=color, alpha=0.55, label=label)

def _detect_speed_columns(params_df: pd.DataFrame, override: list[str] | None = None) -> list[str]:
    """Find columns likely to be speed-related. Allow explicit override."""
    if override:
        cols = [c for c in override if c in params_df.columns]
        return cols

    cols = list(params_df.columns)
    lo = {c: c.lower() for c in cols}

    generic = [c for c in cols if any(tok in lo[c] for tok in ["speed", "step", "v"])]
    specific = [c for c in cols if lo[c] in {
        "speed", "step_length",
        # lognormal:
        "speed_mu", "speed_sigma", "log_speed_mu", "log_speed_sigma",
        # gamma / weibull:
        "speed_k", "speed_theta", "speed_shape", "speed_scale",
        "speed_k_shape", "speed_lambda_scale", "weibull_k", "weibull_lambda",
        # other common aliases:
        "v", "v_mu", "v_sigma"
    }]
    dedup = []
    for c in generic + specific:
        if c not in dedup:
            dedup.append(c)
    return dedup


def plot_speed_posteriors(params_df: pd.DataFrame,
                          weights: np.ndarray,
                          outdir: Path,
                          speed_cols_override: list[str] | None = None,
                          n_pair_resample: int = 4000):
    """
    Robust, version-safe plotting of speed parameter posteriors.

    Outputs (saved in `outdir`):
      - posterior_speed_<param>.png     (1D marginals; weighted if supported)
      - posterior_speed_pairgrid.png    (pairwise hexbin on weighted resample)
      - posterior_speed_ridge.png       (violin ridge of z-scored shapes; optional if >=2)
    """
    outdir.mkdir(parents=True, exist_ok=True)

    speed_cols = _detect_speed_columns(params_df, speed_cols_override)
    if len(speed_cols) == 0:
        print("[speed-plots] No speed parameters detected (or not in override). Skipping.")
        return

    print(f"[speed-plots] Using speed columns: {', '.join(speed_cols)}")

    # ---- 1D marginals (one PNG per parameter)
    for c in speed_cols:
        x = params_df[c].to_numpy(float)
        plt.figure(figsize=(7, 4))
        _safe_kdeplot_1d(x, weights, color="#1f77b4", label=c)
        plt.title(f"Posterior (final population): {c}")
        plt.xlabel(c)
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"posterior_speed_{c}.png", dpi=160)
        plt.close()

    # ---- Pairwise grid (hexbin on weighted resample) if >=2 params
    if len(speed_cols) >= 2:
        sample = _weighted_resample(params_df[speed_cols], weights, n=n_pair_resample, seed=2026)
        k = len(speed_cols)
        fig, axes = plt.subplots(nrows=k, ncols=k, figsize=(2.2*k + 1.2, 2.2*k + 0.8))
        for i, xi in enumerate(speed_cols):
            for j, yj in enumerate(speed_cols):
                ax = axes[i, j] if k > 1 else axes
                if i == j:
                    # diagonal: 1D marginal for xi
                    _safe_kdeplot_1d(sample[xi].to_numpy(float), None, color="#1f77b4")
                    ax.set_ylabel("")
                    ax.set_yticks([])
                elif i > j:
                    # lower triangle: hexbin density
                    ax.hexbin(sample[yj], sample[xi], gridsize=30, cmap="Blues", mincnt=1)
                else:
                    # upper triangle empty
                    ax.axis("off")

                if i == k - 1:
                    ax.set_xlabel(yj if i == j else yj)
                else:
                    ax.set_xlabel("")
                    ax.set_xticklabels([])

                if j == 0 and i != j:
                    ax.set_ylabel(xi)
                elif j == 0 and i == j:
                    ax.set_ylabel(xi)
                else:
                    if i != j:
                        ax.set_ylabel("")
                    ax.set_yticklabels([])

        fig.suptitle("Posterior joint (final population): speed parameters", y=0.995)
        fig.tight_layout()
        fig.savefig(outdir / "posterior_speed_pairgrid.png", dpi=160)
        plt.close(fig)

        # ---- Ridge (violin) of shapes across parameters (z-scored)
        z = (params_df[speed_cols] - params_df[speed_cols].mean()) / params_df[speed_cols].std(ddof=0)
        z = z.melt(var_name="parameter", value_name="z_value")
        plt.figure(figsize=(7.2, 1.2 * len(speed_cols) + 0.8))
        sns.violinplot(
            data=z, y="parameter", x="z_value",
            inner=None, orient="h", cut=0, linewidth=0.7, fill=True, color="#9ecae1"
        )
        plt.title("Posterior shapes (z-scored) — speed params")
        plt.xlabel("z-score")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(outdir / "posterior_speed_ridge.png", dpi=160)
        plt.close()


def plot_pca_final_biplot(params_df: pd.DataFrame, weights: np.ndarray, outdir: Path, top_n: int = 8):
    """Create a single PCA biplot for the final population (PC1 vs PC2)."""
    if params_df.shape[1] < 2:
        print("PCA biplot skipped: <2 parameters in posterior.")
        return

    scaler = StandardScaler().fit(params_df.values)
    X = scaler.transform(params_df.values)
    pca = PCA().fit(X)
    scores = pca.transform(X)[:, :2]  # (N, 2)

    w = np.asarray(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    sizes = 20 + 2800 * (w - w.min()) / (w.max() - w.min() + 1e-12)

    rad = np.percentile(np.sqrt(scores[:, 0]**2 + scores[:, 1]**2), 85)
    loadings2 = pca.components_[:2, :].T  # (P, 2)
    lnorm = np.linalg.norm(loadings2, axis=1)
    top_idx = np.argsort(lnorm)[-min(top_n, len(lnorm)):]  # strongest loadings

    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    hb = ax.scatter(scores[:, 0], scores[:, 1], s=sizes, c=w, cmap="viridis",
                    alpha=0.65, edgecolor="none")
    cb = fig.colorbar(hb, ax=ax, pad=0.01)
    cb.set_label("posterior weight")

    for j in top_idx:
        vec = loadings2[j]
        ax.arrow(0, 0, rad * vec[0], rad * vec[1], color="#d62728", width=0.001,
                 head_width=0.08*rad, length_includes_head=True, alpha=0.9)
        ax.text(rad * vec[0]*1.07, rad * vec[1]*1.07,
                params_df.columns[j], color="#d62728",
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


# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser(
        description="Analyse ABC posterior & PPC (supports dropping g(r) and extra plots)"
    )
    ap.add_argument("--db", type=str, required=True, help="SQLite DB produced by run_abc.py")
    ap.add_argument("--observed_ts", type=str, default="observed/INV_ABM_ready_summary.csv")
    ap.add_argument("--t_start", type=int, default=22)
    ap.add_argument("--total_steps", type=int, default=300)
    ap.add_argument("--pp", type=int, default=100, help="Posterior predictive draws")
    ap.add_argument("--motion", type=str, default="isotropic", choices=["isotropic", "persistent"])
    ap.add_argument("--speed", type=str, default="constant", choices=["constant", "lognorm", "gamma", "weibull"])
    ap.add_argument("--pca_top_k", type=int, default=5, help="How many PCs to plot for loadings at final population")
    ap.add_argument("--save_all_loadings", action="store_true", help="Save PC loadings CSV across populations")
    ap.add_argument("--no_gr", action="store_true", help="Analyse only S0,S1,S2,NND_med (drop g(r))")

    # NEW: explicit override for speed parameter columns
    ap.add_argument("--speed_cols", nargs="+", default=None,
                    help="Explicit list of parameter columns to treat as speed params")

    args = ap.parse_args()
    outdir = Path("results")
    outdir.mkdir(exist_ok=True, parents=True)

    # ---------------- Load observed time-series ----------------
    obs_df = pd.read_csv(args.observed_ts)
    timesteps = obs_df["timestep"].astype(int).to_list()

    stats = ["S0", "S1", "S2", "NND_med"] if args.no_gr else ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
    obs_mat = obs_df[stats].to_numpy(float)

    # ---------------- Load posterior at final population ----------------
    h = pyabc.History(f"sqlite:///{args.db}")
    t_max = h.max_t
    params_df, w = h.get_distribution(m=0, t=t_max)
    if len(params_df) == 0:
        raise RuntimeError("No posterior found. Did ABC finish at least one population?")
    w = np.asarray(w, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)

    # ---------------- Epsilon trajectory ----------------
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

    # ---------------- Parameter marginals (final t) ----------------
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

    # --- Robust speed posteriors (this is the fixed bit) ---
    plot_speed_posteriors(params_df, w, outdir, speed_cols_override=args.speed_cols)

    # ---------------- Posterior predictive checks ----------------
    rng = np.random.default_rng(2025)
    idxs = rng.choice(
        np.arange(len(params_df)),
        size=min(args.pp, len(params_df)),
        replace=False,
        p=w,
    )
    sims = []
    factory = make_model_factory(seed=777)
    full_order = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
    col_idx = [full_order.index(s) for s in stats]

    for i in idxs:
        p = params_df.iloc[i].to_dict()
        params = particle_to_params(p, motion=args.motion, speed_dist=args.speed)
        sim_full = simulate_timeseries(
            factory, params, total_steps=args.total_steps, sample_steps=tuple(timesteps)
        )
        sim = sim_full[:, col_idx]
        sims.append(sim)
    sims = np.asarray(sims)  # N x T x K

    med = np.median(sims, axis=0)
    q05 = np.quantile(sims, 0.05, axis=0)
    q95 = np.quantile(sims, 0.95, axis=0)

    t = np.array(timesteps)
    obs_mat_sel = obs_mat

    for k, s in enumerate(stats):
        pd.DataFrame(med[:, k], columns=[s]).to_csv(outdir / f"ppc_ts_median_{s}.csv", index=False)
        pd.DataFrame(q05[:, k], columns=[s]).to_csv(outdir / f"ppc_ts_q05_{s}.csv", index=False)
        pd.DataFrame(q95[:, k], columns=[s]).to_csv(outdir / f"ppc_ts_q95_{s}.csv", index=False)

        plt.figure(figsize=(10, 4))
        plt.fill_between(t, q05[:, k], q95[:, k], color="#cfe8ff", alpha=0.8, label="5–95% band")
        plt.plot(t, med[:, k], color="#1f77b4", lw=1.8, label="posterior median")
        plt.plot(t, obs_mat_sel[:, k], color="black", lw=1.2, label="observed")
        plt.xlabel("timestep")
        plt.ylabel(s)
        plt.title(f"Time-series PPC — {s}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"ppc_ts_{s}.png")
        plt.close()

    # Discrepancy (selected stats only)
    disc_per_draw = []
    for sim in sims:
        d = np.sqrt(np.sum((sim - obs_mat_sel) ** 2, axis=1))  # per timestep
        disc_per_draw.append(d)
    disc_per_draw = np.asarray(disc_per_draw)
    disc_med = np.median(disc_per_draw, axis=0)
    disc_q05 = np.quantile(disc_per_draw, 0.05, axis=0)
    disc_q95 = np.quantile(disc_per_draw, 0.95, axis=0)
    pd.DataFrame({"timestep": t, "disc_med": disc_med, "disc_q05": disc_q05, "disc_q95": disc_q95}
                 ).to_csv(outdir / "ppc_discrepancy.csv", index=False)

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

    # ---------------- PCA across populations ----------------
    all_pca_rows = []
    all_loadings = []
    # final-population PCA for bar plots
    Xf = StandardScaler().fit_transform(params_df.values)
    pf = PCA().fit(Xf)
    loadings_f = pd.DataFrame(
        pf.components_.T,
        index=params_df.columns,
        columns=[f"PC{k+1}" for k in range(pf.components_.shape[0])],
    )
    top_k = min(args.pca_top_k, loadings_f.shape[1])
    for i in range(top_k):
        pc = f"PC{i+1}"
        plt.figure(figsize=(10, 4))
        loadings_f[pc].plot(kind="bar")
        plt.title(f"PCA loadings at final population — {pc}")
        plt.tight_layout()
        plt.savefig(outdir / f"pca_loadings_final_{pc}.png")
        plt.close()

    # EVR trajectories across populations (optional, kept from your original)
    h_all = h.get_all_populations()
    t_max = h.max_t
    for t_idx in range(t_max + 1):
        df_t, w_t = h.get_distribution(m=0, t=t_idx)
        if len(df_t) == 0:
            continue
        X = StandardScaler().fit_transform(df_t.values)
        pca = PCA().fit(X)
        evr = pca.explained_variance_ratio_
        top_k = min(args.pca_top_k, len(evr))
        for k in range(top_k):
            all_pca_rows.append({"t": t_idx, "PC": k + 1, "explained_variance_ratio": float(evr[k])})
        if args.save_all_loadings:
            L = pd.DataFrame(
                pca.components_.T, index=df_t.columns,
                columns=[f"PC{k+1}" for k in range(len(evr))],
            )
            L.insert(0, "t", t_idx)
            all_loadings.append(L)

    pca_evr_df = pd.DataFrame(all_pca_rows)
    pca_evr_df.to_csv(outdir / "pca_explained_variance_over_time.csv", index=False)
    if not pca_evr_df.empty:
        plt.figure(figsize=(10, 5))
        for k in sorted(pca_evr_df["PC"].unique()):
            sub = pca_evr_df[pca_evr_df["PC"] == k]
            plt.plot(sub["t"], sub["explained_variance_ratio"], marker="o", label=f"PC{k}")
        plt.xlabel("Population t")
        plt.ylabel("Explained variance ratio")
        plt.title("PCA explained variance across populations")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "pca_evr_trajectory.png")
        plt.close()

    if args.save_all_loadings and len(all_loadings) > 0:
        pd.concat(all_loadings, axis=0).to_csv(outdir / "pca_loadings_over_time.csv", index=False)

    # --- Single PCA biplot for final population ---
    plot_pca_final_biplot(params_df, w, outdir, top_n=8)

    print("Analysis complete. Outputs saved to results/ .")


if __name__ == "__main__":
    main()
