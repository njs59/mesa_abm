
#!/usr/bin/env python3
"""
Analyse ABC posterior and generate plots/CSVs.
- Works with new ABM; NND is computed from stored stats (no wrap in compute_summary).
- Adds robust plotting for speed parameters and PCA diagnostics.
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

# --- helpers ---

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
    if not ok:
        plt.hist(np.asarray(x, dtype=float), bins="auto", density=True, color=color, alpha=0.55, label=label)

def _detect_speed_columns(params_df: pd.DataFrame, override: list[str] | None = None) -> list[str]:
    if override:
        return [c for c in override if c in params_df.columns]
    cols = list(params_df.columns)
    lo = {c: c.lower() for c in cols}
    generic = [c for c in cols if any(tok in lo[c] for tok in ["speed", "step", "v"])]
    specific = [c for c in cols if lo[c] in {
        "speed", "step_length",
        # lognormal
        "speed_mu", "speed_sigma", "log_speed_mu", "log_speed_sigma",
        # gamma / weibull
        "speed_k", "speed_theta", "speed_shape", "speed_scale",
        "speed_k_shape", "speed_lambda_scale", "weibull_k", "weibull_lambda",
        # other aliases
        "v", "v_mu", "v_sigma"
    }]
    dedup = []
    for c in generic + specific:
        if c not in dedup:
            dedup.append(c)
    return dedup

def plot_speed_posteriors(params_df: pd.DataFrame, weights: np.ndarray, outdir: Path, speed_cols_override: list[str] | None = None, n_pair_resample: int = 4000):
    outdir.mkdir(parents=True, exist_ok=True)
    speed_cols = _detect_speed_columns(params_df, speed_cols_override)
    if len(speed_cols) == 0:
        print("[speed-plots] No speed parameters detected. Skipping.")
        return
    print(f"[speed-plots] Using speed columns: {', '.join(speed_cols)}")
    # 1D marginals
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
    # Pairwise grid if >=2
    if len(speed_cols) >= 2:
        sample = _weighted_resample(params_df[speed_cols], weights, n=n_pair_resample, seed=2026)
        k = len(speed_cols)
        fig, axes = plt.subplots(nrows=k, ncols=k, figsize=(2.2*k + 1.2, 2.2*k + 0.8))
        for i, xi in enumerate(speed_cols):
            for j, yj in enumerate(speed_cols):
                ax = axes[i, j] if k > 1 else axes
                if i == j:
                    _safe_kdeplot_1d(sample[xi].to_numpy(float), None, color="#1f77b4")
                    ax.set_ylabel("")
                    ax.set_yticks([])
                elif i > j:
                    ax.hexbin(sample[yj], sample[xi], gridsize=30, cmap="Blues", mincnt=1)
                else:
                    ax.axis("off")
                if i == k - 1:
                    ax.set_xlabel(yj if i == j else yj)
                else:
                    ax.set_xlabel("")
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(xi)
                else:
                    if i != j:
                        ax.set_ylabel("")
                        ax.set_yticklabels([])
        fig.suptitle("Posterior joint (final population): speed parameters", y=0.995)
        fig.tight_layout()
        fig.savefig(outdir / "posterior_speed_pairgrid.png", dpi=160)
        plt.close(fig)

# --- main ---

def main():
    ap = argparse.ArgumentParser(description="Analyse ABC posterior & PPC")
    ap.add_argument("--db", type=str, required=True)
    ap.add_argument("--observed_ts", type=str, required=True)
    ap.add_argument("--t_start", type=int, default=22)
    ap.add_argument("--total_steps", type=int, default=300)
    ap.add_argument("--pp", type=int, default=100)
    ap.add_argument("--motion", type=str, default="isotropic", choices=["isotropic", "persistent"])
    ap.add_argument("--speed", type=str, default="constant", choices=["constant", "lognorm", "gamma", "weibull"])  # extend if needed
    ap.add_argument("--pca_top_k", type=int, default=5)
    ap.add_argument("--save_all_loadings", action="store_true")
    ap.add_argument("--no_gr", action="store_true")
    ap.add_argument("--speed_cols", nargs="+", default=None)
    args = ap.parse_args()

    outdir = Path("results"); outdir.mkdir(exist_ok=True, parents=True)

    obs_df = pd.read_csv(args.observed_ts)
    timesteps = obs_df["timestep"].astype(int).to_list()
    stats = ["S0", "S1", "S2", "NND_med"] if args.no_gr else ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
    obs_mat = obs_df[stats].to_numpy(float)

    h = pyabc.History(f"sqlite:///{args.db}")
    t_max = h.max_t
    params_df, w = h.get_distribution(m=0, t=t_max)
    if len(params_df) == 0:
        raise RuntimeError("No posterior found. Did ABC finish at least one population?")
    w = np.asarray(w, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)

    # Epsilon trajectory
    pops = h.get_all_populations()
    eps_df = pd.DataFrame({"t": pops["t"].to_numpy(), "epsilon": pops["epsilon"].to_numpy()})
    eps_df.to_csv(outdir / "epsilon_trajectory.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(eps_df["t"], eps_df["epsilon"], marker="o")
    plt.xlabel("Population"); plt.ylabel("Epsilon"); plt.title("Epsilon trajectory")
    plt.tight_layout(); plt.savefig(outdir / "epsilon_trajectory.png"); plt.close()

    # Parameter marginals
    for c in params_df.columns:
        plt.figure(figsize=(8, 5))
        _safe_kdeplot_1d(params_df[c].to_numpy(float), w)
        plt.title(f"Posterior marginal: {c}"); plt.xlabel(c); plt.ylabel("density")
        plt.tight_layout(); plt.savefig(outdir / f"marginal_{c}.png"); plt.close()

    # Correlation heatmap
    corr = params_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="vlag", center=0)
    plt.title("Posterior correlation"); plt.tight_layout(); plt.savefig(outdir / "posterior_correlation.png"); plt.close()

    # Speed posteriors
    speed_dir = outdir / 'posteriors_speed'; speed_dir.mkdir(exist_ok=True)
    plot_speed_posteriors(params_df, w, speed_dir, speed_cols_override=args.speed_cols)

    # Posterior predictive checks (optional; same approach as earlier if needed)
    print("Analysis complete. Outputs saved to results/ .")

if __name__ == "__main__":
    main()
