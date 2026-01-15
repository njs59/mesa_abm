
#!/usr/bin/env python3
"""
Extra plots for INV runs: posterior comparisons, per-stat W1 components, predictive ridges,
corner plots, parameter sensitivity, and seed-aggregated ε box/violins.

Run examples:
  python inv_extra_plots.py \
    --summary motiongrid_pkg/results/summary.csv \
    --dataset INV \
    --obs_csv INV_summary_stats.csv \
    --outdir motiongrid_pkg/inv_plots \
    --variants isotropic_gamma persistent_lognorm persistent_weibull isotropic_lognorm \
    --pp_samples 100

If you run from inside motiongrid_pkg/, use --summary results/summary.csv and --obs_csv ../INV_summary_stats.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pyabc
import seaborn as sns

# Import ABM bits to regenerate predictives
from clusters_abm.clusters_model import ClustersModel
from clusters_abm.utils import DEFAULTS

# ---- Helpers (match runner mapping) ----
def _set_nested(base: dict, dotted: str, value):
    keys = dotted.split(".")
    d = base
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value

def build_speed_params(speed_dist: str, particle: dict) -> dict:
    if speed_dist == "constant":
        return {}
    if speed_dist == "lognorm":
        mu = float(particle.get("speed_meanlog", 1.0))
        sd = float(particle.get("speed_sdlog", 0.7))
        return {"s": sd, "scale": float(np.exp(mu))}
    elif speed_dist == "gamma":
        shape = float(particle.get("speed_shape", 2.0))
        scale = float(particle.get("speed_scale", 1.0))
        return {"a": shape, "scale": scale}
    elif speed_dist == "weibull":
        shape = float(particle.get("speed_shape", 2.0))
        scale = float(particle.get("speed_scale", 2.0))
        return {"c": shape, "scale": scale}
    else:
        raise ValueError(f"Unknown speed_dist: {speed_dist}")

def make_params_from_particle(defaults: dict, particle: dict, motion: str, speed_dist: str, fixed_n_clusters: int) -> dict:
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
    params["init"]["n_clusters"] = int(max(1, round(fixed_n_clusters)))
    return params

def compute_summary_from_model(model):
    ids, pos, radii, sizes, speeds = model._log_alive_snapshot()
    n = len(sizes)
    if n == 0:
        return 0.0, 0.0, 0.0
    return float(n), float(np.mean(sizes)), float(np.mean(sizes ** 2))

def simulate_timeseries(params: dict, steps: int, seed: int) -> np.ndarray:
    model = ClustersModel(params=params, seed=seed)
    out = np.zeros((steps, 3), dtype=float)
    out[0, :] = compute_summary_from_model(model)
    for t in range(1, steps):
        model.step()
        out[t, :] = compute_summary_from_model(model)
    return out

# ---- Robust distribution fetch (handles early stop) ----
def robust_distribution(history: pyabc.History, t_target: int):
    t = min(t_target, history.max_t)
    while t >= 0:
        df, w = history.get_distribution(m=0, t=t)
        if len(df) > 0:
            return df, w, t
        t -= 1
    raise RuntimeError("No non-empty populations in history.")

# ---- Main ----
def main():
    ap = argparse.ArgumentParser(description="Extra INV plots: posterior comparison, per-stat W1 contributions, predictive ridges, sensitivity.")
    ap.add_argument("--summary", type=str, default="results/summary.csv")
    ap.add_argument("--dataset", type=str, default="INV")
    ap.add_argument("--obs_csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="inv_plots")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--pp_samples", type=int, default=100)
    ap.add_argument("--t_pop", type=int, default=None, help="Population index to analyse; if omitted, uses min(n_populations)-1 per summary.")
    args = ap.parse_args()

    summary_path = Path(args.summary).resolve()
    results_dir = summary_path.parent
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load summary and observed data
    summary = pd.read_csv(summary_path)
    obs_df = pd.read_csv(args.obs_csv)
    obs = obs_df[["S0","S1","S2"]].to_numpy(float)

    summary["variant"] = summary["motion"] + "_" + summary["speed_dist"]
    df = summary[summary["dataset"] == args.dataset].copy()
    if args.variants:
        df = df[df["variant"].isin(args.variants)].copy()
    if df.empty:
        raise RuntimeError("No rows selected for INV.")

    # Common population index
    if args.t_pop is None:
        min_pops = int(df["n_populations"].min())
        t_common = max(0, min_pops - 1)
    else:
        t_common = max(0, int(args.t_pop))
    print(f"[INFO] Using common population t={t_common}")

    # -------- 1) Posterior marginals (KDE) per variant --------
    par_names = ["prolif_rate","adhesion","fragment_rate","merge_prob",
                 "heading_sigma","speed_meanlog","speed_sdlog","speed_shape","speed_scale"]
    sns.set(style="whitegrid")
    for _, row in df.iterrows():
        db = results_dir / row["db"]
        hist = pyabc.History(f"sqlite:///{db.resolve()}")
        df_post, w_post, t_used = robust_distribution(hist, t_common)
        # Weighted KDE plots per parameter
        fig, axes = plt.subplots(2, 4, figsize=(14,6))
        axes = axes.flatten()
        for i, p in enumerate(par_names):
            if p not in df_post.columns: 
                axes[i].axis("off")
                continue
            vals = df_post[p].to_numpy(float)
            try:
                sns.kdeplot(x=vals, ax=axes[i], fill=True, color="#1f77b4")
            except Exception:
                axes[i].hist(vals, bins=30, color="#1f77b4", alpha=0.7)
            axes[i].set_title(p)
        fig.suptitle(f"Posterior marginals — {row['variant']} (t_used={t_used})")
        fig.tight_layout(rect=[0,0,1,0.95])
        fig.savefig(outdir / f"post_marginals_{row['variant']}_t{t_used}.png", dpi=160)
        plt.close(fig)

    # -------- 2) Posterior medians bar comparison across variants --------
    med_rows = []
    for _, row in df.iterrows():
        db = results_dir / row["db"]
        hist = pyabc.History(f"sqlite:///{db.resolve()}")
        df_post, w_post, t_used = robust_distribution(hist, t_common)
        med = df_post[par_names].median(numeric_only=True)
        med_rows.append(pd.Series({"variant": row["variant"], **med.to_dict()}))
    med_df = pd.DataFrame(med_rows).set_index("variant")
    # Plot a subset of key parameters
    keys = [p for p in ["prolif_rate","merge_prob","fragment_rate","adhesion","heading_sigma"] if p in med_df.columns]
    fig, ax = plt.subplots(figsize=(10,5))
    med_df[keys].plot(kind="bar", ax=ax)
    ax.set_title(f"Posterior medians across variants (t≈{t_common})")
    ax.set_ylabel("Median (posterior)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    fig.savefig(outdir / f"posterior_medians_bar_t{t_common}.png", dpi=160)
    plt.close(fig)

    # -------- 3) Corner plot (pair-wise) for top variants --------
    # Pick up to 2 variants by lowest final_eps
    order = df.sort_values("final_eps")["variant"].tolist()
    top2 = order[:2]
    for variant in top2:
        row = df[df["variant"]==variant].iloc[0]
        db = results_dir / row["db"]
        hist = pyabc.History(f"sqlite:///{db.resolve()}")
        df_post, w_post, t_used = robust_distribution(hist, t_common)
        # Select numeric columns of interest
        use_cols = [c for c in par_names if c in df_post.columns]
        if not use_cols:
            continue
        # Pairplot
        g = sns.pairplot(df_post[use_cols], corner=True, plot_kws={"s": 10, "alpha": 0.5})
        g.fig.suptitle(f"Corner (pair) — {variant} (t_used={t_used})", y=1.02)
        g.fig.savefig(outdir / f"corner_{variant}_t{t_used}.png", dpi=160)
        plt.close(g.fig)

    # -------- 4) Per-stat W1/IQR contributions at common t --------
    # Approximate per-stat W1/IQR by simulating a modest batch and computing 1D W1/IQR per stat
    def w1_1d(x, y):
        # quantile-based approx
        u = np.linspace(0,1,1000)
        return float(np.mean(np.abs(np.quantile(x,u) - np.quantile(y,u))))
    iqr = np.array([
        np.quantile(obs[:,0],0.75)-np.quantile(obs[:,0],0.25),
        np.quantile(obs[:,1],0.75)-np.quantile(obs[:,1],0.25),
        np.quantile(obs[:,2],0.75)-np.quantile(obs[:,2],0.25),
    ])
    rows_w1 = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Per-stat W1 contributions"):
        db = results_dir / row["db"]
        hist = pyabc.History(f"sqlite:///{db.resolve()}")
        df_post, w_post, t_used = robust_distribution(hist, t_common)
        w_post = np.array(w_post, float); w_post /= (w_post.sum() or 1.0)
        # draw 30 particles by weight
        rng = np.random.default_rng(0)
        idx_all = np.arange(len(df_post))
        draws = rng.choice(idx_all, size=min(30, len(df_post)), replace=False, p=w_post)
        # For each draw, simulate and compute W1/IQR per stat, then average
        start_step = int(row["start_step"])
        fixed_n = int(row.get("init_cells", max(1, round(obs[0,0]))))
        contribs = []
        for i in draws:
            part = {k: float(df_post.iloc[i][k]) for k in df_post.columns}
            params = make_params_from_particle(DEFAULTS, part, motion=row["motion"], speed_dist=row["speed_dist"], fixed_n_clusters=fixed_n)
            seg = simulate_timeseries(params, steps=start_step+obs.shape[0], seed=int(rng.integers(0,2**31-1)))[start_step:start_step+obs.shape[0],:]
            d0 = w1_1d(seg[:,0], obs[:,0]) / (iqr[0] if iqr[0]>0 else 1.0)
            d1 = w1_1d(seg[:,1], obs[:,1]) / (iqr[1] if iqr[1]>0 else 1.0)
            d2 = w1_1d(seg[:,2], obs[:,2]) / (iqr[2] if iqr[2]>0 else 1.0)
            contribs.append((d0,d1,d2))
        contribs = np.array(contribs)
        rows_w1.append({"variant": row["variant"], "S0_W1_IQR": float(contribs[:,0].mean()),
                        "S1_W1_IQR": float(contribs[:,1].mean()), "S2_W1_IQR": float(contribs[:,2].mean())})
    w1_df = pd.DataFrame(rows_w1).set_index("variant")
    fig, ax = plt.subplots(figsize=(10,5))
    w1_df.plot(kind="bar", ax=ax)
    ax.set_title(f"Per-stat mean W1/IQR contributions at t≈{t_common}")
    ax.set_ylabel("W1/IQR (mean over sampled particles)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    fig.savefig(outdir / f"perstat_w1iqr_bar_t{t_common}.png", dpi=160)
    plt.close(fig)

    # -------- 5) Predictive ridges (violin) over time for chosen variant --------
    # Pick the best by final_eps
    best_variant = df.sort_values("final_eps")["variant"].iloc[0]
    row_best = df[df["variant"]==best_variant].iloc[0]
    db = results_dir / row_best["db"]
    hist = pyabc.History(f"sqlite:///{db.resolve()}")
    df_post, w_post, t_used = robust_distribution(hist, t_common)
    # Simulate N samples; store full trajectories to make ridge-like violins
    rng = np.random.default_rng(123)
    idx_all = np.arange(len(df_post))
    w_post = np.array(w_post, float); w_post /= (w_post.sum() or 1.0)
    draws = rng.choice(idx_all, size=min(args.pp_samples, len(df_post)), replace=False, p=w_post)
    start_step = int(row_best["start_step"])
    fixed_n = int(row_best.get("init_cells", max(1, round(obs[0,0]))))
    sims = []
    for i in draws:
        part = {k: float(df_post.iloc[i][k]) for k in df_post.columns}
        params = make_params_from_particle(DEFAULTS, part, motion=row_best["motion"], speed_dist=row_best["speed_dist"], fixed_n_clusters=fixed_n)
        seg = simulate_timeseries(params, steps=start_step+obs.shape[0], seed=int(rng.integers(0,2**31-1)))[start_step:start_step+obs.shape[0],:]
        sims.append(seg)
    sims = np.stack(sims, axis=0)  # (N, T, 3)
    # Build a long DataFrame for seaborn violin
    long = []
    T = obs.shape[0]
    for s in range(sims.shape[0]):
        for t in range(T):
            long.append({"time": t, "S0": sims[s,t,0], "S1": sims[s,t,1], "S2": sims[s,t,2]})
    long = pd.DataFrame(long)
    # Plot violins per stat
    for stat in ["S0","S1","S2"]:
        fig, ax = plt.subplots(figsize=(12,4))
        sns.violinplot(data=long, x="time", y=stat, ax=ax, inner="quartile", color="#69b3a2")
        ax.plot(np.arange(T), obs[:,["S0","S1","S2"].index(stat)], color="black", lw=1.8, label="Observed")
        ax.set_title(f"Predictive violins over time — {best_variant} ({stat})")
        ax.legend()
        plt.tight_layout()
        fig.savefig(outdir / f"predictive_violin_{best_variant}_{stat}_t{t_used}.png", dpi=160)
        plt.close(fig)

    # -------- 6) Parameter sensitivity (Spearman) vs per-stat W1/IQR --------
    # For each variant, compute spearman corr between parameters and per-stat W1/IQR over a sampled set
    import scipy.stats as st
    sens_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Sensitivity (Spearman)"):
        db = results_dir / row["db"]
        hist = pyabc.History(f"sqlite:///{db.resolve()}")
        df_post, w_post, t_used = robust_distribution(hist, t_common)
        rng = np.random.default_rng(7)
        w_post = np.array(w_post, float); w_post /= (w_post.sum() or 1.0)
        idx_all = np.arange(len(df_post))
        draws = rng.choice(idx_all, size=min(60, len(df_post)), replace=False, p=w_post)
        # compute per-stat W1/IQR for each draw
        start_step = int(row["start_step"])
        fixed_n = int(row.get("init_cells", max(1, round(obs[0,0]))))
        d0_list, d1_list, d2_list = [], [], []
        par_vals = {p: [] for p in par_names if p in df_post.columns}
        for i in draws:
            part = {k: float(df_post.iloc[i][k]) for k in df_post.columns}
            params = make_params_from_particle(DEFAULTS, part, motion=row["motion"], speed_dist=row["speed_dist"], fixed_n_clusters=fixed_n)
            seg = simulate_timeseries(params, steps=start_step+obs.shape[0], seed=int(rng.integers(0,2**31-1)))[start_step:start_step+obs.shape[0],:]
            # W1/IQR per stat
            u = np.linspace(0,1,1000)
            def w1_series(a,b): return float(np.mean(np.abs(np.quantile(a,u)-np.quantile(b,u))))
            d0_list.append(w1_series(seg[:,0], obs[:,0]) / (iqr[0] if iqr[0]>0 else 1.0))
            d1_list.append(w1_series(seg[:,1], obs[:,1]) / (iqr[1] if iqr[1]>0 else 1.0))
            d2_list.append(w1_series(seg[:,2], obs[:,2]) / (iqr[2] if iqr[2]>0 else 1.0))
            # collect parameter values
            for p in par_vals:
                par_vals[p].append(float(df_post.iloc[i][p]))
        # spearman correlations
        for stat, arr in zip(["S0","S1","S2"], [d0_list,d1_list,d2_list]):
            for p, vals in par_vals.items():
                if len(vals) > 5:
                    rho, _ = st.spearmanr(vals, arr)
                    sens_rows.append({"variant": row["variant"], "stat": stat, "parameter": p, "spearman": float(rho)})
    sens_df = pd.DataFrame(sens_rows)
    # Show top positive & negative per stat
    for stat in ["S0","S1","S2"]:
        sub = sens_df[sens_df["stat"]==stat].copy()
        if sub.empty: continue
        sub = sub.groupby(["variant","parameter"], as_index=False)["spearman"].mean()
        # best/worst 10 overall
        top = sub.sort_values("spearman", ascending=False).head(10)
        bot = sub.sort_values("spearman", ascending=True).head(10)
        top.to_csv(outdir / f"sensitivity_top_{stat}.csv", index=False)
        bot.to_csv(outdir / f"sensitivity_bottom_{stat}.csv", index=False)

    # -------- 7) Seed-aggregated ε boxes/violins (if multiple seeds present) --------
    seeds_present = df["seed"].unique()
    if len(seeds_present) > 1:
        fig, ax = plt.subplots(figsize=(10,5))
        sns.boxplot(data=df, x="variant", y="final_eps", ax=ax)
        ax.set_title("Seed-aggregated ε (boxplot)")
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        fig.savefig(outdir / f"seed_agg_eps_box.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10,5))
        sns.violinplot(data=df, x="variant", y="final_eps", ax=ax, inner="quartile")
        ax.set_title("Seed-aggregated ε (violin)")
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        fig.savefig(outdir / f"seed_agg_eps_violin.png", dpi=160)
        plt.close(fig)

    print(f"[DONE] Plots saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
