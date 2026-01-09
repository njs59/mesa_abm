
#!/usr/bin/env python3
"""
Analyze motion-grid runs at a COMMON population index, overlay vs data, and plot diagnostics.

Robust to:
- Different #populations across runs.
- Missing particle distributions at the chosen t_common (falls back to last non-empty t per run).
- Paths: derives results_dir from --summary.

USAGE (from inside motiongrid_pkg/):
  python analyze_motiongrid.py \
    --summary results/summary.csv \
    --obs_csv ../INV_summary_stats.csv \
    --dataset INV \
    --outdir analysis_results \
    --pp_samples 100
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pyabc

# --- Import your ABM and defaults ---
from clusters_abm.clusters_model import ClustersModel
from clusters_abm.utils import DEFAULTS

# ---------- Summary stats ----------
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

# ---------- Parameter mapping ----------
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

# ---------- Robust distribution fetch ----------
def robust_distribution(history: pyabc.History, t_target: int):
    """
    Return (df, w, t_used) for the first non-empty distribution at or below t_target.
    """
    # Clamp to max_t
    t = min(t_target, history.max_t)
    while t >= 0:
        df, w = history.get_distribution(m=0, t=t)
        if len(df) > 0:
            return df, w, t
        t -= 1
    raise RuntimeError("No non-empty populations found in history.")

# ---------- Posterior predictive at a GIVEN population t ----------
def posterior_predictive_at_t(
    db_path: Path,
    obs: np.ndarray,
    start_step: int,
    motion: str,
    speed_dist: str,
    fixed_n_clusters: int,
    t_pop: int,
    n_sims: int = 50,
    seed: int = 0
):
    rng = np.random.default_rng(seed)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    history = pyabc.History(f"sqlite:///{db_path.resolve()}")

    # Find the first available non-empty distribution at or below t_pop
    df, w, t_used = robust_distribution(history, t_pop)

    # Normalise weights and draw indices
    w = np.asarray(w, dtype=float)
    w = w / (w.sum() if w.sum() > 0 else 1.0)
    idx = np.arange(len(df))
    draws = rng.choice(idx, size=min(n_sims, len(df)), replace=False, p=w)

    # Simulate and collect statistics
    T = obs.shape[0]
    sims = np.zeros((len(draws), T, 3), dtype=float)
    for j, i in enumerate(draws):
        particle = {k: float(df.iloc[i][k]) for k in df.columns}
        params = make_params_from_particle(DEFAULTS, particle, motion=motion, speed_dist=speed_dist, fixed_n_clusters=fixed_n_clusters)
        seg = simulate_timeseries(params, steps=start_step + T, seed=int(rng.integers(0, 2**31 - 1)))[start_step : start_step + T, :]
        sims[j, :, :] = seg

    med = np.median(sims, axis=0)
    q5  = np.quantile(sims, 0.05, axis=0)
    q95 = np.quantile(sims, 0.95, axis=0)
    return med, q5, q95, t_used  # return t_used for transparency

# ---------- QQ plot utility ----------
def qq_plot(ax, model_series: np.ndarray, obs_series: np.ndarray, label: str = ""):
    q = np.linspace(0, 1, 1000)
    mq = np.quantile(model_series, q)
    oq = np.quantile(obs_series, q)
    ax.plot(oq, mq, lw=1.5, label=label)
    ax.plot([oq.min(), oq.max()], [oq.min(), oq.max()], color="grey", lw=1, ls="--")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Analyze motion-grid runs at a common population, overlay vs data, and plot diagnostics.")
    ap.add_argument("--summary", type=str, default="results/summary.csv", help="Path to summary.csv")
    ap.add_argument("--obs_csv", type=str, required=True, help="Path to observed CSV with S0,S1,S2")
    ap.add_argument("--dataset", type=str, default="INV")
    ap.add_argument("--outdir", type=str, default="analysis_results")
    ap.add_argument("--pp_samples", type=int, default=100)
    ap.add_argument("--variants", nargs="*", default=None,
                    help="Optional list of variants (e.g. isotropic_gamma persistent_lognorm); if omitted, uses all rows in summary.")
    args = ap.parse_args()

    summary_path = Path(args.summary).resolve()
    results_dir = summary_path.parent  # derive where DBs live
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_path)
    obs_df = pd.read_csv(args.obs_csv)
    if not all(c in obs_df.columns for c in ("S0", "S1", "S2")):
        raise ValueError("Observed CSV must contain S0,S1,S2")
    obs = obs_df[["S0", "S1", "S2"]].to_numpy(dtype=float)

    summary["variant"] = summary["motion"] + "_" + summary["speed_dist"]
    df = summary[summary["dataset"] == args.dataset].copy()
    if args.variants:
        df = df[df["variant"].isin(args.variants)].copy()
    if df.empty:
        raise RuntimeError("No rows selected from summary for the given dataset/variants.")

    # COMMON population index (zero-based)
    min_pops = int(df["n_populations"].min())
    t_common = max(0, min_pops - 2)
    print(f"\nComparing all runs at COMMON population t = {t_common} (min of n_populations - 2).")
    print(f"Results directory detected: {results_dir}")

    # --- Epsilon trajectories ---
    plt.figure(figsize=(8, 5))
    for _, row in df.iterrows():
        db = results_dir / row["db"]
        if not db.exists():
            raise FileNotFoundError(f"Cannot find DB: {db}")
        history = pyabc.History(f"sqlite:///{db.resolve()}")
        pops = history.get_all_populations()
        eps = pops["epsilon"].to_numpy()
        plt.plot(np.arange(len(eps)), eps, marker="o", lw=1.5, label=row["variant"])
    plt.xlabel("Population t"); plt.ylabel("ε (IQR-scaled)")
    plt.title(f"Epsilon trajectories ({args.dataset})")
    plt.legend(ncol=2, fontsize=8); plt.tight_layout()
    plt.savefig(outdir / f"{args.dataset}_epsilon_trajectories.png", dpi=160); plt.close()

    # --- Posterior predictive medians/bands at common t (with per-run fallback info) ---
    colours = {"isotropic": "#1f77b4", "persistent": "#ff7f0e"}
    store = {}
    stats_names = ["S0", "S1", "S2"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    T = obs.shape[0]; x = np.arange(T)

    used_ts = []  # track t_used per run
    for _, row in tqdm(df.iterrows(), desc="Posterior predictive (common population)", total=len(df)):
        db = results_dir / row["db"]
        motion, speed = row["motion"], row["speed_dist"]
        start_step = int(row["start_step"])
        fixed_n = int(row.get("init_cells", max(1, round(obs[0, 0]))))

        med, q5, q95, t_used = posterior_predictive_at_t(
            db_path=db, obs=obs, start_step=start_step, motion=motion, speed_dist=speed,
            fixed_n_clusters=fixed_n, t_pop=t_common, n_sims=args.pp_samples
        )
        used_ts.append((row["variant"], t_used))
        store[row["variant"]] = (med, q5, q95)

        for k, ax in enumerate(axes):
            ax.plot(x, med[:, k], lw=1.6, label=row["variant"], color=colours.get(motion, "grey"))

    # Overlay observed data
    for k, ax in enumerate(axes):
        ax.plot(x, obs[:, k], lw=2.0, color="black", label="Observed")
        ax.set_title(stats_names[k]); ax.set_xlabel("t (aligned steps)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Value")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle(f"Overlay vs data (common population t={t_common}, with per-run fallback if needed) — {args.dataset}")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(outdir / f"{args.dataset}_overlay_vs_data_common_t{t_common}.png", dpi=160)
    plt.close()

    # Print any fallbacks that happened
    fallbacks = [(v, tu) for v, tu in used_ts if tu != t_common]
    if fallbacks:
        print("\n⚠️ Some runs had no particles at t_common; fell back to:")
        for v, tu in fallbacks:
            print(f" - {v}: used t={tu}")

    # --- ε at common t (bar; clamped as needed) ---
    eps_at_common = []
    for _, row in df.iterrows():
        db = results_dir / row["db"]
        history = pyabc.History(f"sqlite:///{db.resolve()}")
        pops = history.get_all_populations()
        t_use = min(t_common, len(pops) - 1)
        eps_at_common.append((row["variant"], float(pops.iloc[t_use]["epsilon"])))
    eps_df = pd.DataFrame(eps_at_common, columns=["variant", "epsilon_common"])

    order = eps_df.sort_values("epsilon_common")["variant"]
    plt.figure(figsize=(9, 5))
    colours_bars = [colours[v.split("_")[0]] for v in order]
    plt.bar(order, eps_df.set_index("variant").loc[order, "epsilon_common"], color=colours_bars)
    plt.axhline(0.30, color="green", ls="--", lw=1, label="Excellent ≤ 0.30")
    plt.axhline(0.50, color="orange", ls="--", lw=1, label="Good 0.31–0.50")
    plt.axhline(0.85, color="red", ls="--", lw=1, label="Acceptable 0.51–0.85")
    plt.title(f"ε at common population t={t_common} — {args.dataset}")
    plt.ylabel("ε (IQR-scaled)"); plt.xticks(rotation=45, ha="right"); plt.legend()
    plt.tight_layout(); plt.savefig(outdir / f"{args.dataset}_epsilon_bar_common_t{t_common}.png", dpi=160); plt.close()

    # --- Coverage bars (from summary at final population) ---
    plt.figure(figsize=(10, 5))
    idx = np.arange(len(df)); w = 0.28
    plt.bar(idx - w, df["coverage_S0"], width=w, label="S0")
    plt.bar(idx       , df["coverage_S1"], width=w, label="S1")
    plt.bar(idx + w, df["coverage_S2"], width=w, label="S2")
    plt.xticks(idx, df["variant"], rotation=45, ha="right")
    plt.ylim(0, 1.05); plt.ylabel("Coverage (fraction within 5–95% band)")
    plt.title(f"Posterior predictive coverage (final-pop summary) — {args.dataset}")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"{args.dataset}_coverage_bars_summary.png", dpi=160); plt.close()

    # --- QQ plots for top-2 variants at (fallback) t_used ---
    top2 = order.tolist()[:2]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for k, ax in enumerate(axes):
        for v in top2:
            med, q5, q95 = store[v]
            qq_plot(ax, med[:, k], obs[:, k], label=v)
        ax.set_title(f"QQ: {['S0','S1','S2'][k]}")
        ax.set_xlabel("Observed quantiles"); ax.set_ylabel("Model quantiles")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"QQ plots (median vs observed) at common/fallback t — {args.dataset}")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(outdir / f"{args.dataset}_qq_common_fallback.png", dpi=160); plt.close()

    print(f"\nSaved figures to: {outdir.resolve()}")
    for p in sorted(outdir.glob("*.png")):
        print(" -", p.name)

if __name__ == "__main__":
    main()
