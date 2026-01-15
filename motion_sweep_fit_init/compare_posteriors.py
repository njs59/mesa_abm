
#!/usr/bin/env python3
"""
Compare posterior distributions across selected runs (variants).

- Loads particles from pyABC DBs via summary.csv.
- Compares at common population t = min(n_populations)-1, with robust fallback per run.
- Overlay KDE/hist per parameter (weighted resampling to unweighted draws).
- Corner/pair plot across variants (hue=variant).
- Divergence metrics (overlap, JS) only where both variants have the parameter.
"""

import argparse
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyabc

# Candidate parameters (script will filter to those actually present)
PARAMS_ORDER = [
    "prolif_rate", "adhesion", "fragment_rate", "merge_prob",
    "heading_sigma", "speed_meanlog", "speed_sdlog", "speed_shape", "speed_scale", "init_n_clusters",
]

def robust_distribution(history: pyabc.History, t_target: int):
    """Return (df, w, t_used) for the first non-empty distribution at or below t_target."""
    t = min(int(t_target), int(history.max_t))
    while t >= 0:
        df, w = history.get_distribution(m=0, t=t)
        if len(df) > 0:
            return df, w, t
        t -= 1
    raise RuntimeError("No non-empty populations in history.")

def weighted_resample(df: pd.DataFrame, weights: np.ndarray, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Resample n rows from df with probability proportional to weights (with replacement)."""
    w = np.asarray(weights, float)
    p = w / (w.sum() if w.sum() > 0 else 1.0)
    idx = rng.choice(np.arange(len(df)), size=max(1, n), replace=True, p=p)
    return df.iloc[idx].reset_index(drop=True)

def binned_metrics(x: np.ndarray, y: np.ndarray, bins: int = 50):
    """Binned overlap and Jensen–Shannon divergence (base-2)."""
    rng = np.random.default_rng(0)
    pooled = np.concatenate([x, y])
    if np.allclose(pooled.min(), pooled.max()):
        pooled = pooled + rng.normal(0, 1e-9, size=pooled.shape)
    edges = np.linspace(pooled.min(), pooled.max(), bins + 1)
    p_hist, _ = np.histogram(x, bins=edges, density=True)
    q_hist, _ = np.histogram(y, bins=edges, density=True)
    p = p_hist / (p_hist.sum() if p_hist.sum() > 0 else 1.0)
    q = q_hist / (q_hist.sum() if q_hist.sum() > 0 else 1.0)
    m = 0.5 * (p + q)
    eps = 1e-12
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2((a[mask] + eps) / (b[mask] + eps)))
    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    overlap = np.sum(np.minimum(p, q))
    return float(overlap), float(js)

def main():
    ap = argparse.ArgumentParser(description="Compare posterior distributions for selected runs.")
    ap.add_argument("--summary", type=str, default="results/summary.csv", help="Path to summary.csv")
    ap.add_argument("--dataset", type=str, required=True, help="Dataset label (e.g., INV, PRO)")
    ap.add_argument("--variants", nargs="+", default=None, help="List of variant labels, e.g., isotropic_gamma persistent_lognorm")
    ap.add_argument("--outdir", type=str, default="post_compare", help="Output directory")
    ap.add_argument("--pp_samples", type=int, default=5000, help="Weighted resample size per posterior")
    ap.add_argument("--t_pop", type=int, default=None, help="Population index (0-based). If omitted, uses min(n_populations)-1.")
    ap.add_argument("--downsample_pairplot", type=int, default=3000, help="Max rows per variant for the pair plot")
    args = ap.parse_args()

    summary_path = Path(args.summary).resolve()
    results_dir = summary_path.parent
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_path)
    summary["variant"] = summary["motion"] + "_" + summary["speed_dist"]
    rows = summary[summary["dataset"] == args.dataset].copy()
    if args.variants:
        rows = rows[rows["variant"].isin(args.variants)].copy()
    if rows.empty:
        raise RuntimeError("No rows selected. Check dataset and variant names.")

    # Common population index
    if args.t_pop is None:
        min_pops = int(rows["n_populations"].min())
        t_common = max(0, min_pops - 1)
    else:
        t_common = max(0, int(args.t_pop))
    print(f"[INFO] Comparing at common population t={t_common}")

    # Colours
    variants = rows["variant"].tolist()
    palette = sns.color_palette("tab10", n_colors=len(variants))
    variant_colours = dict(zip(variants, palette))

    # Collect resampled posteriors
    rng = np.random.default_rng(42)
    posterior_samples = {}   # variant -> DataFrame (resampled)
    used_population = {}     # variant -> t_used (after fallback)
    params_present = {}      # variant -> set of parameter columns available

    for _, r in rows.iterrows():
        db = results_dir / r["db"]
        if not db.exists():
            raise FileNotFoundError(f"DB not found: {db}")
        hist = pyabc.History(f"sqlite:///{db.resolve()}")
        df_post, w_post, t_used = robust_distribution(hist, t_common)
        used_population[r["variant"]] = t_used
        # Filter to numeric parameters in our candidate list
        use_cols = [p for p in PARAMS_ORDER if p in df_post.columns]
        params_present[r["variant"]] = set(use_cols)
        samples = weighted_resample(df_post[use_cols], w_post, n=args.pp_samples, rng=rng)
        samples["variant"] = r["variant"]
        posterior_samples[r["variant"]] = samples

    # Fallback notice
    fb = [(v, t) for v, t in used_population.items() if t != t_common]
    if fb:
        print("[WARN] Fallback populations used (no particles at t_common):")
        for v, t in fb:
            print(f"  - {v}: used t={t}")

    # Build the global plotting parameter set dynamically
    params_plot = sorted(set().union(*(params_present[v] for v in variants)))
    if not params_plot:
        raise RuntimeError("No common parameters available to plot across selected variants.")

    # --- Overlay KDEs per parameter (skip absent ones per variant) ---
    sns.set(style="whitegrid")
    for p in params_plot:
        plt.figure(figsize=(9, 5))
        plotted_any = False
        for v in variants:
            vsamp = posterior_samples[v]
            if p not in vsamp.columns:
                continue
            plotted_any = True
            try:
                sns.kdeplot(x=vsamp[p], label=v, color=variant_colours[v], lw=1.8)
            except Exception:
                sns.histplot(vsamp[p], label=v, color=variant_colours[v],
                             stat="density", bins=40, alpha=0.35)
        if not plotted_any:
            plt.close()
            continue
        plt.title(f"Posterior overlay — {p} (t≈{t_common})")
        plt.xlabel(p); plt.ylabel("Density"); plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"post_overlay_{p}_t{t_common}.png", dpi=160)
        plt.close()

    # --- Corner / pair plot with hue=variant ---
    concat_rows = []
    for v in variants:
        dfv = posterior_samples[v]
        if args.downsample_pairplot and len(dfv) > args.downsample_pairplot:
            dfv = dfv.sample(n=args.downsample_pairplot, random_state=0)
        concat_rows.append(dfv)
    big = pd.concat(concat_rows, axis=0, ignore_index=True)

    # Only keep columns present in at least one variant and not the label
    use_cols = [p for p in params_plot if p in big.columns]
    if use_cols:
        g = sns.pairplot(big[use_cols + ["variant"]], corner=True, hue="variant",
                         plot_kws={"s": 10, "alpha": 0.4})
        g.fig.suptitle(f"Corner (pair) plot across selected variants (t≈{t_common})", y=1.02)
        g.fig.savefig(outdir / "post_pairplot_selected.png", dpi=160)
        plt.close(g.fig)

    # --- Divergence metrics per parameter where both have it ---
    metric_rows = []
    for p in params_plot:
        for v1, v2 in combinations(variants, 2):
            df1 = posterior_samples[v1]; df2 = posterior_samples[v2]
            if p not in df1.columns or p not in df2.columns:
                continue
            x = df1[p].to_numpy(float)
            y = df2[p].to_numpy(float)
            overlap, js = binned_metrics(x, y, bins=60)
            metric_rows.append({"parameter": p, "variant_a": v1, "variant_b": v2,
                                "overlap": overlap, "js_divergence": js})
    if metric_rows:
        mdf = pd.DataFrame(metric_rows)
        mdf.to_csv(outdir / "post_metrics.csv", index=False)

    print(f"[DONE] Posterior comparison saved to: {outdir.resolve()}")
    # List a few outputs
    for p in params_plot[:5]:
        fp = outdir / f"post_overlay_{p}_t{t_common}.png"
        if fp.exists():
            print(f" - {fp.name}")
    pair_path = outdir / "post_pairplot_selected.png"
    if pair_path.exists():
        print(f" - {pair_path.name}")
    metrics_path = outdir / "post_metrics.csv"
    if metrics_path.exists():
        print(f"Metrics: {metrics_path.name}")

if __name__ == "__main__":
    main()
