#!/usr/bin/env python3
"""
Richer PRO vs INV comparison per movement model (variant), reading TWO summary CSVs.
- For each variant (motion + "_" + speed_dist), creates its own subdirectory of plots.
- Compares posteriors at a common population index t = min(n_populations across INV/PRO) - 1,
  with robust fallback per dataset if particles are missing at that t.
- Outputs:
  * Overlay KDE/hist per parameter with hue = dataset (INV/PRO)
  * Corner/pair plot with hue = dataset (downsampled per dataset)
  * Divergence metrics (binned overlap, Jensen–Shannon divergence) saved to CSV

Required columns in each summary CSV:
  motion, speed_dist, db, n_populations
(dataset column in those CSVs is ignored; we tag rows as INV/PRO based on the file.)

Variant definition:
  variant = motion + "_" + speed_dist
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyabc

# Candidate parameters (script will filter to those actually present)
PARAMS_ORDER = [
    "prolif_rate", "adhesion", "fragment_rate", "merge_prob",
    "heading_sigma", "speed_meanlog", "speed_sdlog", "speed_shape", "speed_scale",
]

DATASETS = ("INV", "PRO")


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


def binned_metrics(x: np.ndarray, y: np.ndarray, bins: int = 60):
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


def load_and_tag_summary(path: Path, dataset_label: str) -> pd.DataFrame:
    """Load a summary.csv and ensure required columns exist; tag with dataset_label."""
    df = pd.read_csv(path)
    required = {"motion", "speed_dist", "db", "n_populations"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
    df["dataset"] = dataset_label
    df["variant"] = df["motion"] + "_" + df["speed_dist"]
    return df


def pick_best_row(rows: pd.DataFrame) -> pd.Series:
    """Pick a single summary row for a dataset+variant (prefer highest n_populations)."""
    if len(rows) == 1:
        return rows.iloc[0]
    return rows.sort_values(by="n_populations", ascending=False).iloc[0]


def main():
    ap = argparse.ArgumentParser(description="Richer PRO vs INV comparison per variant from two summaries.")
    ap.add_argument("--summary-inv", required=True, help="Path to summary CSV for INV")
    ap.add_argument("--summary-pro", required=True, help="Path to summary CSV for PRO")
    ap.add_argument("--outdir", default="post_PRO_INV", help="Base output directory")
    ap.add_argument("--pp_samples", type=int, default=5000, help="Weighted resample size per posterior")
    ap.add_argument("--t_pop", type=int, default=None, help="Population index (0-based). If omitted, uses min(n_populations)-1 per variant.")
    ap.add_argument("--downsample_pairplot", type=int, default=3000, help="Max rows per dataset for the pair plot")
    ap.add_argument("--variants", nargs="+", default=None, help="Optional list of variants to include (format: motion_speeddist)")
    args = ap.parse_args()

    inv_path = Path(args.summary_inv).resolve()
    pro_path = Path(args.summary_pro).resolve()

    base_outdir = Path(args.outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)

    inv_summary = load_and_tag_summary(inv_path, "INV")
    pro_summary = load_and_tag_summary(pro_path, "PRO")

    # Variants to process = union of both summaries
    all_variants = sorted(set(inv_summary["variant"]).union(set(pro_summary["variant"])))
    if args.variants:
        variants = [v for v in args.variants if v in all_variants]
        if not variants:
            raise RuntimeError("No matching variants found for the provided list.")
    else:
        variants = all_variants

    sns.set(style="whitegrid")
    rng = np.random.default_rng(42)

    for variant in variants:
        inv_rows = inv_summary[inv_summary["variant"] == variant]
        pro_rows = pro_summary[pro_summary["variant"] == variant]
        if inv_rows.empty and pro_rows.empty:
            continue

        # Common population index per variant
        if args.t_pop is None:
            nps = []
            if not inv_rows.empty:
                nps.append(int(inv_rows["n_populations"].max()))
            if not pro_rows.empty:
                nps.append(int(pro_rows["n_populations"].max()))
            if not nps:
                continue
            t_common = max(0, min(nps) - 1)
        else:
            t_common = max(0, int(args.t_pop))

        v_outdir = base_outdir / variant
        v_outdir.mkdir(parents=True, exist_ok=True)

        posterior_samples = {}  # dataset -> DataFrame (resampled)
        used_population = {}    # dataset -> t_used (after fallback)
        params_present = {}     # dataset -> set of parameter columns available

        # Resolve DB paths relative to each summary's directory
        inv_root = inv_path.parent
        pro_root = pro_path.parent

        def load_ds(ds_label: str, rows_df: pd.DataFrame, root: Path):
            if rows_df.empty:
                return
            row = pick_best_row(rows_df)
            db_path = (root / row["db"]).resolve() if root.is_dir() else Path(row["db"]).resolve()
            if not db_path.exists():
                print(f"[WARN] DB not found for {variant} ({ds_label}): {db_path}")
                return
            hist = pyabc.History(f"sqlite:///{db_path}")
            df_post, w_post, t_used = robust_distribution(hist, t_common)
            used_population[ds_label] = t_used
            use_cols = [p for p in PARAMS_ORDER if p in df_post.columns]
            if not use_cols:
                return
            params_present[ds_label] = set(use_cols)
            samples = weighted_resample(df_post[use_cols], w_post, n=args.pp_samples, rng=rng)
            samples["dataset"] = ds_label
            posterior_samples[ds_label] = samples

        load_ds("INV", inv_rows, inv_root)
        load_ds("PRO", pro_rows, pro_root)

        if not posterior_samples:
            print(f"[SKIP] {variant}: no datasets available")
            continue

        # Parameters to plot: union of those present in either dataset
        params_plot = sorted(set().union(*(params_present.get(ds, set()) for ds in posterior_samples.keys())))
        if not params_plot:
            print(f"[SKIP] {variant}: no parameters to plot")
            continue

        # Colours for datasets
        ds_list = [ds for ds in DATASETS if ds in posterior_samples]
        palette = sns.color_palette("deep", n_colors=len(ds_list))
        ds_colours = dict(zip(ds_list, palette))

        # --- Overlay KDEs per parameter ---
        for p in params_plot:
            plt.figure(figsize=(9, 5))
            plotted_any = False
            for ds in ds_list:
                vsamp = posterior_samples[ds]
                if p not in vsamp.columns:
                    continue
                plotted_any = True
                try:
                    sns.kdeplot(x=vsamp[p], label=ds, color=ds_colours[ds], lw=1.8)
                except Exception:
                    sns.histplot(vsamp[p], label=ds, color=ds_colours[ds], stat="density", bins=40, alpha=0.35)
            if not plotted_any:
                plt.close(); continue
            # Title and labels (British English)
            approx_t = t_common
            fb = [(ds, t) for ds, t in used_population.items() if t != t_common]
            subtitle = ""
            if fb:
                subtitle = " (fallback t used for: " + ", ".join([f"{ds}=t{t}" for ds, t in fb]) + ")"
            plt.title(f"Posterior overlay — {p} (t≈{approx_t}){subtitle}")
            plt.xlabel(p); plt.ylabel("Density"); plt.legend()
            plt.tight_layout()
            plt.savefig(v_outdir / f"overlay_{p}_t{approx_t}.png", dpi=160)
            plt.close()

        # --- Corner / pair plot with hue=dataset ---
        concat_rows = []
        for ds in ds_list:
            dfv = posterior_samples[ds]
            if args.downsample_pairplot and len(dfv) > args.downsample_pairplot:
                dfv = dfv.sample(n=args.downsample_pairplot, random_state=0)
            concat_rows.append(dfv)
        big = pd.concat(concat_rows, axis=0, ignore_index=True)
        use_cols = [p for p in params_plot if p in big.columns]
        if use_cols:
            g = sns.pairplot(big[use_cols + ["dataset"]], corner=True, hue="dataset",
                             plot_kws={"s": 10, "alpha": 0.4})
            g.fig.suptitle(f"Corner (pair) plot — {variant} (t≈{t_common})", y=1.02)
            g.fig.savefig(v_outdir / "pairplot_selected.png", dpi=160)
            plt.close(g.fig)

        # --- Divergence metrics (only when both datasets present) ---
        if set(ds_list) >= {"INV", "PRO"}:
            metric_rows = []
            xdf = posterior_samples["INV"]
            ydf = posterior_samples["PRO"]
            for p in params_plot:
                if p not in xdf.columns or p not in ydf.columns:
                    continue
                x = xdf[p].to_numpy(float)
                y = ydf[p].to_numpy(float)
                overlap, js = binned_metrics(x, y, bins=60)
                metric_rows.append({
                    "parameter": p,
                    "dataset_a": "INV",
                    "dataset_b": "PRO",
                    "overlap": overlap,
                    "js_divergence": js,
                })
            if metric_rows:
                mdf = pd.DataFrame(metric_rows)
                mdf.to_csv(v_outdir / "metrics_INV_vs_PRO.csv", index=False)

        # Console summary
        print(f"[DONE] {variant}: outputs in {v_outdir.resolve()}")
        # List a few outputs
        for p in params_plot[:5]:
            fp = v_outdir / f"overlay_{p}_t{t_common}.png"
            if fp.exists():
                print(f"  - {fp.name}")
        pair_path = v_outdir / "pairplot_selected.png"
        if pair_path.exists():
            print(f"  - {pair_path.name}")
        metrics_path = v_outdir / "metrics_INV_vs_PRO.csv"
        if metrics_path.exists():
            print(f"  - Metrics: {metrics_path.name}")


if __name__ == "__main__":
    main()
