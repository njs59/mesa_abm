
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Candidate families (SciPy), loc=0 for non-negative support
CANDIDATES = {
    "lognorm": (stats.lognorm, dict(floc=0)),
    "gamma": (stats.gamma, dict(floc=0)),
    "weibull_min": (stats.weibull_min, dict(floc=0)),
    "rayleigh": (stats.rayleigh, dict(floc=0)),
    "invgauss": (stats.invgauss, dict(floc=0)),
    "expon": (stats.expon, dict(floc=0)),
    "gengamma": (stats.gengamma, dict(floc=0)),
}

# Number of free parameters for each distribution
K_MAP = {"lognorm": 2, "gamma": 2, "weibull_min": 2, "rayleigh": 1, "invgauss": 2, "expon": 1, "gengamma": 3}


def load_speeds(csv_path: Path) -> np.ndarray:
    """Load 'speed' column from a CSV and clean to be finite & strictly positive."""
    df = pd.read_csv(csv_path)
    if "speed" not in df.columns:
        raise ValueError(f"CSV '{csv_path.name}' must contain a 'speed' column.")
    x = df["speed"].to_numpy(dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        raise ValueError(f"No valid (finite, positive) speeds found in '{csv_path.name}'.")
    return x


def fit_series(x: np.ndarray, label: str, out_dir: Path) -> pd.DataFrame:
    """Fit candidate distributions to data x, save per-model summary and a 'best-only' plot, and return the summary DataFrame."""
    rows = []
    n = x.size

    for name, (dist, kw) in CANDIDATES.items():
        try:
            params = dist.fit(x, **kw)
            pdf = np.clip(dist.pdf(x, *params), 1e-300, np.inf)
            ll = float(np.sum(np.log(pdf)))
            k = K_MAP[name]
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll
            ks_stat, ks_p = stats.kstest(x, name, args=params)

            rows.append(dict(label=label, dist=name, params=tuple(float(p) for p in params),
                             loglike=ll, k=k, AIC=aic, BIC=bic, KS=ks_stat, KSp=ks_p))
        except Exception as e:
            rows.append(dict(label=label, dist=name, params=(), loglike=np.nan, k=np.nan,
                             AIC=np.inf, BIC=np.inf, KS=np.nan, KSp=np.nan, error=str(e)))

    df = pd.DataFrame(rows).sort_values(["AIC", "BIC"])
    df.to_csv(out_dir / f"fit_{label}.csv", index=False)

    # Plot histogram + best PDF
    xmax = np.percentile(x, 99.5)
    xgrid = np.linspace(0.0, max(xmax, x.max()) * 1.05, 600)
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = int(min(60, max(20, np.sqrt(n))))
    ax.hist(x, bins=bins, density=True, alpha=0.35, color="#1f77b4", edgecolor="white", label=f"{label} data")

    best = df.iloc[0]
    if np.isfinite(best["AIC"]) and len(best["params"]) > 0:
        y = CANDIDATES[best["dist"]][0].pdf(xgrid, *best["params"])
        ax.plot(xgrid, y, color="#d62728", lw=2.5, label=f"best (AIC): {best['dist']} = {best['AIC']:.1f}")

    ax.set_title(f"{label}: best = {best['dist']}   AIC = {best['AIC']:.1f}   BIC = {best['BIC']:.1f}")
    ax.set_xlabel("Instantaneous speed")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"fit_{label}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return df


def plot_all_models(x: np.ndarray, label: str, df: pd.DataFrame, out_dir: Path) -> None:
    """Overlay all converged model PDFs on the histogram, with AIC in the legend."""
    good = df[np.isfinite(df["AIC"]) & df["params"].apply(lambda p: isinstance(p, tuple) and len(p) > 0)].copy()
    if good.empty:
        print(f"⚠️ No successful fits to plot for {label}.")
        return
    good.sort_values("AIC", inplace=True)

    n = x.size
    xmax = np.percentile(x, 99.5)
    xgrid = np.linspace(0.0, max(xmax, x.max()) * 1.05, 600)

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    bins = int(min(60, max(20, np.sqrt(n))))
    ax.hist(x, bins=bins, density=True, alpha=0.25, color="#1f77b4", edgecolor="white", label=f"{label} data")

    colours = plt.get_cmap("tab20").colors
    for idx, row in enumerate(good.itertuples(index=False)):
        dist_name = row.dist
        params = row.params
        aic = row.AIC
        dist = CANDIDATES[dist_name][0]
        y = np.clip(dist.pdf(xgrid, *params), 1e-300, np.inf)
        lw = 2.6 if idx == 0 else 1.6
        ax.plot(xgrid, y, color=colours[idx % len(colours)], lw=lw, alpha=0.9, label=f"{dist_name} (AIC={aic:.1f})")

    ax.set_title(f"{label}: all fitted PDFs (legend shows AIC; lower is better)")
    ax.set_xlabel("Instantaneous speed")
    ax.set_ylabel("Density")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"fit_{label}_all_models.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_aic_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ΔAIC and Akaike weights."""
    df = df.copy()
    aic_min = df["AIC"].min()
    df["DeltaAIC"] = df["AIC"] - aic_min
    weights = np.exp(-0.5 * df["DeltaAIC"])
    df["Weight"] = weights / weights.sum()
    return df.sort_values("AIC")


def save_aic_table(df: pd.DataFrame, label: str, out_dir: Path):
    """Save ranked AIC table as PNG."""
    df_fmt = df[["dist", "AIC", "DeltaAIC", "Weight"]].copy()
    df_fmt["Weight"] = df_fmt["Weight"].map("{:.3f}".format)
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(df_fmt) + 1))
    ax.axis("off")
    table = ax.table(cellText=df_fmt.values, colLabels=df_fmt.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    fig.tight_layout()
    fig.savefig(out_dir / f"aic_table_{label}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def qq_plot(x: np.ndarray, dist, params, label: str, out_dir: Path):
    """Generate QQ plot for a fitted distribution."""
    x_sorted = np.sort(x)
    n = len(x_sorted)
    probs = (np.arange(1, n + 1) - 0.5) / n
    q_theoretical = dist.ppf(probs, *params)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(q_theoretical, x_sorted, s=12, alpha=0.7)
    ax.plot([q_theoretical.min(), q_theoretical.max()],
            [q_theoretical.min(), q_theoretical.max()],
            color="red", lw=1.5)
    ax.set_title(f"QQ Plot: {label} vs {dist.name}")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Observed Quantiles")
    fig.tight_layout()
    fig.savefig(out_dir / f"qq_{label}_{dist.name}.png", dpi=220)
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    out_dir = here / "fit_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    s037_csv = here / "speeds_s037.csv"
    s073_csv = here / "speeds_s073.csv"

    missing = [p.name for p in (s037_csv, s073_csv) if not p.exists()]
    if missing:
        print("❌ Missing input file(s):", ", ".join(missing))
        sys.exit(1)

    s037 = load_speeds(s037_csv)
    s073 = load_speeds(s073_csv)

    df037 = fit_series(s037, "s037", out_dir)
    df073 = fit_series(s073, "s073", out_dir)

    plot_all_models(s037, "s037", df037, out_dir)
    plot_all_models(s073, "s073", df073, out_dir)

    df037 = compute_aic_weights(df037)
    df073 = compute_aic_weights(df073)

    save_aic_table(df037, "s037", out_dir)
    save_aic_table(df073, "s073", out_dir)

    for row in df037.head(3).itertuples(index=False):
        dist = CANDIDATES[row.dist][0]
        qq_plot(s037, dist, row.params, "s037", out_dir)

    for row in df073.head(3).itertuples(index=False):
        dist = CANDIDATES[row.dist][0]
        qq_plot(s073, dist, row.params, "s073", out_dir)

    summary = pd.concat([df037.assign(group="s037"), df073.assign(group="s073")], ignore_index=True)
    summary.to_csv(out_dir / "fit_summary_all_models.csv", index=False)

    combined = summary.groupby("dist")["AIC"].sum().sort_values()
    print("✅ Finished. Outputs written to:", out_dir)
    print("Best overall distribution across both datasets:", combined.index[0])


if __name__ == "__main__":
    main()
