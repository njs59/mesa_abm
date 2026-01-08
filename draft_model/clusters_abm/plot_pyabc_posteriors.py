
# clusters_abm/plot_pyabc_posteriors.py
# Works as a script or a module:  python -m clusters_abm.plot_pyabc_posteriors ...
#
# Features:
# - Posterior marginals and pair plot (seaborn if available, otherwise matplotlib).
# - Experimental vs fitted ABM (posterior-predictive) bands built from
#   FINAL population sum stats stored by pyABC (accepted particles).
# - Robust to pyABC API variants: uses Population.get_accepted_sum_stats()
#   or a dataframe fallback; reads weights from particles or final distribution.
# - KDE matrix plotting via pyABC visualization:
#     tries plot_kde_matrix_highlevel(history, x=[...]) and falls back safely.
#
# References:
# - pyABC visualization & History APIs (official docs)  [1](https://pyabc.readthedocs.io/en/latest/api/pyabc.visualization.html)
# - KDE module (shows high-level plotting signatures)     [2](https://pyabc.readthedocs.io/en/latest/_modules/pyabc/visualization/kde.html)
# - History storage defaults (stores_sum_stats=True)      [3](https://colab.research.google.com/github/icb-dcm/pyabc/blob/main/doc/examples/parameter_inference.ipynb)[4](https://pyabc.readthedocs.io/en/latest/index.html)
# - Population / Particle sum stats & weights concepts    [5](https://github.com/ICB-DCM/pyABC/blob/main/doc/examples/myRModel.R)

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: seaborn for nicer plots
try:
    import seaborn as sns
    SEABORN = True
except Exception:
    SEABORN = False

# pyABC: history access and optional viz
import pyabc
from pyabc.storage import History
try:
    from pyabc import visualization as viz  # built-in viz (KDE, matrix, etc.)
    HAS_PYABC_VIZ = True
except Exception:
    HAS_PYABC_VIZ = False


# ------------------------ Utilities & basic plots -----------------------------

def nice_labels():
    return {
        "speed_base": "Speed (base)",
        "prolif_rate": "Proliferation rate",
        "adhesion": "Adhesion",
        "fragment_rate": "Fragmentation rate",
        "merge_prob": "Contact merge probability",
    }


def weighted_quantiles(vals: np.ndarray, weights: np.ndarray, qs=(0.05, 0.5, 0.95)):
    idx = np.argsort(vals)
    v = vals[idx]
    w = weights[idx].astype(float)
    w = w / (w.sum() + 1e-12)
    c = np.cumsum(w)
    outs = []
    for q in qs:
        outs.append(v[np.searchsorted(c, q)])
    return tuple(outs)


def plot_marginals(df: pd.DataFrame, w: np.ndarray, cols, out_path: Path, bins=40):
    labels = nice_labels()
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    if SEABORN:
        sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows))
    axes = np.atleast_2d(axes)
    ax_list = axes.flatten()
    for i, c in enumerate(cols):
        ax = ax_list[i]
        data = df[c].to_numpy(dtype=float)
        if SEABORN:
            sns.histplot(data, bins=bins, stat="density",
                         color="#5271FF", edgecolor="white", ax=ax)
        else:
            ax.hist(data, bins=bins, density=True,
                    color="#5271FF", edgecolor="white", alpha=0.9)
        q05, q50, q95 = weighted_quantiles(data, w, qs=(0.05, 0.5, 0.95))
        for v, ls, lw, col in [(q50, "-", 2.0, "#222222"),
                               (q05, "--", 1.5, "#AA0000"),
                               (q95, "--", 1.5, "#AA0000")]:
            ax.axvline(v, ls=ls, lw=lw, c=col)
        ax.set_xlabel(labels.get(c, c))
        ax.set_ylabel("Density")
        ax.set_title(f"{labels.get(c, c)} posterior")
    # Hide unused panels
    for j in range(i + 1, len(ax_list)):
        ax_list[j].axis("off")
    fig.suptitle("pyABC posterior marginals (median and 5–95% shown)", y=0.995, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[Saved] Marginals → {out_path.resolve()}")
    plt.close(fig)


def plot_pairs(df: pd.DataFrame, w: np.ndarray, cols, out_path: Path, bins=40):
    labels = nice_labels()
    rename_map = {c: labels.get(c, c) for c in cols}
    dfp = df[cols].rename(columns=rename_map).copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if SEABORN:
        sns.set(style="whitegrid")
        g = sns.pairplot(
            dfp,
            corner=True,
            diag_kind="hist",
            plot_kws=dict(s=12, color="#1f77b4", alpha=0.45, edgecolor="none"),
            diag_kws=dict(bins=bins, color="#5271FF"),
        )
        g.fig.suptitle("pyABC posterior (pairwise)", y=1.02)
        g.fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] Pair plot → {out_path.resolve()}")
        plt.close(g.fig)
    else:
        n = len(cols)
        fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n))
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                if i == j:
                    ax.hist(dfp.iloc[:, i].to_numpy(dtype=float), bins=bins,
                            color="#5271FF", edgecolor="white", density=True)
                else:
                    ax.scatter(dfp.iloc[:, j], dfp.iloc[:, i],
                               s=6, alpha=0.4, color="#1f77b4")
                if i == n - 1:
                    ax.set_xlabel(dfp.columns[j])
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(dfp.columns[i])
                else:
                    ax.set_yticklabels([])
        fig.suptitle("pyABC posterior (pairwise, minimal fallback)", y=0.995)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] Pair plot → {out_path.resolve()}")
        plt.close(fig)


# -------- Robust final-population sum-stat extraction & overlay ----------------

def _extract_final_population_sumstats(hist: "History"):
    """
    Retrieve accepted sum statistics and weights for the final ABC population.
    Robust across pyABC versions documented in the API and sources.  [1](https://pyabc.readthedocs.io/en/latest/api/pyabc.visualization.html)[5](https://github.com/ICB-DCM/pyABC/blob/main/doc/examples/myRModel.R)

    Returns:
       sumstats_list : List[dict] with keys like 'S0','S1','S2' (whatever exists)
       weights       : (N,) numpy array, normalised
    """
    t_final = hist.max_t
    pop = hist.get_population(t_final)

    # Preferred API: accepted sum stats as list[dict]  [5](https://github.com/ICB-DCM/pyABC/blob/main/doc/examples/myRModel.R)
    try:
        sumstats_list = pop.get_accepted_sum_stats()  # List[dict]
        if sumstats_list is None or len(sumstats_list) == 0:
            raise RuntimeError("Empty accepted sum stats.")
        # Weights from accepted Particle objects (final-pop normalized)  [5](https://github.com/ICB-DCM/pyABC/blob/main/doc/examples/myRModel.R)
        accepted_particles = [p for p in pop.particles if getattr(p, "accepted", True)]
        weights = np.asarray([float(p.weight) for p in accepted_particles], dtype=float)
        if weights.size != len(sumstats_list):
            # Fallback: weights from posterior distribution dataframe (final pop)  [1](https://pyabc.readthedocs.io/en/latest/api/pyabc.visualization.html)
            _, w = hist.get_distribution(m=0, t=t_final)
            weights = np.asarray(w, dtype=float)[:len(sumstats_list)]
    except Exception:
        # Fallback: dataframe view with 'sum_stat' & 'weight'  [1](https://pyabc.readthedocs.io/en/latest/api/pyabc.visualization.html)
        df_pop = pop.get_for_keys(["sum_stat", "weight"])
        if "sum_stat" not in df_pop.columns or len(df_pop) == 0:
            raise RuntimeError("Could not retrieve sum statistics from final population.")
        sumstats_list = [ss if isinstance(ss, dict) else dict(ss) for ss in df_pop["sum_stat"].tolist()]
        weights = df_pop["weight"].to_numpy(dtype=float)

    # Normalise weights
    weights = weights / (weights.sum() + 1e-12)

    # Diagnostics
    keys_available = set().union(*[set(d.keys()) for d in sumstats_list])
    print(f"[Info] Final population: accepted={len(sumstats_list)}, "
          f"keys={sorted(keys_available)}, stores_sum_stats={hist.stores_sum_stats}")

    return sumstats_list, weights



def plot_fit_vs_data(obs_csv: str,
                     start_step: int,
                     hist: "History",
                     out_dir: Path,
                     title_prefix: str = "Data vs fitted ABM (posterior predictive)",
                     dpi: int = 300):
    """
    Overlay experimental series with posterior predictive bands from the final population.
    Also reports the percentage of observed points falling within the 5–95% band.
    Handles any subset of keys among {'S0','S1','S2'} present in the DB.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load observed segment
    obs_df = pd.read_csv(obs_csv)
    if not all(c in obs_df.columns for c in ("S0", "S1", "S2")):
        raise ValueError("Observed CSV must contain S0,S1,S2")
    obs = obs_df[["S0", "S1", "S2"]].to_numpy(dtype=float)
    obs_seg = obs[:, :]         # (T_obs, 3)
    T_obs = obs_seg.shape[0]

    # 2) Extract final-population sum stats & weights
    sumstats_list, weights = _extract_final_population_sumstats(hist)

    # 3) Decide which series to plot based on available keys
    keys = [k for k in ("S0", "S1", "S2") if all(k in d for d in sumstats_list)]
    if not keys:
        raise RuntimeError("No S0/S1/S2 keys found in final population sum stats.")

    # Helpers
    def stack_key(k: str):
        trajs = [np.asarray(d[k], dtype=float) for d in sumstats_list]
        min_len = min(len(x) for x in trajs)
        T_use = min(min_len, T_obs)
        return np.vstack([x[:T_use] for x in trajs]), T_use

    def timewise_bands(arr: np.ndarray, w: np.ndarray, qs=(0.05, 0.5, 0.95)):
        w = w.astype(float)
        w = w / (w.sum() + 1e-12)
        T = arr.shape[1]
        out = {f"q{int(100*q):02d}": np.empty(T, dtype=float) for q in qs}
        for t in range(T):
            vals = arr[:, t]
            idx = np.argsort(vals)
            v = vals[idx]
            ww = w[idx]
            c = np.cumsum(ww)
            for q in qs:
                out[f"q{int(100*q):02d}"][t] = v[np.searchsorted(c, q)]
        return out

    # 4) Build panels and compute coverage stats
    panels = []
    coverage_rows = []  # for console summary table
    name_to_idx = {"S0": 0, "S1": 1, "S2": 2}
    label_map = {"S0": "Number of clusters, S0",
                 "S1": "Mean cluster size, S1",
                 "S2": "Mean squared size, S2"}

    for k in keys:
        arr, T_use = stack_key(k)
        bands = timewise_bands(arr, weights, qs=(0.05, 0.5, 0.95))
        y_obs = obs_seg[:T_use, name_to_idx[k]]

        # --- NEW: coverage % (observed inside 5–95% band) ---
        inside = (y_obs >= bands["q05"][:T_use]) & (y_obs <= bands["q95"][:T_use])
        coverage_pct = 100.0 * float(np.mean(inside)) if T_use > 0 else 0.0

        # Store for plotting and summary
        label = f"{label_map[k]} (coverage: {coverage_pct:.1f}%)"
        panels.append((k, y_obs, bands, label, T_use))
        coverage_rows.append((k, coverage_pct, T_use))

    # Print a brief coverage summary
    if coverage_rows:
        print("\nCoverage (observed inside posterior 5–95% band):")
        for k, pct, T_use in coverage_rows:
            print(f"  {k}: {pct:.1f}% of {T_use} points")

    # 5) Time axis in hours (0.5 h per step) with trimming per panel
    max_T = max(T_use for *_, T_use in panels)
    t_hours = np.arange(max_T, dtype=float) * 0.5

    # 6) Plot
    if SEABORN:
        sns.set(style="whitegrid")

    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.2), sharex=True)
    axes = np.atleast_1d(axes)
    colors = {"median": "#1f77b4", "band": "#9ecae1", "data": "#222222"}

    for ax, (name, y_obs, bands, label, T_use) in zip(axes, panels):
        # 95% band
        ax.fill_between(t_hours[:T_use], bands["q05"][:T_use], bands["q95"][:T_use],
                        color=colors["band"], alpha=0.5, label="Posterior 5–95%")
        # Median
        ax.plot(t_hours[:T_use], bands["q50"][:T_use], color=colors["median"], lw=2.0,
                label="Posterior median")
        # Data
        ax.plot(t_hours[:T_use], y_obs, color=colors["data"], lw=1.8, ls="--",
                label="Experimental")
        ax.set_title(label)
        ax.set_xlabel("Time since start (hours)")
        ax.set_ylabel(name)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, ncols=min(3, len(lbls)), loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title_prefix, y=1.02, fontsize=12)
    fig.tight_layout()

    out_path = out_dir / "pyabc_fit_vs_data.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[Saved] Data vs fitted ABM → {out_path.resolve()}")
    plt.close(fig)

# ------------------------------- Main ----------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot posterior from pyABC SQLite DB and compare to experimental data."
    )
    parser.add_argument(
        "--db_uri",
        type=str,
        default="sqlite:////Users/nathan/Documents/Oxford/DPhil/mesa_abm/draft_model/clusters_abm/results/pyabc_runs.db",
        help="SQLAlchemy SQLite URI, e.g., sqlite:////absolute/path/to.db",
    )
    parser.add_argument("--out_dir", type=str, default="clusters_abm/results",
                        help="Output directory for figures.")
    parser.add_argument("--bins", type=int, default=40, help="Histogram bins.")
    parser.add_argument("--use_pyabc_kde", action="store_true",
                        help="Also render pyABC built-in KDE plots (1D and matrix).")
    # NEW: inputs for the data-vs-fit overlay
    parser.add_argument("--obs_csv", type=str, default="INV_summary_stats.csv",
                        help="CSV with S0,S1,S2, one row per 30-min step. First row corresponds to step=start_step.")
    parser.add_argument("--start_step", type=int, default=20,
                        help="Start step of the observed segment (e.g., 20).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load history (pyABC storage & visualization documented APIs)  [1](https://pyabc.readthedocs.io/en/latest/api/pyabc.visualization.html)
    hist = History(args.db_uri)
    if hist.n_populations == 0:
        raise RuntimeError("No populations found in DB. Run pyABC first.")

    # Final population posterior over parameters (weighted)
    t_final = hist.max_t
    df, w = hist.get_distribution(m=0, t=t_final)  # documented API  [1](https://pyabc.readthedocs.io/en/latest/api/pyabc.visualization.html)

    # Choose parameters to display (in case DB contains more)
    preferred = ["speed_base", "prolif_rate", "adhesion", "fragment_rate", "merge_prob"]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if not cols:
            raise ValueError("No numeric parameter columns found in the pyABC posterior.")

    # Posterior summary table (weighted quantiles)
    print("\nPosterior summary (weighted, last population):")
    labels = nice_labels()
    rows = []
    for c in cols:
        v = df[c].to_numpy(dtype=float)
        q05, q50, q95 = weighted_quantiles(v, w, qs=(0.025, 0.5, 0.975))
        rows.append((labels.get(c, c), q50, q05, q95))
    post_df = pd.DataFrame(rows, columns=["Parameter", "Median", "5%", "95%"])
    with pd.option_context("display.float_format", "{:,.6g}".format):
        print(post_df.to_string(index=False))

    # Save marginal histograms
    plot_marginals(df, w, cols, out_path=out_dir / "pyabc_fig_posteriors.png", bins=args.bins)

    # Save pair plot
    plot_pairs(df, w, cols, out_path=out_dir / "pyabc_fig_pairs.png", bins=args.bins)

    # Optional: pyABC built-in KDE visualisations (1D and matrix) for the final population.
    # Use high-level / safe signatures as documented & in module sources; fall back gracefully.  [1](https://pyabc.readthedocs.io/en/latest/api/pyabc.visualization.html)[2](https://pyabc.readthedocs.io/en/latest/_modules/pyabc/visualization/kde.html)
    if HAS_PYABC_VIZ and args.use_pyabc_kde:
        try:
            # 1D KDE for each parameter over the final population
            fig, ax = plt.subplots()
            for c in cols:
                viz.plot_kde_1d(df, w, x=c, ax=ax)  # low-level variant: accepts df,w,x  [2](https://pyabc.readthedocs.io/en/latest/_modules/pyabc/visualization/kde.html)
            ax.legend(cols)
            fig.suptitle("pyABC KDE (1D, final population)")
            fig.savefig(out_dir / "pyabc_kde_1d.png", dpi=300, bbox_inches="tight")
            print(f"[Saved] pyABC KDE 1D → {(out_dir / 'pyabc_kde_1d.png').resolve()}")
            plt.close(fig)

            # KDE matrix: prefer high-level (history-based) API when available  [1](https://pyabc.readthedocs.io/en/latest/api/pyabc.visualization.html)
            try:
                fig = viz.plot_kde_matrix_highlevel(hist, x=cols, m=0, t=t_final)
            except Exception:
                # Safe fallback: low-level variant without 'x' keyword (API differences)
                fig = viz.plot_kde_matrix(df, w)  # minimal fallback signature  [1](https://pyabc.readthedocs.io/en/latest/api/pyabc.visualization.html)
            # Save figure (object may be a Matplotlib Figure or a handle depending on version)
            try:
                fig.suptitle("pyABC KDE matrix (final population)")
                fig.savefig(out_dir / "pyabc_kde_matrix.png", dpi=300, bbox_inches="tight")
            except Exception:
                plt.gcf().savefig(out_dir / "pyabc_kde_matrix.png", dpi=300, bbox_inches="tight")
            print(f"[Saved] pyABC KDE matrix → {(out_dir / 'pyabc_kde_matrix.png').resolve()}")
            plt.close('all')
        except Exception as e:
            print(f"[Warn] pyABC KDE plotting failed: {e}")

    # Experimental vs fitted ABM overlay (posterior predictive from stored sum stats)
    try:
        plot_fit_vs_data(
            obs_csv=args.obs_csv,
            start_step=args.start_step,
            hist=hist,
            out_dir=out_dir,
            title_prefix="Data vs fitted ABM (final ABC population)",
        )
    except Exception as e:
        print(f"[Warn] Could not plot data vs fitted ABM: {e}")


if __name__ == "__main__":
    main()
