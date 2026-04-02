#!/usr/bin/env python3
"""
abcp/plot_abc_outputs.py
------------------------
Scan abcp/outputs/<run_tag>/ folders and generate:
  • epsilon schedule plot (epsilon vs population index)
  • posterior 1-D marginals (weighted histograms for each parameter)
  • optional pairwise corner plot (weighted resample)
  • posterior summary CSV (weighted median & 95% CI)

If CSV files are missing, reconstruct them directly from abc.db (pyabc History).
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pyabc is already used by your runner; we reuse it here
import pyabc  # type: ignore


# ----------------------------- IO helpers -----------------------------
def list_runs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def ensure_figs_dir(run_dir: Path) -> Path:
    out = run_dir / "figs"
    out.mkdir(exist_ok=True)
    return out


def open_history(run_dir: Path) -> Optional[pyabc.History]:
    db = run_dir / "abc.db"
    if not db.exists():
        return None
    uri = f"sqlite:///{db}"
    try:
        return pyabc.History(uri)
    except Exception:
        return None


# ----------------------------- Data loaders (with DB fallback) -----------------------------
def load_or_rebuild_epsilon(run_dir: Path) -> Optional[pd.DataFrame]:
    """Return epsilon schedule DataFrame, reconstructing from DB if needed."""
    fp = run_dir / "epsilon_schedule.csv"
    if fp.exists():
        try:
            df = pd.read_csv(fp)
            if {"t", "epsilon"}.issubset(df.columns):
                return df.sort_values("t").reset_index(drop=True)
        except Exception:
            pass

    # fallback to DB
    hist = open_history(run_dir)
    if hist is None:
        return None

    try:
        pops = hist.get_all_populations()  # DataFrame
        col = "epsilon" if "epsilon" in pops.columns else ("eps" if "eps" in pops.columns else None)
        if col is None:
            # last resort: pull from each population object
            epsilons = []
            for t in range(hist.max_t + 1):
                try:
                    pop = hist.get_population(t=t)
                    epsilons.append(getattr(pop, "epsilon", np.nan))
                except Exception:
                    epsilons.append(np.nan)
            df = pd.DataFrame({"t": np.arange(len(epsilons)), "epsilon": epsilons})
        else:
            df = pd.DataFrame({"t": np.arange(len(pops[col].values)),
                               "epsilon": pops[col].values})
        df.to_csv(fp, index=False)
        return df
    except Exception:
        return None


def load_or_rebuild_posterior(run_dir: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Return (posterior_particles_df, weights), reconstructing from abc.db if CSVs are missing.
    """
    fpp = run_dir / "posterior_particles.csv"
    fpw = run_dir / "posterior_weights.csv"

    if fpp.exists() and fpw.exists():
        df = pd.read_csv(fpp)
        wdf = pd.read_csv(fpw)
        if "weight" not in wdf.columns:
            raise ValueError(f"{fpw} must contain a 'weight' column")
        w = wdf["weight"].to_numpy(float)
        w = np.clip(w, 0.0, np.inf)
        if w.sum() <= 0:
            raise ValueError("Posterior weights sum to zero.")
        w = w / w.sum()
        return df, w

    # fallback to DB
    hist = open_history(run_dir)
    if hist is None:
        raise FileNotFoundError(f"No posterior CSVs and no abc.db in {run_dir}")

    try:
        t_max = hist.max_t
        # same API your runner uses when saving posterior CSVs
        df, w = hist.get_distribution(m=0, t=t_max)
        w = np.asarray(w, float)
        if w.sum() <= 0:
            raise ValueError("Posterior weights sum to zero.")
        w = w / w.sum()
        # write back the missing CSVs so future runs find them
        df.to_csv(fpp, index=False)
        pd.DataFrame({"weight": w}).to_csv(fpw, index=False)
        return df, w
    except Exception as e:
        raise RuntimeError(f"Failed to rebuild posterior from DB: {e}")


# ----------------------------- Stats utils -----------------------------
def weighted_quantiles(x: np.ndarray, w: np.ndarray, qs=(0.5, 0.025, 0.975)) -> Tuple[float, float, float]:
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    idx = np.argsort(x)
    x_sorted, w_sorted = x[idx], w[idx]
    cdf = np.cumsum(w_sorted); cdf /= cdf[-1]
    out = [np.interp(q, cdf, x_sorted) for q in qs]
    return float(out[0]), float(out[1]), float(out[2])


def weighted_resample(df: pd.DataFrame, w: np.ndarray, n: int, rng: np.random.Generator) -> pd.DataFrame:
    idx = rng.choice(len(df), size=n, replace=True, p=w)
    return df.iloc[idx].reset_index(drop=True)


# ----------------------------- Plotting -----------------------------
def plot_epsilon(df_eps: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(df_eps["t"], df_eps["epsilon"], "-o", lw=2.0, ms=4)
    ax.set_xlabel("Population index (t)")
    ax.set_ylabel("epsilon")
    ax.set_title(f"Epsilon schedule (final ε = {df_eps['epsilon'].iloc[-1]:.4g})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_marginals(df: pd.DataFrame, w: np.ndarray, out_png: Path, bins: int = 50) -> pd.DataFrame:
    cols = list(df.columns)
    n = len(cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)
    summaries = []

    for i, col in enumerate(cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        x = df[col].to_numpy(float)
        hist, edges = np.histogram(x, bins=bins, weights=w, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, hist, "-", lw=2.0, color="C0")

        med, lo, hi = weighted_quantiles(x, w, qs=(0.5, 0.025, 0.975))
        ax.axvline(med, color="C3", lw=2.0, label=f"median={med:.4g}")
        ax.axvspan(lo, hi, color="C3", alpha=0.15, label=f"95% CI [{lo:.4g}, {hi:.4g}]")
        ax.set_title(col); ax.set_ylabel("density"); ax.grid(alpha=0.25)
        summaries.append({"parameter": col, "median": med, "ci_2p5": lo, "ci_97p5": hi})

    # hide any unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    # global legend (if any)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    return pd.DataFrame(summaries)


def plot_corner(df: pd.DataFrame, w: np.ndarray, out_png: Path, n_samples: int = 3000, alpha: float = 0.15, seed: int = 7) -> None:
    cols = list(df.columns)
    d = len(cols)
    if d < 2:
        return
    rng = np.random.default_rng(seed)
    samp = weighted_resample(df, w, n_samples, rng)

    fig, axes = plt.subplots(d, d, figsize=(2.2 * d, 2.2 * d), squeeze=False)
    for i in range(d):
        for j in range(d):
            ax = axes[i][j]
            if i == j:
                x = samp[cols[j]].to_numpy(float)
                hist, edges = np.histogram(x, bins=40, density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.plot(centers, hist, "-", lw=1.6, color="C0")
                ax.set_yticks([])
            elif i > j:
                x = samp[cols[j]].to_numpy(float)
                y = samp[cols[i]].to_numpy(float)
                ax.scatter(x, y, s=4, alpha=alpha, color="C0", edgecolors="none")
            else:
                ax.axis("off")
            if i == d - 1:
                ax.set_xlabel(cols[j], fontsize=8)
            if j == 0 and i > 0:
                ax.set_ylabel(cols[i], fontsize=8)

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ----------------------------- Main logic -----------------------------
def process_run(run_dir: Path, make_corner: bool, corner_samples: int, bins: int) -> None:
    figs = ensure_figs_dir(run_dir)

    # 1) Epsilon schedule: load or rebuild -> plot
    eps = load_or_rebuild_epsilon(run_dir)
    if eps is not None:
        plot_epsilon(eps, figs / "epsilon_schedule.png")

    # 2) Posterior: load or rebuild -> plots and summary
    df, w = load_or_rebuild_posterior(run_dir)
    summary = plot_marginals(df, w, figs / "posterior_marginals.png", bins=bins)
    summary.to_csv(figs / "posterior_summary.csv", index=False)

    # 3) Optional corner
    if make_corner and df.shape[1] >= 2:
        plot_corner(df, w, figs / "posterior_corner.png", n_samples=corner_samples)


def main():
    ap = argparse.ArgumentParser(description="Plot ABC posteriors and epsilon schedules for runs under abcp/outputs/")
    ap.add_argument("--root", type=str, default="abcp/outputs", help="Root outputs folder")
    ap.add_argument("--run", type=str, default="latest", help="'latest' | 'all' | <run_tag>")
    ap.add_argument("--corner", action="store_true", help="Also draw a simple corner plot (weighted resample)")
    ap.add_argument("--corner-samples", type=int, default=3000, help="Samples for the corner plot resample")
    ap.add_argument("--bins", type=int, default=50, help="Histogram bins for marginals")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    runs = list_runs(root)
    if not runs:
        print(f"[WARN] No runs found under {root}")
        return

    if args.run == "latest":
        targets = [runs[-1]]
    elif args.run == "all":
        targets = runs
    else:
        targets = [root / args.run]
        if not targets[0].exists():
            print(f"[ERROR] Run not found: {targets[0]}")
            return

    for rd in targets:
        print(f"[INFO] Processing: {rd}")
        try:
            process_run(rd, make_corner=args.corner, corner_samples=args.corner_samples, bins=args.bins)
            print(f"[OK] Wrote figures to: {rd/'figs'}")
        except Exception as e:
            print(f"[WARN] Failed on {rd}: {e}")


if __name__ == "__main__":
    main()