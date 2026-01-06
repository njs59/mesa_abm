
# plot_posteriors.py
# Visualise ABC posterior samples: 1D marginals + simple pair plot.
# British English labels and a short summary table.

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    SEABORN = True
except Exception:
    SEABORN = False


def _nice_param_labels() -> dict:
    """Map column names to nicer plot labels."""
    return {
        "phenotypes.proliferative.speed_base": "Speed (base)",
        "phenotypes.proliferative.prolif_rate": "Proliferation rate",
        "phenotypes.proliferative.adhesion": "Adhesion",
        "phenotypes.proliferative.fragment_rate": "Fragmentation rate",
        "merge.prob_contact_merge": "Contact merge probability",
    }


def summarise_series(x: pd.Series) -> dict:
    """Return median and 5–95% quantiles."""
    q = np.quantile(x.to_numpy(dtype=float), [0.05, 0.5, 0.95])
    return {"q05": q[0], "q50": q[1], "q95": q[2]}


def plot_1d_marginals(
    df: pd.DataFrame,
    cols: List[str],
    out_path: str = "results/fig_posteriors.png",
    bins: int = 40,
    kde: bool = True,
) -> None:
    """1D histograms with optional KDE and quantile lines."""
    labels = _nice_param_labels()
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    if SEABORN:
        sns.set(style="whitegrid")

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.atleast_2d(axes)
    ax_list = axes.flatten()

    for i, c in enumerate(cols):
        ax = ax_list[i]
        data = df[c].to_numpy(dtype=float)
        if SEABORN:
            sns.histplot(data, bins=bins, kde=kde, stat="density", color="#5271FF", edgecolor="white", ax=ax)
        else:
            ax.hist(data, bins=bins, density=True, color="#5271FF", edgecolor="white", alpha=0.9)
        qs = np.quantile(data, [0.05, 0.5, 0.95])
        for v, ls, lw, col in [
            (qs[1], "-", 2.0, "#222222"),   # median
            (qs[0], "--", 1.5, "#AA0000"),  # 5%
            (qs[2], "--", 1.5, "#AA0000"),  # 95%
        ]:
            ax.axvline(v, ls=ls, lw=lw, c=col)
        ax.set_xlabel(labels.get(c, c))
        ax.set_ylabel("Density")
        ax.set_title(f"{labels.get(c, c)} posterior")

    # Turn off any unused subplots
    for j in range(i + 1, len(ax_list)):
        ax_list[j].axis("off")

    fig.suptitle("ABC posterior marginals (median and 5–95% shown)", y=0.995, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_pairwise(
    df: pd.DataFrame,
    cols: List[str],
    out_path: str = "results/fig_pairs.png",
    bins: int = 40,
) -> None:
    """Simple pair plot using seaborn if available; fallback to diagonal hist only."""
    labels = _nice_param_labels()
    rename_map = {c: labels.get(c, c) for c in cols}
    dfp = df[cols].rename(columns=rename_map).copy()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if SEABORN:
        sns.set(style="whitegrid")
        g = sns.pairplot(
            dfp,
            corner=True,
            diag_kind="hist",
            plot_kws=dict(s=12, color="#1f77b4", alpha=0.5, edgecolor="none"),
            diag_kws=dict(bins=bins, color="#5271FF"),
        )
