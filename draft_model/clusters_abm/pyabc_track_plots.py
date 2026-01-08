
#!/usr/bin/env python3
# Trace plots for ABC–SMC: show how accepted parameters evolve across populations.
# Works with a pyABC SQLite DB. British-English labelling.

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyabc.storage import History


def _weighted_quantiles(vals: np.ndarray, weights: np.ndarray, qs=(0.05, 0.5, 0.95)):
    """Return weighted quantiles for 1D array vals given weights."""
    vals = np.asarray(vals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if vals.size == 0:
        return tuple(np.nan for _ in qs)
    idx = np.argsort(vals)
    v = vals[idx]
    w = weights[idx]
    w = w / (w.sum() + 1e-12)
    c = np.cumsum(w)
    outs = []
    for q in qs:
        outs.append(v[np.searchsorted(c, q)])
    return tuple(outs)


def main():
    parser = argparse.ArgumentParser(
        description="ABC–SMC trace plots: parameter evolution across populations, plus epsilon vs generation."
    )
    parser.add_argument("--db_uri", type=str, required=True, help="SQLAlchemy SQLite URI, e.g., sqlite:////Users/nathan/Documents/Oxford/DPhil/mesa_abm/draft_model/clusters_abm/results/pyabc_runs_3.db")
    parser.add_argument("--out_dir", type=str, default="clusters_abm/results", help="Where to save figures.")
    parser.add_argument("--max_params", type=int, default=12, help="Max parameters to plot to avoid huge figures.")
    parser.add_argument("--dot_alpha", type=float, default=0.5, help="Scatter transparency for accepted points.")
    parser.add_argument("--point_size", type=float, default=12, help="Scatter point size.")
    parser.add_argument("--bins", type=int, default=40, help="Histogram bins (not used, reserved).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load history (same API family your posterior script uses)
    hist = History(args.db_uri)
    if hist.n_populations == 0:
        raise RuntimeError("No populations found in DB. Run ABC first or check the DB path/URI.")

    # --- Discover parameters present across populations (union of numeric columns from get_distribution) ---
    param_names = set()
    per_pop_data = []  # list of tuples (t, df_t, w_t)
    for t in range(hist.n_populations):
        df_t, w_t = hist.get_distribution(m=0, t=t)  # accepted params + weights, population t
        # Keep only numeric parameter columns
        for c in df_t.columns:
            if np.issubdtype(df_t[c].dtype, np.number):
                param_names.add(c)
        per_pop_data.append((t, df_t, np.asarray(w_t, dtype=float)))

    if not param_names:
        raise RuntimeError("No numeric parameter columns found in accepted particles. "
                           "Check that your model returns parameters and that acceptance occurred.")

    # Sort names for reproducibility and limit to a sensible number
    param_cols = sorted(param_names)
    if len(param_cols) > args.max_params:
        print(f"⚠️ {len(param_cols)} parameters found; limiting to first {args.max_params}.")
        param_cols = param_cols[:args.max_params]

    # --- Build trace plots: scatter per population + median and 5–95% bands ---
    n_params = len(param_cols)
    fig, axes = plt.subplots(n_params, 1, figsize=(8.5, 2.6 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for ax, pname in zip(axes, param_cols):
        # Scatter of accepted values by population (x = t)
        medians, q05s, q95s = [], [], []
        ts = []
        for (t, df_t, w_t) in per_pop_data:
            if pname not in df_t.columns:
                continue
            vals = df_t[pname].to_numpy(dtype=float)
            # Jitter x slightly to reduce overplotting
            x = t + (np.random.rand(len(vals)) - 0.5) * 0.05
            ax.scatter(x, vals, s=args.point_size, alpha=args.dot_alpha, color="#1f77b4")
            q05, q50, q95 = _weighted_quantiles(vals, w_t, qs=(0.05, 0.5, 0.95))
            medians.append(q50)
            q05s.append(q05)
            q95s.append(q95)
            ts.append(t)

        if ts:
            ax.plot(ts, medians, color="#d62728", lw=2.0, label="Median")
            ax.plot(ts, q05s, color="#d62728", lw=1.2, ls="--", alpha=0.8, label="5–95% band")
            ax.plot(ts, q95s, color="#d62728", lw=1.2, ls="--", alpha=0.8)
        ax.set_ylabel(pname)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Population (generation)")
    fig.suptitle("ABC–SMC trace: accepted parameters per population\n(Median and 5–95% band shown)", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "abc_trace_parameters.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_dir / 'abc_trace_parameters.png'}")

    # --- Also plot epsilon vs generation from population metadata ---
    pops = hist.get_all_populations()
    if "epsilon" in pops.columns:
        fig2, ax2 = plt.subplots(figsize=(7.0, 3.6))
        ax2.plot(pops["t"], pops["epsilon"], marker="o", color="#9467bd")
        ax2.set_xlabel("Population (generation)")
        ax2.set_ylabel("Epsilon (distance threshold)")
        ax2.set_title("ABC–SMC schedule: epsilon vs generation")
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(out_dir / "abc_trace_epsilon.png", dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"[Saved] {out_dir / 'abc_trace_epsilon.png'}")
    else:
        print("⚠️ Population table has no 'epsilon' column; skipping epsilon plot.")


if __name__ == "__main__":
    main()
