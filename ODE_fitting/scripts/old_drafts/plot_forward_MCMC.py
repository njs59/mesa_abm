#!/usr/bin/env python3
"""
Plot diagnostics and posterior predictive for MCMC fitted to FORWARD-ABM stats.

Inputs (defaults):
- ODE_fitting/results/MCMC_forward_ABM_chains_parallel.csv  (or _chains.csv)
- ODE_fitting/results/forward_means_stats.csv               (ABM mean stats)
- ODE model: Model_structures/Model_class_Cell_num_Prolif_cst_Shed.py

Outputs (to results/):
- MCMC_param_summary.csv
- mcmc_trace.png
- mcmc_posterior.png
- mcmc_corner.png
- posterior_predictive_S0_S1_S2.png
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAVE_SEABORN = True
except Exception:
    HAVE_SEABORN = False

# Import the ODE model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Model_structures.Model_class_Cell_num_Prolif_cst_Shed import SmolModel


# ----------------------- helpers -----------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def infer_shape_from_flat(flat: np.ndarray, candidates=(7, 4)) -> tuple[int, int]:
    M = flat.shape[1]
    for p in candidates:
        if M % p == 0:
            return (M // p, p)
    raise ValueError(
        f"Columns={M} not divisible by any of {candidates} — cannot infer (iters, params)."
    )

def gelman_rhat(chains_3d: np.ndarray) -> np.ndarray:
    m, n, p = chains_3d.shape
    rhat = np.zeros(p, dtype=float)
    for j in range(p):
        chain = chains_3d[:, :, j]
        chain_means = chain.mean(axis=1)
        overall = chain_means.mean()
        B = n * np.sum((chain_means - overall) ** 2) / (m - 1) if m > 1 else 0.0
        W = np.sum(np.var(chain, axis=1, ddof=1)) / m
        var_hat = ((n - 1) / n) * W + (B / n) if m > 1 else W
        rhat[j] = np.sqrt(var_hat / W) if W > 0 else np.nan
    return rhat


# ----------------------- CLI -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Plot MCMC diagnostics and posterior predictive.")
    ap.add_argument("--chains",
                    type=str,
                    default=None,
                    help="Path to chains CSV. If omitted, will try parallel then non-parallel default.")
    ap.add_argument("--results-dir",
                    type=str,
                    default=None,
                    help="Output dir for figures/CSVs. Default: ODE_fitting/results")
    ap.add_argument("--forward-means",
                    type=str,
                    default=None,
                    help="Path to forward_means_stats.csv. Default: ODE_fitting/results/forward_means_stats.csv")
    ap.add_argument("--burn",
                    type=int,
                    default=0,
                    help="Burn-in iterations to discard from each chain (default 0).")
    ap.add_argument("--thin",
                    type=int,
                    default=1,
                    help="Keep every THIN-th sample (default 1 = no thinning).")
    return ap.parse_args()


# ----------------------- main -----------------------
def main():
    module_dir   = os.path.dirname(os.path.abspath(__file__))          # .../ODE_fitting/scripts
    project_root = os.path.abspath(os.path.join(module_dir, '../..'))  # project root
    default_results = os.path.join(project_root, 'ODE_fitting/results')

    args = parse_args()
    results_dir = args.results_dir or default_results
    ensure_dir(results_dir)

    # Resolve chains path
    if args.chains is not None:
        chains_path = args.chains
    else:
        candidates = [
            # os.path.join(results_dir, 'MCMC_forward_ABM_chains_parallel.csv'),
            os.path.join(results_dir, 'MCMC_forward_ABM_adaptive_chain1.csv'),
            os.path.join(results_dir, 'MCMC_forward_ABM_chains.csv'),
        ]
        chains_path = next((p for p in candidates if os.path.exists(p)), None)
        if chains_path is None:
            raise FileNotFoundError("No chains file found. Tried:\n  - " + "\n  - ".join(candidates))

    forward_means_path = args.forward_means or os.path.join(results_dir, 'forward_means_stats.csv')
    if not os.path.exists(forward_means_path):
        raise FileNotFoundError(f"Missing forward means stats CSV: {forward_means_path}")

    print(f"Chains: {chains_path}")
    print(f"ABM means: {forward_means_path}")
    print(f"Burn={args.burn}, Thin={args.thin}")

    # ---------- load & reshape chains ----------
    flat = pd.read_csv(chains_path, header=None).to_numpy()  # (n_chains, n_iters*n_params)
    n_chains = flat.shape[0]
    n_iters, n_params = infer_shape_from_flat(flat, candidates=(7, 4))
    chains = flat.reshape(n_chains, n_iters, n_params)

    # Apply burn-in / thinning
    start = args.burn
    step  = max(1, args.thin)
    if start >= n_iters:
        raise ValueError(f"Burn-in ({start}) >= n_iters ({n_iters}). Choose a smaller burn.")
    chains = chains[:, start::step, :]
    n_iters_eff = chains.shape[1]

    print(f"Shape after burn/thin: chains={n_chains}, iters={n_iters_eff}, params={n_params}")

    # Parameter names
    if n_params == 7:
        param_names = ['b', 'p', 's', 'N', 'sigma_S0', 'sigma_S1', 'sigma_S2']
    elif n_params == 4:
        param_names = ['b', 'p', 's', 'N']
    else:
        param_names = [f'θ{i+1}' for i in range(n_params)]

    # Summary stats
    all_samples = chains.reshape(-1, n_params)
    means   = np.mean(all_samples, axis=0)
    medians = np.median(all_samples, axis=0)
    lowers  = np.percentile(all_samples, 2.5, axis=0)
    uppers  = np.percentile(all_samples, 97.5, axis=0)
    rhat    = gelman_rhat(chains)

    summary_df = pd.DataFrame({
        'parameter': param_names,
        'mean': means,
        'median': medians,
        'lower_95': lowers,
        'upper_95': uppers,
        'rhat': rhat,
    })
    summary_csv = os.path.join(results_dir, 'MCMC_param_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # ---------- Trace plots ----------
    rows = int(np.ceil(n_params / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.2 * rows), sharex=True)
    axes = axes.ravel() if n_params > 1 else [axes]
    x = np.arange(n_iters_eff)
    for j in range(n_params):
        ax = axes[j]
        for c in range(n_chains):
            ax.plot(x, chains[c, :, j], lw=0.6, alpha=0.9, label=f'chain {c+1}' if j == 0 else None)
        ax.set_title(f"Trace: {param_names[j]}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(param_names[j])
    if n_params > 0:
        axes[0].legend(ncol=min(3, n_chains))
    for k in range(n_params, len(axes)):
        fig.delaxes(axes[k])
    fig.tight_layout()
    out_trace = os.path.join(results_dir, 'mcmc_trace.png')
    fig.savefig(out_trace, dpi=150); plt.close(fig)

    # ---------- Posterior marginals ----------
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.2 * rows))
    axes = axes.ravel() if n_params > 1 else [axes]
    for j in range(n_params):
        ax = axes[j]
        data = all_samples[:, j]
        ax.hist(data, bins=40, density=True, color='steelblue', alpha=0.6, edgecolor='black')
        if HAVE_SEABORN:
            sns.kdeplot(data=data, ax=ax, color='darkred', lw=1.8)
        ax.set_title(f"Posterior: {param_names[j]}")
        ax.set_xlabel(param_names[j]); ax.set_ylabel("Density")
    for k in range(n_params, len(axes)):
        fig.delaxes(axes[k])
    fig.tight_layout()
    out_post = os.path.join(results_dir, 'mcmc_posterior.png')
    fig.savefig(out_post, dpi=150); plt.close(fig)

    # ---------- Corner plot ----------
    out_corner = os.path.join(results_dir, 'mcmc_corner.png')
    if HAVE_SEABORN and n_params >= 2:
        df_params = pd.DataFrame(all_samples, columns=param_names)
        g = sns.pairplot(df_params, corner=True, diag_kind='kde', plot_kws={'s': 8, 'alpha': 0.3})
        g.fig.suptitle("Posterior pairwise (corner) plot", y=1.02)
        g.fig.savefig(out_corner, dpi=150, bbox_inches='tight')
        plt.close(g.fig)
    elif n_params >= 2:
        fig, axes = plt.subplots(n_params, n_params, figsize=(1.9*n_params, 1.9*n_params))
        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]
                if i == j:
                    ax.hist(all_samples[:, j], bins=40, color='steelblue', alpha=0.6, edgecolor='black')
                elif i > j:
                    ax.scatter(all_samples[:, j], all_samples[:, i], s=4, alpha=0.25, color='black')
                else:
                    ax.axis('off')
                if i == n_params - 1: ax.set_xlabel(param_names[j], fontsize=8)
                if j == 0 and i > 0: ax.set_ylabel(param_names[i], fontsize=8)
        plt.suptitle("Posterior pairwise (corner) plot")
        plt.tight_layout()
        fig.savefig(out_corner, dpi=150); plt.close(fig)

    # ---------- Posterior predictive overlay ----------
    # ABM forward means
    df_mean = pd.read_csv(forward_means_path)
    times = df_mean['step'].to_numpy()
    target = df_mean[['num_clusters', 'mean_cluster_size', 'mean_squared_cluster_size']].to_numpy()

    # ODE simulate at posterior means (first 4 params are b, p, s, N)
    b, p, s, N = [float(x) for x in means[:4]]
    sim = SmolModel(None, None).simulate([b, p, s, N], times)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    labels = ['S0: number of clusters', 'S1: mean cluster size', 'S2: mean squared size']
    for k in range(3):
        axs[k].plot(times, target[:, k], 'k-', lw=2, label='ABM forward means')
        axs[k].plot(times, sim[:, k], 'r--', lw=2, label='ODE posterior mean')
        axs[k].set_ylabel(labels[k]); axs[k].legend(loc='best')
    axs[-1].set_xlabel('Step')
    fig.suptitle('Posterior predictive check (means)')
    fig.tight_layout()
    out_pp = os.path.join(results_dir, 'posterior_predictive_S0_S1_S2.png')
    fig.savefig(out_pp, dpi=150); plt.close(fig)

    print("\nSaved:")
    print(" ", os.path.join(results_dir, 'MCMC_param_summary.csv'))
    print(" ", out_trace)
    print(" ", out_post)
    print(" ", out_corner)
    print(" ", out_pp)


if __name__ == "__main__":
    main()