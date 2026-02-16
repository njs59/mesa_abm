#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pints

# -------------------------------------------------------------------
# Import model
# -------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Model_structures.Model_class_Cell_num_Prolif_cst_Shed import SmolModel


# ===================================================================
# Load chains and data
# ===================================================================
def load_data_and_chains():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    results_dir = os.path.join(project_root, "ODE_fitting/results")

    # Load full 3D chains: (n_chains, n_iters, n_params)
    npz_path = os.path.join(results_dir, "MCMC_forward_ABM_adaptive_chains.npz")
    chains = np.load(npz_path)["chains"]

    # Load ABM summary stats
    stats_path = os.path.join(results_dir, "forward_means_stats.csv")
    df = pd.read_csv(stats_path)
    times = df["step"].to_numpy()
    data_values = df[[
        "num_clusters",
        "mean_cluster_size",
        "mean_squared_cluster_size"
    ]].to_numpy()

    return chains, times, data_values, results_dir


# ===================================================================
# Trace plot
# ===================================================================
def plot_traces(chains, param_names, save_dir):
    n_chains, iters, n_params = chains.shape
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2*n_params), sharex=True)

    for p in range(n_params):
        for c in range(n_chains):
            axes[p].plot(chains[c, :, p], lw=0.5, alpha=0.7)
        axes[p].set_ylabel(param_names[p])

    axes[-1].set_xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "trace_plots.png"), dpi=150)
    plt.close()


# ===================================================================
# Posterior marginals
# ===================================================================
def plot_marginals(flat_samples, param_names, save_dir):
    n_params = flat_samples.shape[1]
    fig, axes = plt.subplots(n_params, 1, figsize=(6, 2*n_params))

    for j in range(n_params):
        sns.histplot(flat_samples[:, j], bins=40, kde=True, ax=axes[j])
        axes[j].set_xlabel(param_names[j])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "posterior_marginals.png"), dpi=150)
    plt.close()


# ===================================================================
# Pairwise posteriors (corner plot)
# ===================================================================
def plot_pairwise(flat_samples, param_names, save_dir):
    df = pd.DataFrame(flat_samples, columns=param_names)
    grid = sns.pairplot(df, corner=True, diag_kind="kde",
                        plot_kws={"alpha": 0.25, "s": 8})
    grid.fig.set_size_inches(12, 12)
    grid.savefig(os.path.join(save_dir, "pairwise_posteriors.png"), dpi=150)
    plt.close()


# ===================================================================
# Posterior Predictive Checks WITH lognormal noise
# ===================================================================
def posterior_predictive_checks(model, flat_samples, times, data, save_dir,
                                nsamples=300, random_seed=42):
    """
    Posterior predictive simulation with multiplicative lognormal noise:

        y_rep = y_clean * exp(mu + sigma * eps)

    where:
        - y_clean is simulated from the model using the first 4 params
          (b, p, q, N_0).
        - sigma = (Sigma_0, sigma_1, sigma_2) are the last 3 params.
        - Default mu = 0 for mean-one multiplicative noise.
    """
    rng = np.random.default_rng(random_seed)

    # Sample parameter vectors from posterior
    n_total = flat_samples.shape[0]
    idx = rng.choice(n_total, size=nsamples, replace=False)
    draws = flat_samples[idx]  # (nsamples, 7)

    # Split parameters: first 4 = model; last 3 = noise std devs
    theta_draws = draws[:, :4]            # (nsamples, 4) -> (b, p, q, N_0)
    sigma_draws = np.abs(draws[:, 4:7])   # (nsamples, 3) -> (Sigma_0, sigma_1, sigma_2)
    mu_draws = np.zeros_like(sigma_draws) # set to -0.5*sigma^2 if you used mean-one on log-scale

    T = len(times)
    y_rep = np.zeros((nsamples, T, 3))

    # Generate predictive simulations
    for i in range(nsamples):
        params = theta_draws[i]
        sig = sigma_draws[i]
        mu = mu_draws[i]

        # Clean model simulation (T, 3)
        y_clean = model.simulate(params, times)

        # Lognormal multiplicative noise
        eps = rng.standard_normal(size=(T, 3))
        mu_mat = np.broadcast_to(mu, (T, 3))
        sig_mat = np.broadcast_to(sig, (T, 3))
        noise = np.exp(mu_mat + sig_mat * eps)

        y_rep[i] = y_clean * noise

    # Predictive intervals
    med   = np.median(y_rep, axis=0)
    low95 = np.percentile(y_rep, 2.5, axis=0)
    high95= np.percentile(y_rep, 97.5, axis=0)

    labels = ["num_clusters", "mean_cluster_size", "mean_squared_cluster_size"]

    # Plot PPCs
    for k in range(3):
        plt.figure(figsize=(10, 4))

        # 95% predictive interval (posterior predictive)
        plt.fill_between(times, low95[:, k], high95[:, k],
                         alpha=0.35, color="skyblue",
                         label="95% predictive interval")

        # Observed data
        plt.plot(times, data[:, k], "kx-", markersize=4, label="Observed")

        plt.xlabel("Time")
        plt.ylabel(labels[k])
        plt.title(f"Posterior Predictive Check — {labels[k]}")
        plt.legend()

        out = os.path.join(save_dir, f"ppc_{labels[k]}.png")
        plt.savefig(out, dpi=150)
        plt.close()


# ===================================================================
# Main Execution
# ===================================================================
if __name__ == "__main__":

    chains, times, data_values, save_dir = load_data_and_chains()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Parameter names as requested:
    # b, p, q, N_0, Sigma_0, sigma_1, sigma_2
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    param_names = [r"$b$",r"$p$",r"$q$",r"$N_0$",r"$\sigma_0$",r"$\sigma_1$",r"$\sigma_2$"]

    # Burn-in: last 50% kept
    n_chains, n_iters, n_params = chains.shape
    burn = n_iters // 2
    post = chains[:, burn:, :]
    flat = post.reshape(-1, n_params)

    print("Plotting trace plots…")
    plot_traces(chains, param_names, save_dir)

    print("Plotting marginal posteriors…")
    plot_marginals(flat, param_names, save_dir)

    print("Plotting pairwise posteriors…")
    plot_pairwise(flat, param_names, save_dir)

    print("Running posterior predictive checks…")
    model = SmolModel(None, None)
    posterior_predictive_checks(model, flat, times, data_values, save_dir)

    print("\nAll plots saved in:", save_dir)