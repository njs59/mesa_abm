#!/usr/bin/env python3
"""
Adaptive-parallel MCMC inference for FORWARD-ABM summary statistics.

Sampler:
    AdaptiveCovarianceMCMC (gradient-free, efficient for ~7 parameters)

Method:
    - 3 parallel chains
    - Controller recreated each block (Method C)
    - R-hat computed on last 50% of MASTER chain
    - Chunked sampling (e.g. 5000 iterations per block)
    - Controlled internal printing every N iterations
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pints
import pints.plot

# --------------------------------------------------------------
# Import ODE model
# --------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Model_structures.Model_class_Cell_num_Prolif_cst_Shed import SmolModel


# ===================================================================
# MAIN
# ===================================================================
def main():

    # ---------------------------------------------------------------
    # Load forward ABM summary statistics
    # ---------------------------------------------------------------
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    stats_path = os.path.join(project_root, 'ODE_fitting/results/forward_means_stats.csv')

    df = pd.read_csv(stats_path)
    data_values = df[[
        "num_clusters",
        "mean_cluster_size",
        "mean_squared_cluster_size"
    ]].to_numpy()
    times = df["step"].to_numpy()

    print("Loaded forward ABM stats:", data_values.shape)

    # ---------------------------------------------------------------
    # Build problem + log-posterior
    # ---------------------------------------------------------------
    model = SmolModel(None, None)
    problem = pints.MultiOutputProblem(model, times, data_values)
    log_likelihood = pints.LogNormalLogLikelihood(problem)

    # Priors
    true_params = [0.0003, 0.001, 0.0001, 900, 1, 1, 1]
    prior_lower = [0, 0, 0, 1, 0, 0, 0]
    prior_upper = [0.001, 0.1, 0.1, 4000, 100, 100, 100]
    log_prior = pints.UniformLogPrior(prior_lower, prior_upper)
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    print("Parameter dimension:", problem.n_parameters())

    # ---------------------------------------------------------------
    # Initial positions for 3 chains
    # ---------------------------------------------------------------
    x0 = [
        0.9 * np.array(true_params),
        1.05 * np.array(true_params),
        1.15 * np.array(true_params),
    ]
    n_chains = len(x0)

    # ---------------------------------------------------------------
    # Settings
    # ---------------------------------------------------------------
    chunk = 5000
    print_every = 1000
    rhat_threshold = 1.01
    max_total_iters = 200000

    total_iters = 0
    converged = False
    all_chains = None

    print("\n=== Starting adaptive AdaptiveCovarianceMCMC (3 chains) ===")

    # ===============================================================
    # ADAPTIVE LOOP
    # ===============================================================
    while not converged and total_iters < max_total_iters:

        print(f"\n--- Running AdaptiveCovarianceMCMC block of {chunk} iterations ---")

        # Fresh controller per block
        controller = pints.MCMCController(
            log_posterior,
            chains=n_chains,
            x0=x0,
            method=pints.AdaptiveCovarianceMCMC
        )
        controller.set_parallel(True)
        controller.set_log_interval(print_every)
        controller.set_max_iterations(chunk)

        start_time = time.time()
        part = controller.run()   # shape (3, chunk, n_params)
        elapsed = time.time() - start_time

        print(f"Block completed in {elapsed:.2f} seconds.")

        # Append
        if all_chains is None:
            all_chains = part
        else:
            all_chains = np.concatenate([all_chains, part], axis=1)

        total_iters = all_chains.shape[1]
        print(f"Total accumulated iterations: {total_iters}")

        # New starting positions for next block
        x0 = [all_chains[c, -1, :] for c in range(n_chains)]

        # --------------------------------------------------------------
        # R-hat on last 50% of MASTER chain
        # --------------------------------------------------------------
        n_keep = total_iters // 2
        recent = all_chains[:, -n_keep:, :]   # 3D array

        try:
            rhat = pints.rhat(recent)
        except Exception:
            rhat = pints.rhat(np.asarray(recent))

        rhat = np.asarray(rhat).ravel()
        print("R-hat (last 50%):", ", ".join(f"{v:.4f}" for v in rhat))
        print(f"max R-hat = {np.max(rhat):.5f}")

        if np.all(np.isfinite(rhat)) and np.all(rhat < rhat_threshold):
            converged = True
            print("\n=== Convergence achieved! ===")
            break

    if not converged:
        print("\nWARNING: Max total iterations reached without convergence.")

    # ===============================================================
    # SAVE OUTPUTS
    # ===============================================================
    save_dir = os.path.join(project_root, "ODE_fitting/results")
    os.makedirs(save_dir, exist_ok=True)

    # Save full 3D master chain
    npz_path = os.path.join(save_dir, "MCMC_forward_ABM_adaptive_chains.npz")
    np.savez_compressed(npz_path, chains=all_chains)
    print("Saved NPZ chain file:", npz_path)

    # Save each chain individually
    for c in range(n_chains):
        csv_path = os.path.join(save_dir, f"MCMC_forward_ABM_adaptive_chain{c+1}.csv")
        pd.DataFrame(all_chains[c]).to_csv(csv_path, index=False, header=False)
        print(f"Saved chain {c+1} to:", csv_path)

    # ---------------------------------------------------------------
    # Final summary (post burn-in)
    # ---------------------------------------------------------------
    burn = total_iters // 2
    post = all_chains[:, burn:, :]

    summary = pints.MCMCSummary(post)
    summary_text = str(summary)

    summary_path = os.path.join(save_dir, "MCMC_adaptive_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print("\n=== Final Summary (post-burn-in) ===")
    print(summary_text)
    print("Saved summary:", summary_path)

    # ---------------------------------------------------------------
    # Trace plot
    # ---------------------------------------------------------------
    trace_path = os.path.join(save_dir, "MCMC_adaptive_trace.png")
    pints.plot.trace(all_chains)
    plt.tight_layout()
    plt.savefig(trace_path, dpi=150)
    plt.close()
    print("Saved trace plot:", trace_path)


# ===================================================================
# Safe entry point
# ===================================================================
if __name__ == "__main__":
    main()