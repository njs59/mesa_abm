#!/usr/bin/env python3
"""
Parallel MCMC inference for FORWARD-ABM summary statistics.
"""

import numpy as np
import pandas as pd
import pints
import sys, os
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Import ODE model
# -------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Model_structures.Model_class_Cell_num_Prolif_cst_Shed import SmolModel


def main():

    # -------------------------------------------------------------
    # Load FORWARD ABM OUTPUT
    # -------------------------------------------------------------
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    forward_stats_path = os.path.join(project_root,
                                      "ODE_fitting/results/forward_means_stats.csv")

    df = pd.read_csv(forward_stats_path)

    data_values = df[["num_clusters",
                      "mean_cluster_size",
                      "mean_squared_cluster_size"]].to_numpy()

    times = df["step"].to_numpy()

    print("Loaded ABM forward stats:", data_values.shape)

    # -------------------------------------------------------------
    # Build model + problem
    # -------------------------------------------------------------
    model = SmolModel(None, None)
    problem = pints.MultiOutputProblem(model, times, data_values)

    log_likelihood = pints.LogNormalLogLikelihood(problem)

    # Priors same as before
    true_params = [0.0003, 0.001, 0.0001, 900, 1, 1, 1]
    prior_lower = [0, 0, 0, 1, 0, 0, 0]
    prior_upper = [0.001, 0.1, 0.1, 4000, 100, 100, 100]

    log_prior = pints.UniformLogPrior(prior_lower, prior_upper)
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    # -------------------------------------------------------------
    # Initial MCMC positions
    # -------------------------------------------------------------
    xs = [
        0.9 * np.array(true_params),
        1.05 * np.array(true_params),
        1.15 * np.array(true_params),
    ]

    print("Parameter dimension:", problem.n_parameters())

    # -------------------------------------------------------------
    # PARALLEL MCMC
    # -------------------------------------------------------------
    ITERS = 100000
    method = pints.DifferentialEvolutionMCMC

    mcmc = pints.MCMCController(log_posterior,
                                chains=3,
                                x0=xs,
                                method=method)

    # Enable multi-core execution
    mcmc.set_parallel(True)

    mcmc.set_max_iterations(ITERS)
    print("Running parallel MCMC...")

    chains = mcmc.run()

    print("Done MCMC.")

    # -------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------
    save_dir = os.path.join(project_root, "ODE_fitting/results")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "MCMC_forward_ABM_chains_parallel.csv")

    chains_reshaped = chains.reshape(chains.shape[0], -1)
    pd.DataFrame(chains_reshaped).to_csv(save_path, index=False, header=False)

    print("Saved parallel MCMC output to:", save_path)


# -------------------------------------------------------------
# REQUIRED FOR MACOS + MULTIPROCESSING
# -------------------------------------------------------------
if __name__ == "__main__":
    main()