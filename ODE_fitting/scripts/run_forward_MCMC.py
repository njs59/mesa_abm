#!/usr/bin/env python3
"""
Run MCMC inference on FORWARD-ABM summary statistics.

Uses:
- forward_means_stats.csv  (S0, S1, S2 from forward sims)
- SmolModel ODE model structure
- Same prior and likelihood as original MCMC code
"""

import numpy as np
import pandas as pd
import pints
import pints.plot
import sys, os
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Import your ODE model
# -------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Model_structures.Model_class_Cell_num_Prolif_cst_Shed import SmolModel

# -------------------------------------------------------------
# Load FORWARD ABM OUTPUT
# -------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
forward_stats_path = os.path.join(project_root,
                                  "ODE_fitting/results/forward_means_stats.csv")

df = pd.read_csv(forward_stats_path)

# Extract the summary statistics:
#   S0 = num_clusters
#   S1 = mean_cluster_size
#   S2 = mean_squared_cluster_size
data_values = df[["num_clusters", "mean_cluster_size", "mean_squared_cluster_size"]].to_numpy()
times = df["step"].to_numpy()

print("Loaded ABM forward stats:")
print("Times:", times.shape)
print("Data values:", data_values.shape)

# -------------------------------------------------------------
# Instantiate ODE model
# -------------------------------------------------------------
model = SmolModel(None, None)

# -------------------------------------------------------------
# Build PINTS problem
# -------------------------------------------------------------
problem = pints.MultiOutputProblem(model, times, data_values)

# Use LogNormal likelihood (your choice in original code)
log_likelihood = pints.LogNormalLogLikelihood(problem)

# -------------------------------------------------------------
# Priors — SAME as your experimental-fitting script
# -------------------------------------------------------------
true_parameters = [0.0003, 0.001, 0.0001, 900, 1, 1, 1]

prior_lower = [0,     0,     0,     1,     0, 0, 0]
prior_upper = [0.001, 0.1,   0.1,   4000, 100,100,100]

log_prior = pints.UniformLogPrior(prior_lower, prior_upper)

log_posterior = pints.LogPosterior(log_likelihood, log_prior)

# -------------------------------------------------------------
# Initial MCMC positions
# -------------------------------------------------------------
xs = [
    0.9  * np.array(true_parameters),
    1.05 * np.array(true_parameters),
    1.15 * np.array(true_parameters),
]

print("Parameter dimension =", problem.n_parameters())

# -------------------------------------------------------------
# Run MCMC
# -------------------------------------------------------------
ITERS = 1000
mcmc = pints.MCMCController(log_posterior,
                            chains=3,
                            x0=xs,
                            method=pints.SliceDoublingMCMC)
mcmc.set_max_iterations(ITERS)

print("Running MCMC...")
chains = mcmc.run()
print("Done.")

# -------------------------------------------------------------
# Save output
# -------------------------------------------------------------
save_dir = os.path.join(project_root, "ODE_fitting/results")
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "MCMC_forward_ABM_chains.csv")

chains_reshaped = chains.reshape(chains.shape[0], -1)
pd.DataFrame(chains_reshaped).to_csv(save_path, index=False, header=False)

print("Saved MCMC chains to:\n   ", save_path)