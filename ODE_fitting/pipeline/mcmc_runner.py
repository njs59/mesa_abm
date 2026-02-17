import os
import time
import numpy as np
import pandas as pd
import pints
import matplotlib.pyplot as plt

from .model_registry import load_model_class, get_model_meta


def build_problem(model_key: str, times: np.ndarray, data_values: np.ndarray):
    ModelClass = load_model_class(model_key)
    model = ModelClass(None, None)  # your SmolModels accept (None, None)
    problem = pints.MultiOutputProblem(model, times, data_values)
    return model, problem


def build_log_posterior(meta: dict, problem: pints.MultiOutputProblem):
    """
    meta: dict with 'priors' (for model params only).
    Appends 3 noise std devs for the three outputs.
    """
    k_model = len(meta['priors'])
    lower = [lo for lo, hi in meta['priors']]
    upper = [hi for lo, hi in meta['priors']]
    # Append 3 lognormal noise std devs (for the 3 outputs)
    lower += [0.0, 0.0, 0.0]
    upper += [100.0, 100.0, 100.0]
    prior = pints.UniformLogPrior(lower, upper)

    loglik = pints.LogNormalLogLikelihood(problem)
    logpost = pints.LogPosterior(loglik, prior)
    return logpost, k_model + 3


def initial_positions(meta: dict, n_params: int, scale: float = 0.1):
    init_model = meta.get('initial', [0.1] * (n_params - 3))
    init = np.array(list(init_model) + [0.5, 0.5, 0.5], dtype=float)
    return [
        (1.00 - scale) * init,
        (1.00 + scale) * init,
        (1.15) * init,
    ]


def run_adaptive_mcmc(model_key: str, times: np.ndarray, data_values: np.ndarray,
                      mcmc_cfg: dict, out_dir: str, model_meta_overrides: dict | None = None,
                      random_seed: int = 12345):
    os.makedirs(out_dir, exist_ok=True)

    # Build problem and log-posterior
    model, problem = build_problem(model_key, times, data_values)
    meta = get_model_meta(model_key, overrides=model_meta_overrides or {})
    logpost, n_params = build_log_posterior(meta, problem)

    # Initial positions for 3 chains
    x0 = initial_positions(meta, n_params)
    n_chains = len(x0)

    # Settings
    chunk = int(mcmc_cfg.get('chunk', 5000))
    print_every = int(mcmc_cfg.get('print_every', 1000))
    rhat_threshold = float(mcmc_cfg.get('rhat_threshold', 1.01))
    max_total_iters = int(mcmc_cfg.get('max_total_iters', 200000))

    total_iters = 0
    converged = False
    all_chains = None

    print(f"\n=== [{model_key}] Starting AdaptiveCovarianceMCMC (3 chains) ===")

    while not converged and total_iters < max_total_iters:
        print(f"\n--- [{model_key}] Running block of {chunk} iterations ---")
        controller = pints.MCMCController(
            logpost, chains=n_chains, x0=x0, method=pints.AdaptiveCovarianceMCMC
        )
        controller.set_parallel(True)
        controller.set_log_interval(print_every)
        controller.set_max_iterations(chunk)
        controller.set_random_seed(random_seed)

        start = time.time()
        part = controller.run()  # (chains, chunk, n_params)
        print(f"Block time: {time.time()-start:.2f}s")

        all_chains = part if all_chains is None else np.concatenate([all_chains, part], axis=1)
        total_iters = all_chains.shape[1]
        x0 = [all_chains[c, -1, :] for c in range(n_chains)]

        # R-hat on last 50%
        n_keep = max(1, total_iters // 2)
        recent = all_chains[:, -n_keep:, :]
        try:
            rhat = pints.rhat(recent)
        except Exception:
            rhat = pints.rhat(np.asarray(recent))
        rhat = np.asarray(rhat).ravel()
        print("R-hat (last 50%):", ", ".join(f"{v:.4f}" for v in rhat))
        if np.all(np.isfinite(rhat)) and np.max(rhat) < rhat_threshold:
            converged = True
            print(f"\n=== [{model_key}] Convergence achieved. ===")
            break

    # Save chains
    npz_path = os.path.join(out_dir, f"mcmc_{model_key}_chains.npz")
    np.savez_compressed(npz_path, chains=all_chains)

    # Per-chain CSVs
    for c in range(n_chains):
        csv_path = os.path.join(out_dir, f"mcmc_{model_key}_chain{c+1}.csv")
        pd.DataFrame(all_chains[c]).to_csv(csv_path, index=False, header=False)

    # Summary after burn-in
    burn = int(mcmc_cfg.get('burn_in_fraction', 0.5) * total_iters)
    post = all_chains[:, burn:, :]
    try:
        summary = pints.MCMCSummary(post)
        with open(os.path.join(out_dir, f"mcmc_{model_key}_summary.txt"), 'w') as f:
            f.write(str(summary))
    except Exception as e:
        print("Warning: Could not generate PINTS summary:", e)

    # Trace plot
    try:
        pints.plot.trace(all_chains)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"mcmc_{model_key}_trace.png"), dpi=150)
        plt.close()
    except Exception as e:
        print("Trace plot failed:", e)

    # Approximate MLE via best posterior sample (likelihood only)
    loglik = pints.LogNormalLogLikelihood(problem)
    flat = post.reshape(-1, n_params)
    loglik_vals = np.array([loglik(v) for v in flat])
    max_ll = float(np.max(loglik_vals))

    # AIC = 2k - 2 log L  (k includes 3 noise std devs)
    aic = 2 * n_params - 2 * max_ll

    return {
        'chains': all_chains,
        'npz_path': npz_path,
        'n_params': n_params,
        'max_loglik': max_ll,
        'AIC': aic,
        'post_burn_flat': flat,
        'times': times,
        'data': data_values,
        'model_key': model_key,
    }