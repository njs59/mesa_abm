import os
import time
import random
import numpy as np
import pandas as pd
import pints
import matplotlib.pyplot as plt

from .model_registry import load_model_class, get_model_meta


def build_problem(model_key: str, times: np.ndarray, data_values: np.ndarray):
    ModelClass = load_model_class(model_key)
    model = ModelClass(None, None)
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
    init_model = meta.get("initial", [0.1] * (n_params - 3))
    init = np.array(list(init_model) + [0.5, 0.5, 0.5], dtype=float)
    return [
        (1.00 - scale) * init,
        (1.00 + scale) * init,
        1.15 * init,
    ]


def run_adaptive_mcmc(
    model_key: str,
    times: np.ndarray,
    data_values: np.ndarray,
    mcmc_cfg: dict,
    out_dir: str,
    model_meta_overrides: dict | None = None,
    random_seed: int = 12345,
    progress_callback=None,      # NEW: live progress hook (optional)
):
    os.makedirs(out_dir, exist_ok=True)

    # Global seeding for reproducibility (sufficient for this setup)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Build model + posterior
    model, problem = build_problem(model_key, times, data_values)
    meta = get_model_meta(model_key, model_meta_overrides or {})
    logpost, n_params = build_log_posterior(meta, problem)

    # Initial positions for 3 chains
    x0 = initial_positions(meta, n_params)
    n_chains = len(x0)

    # Settings
    chunk = int(mcmc_cfg.get("chunk", 5000))
    print_every = int(mcmc_cfg.get("print_every", 1000))
    rhat_threshold = float(mcmc_cfg.get("rhat_threshold", 1.01))
    max_total_iters = int(mcmc_cfg.get("max_total_iters", 200000))

    total_iters = 0
    converged = False
    all_chains = None
    block_idx = 0

    print(f"\n=== [{model_key}] Starting HaarioBardenetACMC (3 chains) ===")

    while not converged and total_iters < max_total_iters:
        block_idx += 1
        print(f"\n--- [{model_key}] Running block {block_idx} of {chunk} iterations ---")

        # Use the non-deprecated sampler
        controller = pints.MCMCController(
            logpost,
            chains=n_chains,
            x0=x0,
            method=pints.HaarioBardenetACMC,
        )
        controller.set_parallel(True)
        controller.set_log_interval(print_every)
        controller.set_max_iterations(chunk)

        start = time.time()
        part = controller.run()  # (chains, chunk, n_params)
        elapsed = time.time() - start
        print(f"[{model_key}] Block {block_idx} time: {elapsed:.2f}s")

        all_chains = part if all_chains is None else np.concatenate([all_chains, part], axis=1)
        total_iters = all_chains.shape[1]

        # Update positions for next block
        x0 = [all_chains[c, -1, :] for c in range(n_chains)]

        # R-hat on last 50% of samples
        n_keep = max(1, total_iters // 2)
        recent = all_chains[:, -n_keep:, :]
        try:
            rhat = pints.rhat(recent)
        except Exception:
            rhat = pints.rhat(np.asarray(recent))
        rhat = np.asarray(rhat).ravel()

        # ✅ Extra prints requested
        max_rhat = float(np.max(rhat)) if rhat.size else float('nan')
        print(f"[{model_key}] R-hat (last 50%): " + ", ".join(f"{v:.4f}" for v in rhat))
        print(f"[{model_key}] Max R-hat this block: {max_rhat:.5f} | Total iterations so far: {total_iters}")

        # Live progress callback (optional)
        if callable(progress_callback):
            try:
                progress_callback(block_idx, total_iters, rhat.tolist())
            except Exception as e:
                print("Progress callback failed:", e)

        if np.all(np.isfinite(rhat)) and max_rhat < rhat_threshold:
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

    # Burn-in and flat posterior
    burn = int(mcmc_cfg.get("burn_in_fraction", 0.5) * total_iters)
    post = all_chains[:, burn:, :]
    flat = post.reshape(-1, n_params)

    # Summary (best effort)
    try:
        summary = pints.MCMCSummary(post)
        with open(os.path.join(out_dir, f"mcmc_{model_key}_summary.txt"), "w") as f:
            f.write(str(summary))
    except Exception as e:
        print("Warning: Could not generate PINTS summary:", e)

    # Trace plot
    try:
        # Prefer the pints plotting submodule if available
        try:
            import pints.plot as pplot
            pplot.trace(all_chains)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"mcmc_{model_key}_trace.png"), dpi=150)
            plt.close()
        except Exception:
            # Fallback: minimal Matplotlib trace plot
            n_chains, n_iters, n_params = all_chains.shape
            fig, axes = plt.subplots(n_params, 1, figsize=(10, 2 * n_params), sharex=True)
            if n_params == 1:
                axes = [axes]
            for p_idx in range(n_params):
                ax = axes[p_idx]
                for c_idx in range(n_chains):
                    ax.plot(all_chains[c_idx, :, p_idx], lw=0.5, alpha=0.8)
                ax.set_ylabel(f"θ{p_idx+1}")
            axes[-1].set_xlabel("Iteration")
            fig.tight_layout()
            plt.savefig(os.path.join(out_dir, f"mcmc_{model_key}_trace.png"), dpi=150)
            plt.close()
    except Exception as e:
        print("Trace plot failed (fallback also failed):", e)

    # # Approximate MLE from posterior samples (likelihood only)
    # loglik = pints.LogNormalLogLikelihood(problem)
    # loglik_vals = np.array([loglik(v) for v in flat])
    # max_ll = float(np.max(loglik_vals))

    # # AIC with noise params included in k
    # aic = 2 * n_params - 2 * max_ll

    return {
        "chains": all_chains,
        "post_burn_flat": flat,
        "n_params": n_params,
        # "max_loglik": max_ll,
        # "AIC": aic,
        "times": times,
        "data": data_values,
        "model_key": model_key,
    }