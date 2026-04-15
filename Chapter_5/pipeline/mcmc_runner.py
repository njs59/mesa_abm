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
    k_model = len(meta['priors'])
    lower = [lo for lo, hi in meta['priors']]
    upper = [hi for lo, hi in meta['priors']]

    # append noise parameters
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
    progress_callback=None,      # NEW
):
    os.makedirs(out_dir, exist_ok=True)

    random.seed(random_seed)
    np.random.seed(random_seed)

    model, problem = build_problem(model_key, times, data_values)
    meta = get_model_meta(model_key, model_meta_overrides or {})
    logpost, n_params = build_log_posterior(meta, problem)

    x0 = initial_positions(meta, n_params)
    n_chains = len(x0)

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
        part = controller.run()
        elapsed = time.time() - start
        print(f"[{model_key}] Block {block_idx} time: {elapsed:.2f}s")

        all_chains = part if all_chains is None else np.concatenate([all_chains, part], axis=1)
        total_iters = all_chains.shape[1]

        x0 = [all_chains[c, -1, :] for c in range(n_chains)]

        n_keep = max(1, total_iters // 2)
        recent = all_chains[:, -n_keep:, :]
        try:
            rhat = pints.rhat(recent)
        except Exception:
            rhat = pints.rhat(np.asarray(recent))
        rhat = np.asarray(rhat).ravel()

        max_rhat = float(np.max(rhat)) if rhat.size else float('nan')
        print(f"[{model_key}] R-hat (last 50%): " + ", ".join(f"{v:.4f}" for v in rhat))
        print(f"[{model_key}] Max R-hat this block: {max_rhat:.5f} | Total iterations so far: {total_iters}")

        if callable(progress_callback):
            try:
                progress_callback(block_idx, total_iters, rhat.tolist())
            except Exception as e:
                print("Progress callback failed:", e)

        if np.all(np.isfinite(rhat)) and max_rhat < rhat_threshold:
            converged = True
            print(f"\n=== [{model_key}] Convergence achieved. ===")
            break

    # save chains
    npz_path = os.path.join(out_dir, f"mcmc_{model_key}_chains.npz")
    np.savez_compressed(npz_path, chains=all_chains)

    # per chain csv
    for c in range(n_chains):
        csv_path = os.path.join(out_dir, f"mcmc_{model_key}_chain{c+1}.csv")
        pd.DataFrame(all_chains[c]).to_csv(csv_path, index=False, header=False)

    burn = int(mcmc_cfg.get("burn_in_fraction", 0.5) * total_iters)
    post = all_chains[:, burn:, :]
    flat = post.reshape(-1, n_params)

    try:
        summary = pints.MCMCSummary(post)
        with open(os.path.join(out_dir, f"mcmc_{model_key}_summary.txt"), "w") as f:
            f.write(str(summary))
    except Exception as e:
        print("Warning: Could not generate PINTS summary:", e)

    # trace plot
    try:
        try:
            import pints.plot as pplot
            pplot.trace(all_chains)
        except Exception:
            fig, axes = plt.subplots(n_params, 1, figsize=(10, 2*n_params), sharex=True)
            if n_params == 1:
                axes = [axes]
            for p_idx in range(n_params):
                ax = axes[p_idx]
                for c_idx in range(n_chains):
                    ax.plot(all_chains[c_idx, :, p_idx], lw=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"mcmc_{model_key}_trace.png"), dpi=150)
        plt.close()
    except Exception as e:
        print("Trace plot failed:", e)

    # MCMC no longer computes AIC — MLE does that now
    return {
        "chains": all_chains,
        "post_burn_flat": flat,
        "n_params": n_params,
        "times": times,
        "data": data_values,
        "model_key": model_key,
    }