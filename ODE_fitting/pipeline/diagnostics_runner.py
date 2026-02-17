import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Enable nicer mathtext fonts globally
import matplotlib as mpl
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams["text.usetex"] = False   # change to True only if full LaTeX installed

from .model_registry import load_model_class, get_model_meta


# ----------------------------------------------------------------------
#  PARAMETER NAME HANDLING
# ----------------------------------------------------------------------
def _param_names_with_noise(model_key: str, overrides: dict | None = None):
    """
    Returns the model parameter names + noise parameters, all wrapped
    in LaTeX syntax so plots show things like $b$, $p$, $s$, $N_0$,
    and noise parameters $\\sigma_0$, $\\sigma_1$, $\\sigma_2$.

    Uses the names from models_meta (pipeline_config.yaml).
    """
    meta = get_model_meta(model_key, overrides or {})
    base_names = meta["param_names"]

    # Wrap model parameters in mathtext $...$
    latex_params = [f"${name}$" for name in base_names]

    latex_noise = [r"$\sigma_0$", r"$\sigma_1$", r"$\sigma_2$"]

    return latex_params + latex_noise


# ----------------------------------------------------------------------
#  TRACE PLOTS
# ----------------------------------------------------------------------
def plot_traces(chains: np.ndarray, param_names, out_path: str):
    n_chains, iters, n_params = chains.shape

    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for p in range(n_params):
        for c in range(n_chains):
            axes[p].plot(chains[c, :, p], lw=0.5, alpha=0.7)
        axes[p].set_ylabel(param_names[p])

    axes[-1].set_xlabel("Iteration")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------
#  MARGINAL POSTERIORS
# ----------------------------------------------------------------------
def plot_marginals(flat_samples: np.ndarray, param_names, out_path: str):
    n_params = flat_samples.shape[1]
    fig, axes = plt.subplots(n_params, 1, figsize=(6, 2 * n_params))

    if n_params == 1:
        axes = [axes]

    for j in range(n_params):
        sns.histplot(flat_samples[:, j], bins=40, kde=True, ax=axes[j])
        axes[j].set_xlabel(param_names[j])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------
#  PAIRWISE POSTERIORS
# ----------------------------------------------------------------------
def plot_pairwise(flat_samples: np.ndarray, param_names, out_path: str):
    df = pd.DataFrame(flat_samples, columns=param_names)
    grid = sns.pairplot(
        df,
        corner=True,
        diag_kind="kde",
        plot_kws={"alpha": 0.25, "s": 8},
    )
    grid.fig.set_size_inches(12, 12)
    grid.savefig(out_path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------
#  POSTERIOR PREDICTIVE CHECKS
# ----------------------------------------------------------------------
def posterior_predictive_checks(
    model_key: str,
    flat_samples: np.ndarray,
    times: np.ndarray,
    data: np.ndarray,
    out_dir: str,
    nsamples: int = 300,
    seed: int = 42
):
    rng = np.random.default_rng(seed)
    ModelClass = load_model_class(model_key)
    model = ModelClass(None, None)

    n_total = flat_samples.shape[0]
    idx = rng.choice(n_total, size=min(nsamples, n_total), replace=False)
    draws = flat_samples[idx]

    # Split parameters: model vs noise
    k_model = int(model.n_parameters()) if hasattr(model, "n_parameters") else (draws.shape[1] - 3)
    theta = draws[:, :k_model]
    sigmas = np.abs(draws[:, k_model:k_model+3])

    T = len(times)
    y_rep = np.zeros((len(idx), T, 3))

    for i in range(len(idx)):
        y_clean = model.simulate(theta[i], times)
        eps = rng.standard_normal(size=(T, 3))
        noise = np.exp(sigmas[i] * eps)   # multiplicative log-normal
        y_rep[i] = y_clean * noise

    med = np.median(y_rep, axis=0)
    low95 = np.percentile(y_rep, 2.5, axis=0)
    high95 = np.percentile(y_rep, 97.5, axis=0)

    labels = ["num_clusters", "mean_cluster_size", "mean_squared_cluster_size"]
    latex_labels = [r"$n$", r"$\bar{s}$", r"$\overline{s^2}$"]

    for k in range(3):
        plt.figure(figsize=(10, 4))
        plt.fill_between(times, low95[:, k], high95[:, k],
                         alpha=0.35, color="skyblue",
                         label="95% predictive interval")
        plt.plot(times, data[:, k], "kx-", ms=4, label="Observed")
        plt.xlabel("Time")
        plt.ylabel(latex_labels[k])
        plt.title(f"PPC for {latex_labels[k]}")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"ppc_{labels[k]}.png"), dpi=150)
        plt.close()


# ----------------------------------------------------------------------
#  ENTRY POINT
# ----------------------------------------------------------------------
def run_diagnostics(
    model_key: str,
    chains: np.ndarray,
    flat_post: np.ndarray,
    times: np.ndarray,
    data: np.ndarray,
    out_dir: str,
    nsamples_ppc: int = 300,
    seed: int = 42
):
    os.makedirs(out_dir, exist_ok=True)

    # Parameter names in LaTeX
    names = _param_names_with_noise(model_key)

    # Standard plots
    plot_traces(chains, names, os.path.join(out_dir, "trace_plots.png"))
    plot_marginals(flat_post, names, os.path.join(out_dir, "posterior_marginals.png"))

    try:
        plot_pairwise(flat_post, names, os.path.join(out_dir, "pairwise_posteriors.png"))
    except Exception as e:
        print("Pairwise plot failed:", e)

    posterior_predictive_checks(
        model_key,
        flat_post,
        times,
        data,
        out_dir,
        nsamples=nsamples_ppc,
        seed=seed
    )