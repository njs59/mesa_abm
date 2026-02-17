# pipeline/mle_runner.py
import os
import numpy as np
import pints
import scipy.stats
import math

from .model_registry import load_model_class, get_model_meta

def run_mle(model_key, times, data_values, out_dir, model_meta_overrides=None):
    os.makedirs(out_dir, exist_ok=True)

    ModelClass = load_model_class(model_key)
    model = ModelClass(None, None)

    problem = pints.MultiOutputProblem(model, times, data_values)
    loglik = pints.LogNormalLogLikelihood(problem)

    meta = get_model_meta(model_key, model_meta_overrides or {})
    param_names = meta["param_names"]

    k_model = len(param_names)
    k_total = k_model + 3

    theta0 = np.array(meta["initial"] + [0.5, 0.5, 0.5], dtype=float)

    lower = [p[0] for p in meta["priors"]] + [0.0, 0.0, 0.0]
    upper = [p[1] for p in meta["priors"]] + [100.0, 100.0, 100.0]
    boundaries = pints.RectangularBoundaries(lower, upper)

    opt = pints.OptimisationController(
        loglik, theta0, boundaries=boundaries, method=pints.CMAES
    )
    opt.set_log_to_screen(False)

    theta_hat, max_ll = opt.run()
    max_ll = float(max_ll)

    AIC = 2 * k_total - 2 * max_ll
    n = data_values.shape[0]
    BIC = k_total * math.log(n) - 2 * max_ll

    result = {
        "theta_hat": theta_hat.tolist(),
        "max_loglik": max_ll,
        "AIC": AIC,
        "BIC": BIC,
        "k_params": k_total,
        "param_names": param_names + ["sigma0", "sigma1", "sigma2"]
    }

    import yaml
    with open(os.path.join(out_dir, "mle_results.yaml"), "w") as f:
        yaml.safe_dump(result, f, sort_keys=False)

    return result