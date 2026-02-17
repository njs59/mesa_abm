"""
Model registry adaptor used by the pipeline.

- Imports your existing Model_structures/model_registry.py (class locations).
- Lets you provide optional per-model metadata (names / priors / initials)
  via pipeline_config.yaml.
- If meta aren't provided, it infers the parameter count from the class
  and uses generic non-negative priors, assuming the last parameter is N0.

Noise parameters (sigma_0, sigma_1, sigma_2) for the 3 outputs are appended
inside mcmc_runner and don't need to be listed here.
"""
from importlib import import_module


def _import_external_registry():
    """
    Try importing your existing registry from both likely import paths.
    Returns (module, registry_dict, load_fn).
    """
    candidates = [
        "ODE_fitting.Model_structures.model_registry",
        "Model_structures.model_registry",
    ]
    last_err = None
    for path in candidates:
        try:
            mod = import_module(path)
            reg = getattr(mod, "MODEL_REGISTRY")
            load_fn = getattr(mod, "load_model")
            return mod, reg, load_fn
        except Exception as e:
            last_err = e
            continue
    raise ImportError(
        f"Could not import Model_structures.model_registry via {candidates}: {last_err}"
    )


# Import your external registry once
_EXT_MOD, EXTERNAL_REGISTRY, EXTERNAL_LOAD = _import_external_registry()


def load_model_class(model_key: str):
    """Return the SmolModel class for the given key (delegates to your registry)."""
    return EXTERNAL_LOAD(model_key)


def infer_default_meta(model_key: str):
    """
    If no explicit meta are provided in pipeline_config, infer:
      - n_params from model.n_parameters()
      - param_names: ["theta1", ... "thetaK"] (last one labelled "N0")
      - priors: non-negative Uniform; last param wide [1, 4000] for N0
      - initial: small positives + N0=500
    """
    cls = load_model_class(model_key)
    try:
        model = cls(None, None)
    except Exception:
        # In case constructor signature differs, fall back without instantiation
        model = None

    if model is not None and hasattr(model, "n_parameters"):
        k = int(model.n_parameters())
    else:
        # default to 3 just to allow pipeline to continue; user should override
        k = 3

    names = [f"theta{i+1}" for i in range(k)]
    # Assume the last parameter behaves like N0 for many of your variants
    names[-1] = "N0"

    priors = [[0.0, 0.1] for _ in range(k)]
    priors[-1] = [1.0, 4000.0]  # N0
    initial = [1.0e-3 for _ in range(k)]
    initial[-1] = 500.0         # N0

    return {
        "param_names": names,
        "priors": priors,
        "initial": initial,
    }


def get_model_meta(model_key: str, overrides: dict | None = None):
    """
    Merge overrides (from pipeline_config.yaml) onto inferred defaults.
    Overrides can specify: param_names, priors, initial
    """
    base = infer_default_meta(model_key)
    if overrides:
        base.update({k: v for k, v in overrides.items() if v is not None})
    return base