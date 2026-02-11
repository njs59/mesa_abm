import yaml
import pyabc

DEFAULT_PRIORS = {
    "prolif_rate": [1e-4, 1e-2],
    "fragment_rate": [1e-5, 1e-3],
    "p_merge": [0.5, 1.0],
    "softness": [0.05, 0.4],
    "fragment_minsep_factor": [1.0, 2.0],
    "n_init": [400, 1200],
}

def _to_distribution(bounds):
    parts = {}
    for name, (low, high) in bounds.items():
        low = float(low)
        high = float(high)
        parts[name] = pyabc.RV("uniform", low, high - low)
    return pyabc.Distribution(**parts)

def load_priors(yaml_path=None):
    """
    Load priors from YAML file; fallback to DEFAULT_PRIORS.
    """
    bounds = DEFAULT_PRIORS.copy()

    if yaml_path is not None:
        try:
            with open(yaml_path, "r") as f:
                user_bounds = yaml.safe_load(f) or {}
            bounds.update(user_bounds)
        except Exception as e:
            print(f"[priors] YAML not loaded ({e}); using defaults")

    return _to_distribution(bounds)
