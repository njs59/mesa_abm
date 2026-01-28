
from typing import Dict, Any
import yaml
import pyabc

# Updated DEFAULT_PRIORS for new ABM schema (single merge.strength)
DEFAULT_PRIORS = {
    # biological parameters (proliferative phenotype)
    "prolif_rate": [0.0, 0.02],
    "fragment_rate": [0.0, 0.01],
    # unified merge parameter
    "merge_strength": [0.0, 1.0],
    # initial condition
    "init_n_clusters": [100, 1500],
    # movement priors (generic across models)
    "speed_meanlog": [-1.0, 2.0],  # for lognorm
    "speed_sdlog": [0.1, 1.5],     # for lognorm
    "speed_shape": [1.0, 5.0],     # for gamma/weibull
    "speed_scale": [0.2, 4.0],     # for gamma/weibull/lognorm scale
    "heading_sigma": [0.0, 0.8],
}

def _to_distribution(bounds: Dict[str, Any]) -> pyabc.Distribution:
    parts = {}
    for name, rng in bounds.items():
        low, high = float(rng[0]), float(rng[1])
        parts[name] = pyabc.RV("uniform", low, high - low)
    return pyabc.Distribution(**parts)


def load_priors(yaml_path: str = None) -> pyabc.Distribution:
    """Load priors from YAML if provided, else use DEFAULT_PRIORS."""
    bounds = DEFAULT_PRIORS
    if yaml_path is not None:
        try:
            with open(yaml_path, 'r') as f:
                user_bounds = yaml.safe_load(f) or {}
            bounds = {**DEFAULT_PRIORS, **user_bounds}
        except Exception as e:
            print(f"[priors] YAML not loaded ({e}); using defaults")
    return _to_distribution(bounds)
