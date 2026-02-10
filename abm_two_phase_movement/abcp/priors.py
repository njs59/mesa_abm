# abcp/priors.py
from typing import Dict, Any
import yaml
import pyabc

# Priors ONLY for remaining inferred parameters (no speed/direction!)
DEFAULT_PRIORS: Dict[str, Any] = {
    # biological parameters
    "prolif_rate": [0.0, 0.02],
    "fragment_rate": [0.0, 0.01],

    # unified merge parameter
    "p_merge": [0.0, 1.0],

    # initial condition
    "init_n_clusters": [100, 1500],
}


def _to_distribution(bounds: Dict[str, Any]) -> pyabc.Distribution:
    parts = {}
    for name, rng in bounds.items():
        low, high = float(rng[0]), float(rng[1])
        parts[name] = pyabc.RV("uniform", low, high - low)
    return pyabc.Distribution(**parts)


def load_priors(yaml_path: str = None) -> pyabc.Distribution:
    """
    Load priors from YAML if provided, else use DEFAULT_PRIORS.
    YAML keys override the defaults; keys not present keep default ranges.
    """
    bounds = DEFAULT_PRIORS
    if yaml_path is not None:
        try:
            with open(yaml_path, "r") as f:
                user_bounds = yaml.safe_load(f) or {}
            bounds = {**DEFAULT_PRIORS, **user_bounds}
        except Exception as e:
            print(f"[priors] YAML not loaded ({e}); using defaults")
    return _to_distribution(bounds)