from typing import Dict, Any
import yaml
import pyabc

DEFAULT_PRIORS = {
    # Core biological / interaction parameters
    "prolif_rate": [0.0, 0.02],
    "adhesion": [0.0, 1.0],
    "fragment_rate": [0.0, 0.01],
    "merge_prob": [0.0, 1.0],
    # Initial condition
    "init_n_clusters": [100, 1500],
    # Motion model (constant by default). If you switch to distributions, add priors accordingly.
    # e.g., for lognormal speed magnitude per step:
    "speed_meanlog": [ -1.0, 2.0 ],
    "speed_sdlog":   [  0.1, 1.5 ],
    # heading noise if persistent
    "heading_sigma": [0.0, 0.8],
}

def _to_distribution(bounds: Dict[str, Any]) -> pyabc.Distribution:
    parts = {}
    for name, rng in bounds.items():
        low, high = float(rng[0]), float(rng[1])
        parts[name] = pyabc.RV("uniform", low, high - low)
    return pyabc.Distribution(**parts)

def load_priors(yaml_path: str = None) -> pyabc.Distribution:
    """Load priors from YAML if provided, else use DEFAULT_PRIORS.

    YAML format:
      prolif_rate: [0.0, 0.02]
      adhesion:    [0.0, 1.0]
      ...
    """
    bounds = DEFAULT_PRIORS
    if yaml_path is not None:
        try:
            with open(yaml_path, 'r') as f:
                user_bounds = yaml.safe_load(f) or {}
            # override defaults with any keys the user provided
            bounds = {**DEFAULT_PRIORS, **user_bounds}
        except Exception as e:
            print(f"[priors] YAML not loaded ({e}); using defaults")
    return _to_distribution(bounds)
