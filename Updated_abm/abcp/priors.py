from __future__ import annotations
from pathlib import Path
from typing import Optional

import yaml
import pyabc

def load_priors(yaml_path: Optional[str] = None):
    """
    Load priors from a YAML file. If not given, uses defaults here.
    YAML schema (keys under 'priors'):
      a_gamma:     {type: uniform, low: 0.5, high: 4.0}
      scale_gamma:{type: uniform, low: 0.5, high: 6.0}
      kappa_turn: {type: uniform, low: 0.0, high: 1.0}
      p_merge:    {type: uniform, low: 0.2, high: 0.9}
    """
    priors_default = {
        'a_gamma':     {'type': 'uniform', 'low': 0.5, 'high': 4.0},
        'scale_gamma': {'type': 'uniform', 'low': 0.5, 'high': 6.0},
        'kappa_turn':  {'type': 'uniform', 'low': 0.0, 'high': 1.0},
        'p_merge':     {'type': 'uniform', 'low': 0.2, 'high': 0.9},
    }

    pri = priors_default
    if yaml_path is not None and Path(yaml_path).exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            y = yaml.safe_load(f) or {}
            pri = (y.get('priors') or priors_default)

    # Build pyabc distribution
    dist_kwargs = {}
    for name, spec in pri.items():
        typ = (spec.get('type') or 'uniform').lower()
        if typ == 'uniform':
            low, high = float(spec['low']), float(spec['high'])
            dist_kwargs[name] = pyabc.RV('uniform', low, high - low)  # scipy uniform(loc, scale)
        elif typ == 'norm':
            mu, sd = float(spec['mu']), float(spec['sd'])
            dist_kwargs[name] = pyabc.RV('norm', mu, sd)
        else:
            raise ValueError(f"Unsupported prior type: {typ} for {name}")

    return pyabc.Distribution(**dist_kwargs)