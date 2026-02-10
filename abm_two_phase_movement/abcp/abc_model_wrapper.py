# abcp/abc_model_wrapper.py
from typing import Dict
import copy
from abm.utils import DEFAULTS  # fixed movement_v2 lives here


def _set_nested(base: dict, dotted: str, value):
    keys = dotted.split(".")
    d = base
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def particle_to_params(particle: Dict[str, float]) -> dict:
    """
    Map a sampled particle (only biological/interaction params) -> full params dict
    compatible with ClustersModel using the new fixed two‑phase movement model.

    Movement is NOT parameterised here: DEFAULTS["movement_v2"] is used as-is.
    """
    # Start from a deep copy of DEFAULTS to preserve the fixed movement_v2, space, etc.
    params = copy.deepcopy(DEFAULTS)  # ensures we carry "movement_v2" intact  # <- uses DEFAULTS  [2](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/priors.py)

    # Allowed keys to infer (extend if you decide to fit more):
    #   - prolif_rate, fragment_rate      → phenotypes.proliferative.*
    #   - p_merge                         → merge.p_merge
    #   - init_n_clusters                 → init.n_clusters
    mapping = {
        "prolif_rate": "phenotypes.proliferative.prolif_rate",
        "fragment_rate": "phenotypes.proliferative.fragment_rate",
    }

    for k, v in particle.items():
        if k in mapping:
            _set_nested(params, mapping[k], float(v))
        elif k == "p_merge":
            params["merge"]["p_merge"] = float(v)
        elif k == "init_n_clusters":
            params["init"]["n_clusters"] = int(max(1, round(v)))
        # Any other keys are ignored by default

    # Ensure phenotype of initial agents
    params["init"]["phenotype"] = "proliferative"

    # Coerce merge bounds
    p = float(params["merge"].get("p_merge", 0.9))
    params["merge"]["p_merge"] = max(0.0, min(1.0, p))

    return params