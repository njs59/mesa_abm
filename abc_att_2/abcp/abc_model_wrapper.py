from typing import Dict
import numpy as np
from abm.utils import DEFAULTS

# Mapping helper

def _set_nested(base: dict, dotted: str, value):
    keys = dotted.split(".")
    d = base
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def build_speed_params(speed_dist: str, particle: dict):
    if speed_dist == "constant":
        return {}
    if speed_dist == "lognorm":
        mu = float(particle.get("speed_meanlog", 1.0))
        sd = float(particle.get("speed_sdlog", 0.7))
        return {"s": sd, "scale": float(np.exp(mu))}
    if speed_dist == "gamma":
        shape = float(particle.get("speed_shape", 2.0))
        scale = float(particle.get("speed_scale", 1.0))
        return {"a": shape, "scale": scale}
    if speed_dist == "weibull":
        shape = float(particle.get("speed_shape", 2.0))
        scale = float(particle.get("speed_scale", 2.0))
        return {"c": shape, "scale": scale}
    return {}


def particle_to_params(particle: Dict[str, float], motion: str = "isotropic", speed_dist: str = "constant") -> dict:
    params = {
        "space": dict(DEFAULTS["space"]),
        "time": dict(DEFAULTS["time"]),
        "physics": dict(DEFAULTS["physics"]),
        "phenotypes": {
            "proliferative": dict(DEFAULTS["phenotypes"]["proliferative"]),
            "invasive": dict(DEFAULTS["phenotypes"]["invasive"]),
        },
        "merge": dict(DEFAULTS["merge"]),
        "init": dict(DEFAULTS["init"]),
        "movement": dict(DEFAULTS["movement"]),
    }
    params["movement"]["direction"] = motion
    if speed_dist == "constant":
        params["movement"]["mode"] = "constant"
        params["movement"].pop("distribution", None)
        params["movement"].pop("dist_params", None)
    else:
        params["movement"]["mode"] = "distribution"
        params["movement"]["distribution"] = speed_dist
        params["movement"]["dist_params"] = build_speed_params(speed_dist, particle)
    if motion == "persistent":
        hs = float(particle.get("heading_sigma", params["movement"].get("heading_sigma", 0.25)))
        params["movement"]["heading_sigma"] = max(0.0, hs)
    else:
        params["movement"].pop("heading_sigma", None)

    mapping = {
        "prolif_rate": "phenotypes.proliferative.prolif_rate",
        "adhesion": "phenotypes.proliferative.adhesion",
        "fragment_rate": "phenotypes.proliferative.fragment_rate",
        "merge_prob": "merge.prob_contact_merge",
    }
    for k, v in particle.items():
        if k.startswith("speed_") or k == "heading_sigma":
            continue
        if ("rate" in k or "prob" in k) and v < 0:
            v = 0.0
        if k in mapping:
            _set_nested(params, mapping[k], float(v))
        elif k == "init_n_clusters":
            params["init"]["n_clusters"] = int(max(1, round(v)))
        else:
            try:
                params[k] = float(v)
            except Exception:
                pass
    params["init"]["phenotype"] = "proliferative"
    return params
