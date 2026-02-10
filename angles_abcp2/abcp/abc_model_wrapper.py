from typing import Dict
import numpy as np
from abm.utils import DEFAULTS

def _set_nested(base: dict, dotted: str, value):
    keys=dotted.split('.')
    d=base
    for k in keys[:-1]:
        d=d[k]
    d[keys[-1]]=value

def build_speed_params(speed_dist: str, particle: dict):
    if speed_dist == "constant":
        return {}
    if speed_dist == "lognorm":
        mu=float(particle.get("speed_meanlog",1.0)); sd=float(particle.get("speed_sdlog",0.7))
        return {"s": sd, "scale": float(np.exp(mu))}
    if speed_dist == "gamma":
        shape=float(particle.get("speed_shape",2.0)); scale=float(particle.get("speed_scale",1.0))
        return {"a": shape, "scale": scale}
    if speed_dist == "weibull":
        shape=float(particle.get("speed_shape",2.0)); scale=float(particle.get("speed_scale",2.0))
        return {"c": shape, "scale": scale}
    return {}

def base_params_copy() -> dict:
    return {
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

def particle_to_params_angles(particle: Dict[str, float], *, scenario: int, speed_dist: str = "constant", fixed: Dict[str, float] | None = None) -> dict:
    params = base_params_copy()
    fixed=fixed or {}
    if speed_dist == "constant":
        params["movement"]["mode"] = "constant"
        params["movement"].pop("distribution", None)
        params["movement"].pop("dist_params", None)
    else:
        params["movement"]["mode"] = "distribution"
        params["movement"]["distribution"] = speed_dist
        params["movement"]["dist_params"] = build_speed_params(speed_dist, particle)
    mapping = {
        "prolif_rate": "phenotypes.proliferative.prolif_rate",
        "fragment_rate": "phenotypes.proliferative.fragment_rate",
    }
    if "p_merge" in particle:
        params["merge"]["p_merge"] = max(0.0, min(1.0, float(particle["p_merge"])) )
    for k,v in particle.items():
        if k.startswith("speed_"):
            continue
        if k in mapping:
            _set_nested(params, mapping[k], float(v))
        elif k == "init_n_clusters":
            params["init"]["n_clusters"] = int(max(1, round(v)))
    params["init"]["phenotype"] = "proliferative"

    def set_whole_von_mises(mu, kappa):
        mv=params["movement"]
        mv["direction"] = "von_mises"; mv["mu"] = float(mu); mv["kappa"] = float(kappa)
        mv.pop("heading_sigma", None)
        params.pop("movement_phases", None)
        params.pop("movement_phase_builder", None)

    def set_persistent(hs):
        mv=params["movement"]
        mv["direction"] = "persistent"; mv["heading_sigma"] = float(hs)
        params.pop("movement_phases", None)
        params.pop("movement_phase_builder", None)

    def set_two_phase_builder(t_switch, mu1, kappa1, mu2, kappa2):
        params["movement_phase_builder"] = {
            "scenario": "two_phase_fixed",
            "t_switch": float(t_switch),
            "mu1": float(mu1), "kappa1": float(kappa1),
            "mu2": float(mu2), "kappa2": float(kappa2),
        }
        params.pop("movement_phases", None)

    def set_two_phase_random_switch(mu1,kappa1,mu2,kappa2, switch_cfg: dict):
        cfg={
            "scenario": "two_phase_random_switch",
            "mu1": float(mu1), "kappa1": float(kappa1),
            "mu2": float(mu2), "kappa2": float(kappa2),
        }
        cfg.update(switch_cfg)
        params["movement_phase_builder"]=cfg
        params.pop("movement_phases", None)

    if scenario == 1:
        hs = float(particle.get("heading_sigma", params["movement"].get("heading_sigma", 0.25)))
        set_persistent(hs)
    elif scenario == 2:
        set_whole_von_mises(mu=0.0, kappa=0.0)
    elif scenario == 3:
        kappa = float(fixed.get("kappa", 0.5)); set_whole_von_mises(mu=-np.pi, kappa=kappa)
    elif scenario == 4:
        kappa = float(particle.get("kappa", 0.5)); set_whole_von_mises(mu=-np.pi, kappa=kappa)
    elif scenario == 5:
        kappa = float(fixed.get("kappa", 0.5)); set_whole_von_mises(mu=0.0, kappa=kappa)
    elif scenario == 6:
        kappa = float(particle.get("kappa", 0.5)); set_whole_von_mises(mu=0.0, kappa=kappa)
    elif scenario == 7:
        k1=float(fixed.get("kappa1", 0.5)); k2=float(fixed.get("kappa2", 0.5)); t=float(fixed.get("t_switch", 100.0))
        set_two_phase_builder(t_switch=t, mu1=-np.pi, kappa1=k1, mu2=0.0, kappa2=k2)
    elif scenario == 8:
        k1=float(particle.get("kappa1", 0.5)); k2=float(particle.get("kappa2", 0.5)); t=float(fixed.get("t_switch", 100.0))
        set_two_phase_builder(t_switch=t, mu1=-np.pi, kappa1=k1, mu2=0.0, kappa2=k2)
    elif scenario == 9:
        k1=float(fixed.get("kappa1", 0.5)); k2=float(fixed.get("kappa2", 0.5)); t=float(particle.get("t_switch", 100.0))
        set_two_phase_builder(t_switch=t, mu1=-np.pi, kappa1=k1, mu2=0.0, kappa2=k2)
    elif scenario == 10:
        k1=float(particle.get("kappa1", 0.5)); k2=float(particle.get("kappa2", 0.5)); t=float(particle.get("t_switch", 100.0))
        set_two_phase_builder(t_switch=t, mu1=-np.pi, kappa1=k1, mu2=0.0, kappa2=k2)
    elif scenario == 11:
        dist=fixed.get("switch_dist","lognorm")
        switch_cfg={"switch_dist": dist}
        if dist=="lognorm":
            switch_cfg["switch_meanlog"]=float(fixed.get("switch_meanlog", 4.0))
            switch_cfg["switch_sdlog"]=float(fixed.get("switch_sdlog", 0.6))
        elif dist=="uniform":
            switch_cfg["switch_low"]=float(fixed.get("switch_low", 10.0))
            switch_cfg["switch_high"]=float(fixed.get("switch_high", 200.0))
        set_two_phase_random_switch(-np.pi, float(fixed.get("kappa1",0.5)), 0.0, float(fixed.get("kappa2",0.5)), switch_cfg)
    elif scenario == 12:
        dist=fixed.get("switch_dist","lognorm")
        switch_cfg={"switch_dist": dist}
        if dist=="lognorm":
            switch_cfg["switch_meanlog"]=float(particle.get("switch_meanlog", fixed.get("switch_meanlog", 4.0)))
            switch_cfg["switch_sdlog"]=float(particle.get("switch_sdlog", fixed.get("switch_sdlog", 0.6)))
        elif dist=="uniform":
            switch_cfg["switch_low"]=float(particle.get("switch_low", fixed.get("switch_low", 10.0)))
            switch_cfg["switch_high"]=float(particle.get("switch_high", fixed.get("switch_high", 200.0)))
        k1=float(particle.get("kappa1", fixed.get("kappa1",0.5)))
        k2=float(particle.get("kappa2", fixed.get("kappa2",0.5)))
        set_two_phase_random_switch(-np.pi, k1, 0.0, k2, switch_cfg)
    else:
        params["movement"]["direction"] = "isotropic"
    return params
