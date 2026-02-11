
#!/usr/bin/env python3
from __future__ import annotations
import yaml
from typing import Any, Dict, List
from copy import deepcopy


def _build_timesteps(cfg: Dict[str, Any]) -> List[int]:
    if cfg.get("timesteps") is not None:
        return [int(t) for t in cfg["timesteps"]]
    total_steps = int(cfg.get("total_steps", 145))
    every = int(cfg.get("timesteps_every", 5))
    # include total_steps if divisible; otherwise last < total_steps
    ts = list(range(0, total_steps + 1, every))
    if ts[-1] != total_steps:
        ts.append(total_steps)
    return ts


def load_sweep_config(path: str = "sweeps/sweep_defaults.yaml") -> Dict[str, Any]:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f) or {}

    cfg: Dict[str, Any] = {}
    cfg["observed_csv"]  = raw.get("observed_csv", "observed/INV_ABM_ready_summary.csv")
    cfg["total_steps"]   = int(raw.get("total_steps", 145))
    cfg["timesteps_every"] = int(raw.get("timesteps_every", 5))
    cfg["timesteps"]     = _build_timesteps(raw)
    cfg["summary_stats"] = list(raw.get("summary_stats", ["S0", "S1", "S2", "NND_med"]))
    cfg["replicates"]    = int(raw.get("replicates", 100))
    cfg["workers"]       = int(raw.get("workers", 8))
    cfg["seed_base"]     = int(raw.get("seed_base", 0))

    # Flat baseline parameter values
    params = raw.get("params", {}) or {}
    cfg["params"] = {
        "prolif_rate": float(params.get("prolif_rate", 0.005)),
        "fragment_rate": float(params.get("fragment_rate", 0.0005)),
        "p_merge": float(params.get("p_merge", 0.9)),
        "softness": float(params.get("softness", 0.15)),
        "fragment_minsep_factor": float(params.get("fragment_minsep_factor", 1.1)),
        "n_init": int(params.get("n_init", 800)),
    }
    return cfg


def deep_update(base: Dict[str, Any], upd: Dict[str, Any] | None) -> Dict[str, Any]:
    out = deepcopy(base)
    if not upd:
        return out
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    if not overrides:
        return cfg
    merged = deep_update(cfg, overrides)
    # Rebuild timesteps if timing overrides were provided
    timing_keys = {"timesteps", "timesteps_every", "total_steps"}
    if any(k in overrides for k in timing_keys):
        merged["timesteps"] = _build_timesteps(merged)
    return merged
