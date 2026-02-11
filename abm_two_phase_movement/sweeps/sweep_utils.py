#!/usr/bin/env python3
from __future__ import annotations

import copy
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

from abm.utils import DEFAULTS


# ------------------------------
# FLAT → NESTED PARAM MAPPING
# ------------------------------

def flat_to_nested_params(flat: Dict[str, Any]) -> Dict[str, Any]:
    ph = {
        "prolif_rate": flat.get("prolif_rate"),
        "fragment_rate": flat.get("fragment_rate"),
    }
    ph = {k: v for k, v in ph.items() if v is not None}

    physics = {
        "softness": flat.get("softness"),
        "fragment_minsep_factor": flat.get("fragment_minsep_factor"),
    }
    physics = {k: v for k, v in physics.items() if v is not None}

    merge = {"p_merge": flat.get("p_merge")}
    merge = {k: v for k, v in merge.items() if v is not None}

    init = {
        "n_clusters": flat.get("n_init"),
        "size": 1,
        "phenotype": "proliferative",
    }
    init = {k: v for k, v in init.items() if v is not None}

    out = {}
    if ph:
        out.setdefault("phenotypes", {}).setdefault("proliferative", {}).update(ph)
    if physics:
        out.setdefault("physics", {}).update(physics)
    if merge:
        out.setdefault("merge", {}).update(merge)
    if init:
        out.setdefault("init", {}).update(init)
    return out


# ------------------------------
# MERGE WITH ABM DEFAULTS
# ------------------------------

def merge_baseline_and_override(baseline_flat: Dict[str, Any],
                                override_flat: Dict[str, Any]) -> Dict[str, Any]:
    nested = copy.deepcopy(DEFAULTS)

    flat = dict(baseline_flat or {})
    flat.update({k: v for k, v in (override_flat or {}).items() if v is not None})

    override_nested = flat_to_nested_params(flat)

    def deep_merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict):
                a.setdefault(k, {})
                deep_merge(a[k], v)
            else:
                a[k] = v
        return a

    return deep_merge(nested, override_nested)


# ------------------------------
# OBSERVED DATA LOADING
# ------------------------------

def load_observed_with_scaler(
    observed_csv: str,
    summary_stats: List[str],
    requested_timesteps: List[int],
) -> Tuple[np.ndarray, List[int], MaxAbsScaler]:

    obs = pd.read_csv(observed_csv).sort_values("timestep").reset_index(drop=True)

    data_ts = set(int(t) for t in obs["timestep"])
    req_ts = [int(t) for t in requested_timesteps]

    valid_ts = [t for t in req_ts if t in data_ts]
    missing = [t for t in req_ts if t not in data_ts]

    if missing:
        print(f"[sweeps] WARNING: dropping missing timesteps: {missing[:10]}"
              + (" ..." if len(missing) > 10 else ""))

    if not valid_ts:
        raise ValueError("No overlapping timesteps between observed and sweep request.")

    obs_sel = obs.set_index("timestep").loc[valid_ts].reset_index()

    # matrix version (T × 4 stats)
    obs_mat = obs_sel[summary_stats].to_numpy()

    scaler = MaxAbsScaler().fit(obs_mat)
    obs_scaled = scaler.transform(obs_mat).flatten()

    return obs_scaled, valid_ts, scaler


# ------------------------------
# DISTANCE
# ------------------------------

def distance_l2_scaled(sim_df: pd.DataFrame,
                       scaler: MaxAbsScaler,
                       observed_scaled_vec: np.ndarray,
                       summary_stats: List[str]) -> float:

    sim_mat = sim_df[summary_stats].to_numpy()
    sim_scaled = scaler.transform(sim_mat).flatten()

    return float(np.linalg.norm(sim_scaled - observed_scaled_vec))