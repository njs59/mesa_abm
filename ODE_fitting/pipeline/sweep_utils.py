#!/usr/bin/env python3
"""
Utilities for expanding ABM parameter sweeps, naming scenarios,
loading configuration, slicing data, and writing summaries.
"""
from __future__ import annotations
import os
import csv
import itertools
from typing import Dict, Any, List, Iterable

import yaml
import pandas as pd


# -------------------- config & IO helpers --------------------

def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def slice_data(mean_csv: str, mode: str):
    """Replicates the slicing logic from the original pipeline."""
    df = pd.read_csv(mean_csv)
    if mode in [
        "data_t71_phase2",
        "singletons_phase2_fit71plus",
        "singletons_phase1_to_2_fit71plus",
    ]:
        df = df[df["step"] >= 71].reset_index(drop=True)
    return (
        df,
        df["step"].to_numpy(),
        df[["num_clusters", "mean_cluster_size", "mean_squared_cluster_size"]].to_numpy(),
    )


# -------------------- sweep helpers --------------------
def _flatten(d: Dict[str, Any] | None, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not d:
        return out
    for k, v in d.items():
        kk = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, kk))
        else:
            out[kk] = v
    return out


def write_abm_sweep_summary(run_dir: str, rows: List[Dict[str, Any]]):
    if not rows:
        return None
    allkeys = sorted({k for r in rows for k in r.keys()})
    path = os.path.join(run_dir, "abm_sweep_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allkeys)
        w.writeheader()
        w.writerows(rows)
    return path


def _cartesian(overrides_dict: Dict[str, Any]):
    if not overrides_dict:
        return [{}]
    keys = list(overrides_dict.keys())
    vals = [(v if isinstance(v, (list, tuple)) else [v]) for v in overrides_dict.values()]
    return [{k: v for k, v in zip(keys, combo)} for combo in itertools.product(*vals)]


def _format_val_for_name(v):
    return str(v).replace(".", "p") if isinstance(v, float) else str(v)


def make_scenario_name(idx: int, mode: str, overrides: Dict[str, Any]):
    parts = [f"abm_{idx:02d}_{mode}"]
    for k in sorted(overrides.keys()):
        leaf = k.split(".")[-1]
        parts.append(f"{leaf}_{_format_val_for_name(overrides[k])}")
    return "__".join(parts)


def expand_param_sweep(cfg: Dict[str, Any], base_scenarios: List[Dict[str, Any]]):
    sweep = cfg.get("abm_param_sweep")
    if not sweep:
        return base_scenarios
    expanded: List[Dict[str, Any]] = []
    for block in sweep:
        apply_modes = block.get("apply_to_modes", ["*"])
        overrides = block.get("overrides", {})
        combos = _cartesian(overrides)
        for base in base_scenarios:
            mode = base["mode"]
            applies = ("*" in apply_modes) or (mode in apply_modes)
            if not applies:
                expanded.append(base)
                continue
            for ov in combos:
                ap = dict(base.get("abm_params", {}))
                ap["overrides"] = ov
                expanded.append({"mode": mode, "abm_params": ap})

    # de-duplicate
    unique: List[Dict[str, Any]] = []
    seen = set()

    def _froz(d):
        if not d:
            return ()
        t = []
        for k, v in sorted(d.items()):
            if isinstance(v, dict):
                t.append((k, tuple(sorted(_flatten(v).items()))))
            else:
                t.append((k, v))
        return tuple(t)

    for sc in expanded:
        key = (sc["mode"], _froz(sc.get("abm_params")))
        if key not in seen:
            seen.add(key)
            unique.append(sc)
    return unique


def extract_overrides(scenario: Dict[str, Any]) -> Dict[str, Any]:
    return dict(scenario.get("abm_params", {}).get("overrides", {}))


def deduce_swept_param_keys(scenarios: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for sc in scenarios:
        keys.update(extract_overrides(sc).keys())
    return sorted(list(keys))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


__all__ = [
    "load_cfg",
    "slice_data",
    "write_abm_sweep_summary",
    "_flatten",
    "make_scenario_name",
    "expand_param_sweep",
    "extract_overrides",
    "deduce_swept_param_keys",
    "ensure_dir",
]