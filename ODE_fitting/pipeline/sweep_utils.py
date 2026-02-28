#!/usr/bin/env python3
"""
Utilities for expanding ABM parameter sweeps, naming scenarios,
loading configuration, slicing data, and writing summaries.

Adds auto-generated range support for override values:
- {range:    {start: s, stop: t, step: h}}
- {linspace: {start: s, stop: t, num: n}}
- {logspace: {start: a, stop: b, num: n, base: 10}}   # exponents; value = base ** exp
- "RANGE(s, t, h)"
- "LINSPACE(s, t, n)"
- "LOGSPACE(a, b, n, base=10)"
- "start:step:stop"  (inclusive)
"""

from __future__ import annotations
import os
import csv
import itertools
import math
import re
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


# -------- Range expansion logic (new) --------

_NUM_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

def _frange(start: float, stop: float, step: float) -> List[float]:
    """
    Inclusive floating-step range with robust rounding (mitigates float drift).
    Assumes step != 0 and that |(stop - start)/step| is reasonable.
    """
    if step == 0:
        raise ValueError("range step cannot be 0")
    # compute number of steps (floor) and build inclusive sequence
    n_steps = int(math.floor((stop - start) / step + 1e-12))
    values = [start + i * step for i in range(n_steps + 1)]
    # include stop if still meaningfully beyond last value due to rounding
    if (step > 0 and values[-1] < stop - 1e-12) or (step < 0 and values[-1] > stop + 1e-12):
        values.append(stop)
    # round to collapse things like 0.30000000004 -> 0.3
    return [float(f"{x:.12g}") for x in values]


def _parse_range_string(spec: str):
    """
    Parse string-based compact specs and return a list of floats if matched.
    Supported:
      - "start:step:stop"              (inclusive)
      - "RANGE(s, t, h)"
      - "LINSPACE(s, t, n)"
      - "LOGSPACE(a, b, n, base=10)"   (a..b are exponents)
    """
    s = spec.strip()
    s_upper = s.upper()

    # Colon stride: "start:step:stop"
    m = re.fullmatch(rf"\s*({_NUM_RE})\s*:\s*({_NUM_RE})\s*:\s*({_NUM_RE})\s*", s)
    if m:
        a, h, b = map(float, m.groups())
        return _frange(a, b, h)

    # RANGE(a,b,h)
    m = re.fullmatch(rf"\s*RANGE\(\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*\)\s*", s_upper)
    if m:
        a, b, h = map(float, m.groups())
        return _frange(a, b, h)

    # LINSPACE(a,b,n)
    m = re.fullmatch(rf"\s*LINSPACE\(\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*,\s*(\d+)\s*\)\s*", s_upper)
    if m:
        a, b, n = m.groups()
        a, b, n = float(a), float(b), int(n)
        if n <= 1:
            return [float(f"{a:.12g}")]
        vals = [a + (b - a) * i / (n - 1) for i in range(n)]
        return [float(f"{v:.12g}") for v in vals]

    # LOGSPACE(a,b,n[,base=...])  where a,b are exponents
    m = re.fullmatch(
        rf"\s*LOGSPACE\(\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*,\s*(\d+)\s*(?:,\s*BASE\s*=\s*({_NUM_RE})\s*)?\)\s*",
        s_upper
    )
    if m:
        a, b, n, base = m.groups()
        a, b, n = float(a), float(b), int(n)
        base = float(base) if base is not None else 10.0
        if n <= 1:
            vals = [base ** a]
        else:
            exps = [a + (b - a) * i / (n - 1) for i in range(n)]
            vals = [base ** e for e in exps]
        return [float(f"{v:.12g}") for v in vals]  # noqa: E999 (editor hint)

    return None  # not a recognized string spec


def _expand_value_spec(v):
    """
    Expand a single override value 'v' into a list if it encodes a range spec.
    Otherwise:
      - if v is already a list/tuple -> return list(v)
      - else -> return [v]

    Supported encodings:
      - dict: {'range': {...}} / {'linspace': {...}} / {'logspace': {...}}
      - string: 'RANGE(...)' / 'LINSPACE(...)' / 'LOGSPACE(...)' / 'a:h:b'
    """
    # Already a sequence
    if isinstance(v, (list, tuple)):
        return list(v)

    # Dict-based specs
    if isinstance(v, dict):
        if "range" in v:
            d = v["range"] or {}
            a = float(d["start"])
            b = float(d["stop"])
            h = float(d["step"])
            return _frange(a, b, h)

        if "linspace" in v:
            d = v["linspace"] or {}
            a = float(d["start"])
            b = float(d["stop"])
            n = int(d["num"])
            if n <= 1:
                return [float(f"{a:.12g}")]
            vals = [a + (b - a) * i / (n - 1) for i in range(n)]
            return [float(f"{val:.12g}") for val in vals]

        if "logspace" in v:
            d = v["logspace"] or {}
            a = float(d["start"])
            b = float(d["stop"])
            n = int(d["num"])
            base = float(d.get("base", 10.0))
            if n <= 1:
                vals = [base ** a]
            else:
                exps = [a + (b - a) * i / (n - 1) for i in range(n)]
                vals = [base ** e for e in exps]
            return [float(f"{val:.12g}") for val in vals]

    # String-based specs
    if isinstance(v, str):
        out = _parse_range_string(v)
        if out is not None:
            return out

    # Fallback: scalar -> wrap in list
    return [v]


def _normalize_overrides(overrides: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    For each override key, ensure the value is a LIST.
    Expands any range encodings to a list of floats.
    """
    norm: Dict[str, List[Any]] = {}
    for k, v in (overrides or {}).items():
        vals = _expand_value_spec(v)
        norm[k] = vals
    return norm


def _cartesian(overrides_dict: Dict[str, Any]):
    """
    Cartesian product over normalized overrides (each value is a list).
    """
    if not overrides_dict:
        return [{}]
    norm = _normalize_overrides(overrides_dict)
    keys = list(norm.keys())
    vals = [norm[k] for k in keys]
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
        combos = _cartesian(overrides)  # <— uses range-aware expansion
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