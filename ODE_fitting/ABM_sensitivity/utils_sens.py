"""
Utilities for the ABM Sensitivity Pipeline (with 5 movement modes)
-------------------------------------------------------------------

Changes in this version:
- sweep expansion now supports:
    * range: {start, stop, step}
    * values: [ ... ]
    * BARE LIST shorthand:   key: [ ... ]   <-- NEW
- scenario naming still uses pretty float formatting (no FP artefacts)
"""

import os
import yaml
from copy import deepcopy
from datetime import datetime
from decimal import Decimal, getcontext

# Use ample precision for clean step arithmetic
getcontext().prec = 28


# -----------------------------------------------------------------------------
# YAML Loading
# -----------------------------------------------------------------------------
def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
# Directory Helpers
# -----------------------------------------------------------------------------
def make_timestamped_dir(root: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = os.path.join(root, ts)
    os.makedirs(out, exist_ok=True)
    return out


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


# -----------------------------------------------------------------------------
# Modes (the 5 movement presets)
# -----------------------------------------------------------------------------
def expand_modes(cfg: dict):
    modes = cfg.get("modes", None)
    if not modes:
        return [{
            "name": "vanilla",
            "movement_phase": 1,
            "analysis_start_step": 0,
            "init": None,
        }]
    out = []
    for m in modes:
        out.append({
            "name": str(m.get("name", "mode")),
            "movement_phase": int(m.get("movement_phase", 1)),
            "analysis_start_step": int(m.get("analysis_start_step", 0)),
            "init": m.get("init", None),
        })
    return out


# -----------------------------------------------------------------------------
# Parameter Sweep Tools (Decimal-safe)
# -----------------------------------------------------------------------------
def _expand_range_decimal(block):
    """
    Decimal-accurate range expansion:
        {'range': {'start': 0.5, 'stop': 2.5, 'step': 0.2}}
    Returns a list of clean floats (0.5, 0.7, ..., 2.5) without FP drift.
    """
    r = block["range"]
    start = Decimal(str(r["start"]))
    stop  = Decimal(str(r["stop"]))
    step  = Decimal(str(r["step"]))
    if step == 0:
        raise ValueError("range.step must be non-zero")

    n = int(((stop - start) / step).to_integral_value(rounding="ROUND_FLOOR")) + 1
    vals = [float(start + i * step) for i in range(n)]
    if vals and abs(vals[-1] - float(stop)) > 1e-12:
        vals[-1] = float(stop)
    return vals


def expand_sweep_dict(sweep_cfg: dict):
    """
    Build Cartesian product of sweep values supporting:
      - {key: {range: {...}}}
      - {key: {values: [...]}}
      - {key: [ ... ]}            <-- NEW (bare list shorthand)
    Returns: list[dict] of dotted-key overrides for each scenario.
    """
    if not sweep_cfg:
        return [{}]

    keys = list(sweep_cfg.keys())
    value_lists = []
    for key in keys:
        block = sweep_cfg[key]

        # Bare list or tuple shorthand
        if isinstance(block, (list, tuple)):
            vals = [float(Decimal(str(v))) for v in block]
            value_lists.append(vals)
            continue

        # Dict formats
        if not isinstance(block, dict):
            raise ValueError(f"Unsupported sweep block for {key}: {type(block)}")

        if "range" in block:
            value_lists.append(_expand_range_decimal(block))
        elif "values" in block:
            vals = [float(Decimal(str(v))) for v in block["values"]]
            value_lists.append(vals)
        else:
            raise ValueError(f"Unsupported sweep block for {key}; use 'range', 'values', or bare list.")

    # Cartesian product without numpy (avoid float coercion)
    combos = []

    def _cartesian(lists, idx=0, acc=None):
        acc = [] if acc is None else acc
        if idx == len(lists):
            yield acc
            return
        for v in lists[idx]:
            yield from _cartesian(lists, idx + 1, acc + [v])

    for combo in _cartesian(value_lists):
        combos.append({str(k): v for k, v in zip(keys, combo)})
    return combos


# -----------------------------------------------------------------------------
# Nested Dict Override System
# -----------------------------------------------------------------------------
def set_nested(d: dict, dotted_key: str, value):
    parts = dotted_key.split(".")
    x = d
    for p in parts[:-1]:
        if p not in x:
            x[p] = {}
        x = x[p]
    x[parts[-1]] = value


def apply_overrides(base: dict, overrides: dict):
    out = deepcopy(base)
    for k, v in overrides.items():
        set_nested(out, k, v)
    return out


# -----------------------------------------------------------------------------
# Scenario Naming Helpers (pretty, FP-safe)
# -----------------------------------------------------------------------------
def _format_number_nicely(v: float, sig: int = 6) -> str:
    if v == 0.0:
        return "0"
    a = abs(v)
    if 1e-3 <= a < 1e4:
        s = f"{v:.{sig}g}"
    else:
        s = f"{v:.{sig}e}"
    if "e" not in s and "." in s:
        s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s


def format_value_for_name(v):
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float) and float(v).is_integer():
        return str(int(v))
    if isinstance(v, float):
        s = _format_number_nicely(v, sig=6)
        if "e" in s or "E" in s:
            return s.replace("E", "e")
        return s.replace(".", "p")
    return str(v)


def make_scenario_name(index: int, mode_name: str, override_dict: dict):
    parts = [f"scenario_{index:02d}", f"mode_{mode_name}"]
    for key in sorted(override_dict.keys()):
        leaf = key.split(".")[-1]
        val = format_value_for_name(override_dict[key])
        parts.append(f"{leaf}_{val}")
    return "__".join(parts)


# -----------------------------------------------------------------------------
# Seed Utility
# -----------------------------------------------------------------------------
def derive_seed(base_seed: int, scenario_idx: int, repeat_idx: int):
    return base_seed + scenario_idx * 10000 + repeat_idx


# -----------------------------------------------------------------------------
# ABM Parameter Injection
# -----------------------------------------------------------------------------
def prepare_model_params(defaults: dict, overrides: dict, mode_init: dict | None):
    newp = deepcopy(defaults)
    if mode_init:
        newp["init"] = deepcopy(mode_init)
    for key, val in overrides.items():
        set_nested(newp, key, val)
    return newp