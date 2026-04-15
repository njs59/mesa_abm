#!/usr/bin/env python3
"""pipeline.abm_runner

This module provides the ABM forward-simulation entry point used by pipeline.py:

    run_forward_sims(abm_cfg: dict, out_dir: str) -> str

It replaces older code that imported from a top-level `scripts` package.
Instead, it directly runs the Mesa ABM (abm/ClustersModel) and writes the
mean summary CSV that pipeline.py expects.

Expected output columns (consumed by pipeline.py):
    - step
    - num_clusters
    - mean_cluster_size
    - mean_squared_cluster_size

Notes
-----
* Supports the pipeline's scenario overrides via dotted keys (e.g.
  phenotypes.proliferative.prolif_rate).
* Implements the 'singletons_phase2_fit_all' scenario by forcing agents
  into movement phase 2 at t=0.

"""

from __future__ import annotations

import os
import json
import yaml
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from multiprocessing import Pool

# ABM imports (these are sibling packages to `pipeline/` in your repo)
from abm.clusters_model import ClustersModel
from abm.utils import DEFAULTS


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _deepcopy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """A safe-ish deep copy for nested dicts/lists without pulling in copy.deepcopy."""
    if isinstance(d, dict):
        return {k: _deepcopy_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_deepcopy_dict(v) for v in d]
    return d


def _merge_dicts(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `patch` into `base` (returns new dict)."""
    out = _deepcopy_dict(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = _deepcopy_dict(v) if isinstance(v, (dict, list)) else v
    return out


def _set_by_dotted_path(d: Dict[str, Any], dotted: str, value: Any) -> None:
    """Set d['a']['b']['c']=value for dotted='a.b.c'. Creates dicts as needed."""
    parts = dotted.split('.')
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _apply_overrides(params: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = _deepcopy_dict(params)
    for k, v in (overrides or {}).items():
        _set_by_dotted_path(out, k, v)
    return out


def _resolve_path(maybe_path: str, base_dir: str) -> str:
    if not maybe_path:
        return maybe_path
    if os.path.isabs(maybe_path):
        return maybe_path
    return os.path.abspath(os.path.join(base_dir, maybe_path))


# -----------------------------------------------------------------------------
# Core simulation
# -----------------------------------------------------------------------------

@dataclass
class _WorkerArgs:
    params: Dict[str, Any]
    steps: int
    seed: int
    init_clusters: List[Dict[str, Any]]
    force_phase2: bool


def _run_one(worker: _WorkerArgs) -> np.ndarray:
    """Run one replicate and return an array (steps+1, 3)."""
    model = ClustersModel(params=worker.params, seed=worker.seed, init_clusters=worker.init_clusters)

    # Force movement phase 2 if requested (used for singletons_phase2_fit_all)
    if worker.force_phase2:
        for a in list(model.agent_set):
            if getattr(a, 'alive', True):
                a.movement_phase = 2
                # also make the switch time immediate (defensive)
                try:
                    a.phase_switch_time = 0.0
                except Exception:
                    pass

    # The model logs initial state already
    for _ in range(worker.steps):
        model.step()

    # Summaries per logged step
    out = np.zeros((worker.steps + 1, 3), dtype=float)
    for t in range(worker.steps + 1):
        sizes = np.asarray(model.size_log[t], dtype=float)
        if sizes.size == 0:
            out[t, :] = 0.0
        else:
            out[t, 0] = float(sizes.size)
            out[t, 1] = float(sizes.mean())
            out[t, 2] = float((sizes * sizes).mean())
    return out


def run_forward_sims(abm_cfg: Dict[str, Any], out_dir: str) -> str:
    """Run forward ABM simulations and write the mean summary CSV.

    Parameters
    ----------
    abm_cfg:
        Dict assembled by pipeline.py, typically:
            {**cfg['abm'], 'mode': <mode>, 'abm_params': {...}}
    out_dir:
        Output directory (created if needed)

    Returns
    -------
    str
        Path to the mean summary CSV (the one consumed by pipeline.slice_data)
    """

    os.makedirs(out_dir, exist_ok=True)

    # Determine where this module lives (used to resolve relative paths)
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # Load defaults: prefer YAML if provided, else fallback to abm.utils.DEFAULTS
    defaults_yaml = abm_cfg.get('defaults_yaml', None)
    defaults_path = _resolve_path(defaults_yaml, this_dir) if defaults_yaml else None

    if defaults_path and os.path.exists(defaults_path):
        with open(defaults_path, 'r') as f:
            defaults_params = yaml.safe_load(f) or {}
    else:
        defaults_params = _deepcopy_dict(DEFAULTS)

    # Scenario mode and per-scenario params
    mode = str(abm_cfg.get('mode', '')).strip()
    abm_params = abm_cfg.get('abm_params', {}) or {}
    overrides = abm_params.get('overrides', {}) or {}

    params = _apply_overrides(defaults_params, overrides)

    # Establish run settings
    n_runs = int(abm_cfg.get('n_runs', 1))
    n_workers = int(abm_cfg.get('n_workers', 1))
    base_seed = int(abm_cfg.get('seed', 42))

    # Simulation steps (prefer YAML defaults if present)
    steps = int(params.get('time', {}).get('steps', 145))

    # Build initial clusters list
    init_cfg = params.get('init', {}) or {}

    # Pipeline-specific singleton count knob
    if 'initial_singleton_count' in abm_params and abm_params['initial_singleton_count'] is not None:
        n0 = int(abm_params['initial_singleton_count'])
    else:
        n0 = int(init_cfg.get('n_clusters', 1000))

    size0 = int(init_cfg.get('size', 1))
    phenotype0 = str(init_cfg.get('phenotype', 'proliferative'))

    init_clusters = [{'size': size0, 'phenotype': phenotype0} for _ in range(n0)]

    # Mode logic
    force_phase2 = (mode == 'singletons_phase2_fit_all')

    # Snapshot inputs actually used (helpful for provenance)
    with open(os.path.join(out_dir, 'abm_input.yaml'), 'w') as f:
        yaml.safe_dump({
            'mode': mode,
            'n_runs': n_runs,
            'n_workers': n_workers,
            'seed': base_seed,
            'steps': steps,
            'abm_params': abm_params,
            'overrides': overrides,
            'resolved_defaults_yaml': defaults_path if defaults_path else None,
            'final_params': params,
        }, f, sort_keys=False)

    # Run replicates
    worker_args: List[_WorkerArgs] = []
    for r in range(n_runs):
        worker_args.append(_WorkerArgs(
            params=params,
            steps=steps,
            seed=base_seed + r,
            init_clusters=init_clusters,
            force_phase2=force_phase2,
        ))

    if n_workers <= 1:
        sims = [_run_one(a) for a in worker_args]
    else:
        with Pool(processes=n_workers) as pool:
            sims = pool.map(_run_one, worker_args)

    sims = np.stack(sims, axis=0)  # (n_runs, steps+1, 3)

    mean = sims.mean(axis=0)

    df = pd.DataFrame({
        'step': np.arange(steps + 1, dtype=int),
        'num_clusters': mean[:, 0],
        'mean_cluster_size': mean[:, 1],
        'mean_squared_cluster_size': mean[:, 2],
    })

    out_csv = os.path.join(out_dir, 'forward_means_stats.csv')
    df.to_csv(out_csv, index=False)

    return out_csv
