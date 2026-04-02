# Multi_phase_analysis/analysis_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
import copy
import csv
import numpy as np
import pandas as pd

# Honour user's convention: dt=1.0 corresponds to 30 minutes
MIN_PER_DT = 30.0

TRAJ_COLUMNS = [
    "condition","repeat","step","time_min",
    "agent_id","phenotype","movement_phase",
    "x","y","size","radius","speed"
]

# -----------------------------
# Deep merge DEFAULTS + overrides
# -----------------------------
def deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d

# -----------------------------
# Spatial helpers (torus)
# -----------------------------
def _torus_delta(dx: np.ndarray, L: float) -> np.ndarray:
    return (dx + L/2.0) % L - L/2.0

def _pairwise_distances_torus(pos: np.ndarray, W: float, H: float) -> np.ndarray:
    n = pos.shape[0]
    if n < 2:
        return np.zeros((0,), dtype=float)
    dx = _torus_delta(pos[:,None,0] - pos[None,:,0], W)
    dy = _torus_delta(pos[:,None,1] - pos[None,:,1], H)
    d2 = dx*dx + dy*dy
    iu = np.triu_indices(n, k=1)
    return np.sqrt(d2[iu])

def nearest_neighbour_distances(pos: np.ndarray, W: float, H: float) -> np.ndarray:
    n = pos.shape[0]
    if n < 2:
        return np.zeros((0,), dtype=float)
    dx = _torus_delta(pos[:,None,0] - pos[None,:,0], W)
    dy = _torus_delta(pos[:,None,1] - pos[None,:,1], H)
    d2 = dx*dx + dy*dy
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(np.min(d2, axis=1))

def clark_evans_R_z(pos: np.ndarray, W: float, H: float) -> Tuple[float,float]:
    n = pos.shape[0]
    if n < 2:
        return (float('nan'), float('nan'))
    area = W*H
    lam = n/area
    if lam <= 0:
        return (float('nan'), float('nan'))
    nn = nearest_neighbour_distances(pos, W, H)
    if nn.size == 0:
        return (float('nan'), float('nan'))
    r_obs = float(nn.mean())
    r_exp = 1.0/(2.0*np.sqrt(lam))
    se = 0.26136/np.sqrt(n*lam)
    if se <= 0:
        return (float('nan'), float('nan'))
    z = (r_obs - r_exp)/se
    R = r_obs/r_exp if r_exp>0 else float('nan')
    return (float(R), float(z))

# -----------------------------
# Snapshot writer
# -----------------------------
def write_snapshot_csv(writer, model, condition: str, rep: int, step: int):
    # Convert dt to minutes
    time_min = step * float(model.dt) * MIN_PER_DT
    for a in list(model.agent_set):
        if not getattr(a, 'alive', True):
            continue
        p = getattr(a, 'pos', None)
        if p is None:
            continue
        v = getattr(a, 'vel', np.zeros(2))
        speed = float(np.linalg.norm(v))
        writer.writerow(dict(
            condition=condition,
            repeat=rep,
            step=step,
            time_min=time_min,
            agent_id=int(a.unique_id),
            phenotype=str(a.phenotype),
            movement_phase=int(getattr(a,'movement_phase',0)),
            x=float(p[0]),
            y=float(p[1]),
            size=float(a.size),
            radius=float(a.radius),
            speed=speed,
        ))

# -----------------------------
# One simulation helper (returns per-time summaries)
# -----------------------------
def run_single_sim(
    *,
    condition: str,
    rep: int,
    seed: int,
    steps: int,
    base_params: dict,
    overrides: dict,
    phase2_only: bool,
    allow_cross: bool,
    n_init: int,
    init_size: int,
    traj_csv: Path,
):
    from abm.clusters_model import ClustersModel
    from abm.utils import DEFAULTS
    # Build params
    merged = copy.deepcopy(DEFAULTS)
    merged = deep_update(merged, copy.deepcopy(base_params))
    merged = deep_update(merged, copy.deepcopy(overrides))
    merged['time']['steps'] = int(steps)
    merged['interactions']['allow_cross_phase_interactions'] = bool(allow_cross)
    merged['init'] = dict(n_clusters=int(n_init), size=int(init_size), phenotype='invasive')

    model = ClustersModel(params=merged, seed=seed)

    # Phase-2 only override at t=0
    if phase2_only:
        for a in list(model.agent_set):
            if getattr(a, 'alive', True):
                a.movement_phase = 2
                a.phase_switch_time = np.inf

    # Geometry
    W = float(model.space.x_max)
    H = float(model.space.y_max)

    # Output trajectories
    traj_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_ts: List[dict] = []

    with traj_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=TRAJ_COLUMNS)
        w.writeheader()
        # t=0 snapshot
        write_snapshot_csv(w, model, condition, rep, 0)
        # initial summary
        alive = [a for a in model.agent_set if getattr(a,'alive',True) and getattr(a,'pos',None) is not None]
        if len(alive)==0:
            rows_ts.append(dict(condition=condition, repeat=rep, step=0, time_min=0.0,
                                n_clusters=0, mean_size=np.nan, var_size=np.nan,
                                median_nnd=np.nan, R=np.nan, z=np.nan))
        else:
            pos0 = np.array([[float(a.pos[0]), float(a.pos[1])] for a in alive], float)
            sizes0 = np.array([float(a.size) for a in alive], float)
            nn0 = nearest_neighbour_distances(pos0, W, H)
            R0, z0 = clark_evans_R_z(pos0, W, H)
            rows_ts.append(dict(condition=condition, repeat=rep, step=0, time_min=0.0,
                                n_clusters=len(alive), mean_size=float(sizes0.mean()),
                                var_size=float(sizes0.var(ddof=1)) if len(alive)>1 else 0.0,
                                median_nnd=float(np.median(nn0)) if nn0.size else np.nan,
                                R=R0, z=z0))
        # steps 1..T
        for t in range(1, steps+1):
            model.step()
            write_snapshot_csv(w, model, condition, rep, t)
            alive = [a for a in model.agent_set if getattr(a,'alive',True) and getattr(a,'pos',None) is not None]
            if len(alive)==0:
                rows_ts.append(dict(condition=condition, repeat=rep, step=t, time_min=t*float(model.dt)*MIN_PER_DT,
                                    n_clusters=0, mean_size=np.nan, var_size=np.nan,
                                    median_nnd=np.nan, R=np.nan, z=np.nan))
                continue
            pos = np.array([[float(a.pos[0]), float(a.pos[1])] for a in alive], float)
            sizes = np.array([float(a.size) for a in alive], float)
            nn = nearest_neighbour_distances(pos, W, H)
            R, z = clark_evans_R_z(pos, W, H)
            rows_ts.append(dict(condition=condition, repeat=rep, step=t, time_min=t*float(model.dt)*MIN_PER_DT,
                                n_clusters=len(alive), mean_size=float(sizes.mean()),
                                var_size=float(sizes.var(ddof=1)) if len(alive)>1 else 0.0,
                                median_nnd=float(np.median(nn)) if nn.size else np.nan,
                                R=R, z=z))

    # Final quick summary (for convenience)
    last = rows_ts[-1]
    summary = dict(condition=condition, repeat=rep,
                   n_clusters=last['n_clusters'], mean_size=last['mean_size'], var_size=last['var_size'])
    return rows_ts, summary