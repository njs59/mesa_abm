#!/usr/bin/env python3
from __future__ import annotations
from typing import Callable, Tuple
import numpy as np

# -------- geometry helpers (torus) --------
def _torus_dist(dx: np.ndarray, W: float) -> np.ndarray:
    dx = np.abs(dx)
    return np.minimum(dx, W - dx)

def _pair_distances_torus(pos: np.ndarray, W: float, H: float) -> np.ndarray:
    if len(pos) < 2:
        return np.zeros((0,), dtype=float)
    dx = _torus_dist(pos[:, None, 0] - pos[None, :, 0], W)
    dy = _torus_dist(pos[:, None, 1] - pos[None, :, 1], H)
    d = np.sqrt(dx*dx + dy*dy)
    iu = np.triu_indices(len(pos), k=1)
    return d[iu]

def _nearest_neighbour_torus(pos: np.ndarray, W: float, H: float) -> np.ndarray:
    n = len(pos)
    if n <= 1:
        return np.zeros((0,), dtype=float)
    dx = _torus_dist(pos[:, None, 0] - pos[None, :, 0], W)
    dy = _torus_dist(pos[:, None, 1] - pos[None, :, 1], H)
    d = np.sqrt(dx*dx + dy*dy)
    d[np.arange(n), np.arange(n)] = np.inf
    return d.min(axis=1)

def _g_of_r(pos: np.ndarray, W: float, H: float, r: float, dr: float = 5.0) -> float:
    n = len(pos)
    if n < 2:
        return 0.0
    area = W * H
    rho = n / area
    d = _pair_distances_torus(pos, W, H)
    r_lo, r_hi = max(0.0, r - dr/2.0), r + dr/2.0
    count = np.count_nonzero((d >= r_lo) & (d < r_hi))
    shell_area = 2.0 * np.pi * r * dr
    expected = rho * shell_area * n / 2.0
    if expected <= 0:
        return 0.0
    return float(count / expected)

def _summaries_from_state(positions: np.ndarray, sizes: np.ndarray, W: float, H: float) -> Tuple[float, ...]:
    # [S0,S1,S2,NND_med,g_r40,g_r80]
    s0 = int(np.count_nonzero(sizes == 1))
    s1 = int(np.count_nonzero(sizes == 2))
    s2 = int(np.count_nonzero(sizes >= 3))
    nnd = _nearest_neighbour_torus(positions, W, H)
    nnd_med = float(np.median(nnd)) if nnd.size else 0.0
    g_r40 = _g_of_r(positions, W, H, r=40.0, dr=5.0)
    g_r80 = _g_of_r(positions, W, H, r=80.0, dr=5.0)
    return (s0, s1, s2, nnd_med, g_r40, g_r80)

# -------- main simulator (FAST path: no per-step logs) --------
def simulate_timeseries(
    model_factory: Callable,
    *,
    params: dict,
    total_steps: int,
    sample_steps: Tuple[int, ...],
    init_phenotype: str = "invasive",
    force_phase2: bool = True,
    disable_logging: bool = True,   # NEW: avoid per-step logging for speed
) -> np.ndarray:
    """
    Simulate ABM and return a matrix (T, 6) of summaries at 'sample_steps':
      [S0, S1, S2, NND_med, g_r40, g_r80]
    Speed-ups:
      • Disable model logging (if supported) to avoid large per-step arrays.
      • Build positions/sizes directly from live agents only at sampling steps.
    """
    m = model_factory(params)

    # Try to disable the model's per-step logging (safe no-op if not present)
    if disable_logging and hasattr(m, "enable_logging"):
        try:
            m.enable_logging = False
        except Exception:
            pass

    # Seed singletons of chosen phenotype
    n0 = int(params.get('init', {}).get('n_clusters', 800))
    for _ in range(n0):
        a = m.spawn_cluster(size=1, phenotype=init_phenotype, phase_switch_time=0.0)
        if force_phase2:
            a.movement_phase = 2
            a.phase_switch_time = float('-inf')

    W = float(m.space.x_max)
    H = float(m.space.y_max)
    sample_set = set(int(s) for s in sample_steps)
    out = []
    # Step and sample on demand
    for step in range(1, int(total_steps) + 1):
        m.step()
        if step in sample_set:
            # Build positions/sizes directly from live agents (no log arrays)
            pos = []
            size = []
            # (assuming model has .agent_set like in your codebase)
            for ag in getattr(m, "agent_set", []):
                if getattr(ag, "alive", True) and getattr(ag, "pos", None) is not None:
                    pos.append(ag.pos)
                    size.append(float(getattr(ag, "size", 1.0)))
            P = np.asarray(pos, dtype=float) if pos else np.zeros((0, 2), dtype=float)
            S = np.asarray(size, dtype=float) if size else np.zeros((0,), dtype=float)
            out.append(_summaries_from_state(P, S, W, H))

    if not out:
        return np.zeros((0, 6), dtype=float)
    return np.asarray(out, dtype=float)