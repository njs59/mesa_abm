#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# Surface-to-surface NND helper
# --------------------------------------------------------------------
def nearest_neighbor_surface_median(pos: np.ndarray, radii: np.ndarray) -> float:
    if pos is None or len(pos) < 2:
        return float("nan")
    diff = pos[:, None, :] - pos[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    radii = np.asarray(radii, dtype=float).reshape(-1, 1)
    rsum = radii + radii.T
    s2s = dists - rsum
    np.fill_diagonal(s2s, np.inf)
    s2s = np.maximum(s2s, 0.0)  # clamp overlaps to 0
    nn = np.min(s2s, axis=1)
    return float(np.median(nn))

# --------------------------------------------------------------------
# Example simulate_timeseries driving your ClustersModel
# NOTE: This is a scaffold; keep your own implementation if you already
#       compute S0,S1,S2,g_r40,g_r80 consistently with your observed CSV.
# --------------------------------------------------------------------
def simulate_timeseries(model_factory, params, total_steps: int, sample_steps: tuple[int, ...]):
    """
    Returns a NumPy array of shape (T, K) in the fixed column order:
      ["S0","S1","S2","NND_med","g_r40","g_r80"]
    Replace g_r40/g_r80 with your own implementation if you use them.
    """
    m = model_factory(params)
    sample_steps = tuple(sorted(set(int(s) for s in sample_steps if 0 <= s <= total_steps)))
    out_rows = []

    # Assumes your model has: ._snapshot_alive() -> (ids, pos, radii, sizes, speeds)
    # and .step(), .time, .dt etc., consistent with your ClustersModel.
    next_idx = 0
    for step in range(total_steps + 1):
        if next_idx < len(sample_steps) and step == sample_steps[next_idx]:
            ids, pos, radii, sizes, speeds = m._snapshot_alive()
            S0 = float(len(ids))
            S1 = float(np.mean(sizes)) if S0 > 0 else float("nan")
            S2 = float(np.std(sizes)) if S0 > 0 else float("nan")
            NND_med = nearest_neighbor_surface_median(pos, radii)

            # Placeholders for g_r40/g_r80 unless you fill them
            g_r40 = float("nan")
            g_r80 = float("nan")

            out_rows.append([S0, S1, S2, NND_med, g_r40, g_r80])
            next_idx += 1

        if step < total_steps:
            m.step()

    return np.asarray(out_rows, dtype=float)