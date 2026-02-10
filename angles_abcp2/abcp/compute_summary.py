import numpy as np
from typing import Dict, Tuple

def _torus_pairwise_dists(pos: np.ndarray, width: float, height: float) -> np.ndarray:
    """Compute pairwise Euclidean distances on a rectangular torus (minimal image)."""
    if pos.size == 0:
        return np.empty((0, 0), dtype=float)
    dx = pos[:, None, 0] - pos[None, :, 0]
    dy = pos[:, None, 1] - pos[None, :, 1]
    dx = np.minimum(np.abs(dx), width - np.abs(dx))
    dy = np.minimum(np.abs(dy), height - np.abs(dy))
    return np.hypot(dx, dy)

def _plain_pairwise_dists(pos: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances WITHOUT periodic wrap-around."""
    if pos.size == 0:
        return np.empty((0, 0), dtype=float)
    diffs = pos[:, None, :] - pos[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2))

def compute_snapshot_summaries(model, *, stats_torus: bool = False) -> Dict[str, float]:
    """
    Return S0, S1, S2 + SSNND_med (disc approximation) for current model state.
    stats_torus=False -> distances DO NOT wrap
    stats_torus=True  -> distances use minimal-image torus
    """
    ids, pos, radii, sizes, speeds = model._log_alive_snapshot()
    n = len(sizes)
    if n == 0:
        return {"S0": 0.0, "S1": 0.0, "S2": 0.0, "SSNND_med": 0.0}
    S0 = float(n)
    S1 = float(np.mean(sizes))
    S2 = float(np.mean(sizes ** 2))
    if stats_torus:
        W = float(model.params["space"]["width"])
        H = float(model.params["space"]["height"])
        D = _torus_pairwise_dists(pos, W, H)
    else:
        D = _plain_pairwise_dists(pos)
    if D.size == 0:
        ssnnd_med = 0.0
    else:
        Rsum = radii[:, None] + radii[None, :]
        S = D - Rsum
        np.fill_diagonal(S, np.inf)
        S[S < 0] = 0.0
        ssnnd = S.min(axis=1)
        ssnnd_med = float(np.median(ssnnd))
    return {"S0": S0, "S1": S1, "S2": S2, "SSNND_med": ssnnd_med}

def simulate_timeseries(model_factory, params: dict, total_steps: int, sample_steps: Tuple[int, ...]) -> np.ndarray:
    """Simulate and return a (T, 4) matrix for [S0, S1, S2, SSNND_med]."""
    m = model_factory(params)
    K = 4
    T = len(sample_steps)
    out = np.zeros((T, K), dtype=float)
    current = 0
    steps_set = set(sample_steps)
    for step in range(total_steps + 1):
        if step in steps_set:
            s = compute_snapshot_summaries(m, stats_torus=False)
            out[current, :] = [s["S0"], s["S1"], s["S2"], s["SSNND_med"]]
            current += 1
            if current >= T:
                break
        if step < total_steps:
            m.step()
    return out
