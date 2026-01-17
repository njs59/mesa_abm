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


# --- add this next to _torus_pairwise_dists ---
def _plain_pairwise_dists(pos: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances WITHOUT periodic wrap-around."""
    if pos.size == 0:
        return np.empty((0, 0), dtype=float)
    diffs = pos[:, None, :] - pos[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2))


def compute_snapshot_summaries(model, *, stats_torus: bool = False) -> Dict[str, float]:
    """
    Return S0,S1,S2 + spatial stats for current model state.

    stats_torus=False  -> distances DO NOT wrap (your requested behaviour)
    stats_torus=True   -> distances use minimal-image torus (old behaviour)
    """
    ids, pos, radii, sizes, speeds = model._log_alive_snapshot()
    n = len(sizes)
    if n == 0:
        return {"S0": 0.0, "S1": 0.0, "S2": 0.0, "NND_med": 0.0, "g_r40": 0.0, "g_r80": 0.0}

    S0 = float(n)
    S1 = float(np.mean(sizes))
    S2 = float(np.mean(sizes**2))

    # --- choose distance metric for summaries ---
    if stats_torus:
        W = float(model.params["space"]["width"])
        H = float(model.params["space"]["height"])
        D = _torus_pairwise_dists(pos, W, H)
    else:
        D = _plain_pairwise_dists(pos)

    if D.size == 0:
        nnd_med = 0.0; g_r40 = 0.0; g_r80 = 0.0
    else:
        # NND (exclude self)
        D_no_self = D + np.eye(n) * 1e9
        nnd = D_no_self.min(axis=1)
        nnd_med = float(np.median(nnd))

        # If you are still computing g(r), use the SAME D so it matches NND semantics
        def g_of_r(r: float, dr: float = 2.0) -> float:
            if n <= 1:
                return 0.0
            ring = (D_no_self >= (r - dr)) & (D_no_self < (r + dr))
            counts = ring.sum()
            area = float(model.params["space"]["width"]) * float(model.params["space"]["height"])
            rho = n / area
            shell_area = 2.0 * np.pi * r * (2.0 * dr)
            denom = (n * rho * shell_area) + 1e-12
            return float(counts / denom)

        g_r40 = g_of_r(40.0)
        g_r80 = g_of_r(80.0)

    return {"S0": S0, "S1": S1, "S2": S2, "NND_med": nnd_med, "g_r40": g_r40, "g_r80": g_r80}

def simulate_timeseries(model_factory, params: dict, total_steps: int, sample_steps: Tuple[int, ...]) -> np.ndarray:
    m = model_factory(params)
    K = 6
    T = len(sample_steps)
    out = np.zeros((T, K), dtype=float)
    current = 0
    steps_set = set(sample_steps)
    for step in range(total_steps + 1):
        if step in steps_set:
            s = compute_snapshot_summaries(m, stats_torus=False)  # <-- no wrap
            out[current, :] = [s["S0"], s["S1"], s["S2"], s["NND_med"], s["g_r40"], s["g_r80"]]
            current += 1
            if current >= T:
                break
        if step < total_steps:
            m.step()
    return out
