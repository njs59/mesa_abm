# abcp/compute_summary.py
import numpy as np
from typing import Dict, Tuple


def _plain_pairwise_dists(pos: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances WITHOUT periodic wrap-around."""
    if pos.size == 0:
        return np.empty((0, 0), dtype=float)
    diffs = pos[:, None, :] - pos[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2))


def compute_snapshot_summaries(model) -> Dict[str, float]:
    """
    Return S0, S1, S2 + spatial stats for the *current* model state.

    Distances do NOT wrap (non-torus statistics), matching the 'no-wrap' NND
    discussed previously.
    """
    # Correct, existing API from your model:
    # returns (ids, pos, radii, sizes, speeds)  # <- uses _snapshot_alive from ClustersModel  [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/run_abc_maxabs_sweep.py)
    ids, pos, radii, sizes, speeds = model._snapshot_alive()

    n = len(sizes)
    if n == 0:
        return {"S0": 0.0, "S1": 0.0, "S2": 0.0, "NND_med": 0.0, "g_r40": 0.0, "g_r80": 0.0}

    S0 = float(n)
    S1 = float(np.mean(sizes))
    S2 = float(np.mean(sizes ** 2))

    # Distances (no wrap)
    D = _plain_pairwise_dists(pos)
    if D.size == 0:
        nnd_med = 0.0
        g_r40 = 0.0
        g_r80 = 0.0
    else:
        # exclude self
        D_no_self = D + np.eye(n) * 1e9
        nnd = D_no_self.min(axis=1)
        nnd_med = float(np.median(nnd))

        # simple g(r) with same D kernel (optional in fitting)
        def g_of_r(r: float, dr: float = 2.0) -> float:
            if n <= 1:
                return 0.0
            ring = (D_no_self >= (r - dr)) & (D_no_self < (r + dr))
            counts = ring.sum()
            # area / density from model params
            W = float(model.params["space"]["width"])
            H = float(model.params["space"]["height"])
            area = W * H
            rho = n / area
            shell_area = 2.0 * np.pi * r * (2.0 * dr)
            denom = (n * rho * shell_area) + 1e-12
            return float(counts / denom)

        g_r40 = g_of_r(40.0)
        g_r80 = g_of_r(80.0)

    return {"S0": S0, "S1": S1, "S2": S2, "NND_med": nnd_med, "g_r40": g_r40, "g_r80": g_r80}


def simulate_timeseries(
    model_factory, params: dict, total_steps: int, sample_steps: Tuple[int, ...]
) -> np.ndarray:
    """
    Run once; record summaries only at sample_steps (e.g. observed timesteps).
    Returns array [T, 6] in order: S0, S1, S2, NND_med, g_r40, g_r80.
    """
    m = model_factory(params)
    K = 6
    T = len(sample_steps)
    out = np.zeros((T, K), dtype=float)
    current = 0
    steps_set = set(sample_steps)

    for step in range(total_steps + 1):
        if step in steps_set:
            s = compute_snapshot_summaries(m)
            out[current, :] = [s["S0"], s["S1"], s["S2"], s["NND_med"], s["g_r40"], s["g_r80"]]
            current += 1
            if current >= T:
                break
        if step < total_steps:
            m.step()

    return out