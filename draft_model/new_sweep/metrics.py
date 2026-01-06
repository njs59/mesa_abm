
# new_sweep/metrics.py
import numpy as np
from typing import Dict, Tuple

def summary_time_series(model) -> Dict[str, np.ndarray]:
    dt = float(getattr(model, 'dt', 1.0))
    T = len(getattr(model, 'size_log', []))
    S0 = np.zeros(T, dtype=float)
    S1 = np.zeros(T, dtype=float)
    S2 = np.zeros(T, dtype=float)
    total_cells = np.zeros(T, dtype=float)
    mean_speed = np.zeros(T, dtype=float)
    time_min = np.arange(T, dtype=float) * dt
    for t in range(T):
        sizes = np.asarray(model.size_log[t], dtype=float) if t < len(model.size_log) else np.array([])
        speeds = np.asarray(model.speed_log[t], dtype=float) if t < len(model.speed_log) else np.array([])
        n = sizes.size
        S0[t] = float(n)
        if n > 0:
            S1[t] = float(np.mean(sizes))
            S2[t] = float(np.mean(sizes ** 2))
            total_cells[t] = float(np.sum(sizes))
        else:
            S1[t] = 0.0
            S2[t] = 0.0
            total_cells[t] = 0.0
        mean_speed[t] = float(np.mean(speeds)) if speeds.size > 0 else 0.0
    return {
        'S0': S0,
        'S1': S1,
        'S2': S2,
        'total_cells': total_cells,
        'mean_speed': mean_speed,
        'time_min': time_min,
    }


def final_histogram(sizes: np.ndarray, bins: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    if sizes.size == 0:
        return np.zeros(bins, dtype=float), np.linspace(0, 1, bins + 1)
    hist, edges = np.histogram(sizes, bins=bins)
    return hist.astype(float), edges


def speed_size_slope(model) -> float:
    xs, ys = [], []
    for t in range(len(model.size_log)):
        sizes = np.asarray(model.size_log[t], dtype=float)
        speeds = np.asarray(model.speed_log[t], dtype=float)
        if sizes.size and speeds.size and sizes.size == speeds.size:
            xs.append(sizes)
            ys.append(speeds)
    if not xs:
        return 0.0
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if x.size < 2:
        return 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)
