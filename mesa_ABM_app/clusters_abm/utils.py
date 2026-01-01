
import os
from typing import Optional

import numpy as np
import pandas as pd

DEFAULTS = {
    "space": {"width": 1344.0, "height": 1025.0, "torus": True},
    "time": {"dt": 1.0, "steps": 300},
    "physics": {"cell_volume": 1954.0, "density": 1.0},
    "phenotypes": {
        "proliferative": {
            "speed_base": 1.0,
            "speed_size_exp": 0.0,
            "prolif_rate": 0.005,
            "adhesion": 0.6,
            "fragment_rate": 0.0005,
            "frag_size_exp": 0.0,
            "color": (186/256, 29/256, 186/256),
        },
        "invasive": {
            "speed_base": 3.0,
            "speed_size_exp": 0.0,
            "prolif_rate": 0.002,
            "adhesion": 0.9,
            "fragment_rate": 0.0,
            "frag_size_exp": 0.0,
            "color": (70/256, 158/256, 44/256),
        },
    },
    "merge": {"prob_contact_merge": 0.9},

    # NEW: initial condition controls
    "init": {
        "n_clusters": 1000,         # number of agents to spawn at t=0
        "size": 1,                  # each agent starts as a singleton
        "phenotype": "proliferative"  # one of: "proliferative" | "invasive"
    },
}

def radius_from_size_3d(n_cells: int, cell_volume: float = DEFAULTS["physics"]["cell_volume"]) -> float:
    return float(((3.0 / (4.0 * np.pi)) * n_cells * cell_volume) ** (1.0 / 3.0))

def volume_conserving_radius(r1: float, r2: float) -> float:
    return float((r1**3 + r2**3) ** (1.0 / 3.0))

def mass_from_size(
    n_cells: int,
    cell_volume: float = DEFAULTS["physics"]["cell_volume"],
    density: float = DEFAULTS["physics"]["density"],
) -> float:
    return float(density * n_cells * cell_volume)

def momentum_merge(m1: float, v1: np.ndarray, m2: float, v2: np.ndarray) -> np.ndarray:
    return (m1 * v1 + m2 * v2) / (m1 + m2)

def export_timeseries_state(
    model,
    out_csv: str = "results/state_timeseries.csv",
    out_parquet: Optional[str] = None,
) -> pd.DataFrame:
    """
    Export a tidy per-agent, per-timestep table:
        time_min, step, agent_id, x, y, radius, size, speed

    Writes CSV once at the end. If `out_parquet` is provided and pyarrow is installed,
    writes Parquet too (best-effort).
    """
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    rows = []

    dt = float(getattr(model, "dt", 1.0))
    T = len(getattr(model, "size_log", []))

    # sanity: logs must exist
    for attr in ["id_log", "pos_log", "radius_log", "size_log", "speed_log"]:
        if not hasattr(model, attr):
            raise AttributeError(f"Model is missing required log: {attr}")

    for t_idx in range(T):
        ids = np.asarray(model.id_log[t_idx], dtype=int)
        pos = np.asarray(model.pos_log[t_idx], dtype=float)
        radii = np.asarray(model.radius_log[t_idx], dtype=float)
        sizes = np.asarray(model.size_log[t_idx], dtype=float)

        # Guard: if speed_log length lags (e.g., missing t=0 prior to fix), fill zeros
        if t_idx < len(model.speed_log):
            speeds = np.asarray(model.speed_log[t_idx], dtype=float)
        else:
            speeds = np.zeros_like(sizes, dtype=float)

        n_rows = len(ids)
        if not (pos.shape[0] == n_rows == radii.shape[0] == sizes.shape[0] == speeds.shape[0]):
            # Skip malformed tick
            continue

        time_min = t_idx * dt
        for i in range(n_rows):
            rows.append({
                "time_min": float(time_min),
                "step": int(t_idx),
                "agent_id": int(ids[i]),
                "x": float(pos[i, 0]),
                "y": float(pos[i, 1]),
                "radius": float(radii[i]),
                "size": float(sizes[i]),
                "speed": float(speeds[i]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    if out_parquet:
        try:
            os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
            df.to_parquet(out_parquet, index=False, engine="pyarrow")
        except Exception as e:
            print(f"Parquet export skipped: {e}")

    return df
