
import os
from typing import Optional

import numpy as np
import pandas as pd

# -----------------
# Global parameters
# -----------------
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
            "color": (186 / 256, 29 / 256, 186 / 256),
        },
        "invasive": {
            "speed_base": 3.0,
            "speed_size_exp": 0.0,
            "prolif_rate": 0.002,
            "adhesion": 0.9,
            "fragment_rate": 0.0,
            "frag_size_exp": 0.0,
            "color": (70 / 256, 158 / 256, 44 / 256),
        },
    },

    "movement": {
        "mode": "constant",  # "constant" or "distribution"
        "direction": "isotropic",  # "isotropic" or "persistent"
        # Only used if mode == "distribution":
        "distribution": "lognorm",  # "lognorm"|"gamma"|"weibull"|"rayleigh"|"expon"|"invgauss"
        "dist_params": {
            # For lognorm: s (shape, >0) and scale (>0); loc is fixed to 0
            "s": 0.6,
            "scale": 2.0
        },
        # Small angular noise when direction="persistent" (radians, std dev for normal)
        "heading_sigma": 0.25
    },
    "merge": {"prob_contact_merge": 0.9},
    # Initial condition controls
    "init": {
        "n_clusters": 1000,  # agents at t=0
        "size": 1,           # each singleton
        "phenotype": "proliferative",  # "proliferative" or "invasive"
    },
}


# -----------------
# Geometry helpers
# -----------------
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


# -----------------
# Export helpers
# -----------------
def export_timeseries_state(
    model,
    out_csv: str = "results/state_timeseries.csv",
    out_parquet: Optional[str] = None,
) -> pd.DataFrame:
    """
    Export a tidy per-agent, per-timestep table:
      time_min, step, agent_id, x, y, radius, size, speed

    This version is robust to minor per-tick log misalignments:
      * Iterates only up to the SHORTEST log length across id/pos/radius/size/speed.
      * Coerces the arrays at each tick to a COMMON length (min across those arrays)
        rather than dropping the entire tick. This prevents "missing steps" artifacts.
    """
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    rows = []

    # Interpret model.dt = 1.0 as 30 min in your workflow; we keep it unitless here.
    dt = float(getattr(model, "dt", 1.0))

    # --- iterate only up to the shortest log list length
    T = min(
        len(getattr(model, "id_log", [])),
        len(getattr(model, "pos_log", [])),
        len(getattr(model, "radius_log", [])),
        len(getattr(model, "size_log", [])),
        len(getattr(model, "speed_log", [])),
    )

    for t_idx in range(T):
        ids = np.asarray(model.id_log[t_idx], dtype=int)
        pos = np.asarray(model.pos_log[t_idx], dtype=float)
        radii = np.asarray(model.radius_log[t_idx], dtype=float)
        sizes = np.asarray(model.size_log[t_idx], dtype=float)

        # Guard: if speed_log length lags for any reason, fill zeros
        if t_idx < len(model.speed_log):
            speeds = np.asarray(model.speed_log[t_idx], dtype=float)
        else:
            speeds = np.zeros_like(sizes, dtype=float)

        # --- coerce to a common length instead of skipping the entire tick
        n_rows = min(len(ids), pos.shape[0], radii.shape[0], sizes.shape[0], speeds.shape[0])
        if n_rows == 0:
            continue

        ids = ids[:n_rows]
        pos = pos[:n_rows, :]
        radii = radii[:n_rows]
        sizes = sizes[:n_rows]
        speeds = speeds[:n_rows]

        time_min = t_idx * dt
        for i in range(n_rows):
            rows.append(
                {
                    "time_min": float(time_min),
                    "step": int(t_idx),
                    "agent_id": int(ids[i]),
                    "x": float(pos[i, 0]),
                    "y": float(pos[i, 1]),
                    "radius": float(radii[i]),
                    "size": float(sizes[i]),
                    "speed": float(speeds[i]),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    if out_parquet:
        try:
            os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
            df.to_parquet(out_parquet, index=False, engine="pyarrow")
        except Exception as e:
            print(f"Parquet export skipped: {e}")

    return df
