import os
from typing import Optional
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Global parameters (updated for two-phase movement + shifted Gompertz)
# -------------------------------------------------------------------
DEFAULTS = {
    "space": {"width": 1344.0, "height": 1025.0, "torus": True},
    "time": {"dt": 1.0, "steps": 300},

    "physics": {
        "cell_volume": 1954.0,
        "density": 1.0,
        "soft_separate": True,
        "softness": 0.05,
        "fragment_minsep_factor": 1.1,
    },

    "phenotypes": {
        "proliferative": {
            "speed_base": 1.0,
            "speed_size_exp": 0.0,
            "prolif_rate": 0.005,
            "fragment_rate": 0.0005,
            "frag_size_exp": 0.0,
            "color": (186/256, 29/256, 186/256),
        },
        "invasive": {
            "speed_base": 3.0,
            "speed_size_exp": 0.0,
            "prolif_rate": 0.002,
            "fragment_rate": 0.0,
            "frag_size_exp": 0.0,
            "color": (70/256, 158/256, 44/256),
        }
    },

    # Two‑phase movement configuration with best-fit parameters
    "movement_v2": {
        "phase1": {
            "speed_dist": {
                "name": "lognorm",
                "params": {"s": 0.9702953903470796, "scale": 4.517258693059166},
            },
            "turning": {"mu": float(np.pi), "kappa": 0.24187571808790503},
        },
        "phase2": {
            "speed_dist": {
                "name": "gamma",
                "params": {"a": 2.08195513401392, "scale": 3.548763926965852},
            },
            "turning": {"mu": 0.0, "kappa": 0.14698710452005212},
        },
        # Shifted Gompertz transition CDF parameters (tabulated at model init)
        "transition": {
            "p_max": 0.999999999964594,
            "shift": 13.182589764562103,
            "b": 0.027843583207496258,
            "c": 0.03085478819397927,
            "t_max": 400.0,
            "n_points": 3000,
        },
    },

    "interactions": {"allow_cross_phase_interactions": True},

    "merge": {"p_merge": 0.9},

    "init": {"n_clusters": 800, "size": 1, "phenotype": "proliferative"},
}

# -------------------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------------------
def radius_from_size_3d(n_cells: int, *, cell_volume: float = DEFAULTS["physics"]["cell_volume"]) -> float:
    return float(((3.0/(4.0*np.pi))*n_cells*cell_volume)**(1/3))

def volume_conserving_radius(r1: float, r2: float) -> float:
    return float((r1**3 + r2**3)**(1/3))

def mass_from_size(n_cells: int,
                   *,
                   cell_volume: float = DEFAULTS["physics"]["cell_volume"],
                   density: float = DEFAULTS["physics"]["density"]) -> float:
    return float(density*n_cells*cell_volume)

def momentum_merge(m1, v1, m2, v2):
    return (m1*v1 + m2*v2) / (m1 + m2)

# -------------------------------------------------------------------
# Export helper (tidy per-agent, per-step timeseries)
# -------------------------------------------------------------------
def export_timeseries_state(model,
                            out_csv: str = "results/state_timeseries.csv",
                            out_parquet: Optional[str] = None) -> pd.DataFrame:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    rows = []
    dt = float(getattr(model, "dt", 1.0))
    T = min(len(model.id_log), len(model.pos_log), len(model.radius_log), len(model.size_log), len(model.speed_log))
    for t_idx in range(T):
        ids = np.asarray(model.id_log[t_idx], dtype=int)
        pos = np.asarray(model.pos_log[t_idx], dtype=float)
        radii = np.asarray(model.radius_log[t_idx], dtype=float)
        sizes = np.asarray(model.size_log[t_idx], dtype=float)
        speeds = np.asarray(model.speed_log[t_idx], dtype=float)
        n_rows = min(len(ids), pos.shape[0], radii.shape[0], sizes.shape[0], speeds.shape[0])
        if n_rows == 0:
            continue
        ids, pos, radii, sizes, speeds = ids[:n_rows], pos[:n_rows], radii[:n_rows], sizes[:n_rows], speeds[:n_rows]
        time_min = t_idx * dt
        for i in range(n_rows):
            rows.append({
                "time_min": float(time_min),
                "step": int(t_idx),
                "agent_id": int(ids[i]),
                "x": float(pos[i,0]),
                "y": float(pos[i,1]),
                "radius": float(radii[i]),
                "size": float(sizes[i]),
                "speed": float(speeds[i]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    if out_parquet:
        try:
            os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
            df.to_parquet(out_parquet, index=False)
        except Exception as e:
            print(f"Parquet export skipped: {e}")
    return df
