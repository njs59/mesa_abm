from __future__ import annotations

def particle_to_params(particle: dict, *, motion: str = 'isotropic', speed_dist: str = 'gamma') -> dict:
    """
    Map a pyABC particle dict into an ABM params dict for **invasive Phase-2**.

    Parameters supported (presence in priors is optional unless you intend to infer them):
      - a_gamma         : shape for Phase-2 invasive gamma speed
      - scale_gamma     : (optional) scale for Phase-2 invasive gamma speed; if absent, uses a fixed default
      - kappa_turn      : von Mises κ for Phase-2 invasive turning
      - p_merge         : model.merge.p_merge
      - prolif_rate     : phenotypes.invasive.prolif_rate (per step per cell)
      - fragment_rate   : phenotypes.invasive.fragment_rate (per step)
      - init_n_clusters : initial number of invasive singletons (rounded to >=1)

    Notes
    -----
    • Phase 2 is *forced* at t=0 during simulation (see abcp.compute_summary.simulate_timeseries),
      so the movement transition distribution here is inert.
    • The ABM uses movement_v2; speed_base etc. are kept for completeness but not used.
    """
    # Read particle values with sensible fallbacks
    a_gamma = float(particle.get('a_gamma', 2.0))
    # Allow optional gamma scale; keep a stable default if not provided
    scale_gamma = float(particle.get('scale_gamma', 3.55))
    kappa = float(particle.get('kappa_turn', 0.15))
    p_merge = float(particle.get('p_merge', 0.56))
    prolif = float(particle.get('prolif_rate', 0.011))
    frag   = float(particle.get('fragment_rate', 0.0008))
    n0_raw = float(particle.get('init_n_clusters', 800.0))
    n0     = int(max(1, round(n0_raw)))

    params = {
        "movement_v2": {
            "invasive": {
                # Phase 1 is irrelevant since we FORCE Phase 2 at t=0
                "phase1": {
                    "speed_dist": {"name": "lognorm", "params": {"s": 1.0, "scale": 4.5}},
                    "turning": {"mu": 3.141592653589793, "kappa": 0.24},
                },
                "phase2": {
                    "speed_dist": {"name": "gamma", "params": {"a": a_gamma, "scale": scale_gamma}},
                    "turning": {"mu": 0.0, "kappa": kappa},
                },
                "transition": {
                    # Not used because we force Phase 2 from t=0
                    "p_max": 0.9999, "shift": 0.0, "b": 0.03, "c": 0.03,
                    "t_max": 400.0, "n_points": 3000,
                },
            }
        },
        "merge": {"p_merge": p_merge},
        "init": {"n_clusters": n0, "size": 1, "phenotype": "invasive"},
        "interactions": {"allow_cross_phase_interactions": True},
        "phenotypes": {
            "invasive": {
                # Biology parameters inferred via ABC
                "prolif_rate": prolif,
                "fragment_rate": frag,
                "frag_size_exp": 0.0,
                "color": (70/256, 158/256, 44/256),
                # Not used by movement_v2 but kept for completeness
                "speed_base": 1.0,
                "speed_size_exp": 0.0,
            }
        },
    }
    return params