import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ------------------------------------------------------------
# Allow imports from abm/
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from abm.clusters_model import ClustersModel
from abm.utils import DEFAULTS


# ============================================================
# Compute per-cluster NND values (one NND per cluster)
# ============================================================

def compute_full_nnd_list(positions, width, height, torus):
    """
    Return an array of nearest-neighbour distances, one per cluster (agent),
    for the current timestep.
    """
    N = len(positions)
    if N < 2:
        return np.array([])

    if torus:
        # 3×3 tiling to implement periodic distances
        tiles = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tiles.append(positions + np.array([dx * width, dy * height]))
        tiled_positions = np.vstack(tiles)

        tree = cKDTree(tiled_positions)
        dists, _ = tree.query(positions, k=2)
        return dists[:, 1]
    else:
        tree = cKDTree(positions)
        dists, _ = tree.query(positions, k=2)
        return dists[:, 1]


# ============================================================
# Append ghost centres for edge-overlapping clusters
# ============================================================

def append_ghost_centres(positions, radii, width, height):
    """
    If a cluster’s centre is closer to an edge than its radius, add a ghost
    centre on the opposite edge (without removing the original).
    """
    pos = positions.copy()
    ghosts = []

    for (x, y), r in zip(positions, radii):
        if x < r:                 # left overlap
            ghosts.append([x + width, y])
        if (width - x) < r:       # right overlap
            ghosts.append([x - width, y])
        if y < r:                 # bottom overlap
            ghosts.append([x, y + height])
        if (height - y) < r:      # top overlap
            ghosts.append([x, y - height])

    if ghosts:
        pos = np.vstack([pos, np.array(ghosts)])
    return pos


# ============================================================
# Run one simulation and collect 10/50/90 percentiles per method
# ============================================================

def run_single_sim_with_percentiles(params, steps=300):
    model = ClustersModel(params=params)
    width  = params["space"]["width"]
    height = params["space"]["height"]

    # For each method, store 10th / 50th / 90th percentiles per timestep
    eu10 = []; eu50 = []; eu90 = []
    to10 = []; to50 = []; to90 = []
    ea10 = []; ea50 = []; ea90 = []
    ta10 = []; ta50 = []; ta90 = []

    for _ in range(steps):
        model.step()

        positions = model.pos_log[-1]   # (N, 2)
        radii     = model.radius_log[-1]

        # ---- Original centres
        eu_list = compute_full_nnd_list(positions, width, height, torus=False)
        to_list = compute_full_nnd_list(positions, width, height, torus=True)

        # ---- Ghost-augmented centres
        aug_positions = append_ghost_centres(positions, radii, width, height)
        euA_list = compute_full_nnd_list(aug_positions, width, height, torus=False)
        toA_list = compute_full_nnd_list(aug_positions, width, height, torus=True)

        # Percentiles (handle empty cases robustly)
        if len(eu_list) > 0:
            eu10.append(np.percentile(eu_list, 10))
            eu50.append(np.percentile(eu_list, 50))
            eu90.append(np.percentile(eu_list, 90))
        else:
            eu10.append(np.nan); eu50.append(np.nan); eu90.append(np.nan)

        if len(to_list) > 0:
            to10.append(np.percentile(to_list, 10))
            to50.append(np.percentile(to_list, 50))
            to90.append(np.percentile(to_list, 90))
        else:
            to10.append(np.nan); to50.append(np.nan); to90.append(np.nan)

        if len(euA_list) > 0:
            ea10.append(np.percentile(euA_list, 10))
            ea50.append(np.percentile(euA_list, 50))
            ea90.append(np.percentile(euA_list, 90))
        else:
            ea10.append(np.nan); ea50.append(np.nan); ea90.append(np.nan)

        if len(toA_list) > 0:
            ta10.append(np.percentile(toA_list, 10))
            ta50.append(np.percentile(toA_list, 50))
            ta90.append(np.percentile(toA_list, 90))
        else:
            ta10.append(np.nan); ta50.append(np.nan); ta90.append(np.nan)

    return {
        "eu":  (np.array(eu10),  np.array(eu50),  np.array(eu90)),
        "to":  (np.array(to10),  np.array(to50),  np.array(to90)),
        "euA": (np.array(ea10),  np.array(ea50),  np.array(ea90)),
        "toA": (np.array(ta10),  np.array(ta50),  np.array(ta90)),
        "steps": steps,
    }


# ============================================================
# Plot helpers
# ============================================================

def plot_median_only(t, res, colours):
    """Figure A: median (50th) only for four methods."""
    plt.figure(figsize=(10, 6))
    plt.title("Median NND (Single Simulation)")

    plt.plot(t, res["eu"][1],  color=colours["eu"],  lw=2, label="Euclidean (orig)")
    plt.plot(t, res["to"][1],  color=colours["to"],  lw=2, label="Toroidal (orig)")
    plt.plot(t, res["euA"][1], color=colours["euA"], lw=2, label="Euclidean (aug)")
    plt.plot(t, res["toA"][1], color=colours["toA"], lw=2, label="Toroidal (aug)")

    plt.xlabel("Timestep")
    plt.ylabel("NND (median)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_p10_med_p90(t, res, colours):
    """Figure B: 10th (dashed), 50th (solid), 90th (dotted) for four methods."""
    plt.figure(figsize=(12, 7))
    plt.title("NND Percentiles (10th / 50th / 90th) — Single Simulation")

    def triplet(key, label):
        p10, p50, p90 = res[key]
        c = colours[key]
        plt.plot(t, p10, ls="--", color=c, alpha=0.9)
        plt.plot(t, p50, ls="-",  color=c, lw=2, label=label)
        plt.plot(t, p90, ls=":",  color=c, alpha=0.9)

    triplet("eu",  "Euclidean (orig)")
    triplet("to",  "Toroidal (orig)")
    triplet("euA", "Euclidean (aug)")
    triplet("toA", "Toroidal (aug)")

    plt.xlabel("Timestep")
    plt.ylabel("NND")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    params = DEFAULTS
    STEPS = 300

    print("Running single simulation and computing per-timestep percentiles...")
    res = run_single_sim_with_percentiles(params, steps=STEPS)

    t = np.arange(res["steps"])
    colours = {
        "eu":  "tab:blue",
        "to":  "tab:red",
        "euA": "tab:green",
        "toA": "tab:purple",
    }

    # Figure A — median only
    plot_median_only(t, res, colours)

    # Figure B — 10th, median, 90th
    plot_p10_med_p90(t, res, colours)

    print("Done.")