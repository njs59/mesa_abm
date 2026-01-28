import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ------------------------------------------------------------
# Fix imports to work with abm/ directory
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from abm.clusters_model import ClustersModel
from abm.utils import DEFAULTS


# ============================================================
# Compute median NND (Euclidean or Toroidal)
# ============================================================

def compute_median_nnd(positions, width, height, torus_visibility):
    """
    Compute median nearest neighbour distance.
    """
    if len(positions) < 2:
        return np.nan

    if torus_visibility:
        # 3×3 domain tiling for periodic KD-tree
        tiles = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tiles.append(positions + np.array([dx * width, dy * height]))
        tiled = np.vstack(tiles)

        tree = cKDTree(tiled)
        dists, _ = tree.query(positions, k=2)
        nnd = dists[:, 1]
    else:
        # Pure Euclidean
        tree = cKDTree(positions)
        dists, _ = tree.query(positions, k=2)
        nnd = dists[:, 1]

    return np.median(nnd)


# ============================================================
# Ghost expansion
# ============================================================

def append_ghost_centres(positions, radii, width, height):
    """
    Append ghost centres for edge-overlapping clusters.
    """
    pos = positions.copy()
    ghost_points = []

    for (x, y), r in zip(positions, radii):

        if x < r:                  # left edge
            ghost_points.append([x + width, y])
        if (width - x) < r:        # right edge
            ghost_points.append([x - width, y])
        if y < r:                  # bottom edge
            ghost_points.append([x, y + height])
        if (height - y) < r:       # top edge
            ghost_points.append([x, y - height])

    if ghost_points:
        pos = np.vstack([pos, np.array(ghost_points)])

    return pos


# ============================================================
# Run ONE simulation and return 4 NND time-series
# ============================================================

def run_single_sim(params, steps=300):

    model = ClustersModel(params=params)
    width  = params["space"]["width"]
    height = params["space"]["height"]

    # Four definitions of NND:
    nnd_euclid_orig = []
    nnd_torus_orig  = []
    nnd_euclid_aug  = []
    nnd_torus_aug   = []

    for t in range(steps):
        model.step()

        positions = model.pos_log[-1]
        radii     = model.radius_log[-1]

        # Case 1 — Euclidean original
        n1 = compute_median_nnd(positions, width, height, torus_visibility=False)
        nnd_euclid_orig.append(n1)

        # Case 2 — Toroidal original
        n2 = compute_median_nnd(positions, width, height, torus_visibility=True)
        nnd_torus_orig.append(n2)

        # Augment
        aug_positions = append_ghost_centres(positions, radii, width, height)

        # Case 3 — Euclidean augmented
        n3 = compute_median_nnd(aug_positions, width, height, torus_visibility=False)
        nnd_euclid_aug.append(n3)

        # Case 4 — Toroidal augmented
        n4 = compute_median_nnd(aug_positions, width, height, torus_visibility=True)
        nnd_torus_aug.append(n4)

    return (
        np.array(nnd_euclid_orig),
        np.array(nnd_torus_orig),
        np.array(nnd_euclid_aug),
        np.array(nnd_torus_aug),
    )


# ============================================================
# Multiple simulations + mean + CI
# ============================================================

def run_many(params, steps=300, reps=500):

    # Storage arrays: reps × steps
    eu = np.zeros((reps, steps))
    to = np.zeros((reps, steps))
    euA = np.zeros((reps, steps))
    toA = np.zeros((reps, steps))

    for r in range(reps):
        print(f"Running simulation {r+1}/{reps} ...")
        a, b, c, d = run_single_sim(params, steps=steps)

        eu[r, :] = a
        to[r, :] = b
        euA[r, :] = c
        toA[r, :] = d

    return eu, to, euA, toA


# ============================================================
# Plot with CI
# ============================================================

def plot_mean_and_ci(x, data, label, color):
    """Plot mean + 95% CI shaded region."""
    mean = data.mean(axis=0)
    low  = np.percentile(data, 2.5, axis=0)
    high = np.percentile(data, 97.5, axis=0)

    plt.plot(x, mean, lw=2, color=color, label=label)
    plt.fill_between(x, low, high, color=color, alpha=0.25)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    params = DEFAULTS
    STEPS = 300
    REPS = 100

    # -------------------------------
    # Run 500 simulations
    # -------------------------------
    eu, to, euA, toA = run_many(params, steps=STEPS, reps=REPS)

    # x-axis
    t = np.arange(STEPS)

    # -------------------------------
    # Plot FIRST simulation
    # -------------------------------
    plt.figure(figsize=(10,6))
    plt.title("NND Time Series — First Simulation Only")
    plt.plot(t, eu[0], label="Euclidean (orig)")
    plt.plot(t, to[0], label="Toroidal (orig)")
    plt.plot(t, euA[0], label="Euclidean (aug)")
    plt.plot(t, toA[0], label="Toroidal (aug)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # Plot mean + CI for 500 runs
    # -------------------------------
    plt.figure(figsize=(10,6))
    plt.title("NND Means + 95% Confidence Intervals (500 runs)")

    plot_mean_and_ci(t, eu,  "Euclidean (orig)",    "blue")
    plot_mean_and_ci(t, to,  "Toroidal (orig)",     "red")
    plot_mean_and_ci(t, euA, "Euclidean (aug)",     "green")
    plot_mean_and_ci(t, toA, "Toroidal (aug)",      "purple")

    plt.xlabel("Timestep")
    plt.ylabel("Median NND")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("All simulations complete.")