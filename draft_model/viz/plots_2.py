
# viz/plots.py
import os
import copy
import numpy as np
from tqdm import trange

# Use a non-interactive backend (safe on servers/CI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clusters_abm.clusters_model import ClustersModel
from clusters_abm.utils import DEFAULTS, export_timeseries_state


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def run_sim(params, steps: int, seed: int = 123) -> ClustersModel:
    """Run a single simulation and return the model with logs populated."""
    model = ClustersModel(params=params, seed=seed)
    for _ in trange(steps, desc="Simulating", unit="step"):
        model.step()
    return model


def plot_snapshot(model: ClustersModel, out_png: str = "results/snapshot_final.png"):
    """Plot the final spatial snapshot (true discs with current radii)."""
    _ensure_dir(out_png)
    fig, ax = plt.subplots(figsize=(6, 6))

    W = float(model.params["space"]["width"])
    H = float(model.params["space"]["height"])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_title("ABM Snapshot — volume-preserving merge")

    # Iterate AgentSet; draw circles from agent.pos (wrapped by ContinuousSpace)
    for a in list(model.agents):
        if not getattr(a, "alive", True):
            continue
        p = getattr(a, "pos", None)
        if p is None:
            continue
        color = model.params["phenotypes"][a.phenotype]["color"]
        x, y = float(p[0]), float(p[1])
        ax.add_patch(plt.Circle((x, y), float(a.radius), fc=color, ec="k", alpha=0.6))

    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_size_histogram(model: ClustersModel, out_png: str = "results/size_hist_final.png"):
    """Plot the final cluster size distribution. Falls back to live agents if logs are empty."""
    _ensure_dir(out_png)

    # Prefer the last logged sizes; fall back to live snapshot if needed
    sizes = model.size_log[-1] if len(model.size_log) > 0 else []
    if len(sizes) == 0:
        alive_now = [
            a for a in model.agents if getattr(a, "alive", True) and getattr(a, "pos", None) is not None
        ]
        sizes = np.array([getattr(a, "size", 0) for a in alive_now], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    if len(sizes) > 0:
        # Integer-aware bins: 1..max_size (inclusive right edge by +1)
        max_size = int(np.max(sizes))
        bins = np.arange(1, max_size + 2) if max_size >= 1 else np.arange(1, 3)
        ax.hist(sizes, bins=bins, color="#444444")
        ax.set_title("Final cluster size distribution")
        ax.set_xlabel("size (cells)")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.2)
    else:
        # Still produce a figure (empty) so calling code doesn’t fail
        ax.text(0.5, 0.5, "No clusters to display.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_speed_vs_size(model: ClustersModel, out_png: str = "results/speed_vs_size.png"):
    """Scatter of speed vs size at the final snapshot if both are available."""
    _ensure_dir(out_png)

    sizes = model.size_log[-1] if len(model.size_log) > 0 else []
    speeds = model.speed_log[-1] if len(model.speed_log) > 0 else []

    # Graceful fallback to live snapshot if logs are empty
    if len(sizes) == 0 or len(speeds) == 0:
        alive_now = [
            a for a in model.agents if getattr(a, "alive", True) and getattr(a, "pos", None) is not None
        ]
        sizes = np.array([getattr(a, "size", 0) for a in alive_now], dtype=float)
        speeds = np.array([float(np.linalg.norm(getattr(a, "vel", np.zeros(2)))) for a in alive_now], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    if len(sizes) > 0 and len(speeds) > 0:
        ax.scatter(sizes, speeds, s=8, alpha=0.6)
        ax.set_title("Speed vs size (final snapshot)")
        ax.set_xlabel("size (cells)")
        ax.set_ylabel("speed (µm/min)")
        ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, "No data to display.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    # --- Prepare model ---
    params = copy.deepcopy(DEFAULTS)
    params["space"]["torus"] = True  # periodic boundaries

    steps = int(params["time"]["steps"])
    model = run_sim(params=params, steps=steps, seed=123)

    # --- Plots ---
    os.makedirs("results", exist_ok=True)

    # 1) Spatial snapshot
    plot_snapshot(model, out_png="results/snapshot_final.png")

    # 2) Final size histogram (robust to empty logs)
    plot_size_histogram(model, out_png="results/size_hist_final.png")

    # 3) Final speed-size scatter (robust to empty logs)
    plot_speed_vs_size(model, out_png="results/speed_vs_size.png")

    # 4) Tidy per-timestep export (robust exporter in clusters_abm.utils)
    export_timeseries_state(
        model,
        out_csv="results/state_timeseries.csv",
        out_parquet=None,  # set a path if you want Parquet (requires        out_parquet=None,  # set a path if you want Parquet (requires pyarrow)
    )

    print("Saved plots to results/")


if __name__ == "__main__":
    main()