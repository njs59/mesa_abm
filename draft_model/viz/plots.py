
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import copy

from clusters_abm.clusters_model import ClustersModel
from clusters_abm.utils import DEFAULTS, export_timeseries_state


if __name__ == '__main__':
    # --- Prepare model ---
    params = copy.deepcopy(DEFAULTS)
    params["space"]["torus"] = True  # keep toroidal wrapping
    model = ClustersModel(params=params, seed=123)

    steps = int(params["time"]["steps"])
    for _ in trange(steps, desc="Simulating", unit="step"):
        model.step()

    os.makedirs('results', exist_ok=True)

    # --- Spatial snapshot (final step) ---
    fig, ax = plt.subplots(figsize=(6, 6))
    W = params["space"]["width"]; H = params["space"]["height"]
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect('equal')
    ax.set_title('ABM Snapshot — volume-preserving merge')

    for a in model.agents:
        if not getattr(a, "alive", True) or getattr(a, "pos", None) is None:
            continue
        color = DEFAULTS["phenotypes"][a.phenotype]["color"]
        x, y = a.pos
        ax.add_patch(plt.Circle((x, y), a.radius, fc=color, ec='k', alpha=0.6))

    ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')
    fig.tight_layout(); fig.savefig('results/snapshot_final.png', dpi=150); plt.close(fig)

    # --- Final size histogram ---
    sizes = model.size_log[-1] if len(model.size_log) > 0 else []
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    if len(sizes) > 0:
        ax2.hist(sizes, bins=np.arange(1, int(max(sizes)) + 2), color='#444444')
    ax2.set_title('Final cluster size distribution')
    ax2.set_xlabel('size (cells)'); ax2.set_ylabel('count')
    fig2.tight_layout(); fig2.savefig('results/size_hist_final.png', dpi=150); plt.close(fig2)

    # --- Final speed-size scatter ---
    speeds = model.speed_log[-1] if len(model.speed_log) > 0 else []
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    if len(speeds) > 0 and len(sizes) > 0:
        ax3.scatter(sizes, speeds, s=6, alpha=0.5)
    ax3.set_title('Speed vs size (size-independent)')
    ax3.set_xlabel('size (cells)'); ax3.set_ylabel('speed (µm/min)')
    fig3.tight_layout(); fig3.savefig('results/speed_vs_size.png', dpi=150); plt.close(fig3)

    # --- NEW: Export tidy per-timestep state table ---
    # One row per agent per time: time_min, step, agent_id, x, y, radius, size, speed
    df = export_timeseries_state(
        model,
        out_csv="results/state_timeseries.csv",
        out_parquet=None  # set a path here if you also want Parquet (requires pyarrow)
    )
    
    print('Saved plots to results/')
