
# viz/plots_gif.py
import os
import copy
import numpy as np

import matplotlib
matplotlib.use("Agg")  # safe backend for non-GUI environments
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm  # progress bars

from clusters_abm.clusters_model import ClustersModel
from clusters_abm.utils import DEFAULTS


def draw_agents(ax, agents, params):
    """Draw all alive agents as circles."""
    W = params["space"]["width"]; H = params["space"]["height"]
    ax.clear()
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect('equal')
    ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')
    for a in list(agents):
        if not getattr(a, "alive", True) or getattr(a, "pos", None) is None:
            continue
        color = DEFAULTS["phenotypes"][a.phenotype]["color"]
        x, y = a.pos  # ContinuousSpace keeps wrapped positions in agent.pos
        ax.add_patch(plt.Circle((float(x), float(y)), float(a.radius), fc=color, ec='k', alpha=0.6))


def make_progress_cb(total_frames: int, desc: str = "Saving"):
    total_frames = int(max(1, total_frames))
    pbar = tqdm(total=total_frames, desc=desc, unit="frame")

    def _cb(current_frame: int, total: int):
        pbar.n = min(current_frame + 1, total_frames)
        pbar.refresh()
        if pbar.n >= pbar.total:
            pbar.close()
    return _cb


if __name__ == '__main__':
    params = copy.deepcopy(DEFAULTS)
    params["space"]["torus"] = True
    dt = params.get("time", {}).get("dt", 1)

    model = ClustersModel(params=params, seed=123)
    steps = int(params["time"]["steps"])

    os.makedirs('results', exist_ok=True)

    # --- GIF 1: Spatial clusters over time ---
    fig_gif, ax_gif = plt.subplots(figsize=(6, 6))

    def update_spatial(frame_idx):
        model.step()  # advance one step per frame
        draw_agents(ax_gif, model.agents, params)
        ax_gif.set_title(f'ABM — volume-preserving merge t={frame_idx * dt} min (step {frame_idx})')
        return ax_gif.patches

    anim_spatial = FuncAnimation(fig_gif, update_spatial, frames=steps, interval=50, blit=False)
    anim_spatial.save(
        'results/clusters_over_time.gif',
        writer=PillowWriter(fps=10),
        progress_callback=make_progress_cb(steps, desc="Saving spatial GIF")
    )
    plt.close(fig_gif)

    # Logs populated by the spatial GIF stepping
    sizes_over_time = list(model.size_log)
    speeds_over_time = list(model.speed_log)

    # --- GIF 2: Size histogram over time ---
    global_max_size = 0
    if len(sizes_over_time) > 0 and any(len(s) > 0 for s in sizes_over_time):
        global_max_size = int(max(max(s) if len(s) > 0 else 0 for s in sizes_over_time))
    bins = np.arange(1, global_max_size + 2) if global_max_size >= 1 else np.arange(1, 3)

    fig_hist, ax_hist = plt.subplots(figsize=(6, 4))

    def update_hist(frame_idx):
        ax_hist.clear()
        s = sizes_over_time[frame_idx] if frame_idx < len(sizes_over_time) else []
        if len(s) > 0:
            ax_hist.hist(s, bins=bins, color='#444444')
            ax_hist.set_title(f'Cluster size distribution — t={frame_idx * dt} min (step {frame_idx})')
            ax_hist.set_xlabel('size (cells)'); ax_hist.set_ylabel('count')
            ax_hist.grid(True, alpha=0.2)
        else:
            ax_hist.text(0.5, 0.5, "No clusters", ha="center", va="center", transform=ax_hist.transAxes)
            ax_hist.set_axis_off()
        return ax_hist.patches

    frames_hist = max(1, len(sizes_over_time))
    anim_hist = FuncAnimation(fig_hist, update_hist, frames=frames_hist, interval=50, blit=False)
    anim_hist.save(
        'results/size_hist_over_time.gif',
        writer=PillowWriter(fps=10),
        progress_callback=make_progress_cb(frames_hist, desc="Saving size-hist GIF")
    )
    plt.close(fig_hist)

    # --- GIF 3: Speed vs size over time ---
    # Use a common frame count to avoid mismatched indexing
    frames_scatter = max(1, min(len(sizes_over_time), len(speeds_over_time)))

    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))

    def update_scatter(frame_idx):
        ax_scatter.clear()

        # Safe indexing within the common frame range
        sizes = sizes_over_time[frame_idx] if frame_idx < len(sizes_over_time) else []
        speeds = speeds_over_time[frame_idx] if frame_idx < len(speeds_over_time) else []

        # Convert to numpy arrays and align lengths
        sizes = np.asarray(sizes, dtype=float)
        speeds = np.asarray(speeds, dtype=float)
        min_len = min(sizes.size, speeds.size)

        if min_len > 0:
            ax_scatter.scatter(sizes[:min_len], speeds[:min_len], s=6, alpha=0.5)
            ax_scatter.set_title(
                f'Speed vs size (size-independent) — t={frame_idx * dt} min (step {frame_idx})'
            )
            ax_scatter.set_xlabel('size (cells)'); ax_scatter.set_ylabel('speed (µm/min)')
            ax_scatter.grid(True, alpha=0.2)
        else:
            ax_scatter.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_scatter.transAxes)
            ax_scatter.set_axis_off()

        return ax_scatter.collections

    anim_scatter = FuncAnimation(fig_scatter, update_scatter, frames=frames_scatter, interval=50, blit=False)
    anim_scatter.save(
        'results/speed_vs_size_over_time.gif',
        writer=PillowWriter(fps=10),
        progress_callback=make_progress_cb(frames_scatter, desc="Saving speed-size GIF")
    )
    plt.close(fig_scatter)

    # --- Static totals (area & volume curves) ---
    cell_volume = params["physics"]["cell_volume"]
    t = np.arange(len(sizes_over_time), dtype=float) * dt

    total_area = []
    total_volume_count = []
    total_volume_geom = []

    # r(s) = ((3/(4π)) * s * V_cell)^(1/3)
    const = (3.0 / (4.0 * np.pi)) * cell_volume
    for s in sizes_over_time:
        s = np.asarray(s, dtype=float)
        if s.size == 0:
            total_area.append(0.0)
            total_volume_count.append(0.0)
            total_volume_geom.append(0.0)
            continue
        r = (const * s) ** (1.0 / 3.0)
        areas = np.pi * r**2                    # µm²
        vols_geom = (4.0 / 3.0) * np.pi * r**3  # µm³

        total_area.append(float(np.sum(areas)))
        total_volume_count.append(float(cell_volume * np.sum(s)))
        total_volume_geom.append(float(np.sum(vols_geom)))

    total_area = np.asarray(total_area, dtype=float)
    total_volume_count = np.asarray(total_volume_count, dtype=float)
    total_volume_geom = np.asarray(total_volume_geom, dtype=float)

    # Plot 1: area (left) + count-based volume (right)
    fig_av, ax_a = plt.subplots(figsize=(7, 4))
    ax_a.plot(t, total_area, color='tab:blue', lw=2, label='Total area (µm²)')
    ax_a.set_xlabel('time (min)')
    ax_a.set_ylabel('total area (µm²)', color='tab:blue')
    ax_a.tick_params(axis='y', labelcolor='tab:blue')
    ax_a.grid(True, alpha=0.2)
    ax_v = ax_a.twinx()
    ax_v.plot(t, total_volume_count, color='tab:red', lw=2, label='Total volume (counts × V_cell)')
    ax_v.set_ylabel('total volume (µm³)', color='tab:red')
    ax_v.tick_params(axis='y', labelcolor='tab:red')
    lines = ax_a.get_lines() + ax_v.get_lines()
    labels = [l.get_label() for l in lines]
    ax_a.legend(lines, labels, loc='upper left', frameon=False)
    fig_av.tight_layout()
    fig_av.savefig('results/total_area_volume_over_time.png', dpi=150)
    plt.close(fig_av)

    # Plot 2: count-based vs geometric volume
    fig_vv, ax_vv = plt.subplots(figsize=(7, 4))
    ax_vv.plot(t, total_volume_count, color='tab:purple', lw=2, label='Volume (counts × V_cell)')
    ax_vv.plot(t, total_volume_geom, color='tab:orange', lw=2, linestyle='--',
               label='Volume (from areas → (4/3)πr³)')
    ax_vv.set_xlabel('time (min)')
    ax_vv.set_ylabel('total volume (µm³)')
    ax_vv.grid(True, alpha=0.2)
    ax_vv.legend(loc='upper left', frameon=False)
    fig_vv.tight_layout()
    fig_vv.savefig('results/total_volume_comparison_over_time.png', dpi=150)
    plt.close(fig_vv)

    # Save numeric data
    out = np.column_stack([t, total_area, total_volume_count, total_volume_geom])
    np.savetxt(
        'results/total_area_and_volumes_over_time.csv',
        out,
        delimiter=',',
        header='time_min,total_area_um2,volume_counts_um3,volume_from_area_um3',
        comments=''
       )

    print('Saved plots and GIFs')