
#!/usr/bin/env python3
"""
Forward-simulation comparison for cross-phase interactions:
    allow_cross_phase_interactions = True  vs  False

Now runs **all forward simulations in parallel** across both conditions.

Features:
  • Runs N replicates for ON and OFF (default N=100 each) — scheduled together in one Pool
  • Uses scripts_defaults.yaml for baseline parameters
  • Uses the SAME RNG seeds for ON and OFF (rep 0..N-1), only the boolean differs
  • Prints number of workers at start
  • Rich progress bars:
        - Total tasks (2N)
        - Per-condition bars (ON and OFF)
  • Outputs:
      - Summary-stat comparison plots (medians + 5–95% bands)
      - GIF #1: side-by-side histograms (ON vs OFF)
      - GIF #2: overlay histograms (both on one axis)
      - GIF #3: difference histograms (ON – OFF counts per bin)
"""
from __future__ import annotations
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

# Project imports
from abm.clusters_model import ClustersModel
from abm.utils import DEFAULTS, export_timeseries_state
from abcp.compute_summary import simulate_timeseries

ALL_STATS = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def deep_merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict):
            a.setdefault(k, {})
            deep_merge(a[k], v)
        else:
            a[k] = v
    return a

def load_baseline_params(default_yaml_path: str) -> tuple[dict, dict]:
    with open(default_yaml_path, "r") as f:
        d = yaml.safe_load(f)
    flat = d.get("params", {})
    params = deep_merge(dict(DEFAULTS), {})
    # phenotype
    ph = {}
    if "prolif_rate" in flat: ph["prolif_rate"] = float(flat["prolif_rate"]) 
    if "fragment_rate" in flat: ph["fragment_rate"] = float(flat["fragment_rate"]) 
    if ph:
        params.setdefault("phenotypes", {}).setdefault("proliferative", {}).update(ph)
    # physics
    phys = {}
    if "softness" in flat: phys["softness"] = float(flat["softness"]) 
    if "fragment_minsep_factor" in flat: phys["fragment_minsep_factor"] = float(flat["fragment_minsep_factor"]) 
    if phys:
        params.setdefault("physics", {}).update(phys)
    # merge
    if "p_merge" in flat:
        params.setdefault("merge", {})["p_merge"] = float(flat["p_merge"])
    # init
    init = {}
    if "n_init" in flat: init["n_clusters"] = int(flat["n_init"]) 
    if init:
        init.setdefault("size", 1)
        init.setdefault("phenotype", "proliferative")
        params.setdefault("init", {}).update(init)
    return params, d

# ---------------------------------------------------------
# Core single-run logic
# ---------------------------------------------------------
def simulate_one(rep: int, params: dict, total_steps: int, seed_base: int):
    """Run one replicate and return (summary_timeseries, sizes_by_t).

    summary_timeseries: array (T, K)
    sizes_by_t: list length T of 1D arrays of sizes (aggregated within one run)
    """
    seed = int(seed_base) + int(rep)

    def factory(p):
        return ClustersModel(params=p, seed=seed)

    sample_steps = tuple(range(total_steps + 1))
    sim_full = simulate_timeseries(factory, params=params,
                                   total_steps=total_steps,
                                   sample_steps=sample_steps)
    # Capture cluster sizes for GIF
    model = factory(params)
    for _ in range(total_steps):
        model.step()
    df = export_timeseries_state(model)
    sizes_by_t = []
    for t in range(total_steps + 1):
        df_t = df[df["step"] == t]
        sizes_by_t.append(df_t["size"].to_numpy())
    return sim_full, sizes_by_t

# ---------------------------------------------------------
# Parallel worker wrapper: includes condition flag
# ---------------------------------------------------------
def _worker(args):
    label, flag, rep, params_base, total_steps, seed_base = args
    # Build params per task (set the interaction flag)
    params = deep_merge(dict(params_base), {})
    params.setdefault("interactions", {})["allow_cross_phase_interactions"] = bool(flag)
    sim, sizes = simulate_one(rep, params, total_steps, seed_base)
    return label, rep, sim, sizes

# ---------------------------------------------------------
# GIF makers
# ---------------------------------------------------------
def _compute_bins(sizes_on: list[np.ndarray], sizes_off: list[np.ndarray], max_bins: int = 60):
    # determine global max size to set common bins
    max_size = 1
    for arr in sizes_on + sizes_off:
        if arr.size:
            max_size = max(max_size, int(arr.max()))
    # cap number of bins to keep things visible
    bins = np.linspace(1, max_size + 1, num=min(max_bins, max_size) + 1)
    return bins

def gif_side_by_side(outpath: Path, sizes_on_t: list[np.ndarray], sizes_off_t: list[np.ndarray], total_steps: int):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bins = _compute_bins(sizes_on_t, sizes_off_t)

    def animate(t):
        axes[0].clear(); axes[1].clear()
        axes[0].hist(sizes_on_t[t], bins=bins, color="royalblue", alpha=0.8)
        axes[1].hist(sizes_off_t[t], bins=bins, color="crimson", alpha=0.8)
        axes[0].set_title(f"Cross-phase ON — t={t}")
        axes[1].set_title(f"Cross-phase OFF — t={t}")
        for ax in axes:
            ax.set_xlabel("Cluster size"); ax.set_ylabel("Count"); ax.set_ylim(bottom=0)
        fig.suptitle("Cluster size distributions (side-by-side)")

    anim = FuncAnimation(fig, animate, frames=total_steps + 1, interval=150)
    anim.save(outpath, writer=PillowWriter(fps=6))
    plt.close(fig)

def gif_overlay(outpath: Path, sizes_on_t: list[np.ndarray], sizes_off_t: list[np.ndarray], total_steps: int):
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    bins = _compute_bins(sizes_on_t, sizes_off_t)

    def animate(t):
        ax.clear()
        ax.hist(sizes_on_t[t], bins=bins, color="royalblue", alpha=0.5, label="ON")
        ax.hist(sizes_off_t[t], bins=bins, color="crimson", alpha=0.5, label="OFF")
        ax.set_title(f"Overlay: ON vs OFF — t={t}")
        ax.set_xlabel("Cluster size"); ax.set_ylabel("Count"); ax.legend(); ax.set_ylim(bottom=0)

    anim = FuncAnimation(fig, animate, frames=total_steps + 1, interval=150)
    anim.save(outpath, writer=PillowWriter(fps=6))
    plt.close(fig)

def gif_difference(outpath: Path, sizes_on_t: list[np.ndarray], sizes_off_t: list[np.ndarray], total_steps: int):
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4))
    bins = _compute_bins(sizes_on_t, sizes_off_t)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    def animate(t):
        ax.clear()
        counts_on, _ = np.histogram(sizes_on_t[t], bins=bins)
        counts_off, _ = np.histogram(sizes_off_t[t], bins=bins)
        diff = counts_on - counts_off
        ax.bar(bin_centers, diff, width=np.diff(bins), color="darkslateblue", alpha=0.9)
        ax.axhline(0, color="black", lw=1)
        ax.set_title(f"Difference (ON - OFF) — t={t}")
        ax.set_xlabel("Cluster size"); ax.set_ylabel("Count difference")

    anim = FuncAnimation(fig, animate, frames=total_steps + 1, interval=150)
    anim.save(outpath, writer=PillowWriter(fps=6))
    plt.close(fig)

# ---------------------------------------------------------
# Plot summary stats
# ---------------------------------------------------------
def plot_summaries(outdir: Path, tvec: np.ndarray, summary_stats: list[str], sims_on: np.ndarray, sims_off: np.ndarray):
    for k, sname in enumerate(summary_stats):
        plt.figure(figsize=(10, 5))
        for label, sims, color in [("ON", sims_on, "royalblue"), ("OFF", sims_off, "crimson")]:
            arr = sims[:, :, k]
            med = np.median(arr, axis=0)
            q05 = np.quantile(arr, 0.05, axis=0)
            q95 = np.quantile(arr, 0.95, axis=0)
            plt.fill_between(tvec, q05, q95, color=color, alpha=0.2)
            plt.plot(tvec, med, color=color, lw=2, label=f"{label} median")
        plt.title(f"Summary: {sname}")
        plt.xlabel("timestep"); plt.ylabel(sname); plt.legend(); plt.tight_layout()
        plt.savefig(outdir / f"summary_{sname}.png", dpi=200)
        plt.close()

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Forward sim comparison for cross-phase interactions (ON vs OFF) — parallel across both conditions")
    ap.add_argument("--defaults", type=str, default="scripts/scripts_defaults.yaml", help="Path to scripts_defaults.yaml")
    ap.add_argument("--replicates", type=int, default=100, help="Replicates per condition")
    ap.add_argument("--workers", type=int, default=cpu_count(), help="Number of worker processes")
    ap.add_argument("--outdir", type=str, default="results/compare_interactions", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    params_base, cfg = load_baseline_params(args.defaults)
    total_steps = int(cfg.get("total_steps", 145))
    seed_base = int(cfg.get("seed_base", 0))
    summary_stats = list(cfg.get("summary_stats", ["S0", "S1", "S2", "NND_med"]))

    # Workers summary
    print(f"Workers: {args.workers} (cpu_count={cpu_count()})")

    N = int(args.replicates)
    reps = list(range(N))
    tvec = np.arange(total_steps + 1)

    # Build ALL tasks (ON + OFF) and run in a single Pool
    tasks = []
    for label, flag in [("on", True), ("off", False)]:
        for rep in reps:
            tasks.append((label, flag, rep, params_base, total_steps, seed_base))

    # Progress with three bars: total, on, off
    sims_on_list, sims_off_list = [], []
    sizes_on_list, sizes_off_list = [], []

    with Progress(
        "[bold cyan]Simulating (both conditions in parallel)...",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        total_task = progress.add_task("total", total=len(tasks))
        on_task = progress.add_task("ON", total=N)
        off_task = progress.add_task("OFF", total=N)

        with Pool(processes=args.workers) as pool:
            for label, rep, sim, sizes in pool.imap_unordered(_worker, tasks, chunksize=1):
                if label == "on":
                    sims_on_list.append(sim)
                    sizes_on_list.append(sizes)
                    progress.update(on_task, advance=1)
                else:
                    sims_off_list.append(sim)
                    sizes_off_list.append(sizes)
                    progress.update(off_task, advance=1)
                progress.update(total_task, advance=1)

    # Stack results
    sims_on = np.asarray(sims_on_list)  # (N, T, K)
    sims_off = np.asarray(sims_off_list)
    sizes_on_by_t = list(zip(*sizes_on_list))   # (T, list of N arrays)
    sizes_off_by_t = list(zip(*sizes_off_list))

    # Summary plots
    plot_summaries(outdir, tvec, summary_stats, sims_on, sims_off)

    # Aggregate cluster sizes across replicates at each t
    sizes_on_flat = []
    sizes_off_flat = []
    T = total_steps + 1
    for t in range(T):
        so = np.concatenate(sizes_on_by_t[t]) if sizes_on_by_t[t] else np.array([], dtype=int)
        sf = np.concatenate(sizes_off_by_t[t]) if sizes_off_by_t[t] else np.array([], dtype=int)
        sizes_on_flat.append(so)
        sizes_off_flat.append(sf)

    # GIFs
    gif1 = outdir / "cluster_size_side_by_side.gif"
    gif2 = outdir / "cluster_size_overlay.gif"
    gif3 = outdir / "cluster_size_difference.gif"
    gif_side_by_side(gif1, sizes_on_flat, sizes_off_flat, total_steps)
    gif_overlay(gif2, sizes_on_flat, sizes_off_flat, total_steps)
    gif_difference(gif3, sizes_on_flat, sizes_off_flat, total_steps)

    print(f"[done] Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
