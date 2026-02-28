"""
sensitivity_runner.py
=====================

Parallelised + robust ABM sensitivity pipeline with 5 movement modes.

Features
--------
- Modes × Sweep combinations (cartesian).
- Parallel repeats per scenario (spawn context, maxtasksperchild=1).
- Worker-side raw CSV writing (optional; reduces IPC).
- Merge counting via safe monkey-patch with try/finally restoration.
- Summaries sliced by mode.analysis_start_step.
- Plots:
    * 1D CI plots (early/mid/final) or 2D heatmaps (early/mid/final)
    * Time-series overlays (and an extra set with the first timestep removed).

Configuration (config.yaml)
---------------------------
abm:
  n_repeats: 10
  seed: 42
  steps: 300
  dt: 1.0
  width: 1344
  height: 1025
  n_workers: 8        # parallel workers per scenario
  save_raw: true      # workers write per-timestep CSVs
  save_every: 1       # save every k timesteps
  task_timeout: null  # seconds per repeat; null disables

modes:
  - name: data_t71_phase2
    movement_phase: 2
    analysis_start_step: 71
    # init: { n_clusters: 800, size: 1, phenotype: proliferative }

sweep:
  movement_v2.phase2.speed_dist.params.a: [0.7, 2.1]     # bare list OK
  merge.p_merge:
    range: {start: 0.1, stop: 1.0, step: 0.1}           # or 'values: [...]'
"""

from __future__ import annotations

import os
import time
import multiprocessing as mp
from typing import Tuple, List

import numpy as np
import pandas as pd

from .utils_sens import (
    load_yaml, make_timestamped_dir, ensure_dir,
    expand_sweep_dict, expand_modes, prepare_model_params,
    make_scenario_name, derive_seed,
)

from .summary_tools import (
    summarize_repeat, aggregate_over_repeats,
    pick_early_mid_final_indices
)

from .plotting_tools import (
    plot_param_1d_ci, plot_param_2d_heatmaps,
    plot_time_series_over_conditions
)

# Import your ABM:
from abm.clusters_model import ClustersModel
from abm.utils import DEFAULTS as ABM_DEFAULTS


# ---------------------------------------------------------------------------
# Worker: one repeat (top-level for spawn pickling)
# ---------------------------------------------------------------------------
def _repeat_worker(args) -> Tuple[int, pd.DataFrame]:
    """
    Run one ABM repeat in an isolated worker process.

    args = (
        repeat_idx,
        params, steps, seed, movement_phase,
        width, height, grid_x, grid_y,
        rep_dir, save_raw, save_every
    )

    Returns: (repeat_idx, summary_df)
    """
    (repeat_idx, params, steps, seed, movement_phase,
     width, height, grid_x, grid_y,
     rep_dir, save_raw, save_every) = args

    # Create repeat directory (worker-safe)
    os.makedirs(rep_dir, exist_ok=True)

    # Instantiate model
    model = ClustersModel(params=params, seed=seed)

    # Force Phase-2 if requested
    if movement_phase == 2:
        for a in list(getattr(model, "agent_set", [])):
            try:
                a.movement_phase = 2
                a.phase_switch_time = float("inf")
            except Exception:
                pass

    # Patch merge counting
    from abm.cluster_agent import ClusterAgent
    orig_merge_with = ClusterAgent._merge_with
    merge_counts = [0] * steps

    def counted_merge(self, other):
        step = int(round(self.model.time / max(1e-12, float(self.model.dt))))
        if 0 <= step < steps:
            merge_counts[step] += 1
        return orig_merge_with(self, other)

    ClusterAgent._merge_with = counted_merge

    try:
        # Run & optionally write per-timestep CSVs from the worker
        every = max(1, int(save_every))
        for t in range(steps):
            model.step()
            if save_raw and (t % every == 0):
                df = pd.DataFrame({
                    "id": model.id_log[-1],
                    "x": model.pos_log[-1][:, 0],
                    "y": model.pos_log[-1][:, 1],
                    "size": model.size_log[-1],
                })
                df.to_csv(os.path.join(rep_dir, f"t_{t:04d}.csv"), index=False)
    finally:
        # Always restore the original method
        ClusterAgent._merge_with = orig_merge_with

    # Build per-repeat summary (small)
    rep_df = summarize_repeat(
        id_log=model.id_log,
        pos_log=model.pos_log,
        size_log=model.size_log,
        dt=model.dt,
        width=width,
        height=height,
        grid_x=grid_x,
        grid_y=grid_y,
        merges_by_step=merge_counts,
    )

    # Save summary here too (worker writes, parent just aggregates)
    rep_df.to_csv(os.path.join(rep_dir, "summary_repeat.csv"), index=False)

    return repeat_idx, rep_df


# ---------------------------------------------------------------------------
# Main Sensitivity Runner
# ---------------------------------------------------------------------------
def run_sensitivity(config_path: str):
    cfg = load_yaml(config_path)

    # Core config
    results_root = cfg.get("results_root", "ABM_sensitivity_results")
    base_out = make_timestamped_dir(results_root)

    abm_cfg = cfg.get("abm", {})
    n_repeats    = int(abm_cfg.get("n_repeats", 5))
    base_seed    = int(abm_cfg.get("seed", 42))
    steps        = int(abm_cfg.get("steps", 300))
    width        = float(abm_cfg.get("width", 1344))
    height       = float(abm_cfg.get("height", 1025))
    n_workers    = max(1, int(abm_cfg.get("n_workers", 4)))
    save_raw     = bool(abm_cfg.get("save_raw", True))
    save_every   = max(1, int(abm_cfg.get("save_every", 1)))
    task_timeout = abm_cfg.get("task_timeout", None)
    if task_timeout is not None:
        try:
            task_timeout = float(task_timeout)
        except Exception:
            task_timeout = None

    # Morisita grid
    mcfg = cfg.get("morisita", {})
    grid_x = int(mcfg.get("grid_x", 20))
    grid_y = int(mcfg.get("grid_y", 15))

    # Modes + sweep
    modes = expand_modes(cfg)
    sweep_cfg = cfg.get("sweep", {})
    override_combos = expand_sweep_dict(sweep_cfg)

    # Output dirs
    sim_root = ensure_dir(os.path.join(base_out, "simulations"))
    sum_root = ensure_dir(os.path.join(base_out, "summaries"))
    plot_root = ensure_dir(os.path.join(base_out, "plots"))

    total_scenarios = len(modes) * max(1, len(override_combos))
    print(f"\nRunning {total_scenarios} scenarios with up to {n_workers} workers per scenario\n")

    condition_records: List[dict] = []
    scen_idx = 0

    # Use a spawn context – safer with NumPy/BLAS/OpenMP
    ctx = mp.get_context("spawn")

    for mode in modes:
        for overrides in (override_combos or [{}]):
            scenario_name = make_scenario_name(scen_idx, mode["name"], overrides)
            print(f"[Scenario {scen_idx+1}/{total_scenarios}] {scenario_name}")

            cond_simdir = ensure_dir(os.path.join(sim_root, scenario_name))
            cond_sumdir = ensure_dir(os.path.join(sum_root, scenario_name))

            # Prepare ABM parameters
            params = prepare_model_params(ABM_DEFAULTS, overrides, mode_init=mode.get("init"))

            # Prepare work args per repeat
            work_args = []
            for r in range(n_repeats):
                seed = derive_seed(base_seed, scen_idx, r)
                rep_dir = os.path.join(cond_simdir, f"repeat_{r:02d}")
                work_args.append((
                    r, params, steps, seed, mode["movement_phase"],
                    width, height, grid_x, grid_y,
                    rep_dir, save_raw, save_every
                ))

            repeat_frames: List[pd.DataFrame] = [None] * n_repeats  # type: ignore

            # Pool with maxtasksperchild to avoid memory growth in long runs
            with ctx.Pool(processes=min(n_workers, n_repeats), maxtasksperchild=1) as pool:
                # Submit all repeats asynchronously (minimise blocking)
                async_results = [
                    pool.apply_async(_repeat_worker, (wa,))
                    for wa in work_args
                ]

                # Collect results with optional per-task timeout
                for r, res in enumerate(async_results):
                    try:
                        if task_timeout is None:
                            rep_idx, rep_df = res.get()
                        else:
                            rep_idx, rep_df = res.get(timeout=task_timeout)
                    except mp.TimeoutError:
                        # Fallback: run this repeat synchronously to avoid hanging the whole scenario
                        print(f"  [warn] repeat {r:02d} exceeded {task_timeout}s; rerunning synchronously...")
                        rep_idx, rep_df = _repeat_worker(work_args[r])
                    repeat_frames[rep_idx] = rep_df

            # Ensure all summaries exist on disk (workers also wrote to their own repeat folders)
            for r, df in enumerate(repeat_frames):
                if df is None:
                    raise RuntimeError(f"Missing summary for repeat {r} in scenario {scenario_name}")
                # Slice for analysis start step (mode-specific)
                s0 = int(mode.get("analysis_start_step", 0))
                dff = df if s0 <= 0 else df[df["step"] >= s0].reset_index(drop=True)
                dff.to_csv(os.path.join(cond_sumdir, f"summary_repeat_{r:02d}.csv"), index=False)
                repeat_frames[r] = dff

            # Aggregate across repeats (already sliced if applicable)
            agg_df = aggregate_over_repeats(repeat_frames)
            agg_df.to_csv(os.path.join(cond_sumdir, "summary_aggregated.csv"), index=False)

            condition_records.append({
                "name": scenario_name,
                "params": {**overrides, "__mode__": mode["name"]},
                "agg_df": agg_df
            })

            scen_idx += 1

    # -----------------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------------
    if condition_records:
        T = condition_records[0]["agg_df"].shape[0]
        step_indices = pick_early_mid_final_indices(T)
    else:
        step_indices = (0, 0, 0)

    sweep_keys = list(sweep_cfg.keys())

    if len(sweep_keys) == 1:
        plot_param_1d_ci(condition_records, sweep_keys[0], plot_root, step_indices)

    elif len(sweep_keys) == 2:
        plot_param_2d_heatmaps(condition_records, (sweep_keys[0], sweep_keys[1]), plot_root, step_indices)

    # Time-series overlays (and extra set with first timestep removed)
    if len(condition_records) <= 10:
        plot_time_series_over_conditions(condition_records, plot_root, trim_first=False)
        plot_time_series_over_conditions(condition_records, plot_root, trim_first=True)

    print("\nDONE.\n")