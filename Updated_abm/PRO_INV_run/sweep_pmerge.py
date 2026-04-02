#!/usr/bin/env python3
"""
Efficient p_merge sweep for PRO vs INV (Phase 2 only), without redundant reruns.

What this script does
---------------------
1) Defines a p_merge grid (default: 0.1..1.0 step 0.1)
2) Runs each condition exactly once per p_merge:
     - PRO monoculture (n_init=850; baseline proliferation from DEFAULTS)
     - INV monoculture (n_init=1100; proliferation scaled: INV = 0.9 × PRO)
   All agents are forced into Phase 2 from t=0.
3) Exports per-step trajectories (per repeat) to a timestamped run folder:
     PRO_INV_results/sweep_pmerge/<YYYYmmdd_HHMMSS>/trajectories/
4) Builds a summary over all (p_PRO, p_INV) with constraint p_INV >= p_PRO:
     final-time mean cluster size for each phenotype, and
     % difference = 100 × (INV − PRO) / PRO
   Saves:
     summary/sweep_pmerge_summary.csv
     plots/mean_size_pctdiff_heatmap.png

Usage
-----
  python -m PRO_INV_run.sweep_pmerge --repeats 3 --steps 145 --workers 4
  python -m PRO_INV_run.sweep_pmerge --repeats 1 --steps 60 --serial
"""
from __future__ import annotations

import argparse
import json
import os
import multiprocessing as mp
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Light top-level imports (your ABM must be a package: abm/__init__.py present)
from abm.utils import DEFAULTS, export_timeseries_state
from abm.clusters_model import ClustersModel


# -----------------------------
# Core helpers
# -----------------------------
def _force_phase2(model: ClustersModel) -> None:
    """Force all agents into movement phase 2 and prevent switching."""
    import math
    for a in list(model.agent_set):
        if getattr(a, "alive", True):
            a.movement_phase = 2
            a.phase_switch_time = math.inf


def _run_repeat_task(
    kind: str,
    p_merge: float,
    rep: int,
    *,
    steps: int,
    seed: int,
    traj_root: Path,
    pro_n_init: int,
    inv_n_init: int,
    inv_scale: float,
) -> str:
    """Worker: run one repeat for a single condition (either PRO or INV)."""
    # Prevent BLAS oversubscription inside the worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Late import in worker (compatible with spawn)
    from copy import deepcopy
    from abm.utils import DEFAULTS, export_timeseries_state
    from abm.clusters_model import ClustersModel

    phenotype = "proliferative" if kind == "PRO" else "invasive"
    n_init = pro_n_init if kind == "PRO" else inv_n_init

    params = deepcopy(DEFAULTS)
    params["time"]["steps"] = int(steps)
    params["init"] = {
        "n_clusters": int(n_init),
        "size": int(params["init"].get("size", 1)),
        "phenotype": phenotype,
    }
    params["merge"]["p_merge"] = float(p_merge)

    # Full difference for INV: lower proliferation relative to PRO
    if kind == "INV" and inv_scale != 1.0:
        base_pro = float(DEFAULTS["phenotypes"]["proliferative"]["prolif_rate"])
        params["phenotypes"]["invasive"]["prolif_rate"] = float(base_pro * inv_scale)

    model = ClustersModel(params=params, seed=int(seed))
    _force_phase2(model)

    for _ in range(int(steps)):
        model.step()

    out_dir = traj_root / f"{kind}_p{p_merge:.1f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"repeat_{rep:03d}.csv"
    export_timeseries_state(model, out_csv=str(out_csv))
    return str(out_csv)


def _build_grid(pmin: float, pmax: float, pstep: float) -> List[float]:
    grid = []
    p = pmin
    while p <= pmax + 1e-12:
        grid.append(round(p, 3))
        p += pstep
    return grid


def _mean_cluster_size_final(csv_files: List[Path], *, steps: int) -> float:
    """Mean cluster size at final step across repeats (ignores NaN/empty)."""
    vals = []
    for f in csv_files:
        df = pd.read_csv(f, usecols=["step", "size"])
        if df.empty:
            continue
        max_step = int(df["step"].max())
        final_step = steps if steps <= max_step else max_step
        sizes = df[df["step"] == final_step]["size"].to_numpy(float)
        if sizes.size:
            vals.append(float(np.mean(sizes)))
    return float(np.mean(vals)) if len(vals) else float("nan")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Efficient sweep of p_merge for PRO vs INV (full-difference), with timestamped outputs."
    )
    ap.add_argument("--repeats", type=int, default=5, help="Repeats per condition.")
    ap.add_argument("--steps", type=int, default=None, help='Override DEFAULTS["time"]["steps"].')
    ap.add_argument("--seed0", type=int, default=321, help="Base seed used to derive per-task seeds.")
    ap.add_argument("--workers", type=int, default=None, help="Parallel workers (default: cpu_count-1).")
    ap.add_argument("--serial", action="store_true", help="Force serial execution (debugging).")
    ap.add_argument("--pmin", type=float, default=0.1, help="Start of p_merge grid (inclusive).")
    ap.add_argument("--pmax", type=float, default=1.0, help="End of p_merge grid (inclusive).")
    ap.add_argument("--pstep", type=float, default=0.1, help="Step of p_merge grid.")
    args = ap.parse_args()

    # Resolve steps
    steps = int(args.steps) if args.steps is not None else int(DEFAULTS["time"]["steps"])

    # Timestamped run directory (prevents overwriting)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parents[1]
    out_root = root / "PRO_INV_results" / "sweep_pmerge" / run_id
    traj_root = out_root / "trajectories"
    summary_dir = out_root / "summary"
    plots_dir = out_root / "plots"
    for d in (traj_root, summary_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Build grid
    grid = _build_grid(args.pmin, args.pmax, args.pstep)

    # Conditions: each unique PRO p and each unique INV p (no redundant reruns)
    pro_n_init = 850
    inv_n_init = 1100
    inv_scale = 0.9  # INV proliferation relative to PRO

    # Prepare tasks (deterministic seeds per (kind, p, rep))
    tasks = []
    for kind, pvals in [("PRO", grid), ("INV", grid)]:
        base = 10_000 if kind == "PRO" else 20_000
        for p in pvals:
            for r in range(args.repeats):
                seed = int(args.seed0 + base * int(round(100 * p)) + r)
                tasks.append((kind, p, r, steps, seed, str(traj_root), pro_n_init, inv_n_init, inv_scale))

    # Global BLAS thread caps
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Execute
    print(f"[RUN] sweep_pmerge started: run_id={run_id}")
    if args.serial or (args.workers == 1):
        for i, t in enumerate(tasks, 1):
            kind, p, r, steps_, seed, traj_root_str, pro_n, inv_n, inv_scale_ = t
            print(f"[{i}/{len(tasks)}] {kind} p_merge={p:.1f} repeat={r} -> running")
            _run_repeat_task(
                kind,
                p,
                r,
                steps=steps_,
                seed=seed,
                traj_root=Path(traj_root_str),
                pro_n_init=pro_n,
                inv_n_init=inv_n,
                inv_scale=inv_scale_,
            )
    else:
        workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
        ctx = mp.get_context("spawn")  # robust on macOS/Linux with BLAS/NumPy
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = [
                ex.submit(
                    _run_repeat_task,
                    kind,
                    p,
                    r,
                    steps=steps,
                    seed=seed,
                    traj_root=Path(traj_root),
                    pro_n_init=pro_n_init,
                    inv_n_init=inv_n_init,
                    inv_scale=inv_scale,
                )
                for (kind, p, r, steps, seed, traj_root, pro_n_init, inv_n_init, inv_scale) in tasks
            ]
            for i, fut in enumerate(as_completed(futs), 1):
                _ = fut.result()  # raise on error immediately
                if i % max(1, len(futs) // 20) == 0 or i == len(futs):
                    print(f"[RUN] Completed {i}/{len(futs)} repeats...")

    # Save run metadata
    meta = {
        "run_id": run_id,
        "repeats": args.repeats,
        "steps": steps,
        "seed0": args.seed0,
        "workers": None if args.serial else (args.workers or max(1, (os.cpu_count() or 2) - 1)),
        "grid": {"pmin": args.pmin, "pmax": args.pmax, "pstep": args.pstep, "values": grid},
        "pro_n_init": pro_n_init,
        "inv_n_init": inv_n_init,
        "inv_prolif_scale": inv_scale,
        "phase2_only": True,
    }
    with (out_root / "config.json").open("w") as f:
        json.dump(meta, f, indent=2)

    # -----------------------------
    # Summarise and plot
    # -----------------------------
    rows = []
    for p_pro in grid:
        pro_files = sorted((traj_root / f"PRO_p{p_pro:.1f}").glob("repeat_*.csv"))
        m_pro = _mean_cluster_size_final(pro_files, steps=steps)
        for p_inv in grid:
            if p_inv + 1e-12 < p_pro:
                continue  # enforce p_INV >= p_PRO in the table
            inv_files = sorted((traj_root / f"INV_p{p_inv:.1f}").glob("repeat_*.csv"))
            m_inv = _mean_cluster_size_final(inv_files, steps=steps)
            pct = 100.0 * (m_inv - m_pro) / m_pro if (m_pro and np.isfinite(m_pro) and m_pro != 0) else np.nan
            rows.append(
                {
                    "p_merge_PRO": round(p_pro, 1),
                    "p_merge_INV": round(p_inv, 1),
                    "mean_size_PRO_final": m_pro,
                    "mean_size_INV_final": m_inv,
                    "pct_diff_INV_vs_PRO_final": pct,
                }
            )

    df_sum = pd.DataFrame(rows).sort_values(["p_merge_PRO", "p_merge_INV"]).reset_index(drop=True)
    csv_path = summary_dir / "sweep_pmerge_summary.csv"
    df_sum.to_csv(csv_path, index=False)
    print(f"[RUN] Saved summary table: {csv_path}")

    # Heatmap (upper triangle populated)
    grid_sorted = sorted(set([round(x, 1) for x in grid]))
    idx = {v: i for i, v in enumerate(grid_sorted)}
    M = np.full((len(grid_sorted), len(grid_sorted)), np.nan, dtype=float)
    for _, row in df_sum.iterrows():
        i = idx[row["p_merge_PRO"]]
        j = idx[row["p_merge_INV"]]
        M[i, j] = row["pct_diff_INV_vs_PRO_final"]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(M, origin="lower", cmap="coolwarm", vmin=-100, vmax=100)
    ax.set_xticks(range(len(grid_sorted)))
    ax.set_yticks(range(len(grid_sorted)))
    ax.set_xticklabels([f"{v:.1f}" for v in grid_sorted])
    ax.set_yticklabels([f"{v:.1f}" for v in grid_sorted])
    ax.set_xlabel("p_merge (INV)")
    ax.set_ylabel("p_merge (PRO)")
    ax.set_title("% difference in mean cluster size (final) — INV vs PRO")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% (positive = INV > PRO)")
    # Hatch lower triangle (j < i)
    for i in range(len(grid_sorted)):
        for j in range(len(grid_sorted)):
            if j < i:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="//", edgecolor="k", linewidth=0.0))
    fig.tight_layout()
    heat_path = plots_dir / "mean_size_pctdiff_heatmap.png"
    fig.savefig(heat_path, dpi=300)
    plt.close(fig)
    print(f"[RUN] Saved heatmap: {heat_path}")
    print(f"[RUN] sweep_pmerge finished: run_id={run_id}")


if __name__ == "__main__":
    main()