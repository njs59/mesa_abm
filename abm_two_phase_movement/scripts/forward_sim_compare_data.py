#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

# Your ABM & summary
from abm.clusters_model import ClustersModel
from abm.utils import DEFAULTS
from abcp.compute_summary import simulate_timeseries


# --------------------------------------------------------------------
# Constants / helpers
# --------------------------------------------------------------------

# Expected simulator column order (NORMAL NND)
ALL_STATS = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge dict b into a, returning a."""
    for k, v in b.items():
        if isinstance(v, dict):
            a.setdefault(k, {})
            deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def flat_to_nested_params(flat: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert flat CLI-style params into nested ABM params.
    Only override provided keys; the rest come from DEFAULTS.
    """
    out: Dict[str, Any] = {}
    # Phenotype (proliferative) biology
    ph = {}
    if flat.get("prolif_rate") is not None:
        ph["prolif_rate"] = float(flat["prolif_rate"])
    if flat.get("fragment_rate") is not None:
        ph["fragment_rate"] = float(flat["fragment_rate"])
    if ph:
        out.setdefault("phenotypes", {}).setdefault("proliferative", {}).update(ph)

    # Physics
    phys = {}
    if flat.get("softness") is not None:
        phys["softness"] = float(flat["softness"])
    if flat.get("fragment_minsep_factor") is not None:
        phys["fragment_minsep_factor"] = float(flat["fragment_minsep_factor"])
    if phys:
        out.setdefault("physics", {}).update(phys)

    # Merge
    if flat.get("p_merge") is not None:
        out.setdefault("merge", {})["p_merge"] = float(flat["p_merge"])

    # Init
    init = {}
    if flat.get("n_init") is not None:
        init["n_clusters"] = int(flat["n_init"])
    if init:
        init.setdefault("size", 1)
        init.setdefault("phenotype", "proliferative")
        out.setdefault("init", {}).update(init)

    return out


def build_full_params(overrides_flat: Dict[str, Any]) -> Dict[str, Any]:
    """Start from DEFAULTS and apply flat overrides as nested structure."""
    params = deep_merge(dict(DEFAULTS), flat_to_nested_params(overrides_flat))
    # Coercions / sane defaults
    params.setdefault("physics", {})
    params["physics"].setdefault("soft_separate", True)
    params["physics"].setdefault("softness", 0.15)
    params["physics"].setdefault("fragment_minsep_factor", 1.1)
    if "merge" in params and "p_merge" in params["merge"]:
        params["merge"]["p_merge"] = float(np.clip(params["merge"]["p_merge"], 0.0, 1.0))
    return params


def load_observed_nnd(observed_csv: str,
                      requested_timesteps: List[int],
                      summary_stats: List[str]) -> Tuple[pd.DataFrame, List[int]]:
    """
    Load observed CSV and return (filtered_df, valid_timesteps).
    Observed must contain 'timestep' and the four stats:
      ['S0', 'S1', 'S2', 'NND_med']  (NORMAL NND)
    We intersect requested timesteps with those present.
    """
    df = pd.read_csv(observed_csv).sort_values("timestep").reset_index(drop=True)

    # Strict: expect NND_med (normal). If the file has only SSNND_med, raise with guidance.
    if "NND_med" not in df.columns:
        raise ValueError(
            "Observed file must contain 'NND_med' (normal NND). "
            "If your file has 'SSNND_med' (surface-to-surface), use the SSNND version of this script."
        )

    missing_cols = [c for c in summary_stats if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Observed CSV missing required columns {missing_cols}. Got {list(df.columns)}")

    data_ts = set(int(t) for t in df["timestep"])
    req_ts = [int(t) for t in requested_timesteps]
    valid_ts = [t for t in req_ts if t in data_ts]
    dropped = [t for t in req_ts if t not in data_ts]
    if dropped:
        head = ", ".join(map(str, dropped[:10]))
        tail = " ..." if len(dropped) > 10 else ""
        print(f"[forward] WARNING: dropping timesteps not found in observed: [{head}]{tail}")

    df = df.set_index("timestep").loc[valid_ts].reset_index()
    return df, valid_ts


# --------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------
def _simulate_one(args) -> pd.DataFrame:
    """
    Run one replicate and return a DataFrame of the 4 stats across valid timesteps.
    """
    rep, cfg, params = args

    def factory(p: dict):
        return ClustersModel(params=p, seed=int(cfg["seed_base"]) + rep)

    sim_full = simulate_timeseries(
        factory, params=params,
        total_steps=cfg["total_steps"],
        sample_steps=tuple(cfg["timesteps"]),
    )
    # Name columns and slice the 4 fitting stats
    num_cols = min(len(ALL_STATS), sim_full.shape[1])
    df_full = pd.DataFrame(sim_full, columns=ALL_STATS[:num_cols])
    return df_full[cfg["summary_stats"]]


# --------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------
def plot_overlays(outdir: Path,
                  timesteps: List[int],
                  summary_stats: List[str],
                  sims: np.ndarray,
                  obs_matrix: np.ndarray) -> None:
    """
    sims: (N, T, K)  -- N replicates, T timepoints, K stats
    obs_matrix: (T, K)
    """
    outdir.mkdir(parents=True, exist_ok=True)
    T = sims.shape[1]
    t = np.array(timesteps)
    med = np.median(sims, axis=0)          # (T, K)
    q05 = np.quantile(sims, 0.05, axis=0)  # (T, K)
    q95 = np.quantile(sims, 0.95, axis=0)  # (T, K)

    for k, s in enumerate(summary_stats):
        plt.figure(figsize=(10, 4.4))
        plt.fill_between(t, q05[:, k], q95[:, k], color="#cfe8ff", alpha=0.85, label="5–95% band")
        plt.plot(t, med[:, k], color="#1f77b4", lw=2.0, label="replicate median")
        plt.plot(t, obs_matrix[:, k], color="black", lw=1.6, label="observed")
        plt.xlabel("timestep")
        plt.ylabel(s)
        plt.title(f"Forward simulation vs observed — {s}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"forward_overlay_{s}.png", dpi=200)
        plt.close()

        # Save CSVs
        pd.DataFrame({
            "timestep": t,
            "median": med[:, k],
            "q05": q05[:, k],
            "q95": q95[:, k],
            "observed": obs_matrix[:, k],
        }).to_csv(outdir / f"forward_overlay_{s}.csv", index=False)


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Forward simulation with NORMAL NND (NND_med).")
    # I/O
    ap.add_argument("--observed_ts", type=str, required=True,
                    help="Observed CSV with columns: timestep, S0, S1, S2, NND_med")
    ap.add_argument("--outdir", type=str, default="results/forward_sim_nnd",
                    help="Output directory.")

    # Simulation controls
    ap.add_argument("--total_steps", type=int, default=145)
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed_base", type=int, default=777)

    # Fitting stats (NORMAL NND)
    summary_stats = ["S0", "S1", "S2", "NND_med"]

    # Flat parameter overrides (easy to change via CLI)
    ap.add_argument("--prolif_rate", type=float, default=5e-3)
    ap.add_argument("--fragment_rate", type=float, default=5e-4)
    ap.add_argument("--p_merge", type=float, default=0.9)
    ap.add_argument("--softness", type=float, default=0.15)
    ap.add_argument("--fragment_minsep_factor", type=float, default=1.1)
    ap.add_argument("--n_init", type=int, default=800)

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build full params dict from DEFAULTS + overrides
    flat = dict(
        prolif_rate=args.prolif_rate,
        fragment_rate=args.fragment_rate,
        p_merge=args.p_merge,
        softness=args.softness,
        fragment_minsep_factor=args.fragment_minsep_factor,
        n_init=args.n_init,
    )
    params = build_full_params(flat)

    # Load observed (NORMAL NND) and intersect timesteps
    requested_ts = list(range(0, args.total_steps + 1))
    obs_df, valid_ts = load_observed_nnd(args.observed_ts, requested_ts, summary_stats)
    obs_mat = obs_df[summary_stats].to_numpy(float)

    # Config to pass to workers
    cfg = {
        "total_steps": args.total_steps,
        "timesteps": valid_ts,
        "seed_base": args.seed_base,
        "summary_stats": summary_stats,
    }

    N = int(args.replicates)
    print(f"[forward] Running {N} replicates using {args.workers} workers...")
    tasks = [(r, cfg, params) for r in range(N)]

    sims_list: List[pd.DataFrame] = []
    with Progress(
        "[bold cyan]Forward simulation (NND_med)...",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Simulating...", total=N)

        if args.workers > 1:
            with Pool(args.workers) as pool:
                for df in pool.imap_unordered(_simulate_one, tasks):
                    sims_list.append(df)
                    progress.update(task, advance=1)
        else:
            for t in tasks:
                sims_list.append(_simulate_one(t))
                progress.update(task, advance=1)

    # Stack into array (N, T, K)
    sims = np.asarray([df[summary_stats].to_numpy(float) for df in sims_list])  # (N,T,K)

    # Plot overlays and save CSVs
    plot_overlays(outdir, valid_ts, summary_stats, sims, obs_mat)
    print(f"[forward] Done. Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()