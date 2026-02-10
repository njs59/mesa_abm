# abcp/compare_motion_models.py
#!/usr/bin/env python3
"""
Compare multiple ABC runs (DBs) using epsilon & PPCs.
Updated for fixed movement (no motion/speed parsing).

Outputs (in outdir):
 00_epsilon_trajectories.png   # epsilon vs population for each DB
 01_epsilon_final_bar.png      # final epsilon per DB (bar)
 compare_grid.png              # PPC grid (rows=stats, cols=DBs)
 overlay_<stat>.png            # PPC medians overlays for each stat
 overlay_discrepancy.png       # Discrepancy overlays
 summary_per_model.csv         # coverage per stat + discrepancy summaries
 epsilon_trajectories.csv      # optional, all epsilon trajectories

Usage (example):
  python compare_motion_models.py \
    --db_glob 'results/abc_*.db' \
    --observed_ts observed/INV_ABM_ready_summary.csv \
    --pp 80 --workers 8 --downsample 2 \
    --outdir results/compare_runs
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyabc
from concurrent.futures import ProcessPoolExecutor, as_completed

from abm.clusters_model import ClustersModel
from abcp.abc_model_wrapper import particle_to_params
from abcp.compute_summary import simulate_timeseries

plt.style.use("seaborn-v0_8")


def make_model_factory(seed: int = 777):
    def factory(params_dict):
        return ClustersModel(params=params_dict, seed=seed)
    return factory


def detect_stats(observed_ts: Path, drop_gr: bool | None = None) -> List[str]:
    df = pd.read_csv(observed_ts)
    cols = df.columns.tolist()
    base = ["S0", "S1", "S2", "NND_med"]
    has_gr = ("g_r40" in cols) and ("g_r80" in cols)
    if drop_gr is True:
        return base
    if drop_gr is False and has_gr:
        return base + ["g_r40", "g_r80"]
    return base + (["g_r40", "g_r80"] if has_gr else [])


def load_posterior(db_path: Path) -> Tuple[pd.DataFrame, np.ndarray, pyabc.History, int]:
    h = pyabc.History(f"sqlite:///{db_path}")
    t_max = h.max_t
    params_df, w = h.get_distribution(m=0, t=t_max)
    if len(params_df) == 0:
        raise RuntimeError(f"No posterior in {db_path}. Has ABC finished?")
    w = np.asarray(w, float)
    w = np.ones_like(w) / len(w) if w.sum() <= 0 else (w / w.sum())
    return params_df, w, h, t_max


def summarize_bands(sims: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "med": np.median(sims, axis=0),
        "q05": np.quantile(sims, 0.05, axis=0),
        "q95": np.quantile(sims, 0.95, axis=0),
        "N": sims.shape[0],
    }


def _simulate_one(args):
    """Top-level function for multiprocessing. Returns (idx, sim[T,K])."""
    (idx, p_dict, timesteps, total_steps, col_idx, seed_base) = args
    factory = make_model_factory(seed=seed_base + idx)
    params = particle_to_params(p_dict)  # fixed movement
    sim_full = simulate_timeseries(factory, params, total_steps=total_steps, sample_steps=tuple(timesteps))
    sim = sim_full[:, col_idx]
    return idx, sim


def ppc_resim_parallel(
    params_df: pd.DataFrame, w: np.ndarray, timesteps: List[int], stats: List[str],
    total_steps: int, pp: int, workers: int, seed: int = 2026
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    N = len(params_df)
    idxs = rng.choice(np.arange(N), size=pp, replace=True, p=w)  # with replacement
    full_order = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
    col_idx = [full_order.index(s) for s in stats]
    jobs = [(i, params_df.iloc[idx].to_dict(), timesteps, total_steps, col_idx, seed * 1000)
            for i, idx in enumerate(idxs)]
    if workers <= 1:
        sims = [_simulate_one(j)[1] for j in jobs]
        return summarize_bands(np.asarray(sims))
    sims = np.empty((pp, len(timesteps), len(stats)), dtype=float)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_simulate_one, j) for j in jobs]
        for fut in as_completed(futs):
            i, sim = fut.result()
            sims[i] = sim
    return summarize_bands(sims)


def coverage_and_discrepancy(obs_mat: np.ndarray, med: np.ndarray, q05: np.ndarray, q95: np.ndarray):
    inside = (obs_mat >= q05) & (obs_mat <= q95)
    cov = inside.mean(axis=0)  # per stat
    disc_t = np.sqrt(np.sum((med - obs_mat) ** 2, axis=1))
    return pd.Series(cov), pd.Series(disc_t)


def main():
    ap = argparse.ArgumentParser(description="Compare ABC runs via epsilon & PPC (fixed movement)")
    ap.add_argument("--db_glob", type=str, required=True, help="Glob for DBs, e.g. results/abc_*.db")
    ap.add_argument("--observed_ts", type=str, required=True)
    ap.add_argument("--pp", type=int, default=80, help="Posterior predictive draws (for re-sim)")
    ap.add_argument("--total_steps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=1, help="Processes for PPC")
    ap.add_argument("--outdir", type=str, default="results/compare_runs")
    ap.add_argument("--drop_gr", action="store_true", help="Force dropping g(r)")
    ap.add_argument("--limit_models", type=int, default=None, help="Compare only first N DBs")
    ap.add_argument("--downsample", type=int, default=1, help="Plot every kth timestep for speed (PPC plots)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    db_files = sorted([Path(p) for p in Path(".").glob(args.db_glob)])
    if len(db_files) == 0:
        raise FileNotFoundError(f"No DB files matched {args.db_glob}")
    if args.limit_models:
        db_files = db_files[: args.limit_models]

    # Observed data & stats
    obs_df = pd.read_csv(args.observed_ts)
    timesteps_all = obs_df["timestep"].astype(int).to_list()
    stats = detect_stats(Path(args.observed_ts), drop_gr=args.drop_gr)
    obs_mat_all = obs_df[stats].to_numpy(float)  # T x K
    K = len(stats)

    # Optional downsample for PPC plotting speed
    step = max(1, int(args.downsample))
    timesteps = timesteps_all[::step]
    obs_mat = obs_mat_all[::step, :]

    runs = []
    used = []
    for db in db_files:
        label = db.stem  # simple label from filename
        params_df, w, h, t_max = load_posterior(db)

        # Epsilon trajectory
        pops = h.get_all_populations()
        eps_t = pops["t"].to_numpy() if "t" in pops else np.array([])
        eps_vals = pops["epsilon"].to_numpy() if "epsilon" in pops else np.array([])
        eps_final = float(eps_vals[-1]) if eps_vals.size > 0 else np.nan

        # PPC
        bands = ppc_resim_parallel(
            params_df, w, timesteps, stats, total_steps=args.total_steps, pp=args.pp, workers=args.workers
        )
        cov, disc_t = coverage_and_discrepancy(obs_mat, bands["med"], bands["q05"], bands["q95"])
        runs.append({
            "db": db, "label": label,
            "med": bands["med"], "q05": bands["q05"], "q95": bands["q95"], "N_pp": int(bands["N"]),
            "coverage": cov, "disc_t": disc_t,
            "eps_t": eps_t, "eps_vals": eps_vals, "eps_final": eps_final,
            "params_df": params_df, "w": w,
        })
        used.append(label)

    if not runs:
        print("No valid runs to plot.")
        return

    # PLOT 0A: Epsilon trajectories
    palette_models = sns.color_palette("tab10", n_colors=max(10, len(runs)))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for j, r in enumerate(runs):
        if len(r["eps_t"]) == 0:
            continue
        ax.plot(r["eps_t"], r["eps_vals"], marker="o", ms=3, lw=1.5,
                color=palette_models[j], label=r["label"])
    ax.set_xlabel("Population t")
    ax.set_ylabel("Epsilon")
    ax.set_title("Epsilon trajectories per run")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "00_epsilon_trajectories.png", dpi=200)
    plt.close(fig)

    # PLOT 0B: Final epsilon bar
    df_eps = pd.DataFrame([{"label": r["label"], "eps_final": r["eps_final"]} for r in runs])
    df_eps = df_eps.sort_values("eps_final", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(df_eps["label"], df_eps["eps_final"], color="#6baed6")
    ax.set_xlabel("Final epsilon")
    ax.set_title("Final epsilon per run (lower is tighter)")
    for i, v in enumerate(df_eps["eps_final"].to_numpy()):
        ax.text(v, i, f" {v:.3g}", va="center", ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "01_epsilon_final_bar.png", dpi=200)
    plt.close(fig)

    # PLOT 1: PPC GRID
    Kstats = len(stats)
    C = len(runs)
    fig, axes = plt.subplots(nrows=Kstats, ncols=C, figsize=(min(2.6*C, 26), min(2.2*Kstats, 16)), sharex=True)
    if Kstats == 1:
        axes = np.array([axes])
    t = np.array(timesteps)
    for j, r in enumerate(runs):
        axes[0, j].set_title(r["label"])
    for i, s in enumerate(stats):
        for j, r in enumerate(runs):
            ax = axes[i, j]
            ax.fill_between(t, r["q05"][:, i], r["q95"][:, i], color="#cfe8ff", alpha=0.85)
            ax.plot(t, r["med"][:, i], color="#1f77b4", lw=1.7)
            ax.plot(t, obs_mat[:, i], color="black", lw=1.0)
            if i == Kstats - 1:
                ax.set_xlabel("timestep")
            ax.set_ylabel(s if j == 0 else "")
    fig.suptitle("Posterior predictive comparison — grid", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "compare_grid.png", dpi=200)
    plt.close(fig)

    # PLOT 2: PPC overlays per stat
    for i, s in enumerate(stats):
        fig, ax = plt.subplots(figsize=(10, 5))
        for j, r in enumerate(runs):
            ax.plot(t, r["med"][:, i], lw=1.8, color=palette_models[j], label=r["label"])
        ax.plot(t, obs_mat[:, i], color="black", lw=1.5, label="observed", linestyle="--")
        ax.set_title(f"PPC medians overlay — {s}")
        ax.set_xlabel("timestep")
        ax.set_ylabel(s)
        ax.legend(ncols=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"overlay_{s}.png", dpi=200)
        plt.close(fig)

    # PLOT 3: discrepancy overlays
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, r in enumerate(runs):
        ax.plot(t, r["disc_t"], lw=1.5, color=palette_models[j], label=r["label"])
    ax.set_title("Median discrepancy vs time (Euclidean across selected stats)")
    ax.set_xlabel("timestep")
    ax.set_ylabel("discrepancy")
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "overlay_discrepancy.png", dpi=200)
    plt.close(fig)

    # SUMMARY CSV
    rows = []
    for r in runs:
        row = {
            "db": str(r["db"]),
            "label": r["label"],
            "epsilon_final": r["eps_final"],
            "N_pp": r["N_pp"],
        }
        for s_idx, s in enumerate(stats):
            row[f"coverage_{s}"] = float(r["coverage"].iloc[s_idx])
        row["disc_med_median"] = float(np.median(r["disc_t"]))
        row["disc_med_q05"] = float(np.quantile(r["disc_t"], 0.05))
        row["disc_med_q95"] = float(np.quantile(r["disc_t"], 0.95))
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "summary_per_model.csv", index=False)

    # Optional: export epsilon trajectories
    eps_rows = []
    for r in runs:
        if len(r["eps_t"]) == 0:
            continue
        for tt, ev in zip(r["eps_t"], r["eps_vals"]):
            eps_rows.append({"label": r["label"], "t": int(tt), "epsilon": float(ev)})
    if eps_rows:
        pd.DataFrame(eps_rows).to_csv(outdir / "epsilon_trajectories.csv", index=False)

    print(f"Done. Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()