
#!/usr/bin/env python3
"""
Compare behaviour across motion × speed models using PPCs + epsilon + posterior parameters.

Outputs (in this order):
  00_epsilon_trajectories.png       # epsilon vs population for each model
  01_epsilon_final_bar.png          # final epsilon per model (bar)
  posteriors/02_post_<motion>_<param>.png
                                    # posterior KDE overlays per motion & parameter (no re-sim)
  posteriors/03_post_all_<param>.png
                                    # posterior KDE overlays per parameter across ALL models
  compare_grid.png                  # PPC grid (rows=stats, cols=models)
  overlay_<stat>.png                # PPC medians overlays
  overlay_discrepancy.png           # Discrepancy overlays
  summary_per_model.csv             # eps_final, coverage per stat, discrepancy summaries
  posterior_param_summaries.csv     # posterior med/quantiles per parameter per method
  parse_log.txt                     # which DBs were used / skipped

Robust filename parsing:
  Accepts abc_{obs}_{motion}_{speed}_seed{seed}[_{ts}].db
  where {obs} may include underscores.

FAST options for PPC:
  --fast_stored          : use stored final-pop sum_stats (no re-sim)
  --hybrid K R           : K posterior particles × R replicates (K*R ≈ --pp)
  --workers N            : parallel re-simulation

Usage (example):
  python compare_motion_models.py \
    --db_glob 'results/abc_PRO_ABM_ready_summary_*_seed42_*.db' \
    --observed_ts observed/PRO_ABM_ready_summary.csv \
    --drop_gr \
    --pp 80 --workers 8 --downsample 2 \
    --motions_order isotropic persistent \
    --speeds_order constant lognorm gamma weibull \
    --outdir results/compare_PRO
"""

from __future__ import annotations
import argparse, re, os, sys
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyabc
from concurrent.futures import ProcessPoolExecutor, as_completed

# Project imports
from abm.clusters_model import ClustersModel
from abcp.abc_model_wrapper import particle_to_params
from abcp.compute_summary import simulate_timeseries

plt.style.use("seaborn-v0_8")


# ------------------------- helpers -------------------------
def make_model_factory(seed: int = 777):
    def factory(params_dict):
        return ClustersModel(params=params_dict, seed=seed)
    return factory



def parse_model_from_name(p: Path) -> Tuple[str, str]:
    """
    Parse motion and speed from filenames.

    Accepts:
      - abc_{obs}_{motion}_{speed}_seed{seed}[_{ts}].db     (original)
      - abc_{obs}_{motion}_{speed}_noGR.db                  (your current)
      - Or any filename containing the tokens 'isotropic'/'persistent'
        and 'constant'/'lognorm'/'gamma'/'weibull' in any order.

    Returns ("unknown", "unknown") only if neither token is found.
    """
    name = p.name.lower()

    # First try strict regex (original expectation with _seed)
    m = re.search(
        r"abc_.+?_(isotropic|persistent)_(constant|lognorm|gamma|weibull)_seed",
        name
    )
    if m:
        return m.group(1), m.group(2)

    # Try your noGR pattern explicitly
    m2 = re.search(
        r"abc_.+?_(isotropic|persistent)_(constant|lognorm|gamma|weibull)_nogr\.db$",
        name
    )
    if m2:
        return m2.group(1), m2.group(2)

    # Permissive fallback: find tokens anywhere, avoid partial-word matches
    motion = "unknown"
    speed = "unknown"

    if re.search(r"(?:^|[^a-z])isotropic(?:[^a-z]|$)", name):
        motion = "isotropic"
    elif re.search(r"(?:^|[^a-z])persistent(?:[^a-z]|$)", name):
        motion = "persistent"

    if re.search(r"(?:^|[^a-z])constant(?:[^a-z]|$)", name):
        speed = "constant"
    elif re.search(r"(?:^|[^a-z])lognorm(?:[^a-z]|$)", name):
        speed = "lognorm"
    elif re.search(r"(?:^|[^a-z])gamma(?:[^a-z]|$)", name):
        speed = "gamma"
    elif re.search(r"(?:^|[^a-z])weibull(?:[^a-z]|$)", name):
        speed = "weibull"

    return motion, speed



def load_posterior(db_path: Path) -> Tuple[pd.DataFrame, np.ndarray, pyabc.History, int]:
    h = pyabc.History(f"sqlite:///{db_path}")
    t_max = h.max_t
    params_df, w = h.get_distribution(m=0, t=t_max)
    if len(params_df) == 0:
        raise RuntimeError(f"No posterior in {db_path}. Has ABC finished?")
    w = np.asarray(w, float)
    w = np.ones_like(w) / len(w) if w.sum() <= 0 else (w / w.sum())
    return params_df, w, h, t_max


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


def stored_ppc_from_db(h: pyabc.History, t_max: int, T: int, K: int) -> np.ndarray:
    """
    Ultra-fast: pull stored sum_stats for accepted particles at final population.
    Returns array of shape (N_accept, T, K).
    """
    pop = h.get_population(t=t_max)
    sims = []
    for part in pop.particles:
        ss = part.sum_stat  # dict-like
        vec = np.array([ss[f"y_{i}"] for i in range(T * K)], dtype=float)
        sims.append(vec.reshape(T, K))
    return np.asarray(sims)  # N_accept x T x K


def summarize_bands(sims: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "med": np.median(sims, axis=0),
        "q05": np.quantile(sims, 0.05, axis=0),
        "q95": np.quantile(sims, 0.95, axis=0),
        "N": sims.shape[0],
    }


# ------------------------- PPC simulation paths -------------------------
def _simulate_one(args):
    """Top-level function for multiprocessing. Returns (idx, sim[T,K])."""
    (idx, p_dict, motion, speed, timesteps, total_steps, col_idx, seed_base) = args
    factory = make_model_factory(seed=seed_base + idx)
    params = particle_to_params(p_dict, motion=motion, speed_dist=speed)
    sim_full = simulate_timeseries(factory, params, total_steps=total_steps, sample_steps=tuple(timesteps))
    sim = sim_full[:, col_idx]
    return idx, sim


def ppc_resim_parallel(
    params_df: pd.DataFrame, w: np.ndarray, timesteps: List[int], stats: List[str],
    motion: str, speed: str, total_steps: int, pp: int, workers: int, seed: int = 2026
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    N = len(params_df)
    idxs = rng.choice(np.arange(N), size=pp, replace=True, p=w)  # with replacement
    full_order = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
    col_idx = [full_order.index(s) for s in stats]

    jobs = [(i, params_df.iloc[idx].to_dict(), motion, speed, timesteps, total_steps, col_idx, seed*1000)
            for i, idx in enumerate(idxs)]

    if workers <= 1:
        sims = [ _simulate_one(j)[1] for j in jobs ]
        return summarize_bands(np.asarray(sims))

    sims = np.empty((pp, len(timesteps), len(stats)), dtype=float)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_simulate_one, j) for j in jobs]
        for fut in as_completed(futs):
            i, sim = fut.result()
            sims[i] = sim
    return summarize_bands(sims)


def ppc_hybrid_parallel(
    params_df: pd.DataFrame, w: np.ndarray, timesteps: List[int], stats: List[str],
    motion: str, speed: str, total_steps: int, K: int, R: int, workers: int, seed: int = 2026
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    N = len(params_df)
    # Allow K > N when sampling with replacement; do not cap K at N
    # K = min(K, N)  # <-- remove this line
    base_idxs = rng.choice(np.arange(N), size=K, replace=True, p=w)  # now WITH replacement

    full_order = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
    col_idx = [full_order.index(s) for s in stats]

    jobs, jidx = [], 0
    for b in base_idxs:
        for r in range(R):
            jobs.append((jidx, params_df.iloc[b].to_dict(), motion, speed, timesteps, total_steps, col_idx, seed*2000 + r))
            jidx += 1

    M = len(jobs)
    sims = np.empty((M, len(timesteps), len(stats)), dtype=float)

    if workers <= 1:
        for j in jobs:
            i, sim = _simulate_one(j)
            sims[i] = sim
        return summarize_bands(sims)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_simulate_one, j) for j in jobs]
        for fut in as_completed(futs):
            i, sim = fut.result()
            sims[i] = sim
    return summarize_bands(sims)


# ------------------------- plotting & metrics -------------------------
def coverage_and_discrepancy(obs_mat: np.ndarray, med: np.ndarray, q05: np.ndarray, q95: np.ndarray):
    inside = (obs_mat >= q05) & (obs_mat <= q95)
    cov = inside.mean(axis=0)            # per stat
    disc_t = np.sqrt(np.sum((med - obs_mat) ** 2, axis=1))
    return pd.Series(cov), pd.Series(disc_t)


def arrange_models(db_files: List[Path], motions_order: List[str], speeds_order: List[str], limit: int | None) -> List[Path]:
    by_key, unknowns = {}, []
    for p in db_files:
        motion, speed = parse_model_from_name(p)
        if motion in motions_order and speed in speeds_order:
            by_key[(motion, speed)] = p
        else:
            unknowns.append(p)
    ordered = []
    for m in motions_order:
        for s in speeds_order:
            if (m, s) in by_key:
                ordered.append(by_key[(m, s)])
    ordered.extend([p for p in db_files if p not in ordered])
    if limit is not None:
        ordered = ordered[:limit]
    return ordered


def _kde1d_tidy(ax, x: np.ndarray, w: np.ndarray | None, color, label: str):
    """Weighted KDE with tidy data; fallback to weighted resample/hist if needed."""
    import pandas as pd
    ok = False
    try:
        df = pd.DataFrame({"x": np.asarray(x, float)})
        if w is not None:
            ww = np.asarray(w, float)
            ww = ww / (ww.sum() if ww.sum() > 0 else len(ww))
            df["w"] = ww
            sns.kdeplot(data=df, x="x", weights="w", fill=False, color=color, alpha=0.9, ax=ax, label=label)
        else:
            sns.kdeplot(data=df, x="x", fill=False, color=color, alpha=0.9, ax=ax, label=label)
        ok = True
    except Exception:
        ok = False
    if not ok:
        # Weighted resample fallback
        try:
            rng = np.random.default_rng(2026)
            n = min(5000, len(x))
            if w is not None:
                p = np.asarray(w, float)
                p = p / (p.sum() if p.sum() > 0 else len(p))
                idx = rng.choice(np.arange(len(x)), size=n, replace=True, p=p)
            else:
                idx = rng.choice(np.arange(len(x)), size=n, replace=True)
            xs = np.asarray(x, float)[idx]
            sns.kdeplot(xs, fill=False, color=color, alpha=0.9, ax=ax, label=label)
            ok = True
        except Exception:
            ok = False
    if not ok:
        ax.hist(np.asarray(x, float), bins="auto", density=True, color=color, alpha=0.45, label=label)


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser(description="Compare motion×speed models using epsilon, PPCs, and posterior parameters.")
    ap.add_argument("--db_glob", type=str, required=True, help="Glob for DBs, e.g. results/abc_*_seed*.db")
    ap.add_argument("--observed_ts", type=str, required=True)
    ap.add_argument("--pp", type=int, default=80, help="Posterior predictive draws (for re-sim)")
    ap.add_argument("--total_steps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=os.cpu_count()//2 if os.cpu_count() else 2, help="Processes for PPC")
    ap.add_argument("--outdir", type=str, default="results/compare_models")
    ap.add_argument("--motions_order", nargs="+", default=["isotropic", "persistent"])
    ap.add_argument("--speeds_order", nargs="+", default=["constant", "lognorm", "gamma", "weibull"])
    ap.add_argument("--drop_gr", action="store_true", help="Force dropping g(r)")
    ap.add_argument("--limit_models", type=int, default=None, help="Compare only first N models after ordering")
    ap.add_argument("--downsample", type=int, default=1, help="Plot every kth timestep for speed (PPC plots)")
    ap.add_argument("--fast_stored", action="store_true", help="Ultra-fast: use stored final-population sum_stats (no re-sim)")
    ap.add_argument("--hybrid", nargs=2, type=int, default=None, metavar=("K", "R"), help="Hybrid K particles × R replicates")
    ap.add_argument("--assume_motion", type=str, choices=["isotropic","persistent"], default=None,
                    help="If filenames lack motion, assume this value")
    ap.add_argument("--assume_speed", type=str, choices=["constant","lognorm","gamma","weibull"], default=None,
                    help="If filenames lack speed, assume this value")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    db_files = sorted([Path(p) for p in Path().glob(args.db_glob)])
    if len(db_files) == 0:
        raise FileNotFoundError(f"No DB files matched {args.db_glob}")

    db_files = arrange_models(db_files, args.motions_order, args.speeds_order, args.limit_models)

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
    used, skipped = [], []

    for db in db_files:
        motion, speed = parse_model_from_name(db)
        if motion == "unknown" and args.assume_motion:
            motion = args.assume_motion
        if speed == "unknown" and args.assume_speed:
            speed = args.assume_speed
        if motion == "unknown" or speed == "unknown":
            skipped.append(db.name)
            continue

        params_df, w, h, t_max = load_posterior(db)

        # Epsilon trajectory for this model
        pops = h.get_all_populations()
        if "epsilon" in pops.columns:
            eps_t = pops["t"].to_numpy()
            eps_vals = pops["epsilon"].to_numpy()
            eps_final = float(eps_vals[-1])
        else:
            eps_t, eps_vals, eps_final = np.array([]), np.array([]), np.nan

        # ---- Choose PPC path
        try:
            if args.fast_stored:
                sims_full = stored_ppc_from_db(h, t_max, T=len(timesteps_all), K=K)
                sims = sims_full[:, ::step, :]
                bands = summarize_bands(sims)
            elif args.hybrid is not None:
                Kp, Rp = args.hybrid
                bands = ppc_hybrid_parallel(
                    params_df, w, timesteps, stats, motion, speed,
                    total_steps=args.total_steps, K=Kp, R=Rp, workers=args.workers
                )
            else:
                bands = ppc_resim_parallel(
                    params_df, w, timesteps, stats, motion, speed,
                    total_steps=args.total_steps, pp=args.pp, workers=args.workers
                )
        except Exception as e:
            skipped.append(f"{db.name} (error: {e})")
            continue

        cov, disc_t = coverage_and_discrepancy(obs_mat, bands["med"], bands["q05"], bands["q95"])

        runs.append({
            "db": db,
            "label": f"{motion}/{speed}",
            "motion": motion, "speed": speed,
            "med": bands["med"], "q05": bands["q05"], "q95": bands["q95"],
            "N_pp": int(bands["N"]),
            "coverage": cov, "disc_t": disc_t,
            "eps_t": eps_t, "eps_vals": eps_vals, "eps_final": eps_final,
            # store posterior for parameter KDEs
            "params_df": params_df, "w": w,
        })
        used.append(f"{db.name} → {motion}/{speed}")

    # Log which files we used/skipped
    with open(outdir / "parse_log.txt", "w") as f:
        f.write("USED FILES:\n")
        for u in used: f.write(f"  {u}\n")
        if skipped:
            f.write("\nSKIPPED FILES:\n")
            for s in skipped: f.write(f"  {s}\n")

    if not runs:
        print("No valid runs to plot. See parse_log.txt for details.")
        sys.exit(1)

    # ---------- PLOT 0A: Epsilon trajectories (FIRST) ----------
    palette_models = sns.color_palette("tab10", n_colors=max(10, len(runs)))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for j, r in enumerate(runs):
        if len(r["eps_t"]) == 0:
            continue
        ax.plot(r["eps_t"], r["eps_vals"], marker="o", ms=3, lw=1.5,
                color=palette_models[j], label=r["label"])
    ax.set_xlabel("Population t")
    ax.set_ylabel("Epsilon")
    ax.set_title("Epsilon trajectories per method")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "00_epsilon_trajectories.png", dpi=200)
    plt.close(fig)

    # ---------- PLOT 0B: Final epsilon bar (SECOND) ----------
    df_eps = pd.DataFrame([{"label": r["label"], "eps_final": r["eps_final"]} for r in runs])
    df_eps = df_eps.sort_values("eps_final", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(df_eps["label"], df_eps["eps_final"], color="#6baed6")
    ax.set_xlabel("Final epsilon")
    ax.set_title("Final epsilon per method (lower is tighter)")
    for i, v in enumerate(df_eps["eps_final"].to_numpy()):
        ax.text(v, i, f" {v:.3g}", va="center", ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "01_epsilon_final_bar.png", dpi=200)
    plt.close(fig)

    # ---------- Posterior parameters (per motion) ----------
    post_dir = outdir / "posteriors"
    post_dir.mkdir(exist_ok=True, parents=True)

    # palette per speed (consistent across motions)
    speed_levels = args.speeds_order[:]
    speed_palette = dict(zip(speed_levels, sns.color_palette("tab10", n_colors=len(speed_levels))))

    summary_rows = []

    motions_present = sorted(set(r["motion"] for r in runs))
    for motion in motions_present:
        runs_m = [r for r in runs if r["motion"] == motion]
        param_union = set()
        for r in runs_m:
            param_union.update(list(r["params_df"].columns))
        for param in sorted(param_union):
            fig, ax = plt.subplots(figsize=(8.8, 4.8))
            plotted_any = False
            for sp in speed_levels:
                group = [r for r in runs_m if r["speed"] == sp and param in r["params_df"].columns]
                if len(group) == 0:
                    continue
                x_vals, w_vals = [], []
                for r in group:
                    arr = r["params_df"][param].to_numpy(float)
                    x_vals.append(arr)
                    w_vals.append(np.asarray(r["w"], float))
                    summary_rows.append({
                        "motion": r["motion"], "speed": r["speed"], "label": r["label"],
                        "parameter": param,
                        "median": float(np.median(arr)),
                        "q05": float(np.quantile(arr, 0.05)),
                        "q95": float(np.quantile(arr, 0.95)),
                    })
                x = np.concatenate(x_vals, axis=0)
                w = np.concatenate(w_vals, axis=0)
                _kde1d_tidy(ax, x, w, color=speed_palette.get(sp, None), label=sp)
                plotted_any = True
            if plotted_any:
                ax.set_title(f"Posterior comparison — motion={motion} — {param}")
                ax.set_xlabel(param)
                ax.set_ylabel("density")
                ax.legend(title="speed", ncols=2, fontsize=8)
                fig.tight_layout()
                fig.savefig(post_dir / f"02_post_{motion}_{param}.png", dpi=200)
            plt.close(fig)

    # ---------- NEW: Posterior overlays by parameter ACROSS ALL MODELS ----------
    # single plot per parameter: lines for ALL models (motion/speed labels)
    # palette across models (reuse palette_models mapping by label)
    label_list = [r["label"] for r in runs]
    label_to_color = {lab: palette_models[i % len(palette_models)] for i, lab in enumerate(label_list)}

    # union of all parameter names across all runs
    global_params = set()
    for r in runs:
        global_params.update(list(r["params_df"].columns))

    for param in sorted(global_params):
        fig, ax = plt.subplots(figsize=(10.0, 5.2))
        plotted_any = False
        for r in runs:
            if param not in r["params_df"].columns:
                continue
            x = r["params_df"][param].to_numpy(float)
            w = np.asarray(r["w"], float)
            _kde1d_tidy(ax, x, w, color=label_to_color[r["label"]], label=r["label"])
            plotted_any = True
        if plotted_any:
            ax.set_title(f"Posterior comparison across ALL models — {param}")
            ax.set_xlabel(param)
            ax.set_ylabel("density")
            ax.legend(title="model (motion/speed)", fontsize=8, ncols=2)
            fig.tight_layout()
            fig.savefig(post_dir / f"03_post_all_{param}.png", dpi=200)
        plt.close(fig)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(outdir / "posterior_param_summaries.csv", index=False)

    # ---------- PLOT 1: PPC GRID ----------
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
            if i == Kstats - 1: ax.set_xlabel("timestep")
            ax.set_ylabel(s if j == 0 else "")

    fig.suptitle("Posterior predictive comparison — grid", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "compare_grid.png", dpi=200)
    plt.close(fig)

    # ---------- PLOT 2: PPC overlays per stat ----------
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

    # ---------- PLOT 3: discrepancy overlays ----------
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

    # ---------- SUMMARY CSV ----------
    rows = []
    for r in runs:
        row = {
            "db": str(r["db"]),
            "motion": r["motion"], "speed": r["speed"],
            "label": r["label"],
            "epsilon_final": r["eps_final"], "N_pp": r["N_pp"],
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
    print(f"- Epsilon plots saved as 00_* and 01_* (appear first).")
    print(f"- Per-parameter overlays across ALL models saved as posteriors/03_post_all_<parameter>.png .")
    print(f"- See parse_log.txt for which DBs were used / skipped.")


if __name__ == "__main__":
    main()
