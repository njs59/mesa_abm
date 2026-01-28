
#!/usr/bin/env python3
"""
Compare behaviour across motion × speed models using PPCs + epsilon + posterior parameters,
for stats [S0, S1, S2, SSNND_med] only.
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


def make_model_factory(seed: int = 777):
    def factory(params_dict):
        return ClustersModel(params=params_dict, seed=seed)
    return factory


def parse_model_from_name(p: Path) -> Tuple[str, str]:
    name = p.name.lower()
    # Extract motion and speed tokens if present
    motion = "isotropic" if "isotropic" in name else ("persistent" if "persistent" in name else "unknown")
    speed = "unknown"
    for tok in ("constant", "lognorm", "gamma", "weibull"):
        if tok in name:
            speed = tok
            break
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


def stored_ppc_from_db(h: pyabc.History, t_max: int, T: int, K: int) -> np.ndarray:
    pop = h.get_population(t=t_max)
    sims = []
    for part in pop.particles:
        ss = part.sum_stat
        vec = np.array([ss[f"y_{i}"] for i in range(T * K)], dtype=float)
        sims.append(vec.reshape(T, K))
    return np.asarray(sims)


def summarize_bands(sims: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "med": np.median(sims, axis=0),
        "q05": np.quantile(sims, 0.05, axis=0),
        "q95": np.quantile(sims, 0.95, axis=0),
        "N": sims.shape[0],
    }


def _simulate_one(args):
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
    idxs = rng.choice(np.arange(N), size=pp, replace=True, p=w)
    full_order = ["S0", "S1", "S2", "SSNND_med"]
    col_idx = [full_order.index(s) for s in stats]
    jobs = [(i, params_df.iloc[idx].to_dict(), motion, speed, timesteps, total_steps, col_idx, seed * 1000)
            for i, idx in enumerate(idxs)]
    sims = np.empty((pp, len(timesteps), len(stats)), dtype=float)
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


def main():
    ap = argparse.ArgumentParser(description="Compare motion×speed models using epsilon, PPCs, and posterior parameters.")
    ap.add_argument("--db_glob", type=str, required=True, help="Glob for DBs, e.g. results/abc_*_seed*.db")
    ap.add_argument("--observed_ts", type=str, required=True)
    ap.add_argument("--pp", type=int, default=80, help="Posterior predictive draws (re-sim)")
    ap.add_argument("--total_steps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=os.cpu_count()//2 if os.cpu_count() else 2)
    ap.add_argument("--outdir", type=str, default="results/compare_models")
    ap.add_argument("--motions_order", nargs="+", default=["isotropic", "persistent"])
    ap.add_argument("--speeds_order", nargs="+", default=["constant", "lognorm", "gamma", "weibull"])
    ap.add_argument("--limit_models", type=int, default=None)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--fast_stored", action="store_true", help="Use stored final-population sum_stats (no re-sim)")
    ap.add_argument("--assume_motion", type=str, choices=["isotropic","persistent"], default=None)
    ap.add_argument("--assume_speed", type=str, choices=["constant","lognorm","gamma","weibull"], default=None)
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
    stats = ["S0", "S1", "S2", "SSNND_med"]
    for s in stats:
        if s not in obs_df.columns:
            raise ValueError(f"Missing {s} in observed CSV.")
    obs_mat_all = obs_df[stats].to_numpy(float)
    K = len(stats)

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

        pops = h.get_all_populations()
        if "epsilon" in pops.columns:
            eps_t = pops["t"].to_numpy()
            eps_vals = pops["epsilon"].to_numpy()
            eps_final = float(eps_vals[-1])
        else:
            eps_t, eps_vals, eps_final = np.array([]), np.array([]), np.nan

        try:
            if args.fast_stored:
                sims_full = stored_ppc_from_db(h, t_max, T=len(timesteps_all), K=K)
                sims = sims_full[:, ::step, :]
                bands = summarize_bands(sims)
            else:
                bands = ppc_resim_parallel(
                    params_df, w, timesteps, stats, motion, speed, total_steps=args.total_steps, pp=args.pp, workers=args.workers
                )
        except Exception as e:
            skipped.append(f"{db.name} (error: {e})")
            continue

        inside = (obs_mat >= bands["q05"]) & (obs_mat <= bands["q95"])  # T x K
        cov = inside.mean(axis=0)  # per stat
        disc_t = np.sqrt(np.sum((bands["med"] - obs_mat) ** 2, axis=1))

        runs.append({
            "db": db,
            "label": f"{motion}/{speed}",
            "motion": motion, "speed": speed,
            "med": bands["med"], "q05": bands["q05"], "q95": bands["q95"],
            "N_pp": int(bands["N"]),
            "coverage": cov, "disc_t": disc_t,
            "eps_t": eps_t, "eps_vals": eps_vals, "eps_final": eps_final,
            "params_df": params_df, "w": w,
        })
        used.append(f"{db.name} → {motion}/{speed}")

    with open(outdir / "parse_log.txt", "w") as f:
        f.write("USED FILES:")
        for u in used: f.write(f" {u}")
        if skipped:
            f.write("SKIPPED FILES:")
            for s in skipped: f.write(f" {s}")

    if not runs:
        print("No valid runs to plot. See parse_log.txt for details.")
        sys.exit(1)

    # Epsilon trajectories
    palette_models = sns.color_palette("tab10", n_colors=max(10, len(runs)))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for j, r in enumerate(runs):
        if len(r["eps_t"]) == 0:
            continue
        ax.plot(r["eps_t"], r["eps_vals"], marker="o", ms=3, lw=1.5, color=palette_models[j], label=r["label"])
    ax.set_xlabel("Population t")
    ax.set_ylabel("Epsilon")
    ax.set_title("Epsilon trajectories per method")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "00_epsilon_trajectories.png", dpi=200)
    plt.close(fig)

    # Final epsilon bar
    df_eps = pd.DataFrame([{ "label": r["label"], "eps_final": r["eps_final"] } for r in runs])
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

    # Posterior parameter overlays per motion and across all models
    post_dir = outdir / "posteriors"
    post_dir.mkdir(exist_ok=True, parents=True)

    speed_levels = args.speeds_order[:]
    speed_palette = dict(zip(speed_levels, sns.color_palette("tab10", n_colors=len(speed_levels))))

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

    label_list = [r["label"] for r in runs]
    label_to_color = {lab: palette_models[i % len(palette_models)] for i, lab in enumerate(label_list)}
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

    # PPC GRID
    Kstats = len(stats)
    C = len(runs)
    fig, axes = plt.subplots(nrows=Kstats, ncols=C, figsize=(min(2.6 * C, 26), min(2.2 * Kstats, 16)), sharex=True)
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

    # Overlays per stat
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

    # Discrepancy overlays
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, r in enumerate(runs):
        disc_t = np.sqrt(np.sum((r["med"] - obs_mat) ** 2, axis=1))
        ax.plot(t, disc_t, lw=1.5, color=palette_models[j], label=r["label"])
    ax.set_title("Median discrepancy vs time (Euclidean across stats)")
    ax.set_xlabel("timestep")
    ax.set_ylabel("discrepancy")
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "overlay_discrepancy.png", dpi=200)
    plt.close(fig)

    # Summary CSV
    rows = []
    for r in runs:
        row = {
            "db": str(r["db"]),
            "motion": r["motion"], "speed": r["speed"],
            "label": r["label"],
            "epsilon_final": r["eps_final"], "N_pp": r["N_pp"],
        }
        for s_idx, s in enumerate(stats):
            row[f"coverage_{s}"] = float(r["coverage"][s_idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "summary_per_model.csv", index=False)

    # Optional export epsilon trajectories
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
