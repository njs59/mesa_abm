#!/usr/bin/env python3
"""
Analyze motion-grid runs at a COMMON population index, overlay vs data, and plot diagnostics.

This version adds flexible input sources in addition to --summary:
  • --glob: one or more glob patterns for .db files
  • --db: explicit .db files
(You can still pass --summary with a CSV produced earlier.)

When .dbs are provided via --glob/--db, the script parses dataset/motion/speed_dist/seed
from filenames using a format-style pattern (default: {dataset}_{motion}_{speed_dist}_s{seed}.db).
It fills missing fields with sensible defaults and adds columns needed by the analysis.

Usage examples
--------------
# Old (summary-based)
python analyze_motiongrid_glob.py \
  --summary motiongrid_pkg/results/summary.csv \
  --obs_csv INV_summary_stats.csv \
  --dataset INV \
  --outdir motiongrid_pkg/analysis_results \
  --pp_samples 100

# New (glob-based)
python analyze_motiongrid_glob.py \
  --glob "motiongrid_pkg/results/INV_isotropic_lognorm_s*.db" \
        "motiongrid_pkg/results/INV_persistent_lognorm_s*.db" \
  --obs_csv INV_summary_stats.csv \
  --dataset INV \
  --outdir motiongrid_pkg/analysis_results \
  --pp_samples 100 \
  --start_step_default 20

# New (explicit files)
python analyze_motiongrid_glob.py \
  --db motiongrid_pkg/results/INV_isotropic_gamma_s42.db \
      motiongrid_pkg/results/INV_persistent_gamma_s123.db \
  --obs_csv INV_summary_stats.csv \
  --dataset INV \
  --outdir motiongrid_pkg/analysis_results
"""
import argparse
import re
import glob as globlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pyabc

# --- Import your ABM and defaults ---
from clusters_abm.clusters_model import ClustersModel
from clusters_abm.utils import DEFAULTS

# ---------------------- Summary stats ----------------------
def compute_summary_from_model(model):
    ids, pos, radii, sizes, speeds = model._log_alive_snapshot()
    n = len(sizes)
    if n == 0:
        return 0.0, 0.0, 0.0
    return float(n), float(np.mean(sizes)), float(np.mean(sizes ** 2))

def simulate_timeseries(params: dict, steps: int, seed: int) -> np.ndarray:
    model = ClustersModel(params=params, seed=seed)
    out = np.zeros((steps, 3), dtype=float)
    out[0, :] = compute_summary_from_model(model)
    for t in range(1, steps):
        model.step()
        out[t, :] = compute_summary_from_model(model)
    return out

# ---------------------- Parameter mapping ----------------------
def _set_nested(base: dict, dotted: str, value):
    keys = dotted.split(".")
    d = base
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value

def build_speed_params(speed_dist: str, particle: dict) -> dict:
    if speed_dist == "constant":
        return {}
    if speed_dist == "lognorm":
        mu = float(particle.get("speed_meanlog", 1.0))
        sd = float(particle.get("speed_sdlog", 0.7))
        return {"s": sd, "scale": float(np.exp(mu))}
    elif speed_dist == "gamma":
        shape = float(particle.get("speed_shape", 2.0))
        scale = float(particle.get("speed_scale", 1.0))
        return {"a": shape, "scale": scale}
    elif speed_dist == "weibull":
        shape = float(particle.get("speed_shape", 2.0))
        scale = float(particle.get("speed_scale", 2.0))
        return {"c": shape, "scale": scale}
    else:
        raise ValueError(f"Unknown speed_dist: {speed_dist}")

def make_params_from_particle(defaults: dict, particle: dict, motion: str, speed_dist: str, fixed_n_clusters: int) -> dict:
    params = {
        "space": dict(defaults["space"]),
        "time": dict(defaults["time"]),
        "physics": dict(defaults["physics"]),
        "phenotypes": {
            "proliferative": dict(defaults["phenotypes"]["proliferative"]),
            "invasive": dict(defaults["phenotypes"]["invasive"]),
        },
        "merge": dict(defaults["merge"]),
        "init": dict(defaults["init"]),
        "movement": dict(defaults["movement"]),
    }
    params["movement"]["direction"] = motion
    if speed_dist == "constant":
        params["movement"]["mode"] = "constant"
        params["movement"].pop("distribution", None)
        params["movement"].pop("dist_params", None)
    else:
        params["movement"]["mode"] = "distribution"
        params["movement"]["distribution"] = speed_dist
        params["movement"]["dist_params"] = build_speed_params(speed_dist, particle)
    if motion == "persistent":
        hs = float(particle.get("heading_sigma", params["movement"].get("heading_sigma", 0.25)))
        params["movement"]["heading_sigma"] = max(0.0, hs)
    else:
        params["movement"].pop("heading_sigma", None)

    mapping = {
        "prolif_rate": "phenotypes.proliferative.prolif_rate",
        "adhesion": "phenotypes.proliferative.adhesion",
        "fragment_rate": "phenotypes.proliferative.fragment_rate",
        "merge_prob": "merge.prob_contact_merge",
    }
    for k, v in particle.items():
        if k.startswith("speed_") or k == "heading_sigma":
            continue
        if ("rate" in k or "prob" in k) and v < 0:
            v = 0.0
        if k in mapping:
            _set_nested(params, mapping[k], float(v))
        else:
            try:
                params[k] = float(v)
            except Exception:
                pass
    params["init"]["phenotype"] = "proliferative"
    params["init"]["n_clusters"] = int(max(1, round(fixed_n_clusters)))
    return params

# ---------------------- Robust distribution fetch ----------------------
def robust_distribution(history: pyabc.History, t_target: int):
    t = min(int(t_target), int(history.max_t))
    while t >= 0:
        df, w = history.get_distribution(m=0, t=t)
        if len(df) > 0:
            return df, w, t
        t -= 1
    raise RuntimeError("No non-empty populations found in history.")

# ---------------------- Posterior predictive at given t ----------------------
def posterior_predictive_at_t(
    db_path: Path,
    obs: np.ndarray,
    start_step: int,
    motion: str,
    speed_dist: str,
    fixed_n_clusters: int,
    t_pop: int,
    n_sims: int = 50,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    history = pyabc.History(f"sqlite:///{db_path.resolve()}")
    df, w, t_used = robust_distribution(history, t_pop)
    w = np.asarray(w, dtype=float)
    w = w / (w.sum() if w.sum() > 0 else 1.0)
    idx = np.arange(len(df))
    draws = rng.choice(idx, size=min(n_sims, len(df)), replace=False, p=w)
    T = obs.shape[0]
    sims = np.zeros((len(draws), T, 3), dtype=float)
    for j, i in enumerate(draws):
        particle = {k: float(df.iloc[i][k]) for k in df.columns}
        params = make_params_from_particle(DEFAULTS, particle, motion=motion, speed_dist=speed_dist, fixed_n_clusters=fixed_n_clusters)
        seg = simulate_timeseries(params, steps=start_step + T, seed=int(rng.integers(0, 2**31 - 1)))[start_step : start_step + T, :]
        sims[j, :, :] = seg
    med = np.median(sims, axis=0)
    q5 = np.quantile(sims, 0.05, axis=0)
    q95 = np.quantile(sims, 0.95, axis=0)
    return med, q5, q95, t_used

# ---------------------- Helpers for --glob/--db ----------------------
DEFAULT_NAME_PATTERN = r"{dataset}_{motion}_{speed_dist}_s{seed}.db"

def make_regex_from_pattern(pattern: str) -> re.Pattern:
    esc = re.escape(pattern)
    subs = {
        re.escape("{dataset}"): r"(?P<dataset>[^_/\\-]+)",
        re.escape("{motion}"): r"(?P<motion>[^_/\\-]+)",
        re.escape("{speed_dist}"): r"(?P<speed_dist>[^_/\\-]+)",
        re.escape("{seed}"): r"(?P<seed>\\d+)",
    }
    for k, v in subs.items():
        esc = esc.replace(k, v)
    return re.compile(r"^" + esc + r"$")

def parse_from_name(name: str, pat: re.Pattern) -> dict:
    m = pat.match(name)
    if not m:
        return {}
    g = m.groupdict()
    if "seed" in g and g["seed"] is not None:
        try:
            g["seed"] = int(g["seed"]) 
        except Exception:
            pass
    return g

def select_db_paths(globs, dbs, include_hidden=False):
    seen = set()
    out = []
    for gp in globs or []:
        for match in globlib.glob(gp, recursive=True):
            p = Path(match)
            if p.suffix != ".db":
                continue
            if not include_hidden and any(part.startswith('.') for part in p.parts):
                continue
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp); out.append(rp)
    for f in dbs or []:
        p = Path(f)
        if p.exists() and p.suffix == ".db":
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp); out.append(rp)
    return out

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser(description="Analyze motion-grid runs at a common population, overlay vs data, and plot diagnostics (supports --glob/--db or --summary).")
    ap.add_argument("--summary", type=str, default=None, help="Path to summary.csv (optional if using --glob/--db)")
    ap.add_argument("--glob", nargs="*", default=None, help="One or more glob patterns for .db files")
    ap.add_argument("--db", nargs="*", default=None, help="Explicit .db files to include")
    ap.add_argument("--pattern", type=str, default=DEFAULT_NAME_PATTERN, help="Filename pattern for parsing fields when using --glob/--db")
    ap.add_argument("--obs_csv", type=str, required=True, help="Path to observed CSV with S0,S1,S2")
    ap.add_argument("--dataset", type=str, default="INV")
    ap.add_argument("--outdir", type=str, default="analysis_results")
    ap.add_argument("--pp_samples", type=int, default=100)
    ap.add_argument("--variants", nargs="*", default=None, help="Optional list of variants (e.g. isotropic_gamma persistent_lognorm)")
    ap.add_argument("--start_step_default", type=int, default=20, help="Start step to use when rows lack start_step (glob/db mode)")
    ap.add_argument("--init_cells_fixed", type=int, default=0, help="If >0, override initial cluster count (glob/db mode)")
    ap.add_argument("--include_hidden", action="store_true", help="Include hidden files/directories for --glob")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Observed data
    obs_df = pd.read_csv(args.obs_csv)
    if not all(c in obs_df.columns for c in ("S0", "S1", "S2")):
        raise ValueError("Observed CSV must contain S0,S1,S2")
    obs = obs_df[["S0", "S1", "S2"]].to_numpy(dtype=float)

    # Build the working DataFrame "df" either from summary or from globs/dbs
    df = None
    results_dir = None

    if args.summary:
        summary_path = Path(args.summary).resolve()
        results_dir = summary_path.parent
        df = pd.read_csv(summary_path)
        if "variant" not in df.columns:
            df["variant"] = df["motion"] + "_" + df["speed_dist"]
        # Filter dataset
        if "dataset" in df.columns:
            df = df[df["dataset"] == args.dataset].copy()
        if args.variants:
            df = df[df["variant"].isin(args.variants)].copy()
        if df.empty:
            raise RuntimeError("No rows selected from summary for the given dataset/variants.")
    else:
        # Collect .db files
        paths = select_db_paths(args.glob, args.db, include_hidden=args.include_hidden)
        if not paths:
            raise RuntimeError("No .db files selected. Provide --summary or --glob/--db.")
        pat = make_regex_from_pattern(args.pattern)
        rows = []
        for p in paths:
            meta = parse_from_name(p.name, pat)
            dataset = meta.get("dataset") or args.dataset
            motion = meta.get("motion")
            speed = meta.get("speed_dist")
            if args.variants is not None:
                # If variants filter is present, require motion and speed to be parsed
                if motion is None or speed is None:
                    continue
                if f"{motion}_{speed}" not in set(args.variants):
                    continue
            rows.append({
                "dataset": dataset,
                "motion": motion,
                "speed_dist": speed,
                "variant": (f"{motion}_{speed}" if motion and speed else None),
                "db": str(p),  # absolute path OK here
                # Provide defaults used later
                "start_step": args.start_step_default,
            })
        df = pd.DataFrame(rows)
        # Filter dataset
        df = df[df["dataset"] == args.dataset].copy()
        if df.empty:
            raise RuntimeError("No rows after filtering by dataset/variants.")

    # Ensure n_populations for each row (in case it's missing)
    if "n_populations" not in df.columns or df["n_populations"].isna().any():
        n_pops = []
        for _, row in df.iterrows():
            db = Path(row["db"]) if results_dir is None else (results_dir / row["db"])  # summary uses relative
            hist = pyabc.History(f"sqlite:///{db.resolve()}")
            try:
                pops = hist.get_all_populations()
                n_pops.append(int(pops.shape[0]))
            except Exception:
                n_pops.append(int(getattr(hist, "n_populations", 0)))
        df["n_populations"] = n_pops

    # COMMON population index (zero-based). Match prior script's min-2 logic.
    min_pops = int(df["n_populations"].min())
    t_common = max(0, min_pops - 2)
    print(f"\nComparing all runs at COMMON population t = {t_common} (min of n_populations - 2).")

    # ------------- Epsilon trajectories -------------
    plt.figure(figsize=(8, 5))
    for _, row in df.iterrows():
        db = Path(row["db"]) if results_dir is None else (results_dir / row["db"])  # absolute or relative
        if not db.exists():
            raise FileNotFoundError(f"Cannot find DB: {db}")
        history = pyabc.History(f"sqlite:///{db.resolve()}")
        pops = history.get_all_populations()
        eps = pops["epsilon"].to_numpy()
        label = row.get("variant") or f"{row.get('motion')}_{row.get('speed_dist')}"
        plt.plot(np.arange(len(eps)), eps, marker="o", lw=1.5, label=label)
    plt.xlabel("Population t"); plt.ylabel("ε (IQR-scaled)")
    plt.title(f"Epsilon trajectories ({args.dataset})")
    plt.legend(ncol=2, fontsize=8); plt.tight_layout()
    plt.savefig(outdir / f"{args.dataset}_epsilon_trajectories.png", dpi=160); plt.close()

    # ------------- Posterior predictive medians/bands -------------
    colours = {"isotropic": "#1f77b4", "persistent": "#ff7f0e"}
    store = {}
    stats_names = ["S0", "S1", "S2"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    T = obs.shape[0]; x = np.arange(T)
    used_ts = []

    for _, row in tqdm(df.iterrows(), desc="Posterior predictive (common population)", total=len(df)):
        db = Path(row["db"]) if results_dir is None else (results_dir / row["db"])  # absolute or relative
        motion, speed = row.get("motion"), row.get("speed_dist")
        # start_step: from row if available, else default
        start_step = int(row["start_step"]) if "start_step" in row and not pd.isna(row["start_step"]) else int(args.start_step_default)
        # initial clusters: from row if available, else override or derive from obs
        if "init_cells" in row and not pd.isna(row["init_cells"]):
            fixed_n = int(row["init_cells"])
        else:
            fixed_n = int(args.init_cells_fixed) if args.init_cells_fixed > 0 else int(max(1, round(obs[0, 0])))

        med, q5, q95, t_used = posterior_predictive_at_t(
            db_path=db, obs=obs, start_step=start_step, motion=motion, speed_dist=speed,
            fixed_n_clusters=fixed_n, t_pop=t_common, n_sims=args.pp_samples
        )
        used_ts.append(((row.get("variant") or f"{motion}_{speed}"), t_used))
        store[(row.get("variant") or f"{motion}_{speed}")] = (med, q5, q95, motion)

        for k, ax in enumerate(axes):
            ax.plot(x, med[:, k], lw=1.6, label=(row.get("variant") or f"{motion}_{speed}"), color=colours.get(motion, "grey"))

    # Overlay observed data
    for k, ax in enumerate(axes):
        ax.plot(x, obs[:, k], lw=2.0, color="black", label="Observed")
        ax.set_title(stats_names[k]); ax.set_xlabel("t (aligned steps)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Value")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle(f"Overlay vs data (common population t={t_common}, with per-run fallback if needed) — {args.dataset}")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(outdir / f"{args.dataset}_overlay_vs_data_common_t{t_common}.png", dpi=160)
    plt.close()

    # Print fallbacks
    fallbacks = [(v, tu) for v, tu in used_ts if tu != t_common]
    if fallbacks:
        print("\n⚠️ Some runs had no particles at t_common; fell back to:")
        for v, tu in fallbacks:
            print(f" - {v}: used t={tu}")

    # ------------- ε at common t (bar) -------------
    eps_at_common = []
    for _, row in df.iterrows():
        db = Path(row["db"]) if results_dir is None else (results_dir / row["db"])  # absolute or relative
        history = pyabc.History(f"sqlite:///{db.resolve()}")
        pops = history.get_all_populations()
        t_use = min(t_common, len(pops) - 1)
        label = row.get("variant") or f"{row.get('motion')}_{row.get('speed_dist')}"
        eps_at_common.append((label, float(pops.iloc[t_use]["epsilon"])) )
    eps_df = pd.DataFrame(eps_at_common, columns=["variant", "epsilon_common"])
    order = eps_df.sort_values("epsilon_common")["variant"]
    plt.figure(figsize=(9, 5))
    colours_bars = [colours.get(v.split("_")[0], "grey") for v in order]
    plt.bar(order, eps_df.set_index("variant").loc[order, "epsilon_common"], color=colours_bars)
    plt.axhline(0.30, color="green", ls="--", lw=1, label="Excellent ≤ 0.30")
    plt.axhline(0.50, color="orange", ls="--", lw=1, label="Good 0.31–0.50")
    plt.axhline(0.85, color="red", ls="--", lw=1, label="Acceptable 0.51–0.85")
    plt.title(f"ε at common population t={t_common} — {args.dataset}")
    plt.ylabel("ε (IQR-scaled)"); plt.xticks(rotation=45, ha="right"); plt.legend()
    plt.tight_layout(); plt.savefig(outdir / f"{args.dataset}_epsilon_bar_common_t{t_common}.png", dpi=160); plt.close()

    # ------------- Coverage bars (only if summary had coverage columns) -------------
    if all(c in df.columns for c in ("coverage_S0", "coverage_S1", "coverage_S2")):
        plt.figure(figsize=(10, 5))
        idx = np.arange(len(df)); w = 0.28
        plt.bar(idx - w, df["coverage_S0"], width=w, label="S0")
        plt.bar(idx,      df["coverage_S1"], width=w, label="S1")
        plt.bar(idx + w,  df["coverage_S2"], width=w, label="S2")
        tick_labels = [r.get("variant") if isinstance(r, dict) else v for v, r in zip(df.get("variant", df.index), df.to_dict("records"))]
        plt.xticks(idx, tick_labels, rotation=45, ha="right")
        plt.ylim(0, 1.05); plt.ylabel("Coverage (fraction within 5–95% band)")
        plt.title(f"Posterior predictive coverage (final-pop summary) — {args.dataset}")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / f"{args.dataset}_coverage_bars_summary.png", dpi=160); plt.close()

    # ------------- QQ plots for top-2 variants -------------
    def qq_plot(ax, model_series: np.ndarray, obs_series: np.ndarray, label: str = ""):
        q = np.linspace(0, 1, 1000)
        mq = np.quantile(model_series, q)
        oq = np.quantile(obs_series, q)
        ax.plot(oq, mq, lw=1.5, label=label)
        ax.plot([oq.min(), oq.max()], [oq.min(), oq.max()], color="grey", lw=1, ls="--")

    top2 = order.tolist()[:2]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for k, ax in enumerate(axes):
        for v in top2:
            med, q5, q95, motion = store[v]
            qq_plot(ax, med[:, k], obs[:, k], label=v)
        ax.set_title(f"QQ: {'S0 S1 S2'.split()[k]}")
        ax.set_xlabel("Observed quantiles"); ax.set_ylabel("Model quantiles")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"QQ plots (median vs observed) at common/fallback t — {args.dataset}")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(outdir / f"{args.dataset}_qq_common_fallback.png", dpi=160); plt.close()

    print(f"\nSaved figures to: {outdir.resolve()}")
    for p in sorted(outdir.glob("*.png")):
        print(" -", p.name)

if __name__ == "__main__":
    main()
