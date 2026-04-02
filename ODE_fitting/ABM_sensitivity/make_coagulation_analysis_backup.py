#!/usr/bin/env python3
"""
Coagulation analysis using the TRUE rule:

    • Any cluster ID present at timestep t but missing at timestep t+1
      indicates a coagulation event.

Splitting creates NEW IDs but does NOT remove the parent ID → not counted.

This script:
    - Reads raw per-timestep CSVs (t_XXXX.csv)
    - Computes coagulation per repeat
    - Aggregates across repeats (mean + smoothing)
    - Plots per-condition curves + GIFs
    - Produces global comparison plot
    - If 1 param varies → 1D sweep overlay + GIF over time
    - If 2 params vary → 2D heatmap + GIF over time
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio   # Removes deprecation warning
import re
from typing import List, Dict, Tuple, Optional


# -------------------------------------------------------------
# Utility: Parse parameters from scenario folder names
# -------------------------------------------------------------
_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?([eE]-?\d+)?$")
_P_DECIMAL_RE = re.compile(r"^-?\d+p\d+([eE]-?\d+)?$")

def _parse_value(tok: str):
    t = tok.strip()
    if _P_DECIMAL_RE.fullmatch(t):
        t = t.replace("p", ".")
    if _NUMERIC_RE.fullmatch(t):
        try:
            return float(t)
        except Exception:
            return t
    return t

def parse_condition_parameters(name: str) -> Dict[str, object]:
    params = {}
    parts = name.split("__")
    for p in parts:
        if "_" not in p:
            continue
        key, rest = p.split("_", 1)
        params[key] = _parse_value(rest)
    return params


# -------------------------------------------------------------
# Load timestep CSVs
# -------------------------------------------------------------
def load_timestep(fp: Path) -> pd.DataFrame:
    # CSV schema: id,x,y,size  (validated from your example)
    df = pd.read_csv(fp)
    df["id"] = df["id"].astype(str)
    return df[["id"]]   # we only need IDs for coagulation


def list_timesteps(repeat_dir: Path) -> List[Path]:
    files = sorted(repeat_dir.glob("t_*.csv"),
                   key=lambda f: int(re.findall(r"t_(\d+).csv", f.name)[0]))
    return files


# -------------------------------------------------------------
# TRUE coagulation logic (your exact rule)
# -------------------------------------------------------------
def detect_coag_by_id_loss(prev_ids: set[str], curr_ids: set[str]) -> int:
    """
    Number of lost IDs = coagulation events.
    """
    return len(prev_ids - curr_ids)


# -------------------------------------------------------------
# Build per-repeat time series
# -------------------------------------------------------------
def per_repeat_series(repeat_dir: Path, smooth_window: int) -> pd.DataFrame:
    ts_files = list_timesteps(repeat_dir)
    if not ts_files:
        raise RuntimeError(f"No timestep CSVs in {repeat_dir}")

    rows = []

    # Load first step
    first = load_timestep(ts_files[0])
    prev_ids = set(first["id"])
    prev_step = int(re.findall(r"t_(\d+).csv", ts_files[0].name)[0])

    rows.append({"step": prev_step, "P": 0.0})

    # Iterate
    for fp in ts_files[1:]:
        curr = load_timestep(fp)
        curr_ids = set(curr["id"])
        step = int(re.findall(r"t_(\d+).csv", fp.name)[0])

        lost = len(prev_ids - curr_ids)
        nC = len(curr_ids)
        P = lost / max(1, nC)

        rows.append({"step": step, "P": P})
        prev_ids = curr_ids

    df = pd.DataFrame(rows)

    # Smooth
    df["P_smooth"] = df["P"].rolling(window=smooth_window, min_periods=1, center=True).mean()

    # Return ONLY step, P, P_smooth
    return df[["step", "P", "P_smooth"]]

# -------------------------------------------------------------
# Aggregate repeats (average)
# -------------------------------------------------------------
def aggregate_repeats(repeats: List[pd.DataFrame]) -> pd.DataFrame:
    # Join on step (outer) then average
    merged = None
    for i, df in enumerate(repeats):
        df_i = df.set_index("step").rename(columns={
            "P": f"P_r{i}",
            "P_smooth": f"P_s_r{i}"
        })
        merged = df_i if merged is None else merged.join(df_i, how="outer")

    merged = merged.sort_index()

    P_cols = [c for c in merged.columns if c.startswith("P_r")]
    Ps_cols = [c for c in merged.columns if c.startswith("P_s_r")]

    agg = pd.DataFrame(index=merged.index)
    agg["P_mean"] = merged[P_cols].mean(axis=1)
    agg["P_smooth_mean"] = merged[Ps_cols].mean(axis=1)

    agg = agg.reset_index().rename(columns={"index": "step"})
    return agg


# -------------------------------------------------------------
# Basic plotting helpers
# -------------------------------------------------------------
def plot_curve(df: pd.DataFrame, out: Path, title: str):
    plt.figure(figsize=(10,5))
    plt.plot(df["step"], df["P_mean"], alpha=0.35, label="Raw mean")
    plt.plot(df["step"], df["P_smooth_mean"], lw=2, label="Smoothed")
    plt.xlabel("Timestep")
    plt.ylabel("Coagulation probability")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def make_curve_gif(df: pd.DataFrame, out: Path, title: str):
    tmp = out.parent / "_gif_frames"
    tmp.mkdir(exist_ok=True)

    ymax = max(1e-6, df["P_mean"].max()*1.1)

    frames = []
    for i in range(1, len(df)+1):
        plt.figure(figsize=(8,4))
        plt.plot(df["step"][:i], df["P_mean"][:i], color="blue")
        plt.ylim(0, ymax)
        plt.title(f"{title} — t={df['step'].iloc[i-1]}")
        fp = tmp / f"frame_{i:04d}.png"
        plt.savefig(fp, dpi=120)
        plt.close()
        frames.append(imageio.imread(fp))

    imageio.mimsave(out, frames, duration=0.04)


# -------------------------------------------------------------
# Sweep-detection helpers
# -------------------------------------------------------------
def detect_varying_params(param_rows: List[Dict[str, object]]) -> List[str]:
    df = pd.DataFrame(param_rows).fillna("NA")
    cols = [c for c in df.columns if c != "scenario"]
    return [c for c in cols if df[c].nunique(dropna=True)>1]

def numeric_or_cat(values: List[object]):
    s = pd.Series(values)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.8:
        return sorted(num.dropna().unique().tolist()), True
    return sorted(s.astype(str).unique().tolist()), False


# -------------------------------------------------------------
# 1D sweep (single parameter varies)
# -------------------------------------------------------------
def plot_1d_overlay(scen_records, param, outpath: Path):
    """Overlay of smoothed curves, coloured by the varying parameter value."""
    vals = [rec["params"].get(param, "NA") for rec in scen_records]
    levels, is_num = numeric_or_cat(vals)
    cmap = plt.get_cmap("viridis")

    plt.figure(figsize=(11, 7))
    ax = plt.gca()

    # Colour mapping ------------------------------------------
    if is_num:
        numeric_vals = [float(v) for v in vals]
        vmin, vmax = min(numeric_vals), max(numeric_vals)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colours = [cmap(norm(v)) for v in numeric_vals]

        # This scalar mappable MUST be tied to the axes
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

    else:
        cats = levels
        n = max(1, len(cats) - 1)
        colour_map = {c: cmap(i / n) for i, c in enumerate(cats)}
        colours = [colour_map[str(v)] for v in vals]
        sm = None  # no mappable for categorical

    # Plot lines ----------------------------------------------
    for rec, col in zip(scen_records, colours):
        df = rec["agg"]
        ax.plot(df["step"], df["P_smooth_mean"], lw=1.8, color=col)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Smoothed coagulation probability")
    ax.set_title(f"1D Sweep — coloured by {param}")

    # Colourbar or legend --------------------------------------
    if is_num:
        # Explicitly bind the colourbar to THIS axes
        plt.colorbar(sm, ax=ax, label=param)
    else:
        # Categorical legend
        handles = [
            plt.Line2D([0], [0], lw=2, color=colour_map[c], label=str(c))
            for c in levels
        ]
        ax.legend(handles=handles, title=param, fontsize=8, ncol=1)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_1d_overlay_metric(scen_records, param, outpath, metric_col, ylabel, title_suffix):
    """
    Generic 1D sweep overlay for any smoothed metric.
    Produces coloured-by-parameter curves over time, just like the original per‑cluster overlay.
    """
    vals = [rec["params"].get(param, "NA") for rec in scen_records]
    levels, is_num = numeric_or_cat(vals)

    cmap = plt.get_cmap("viridis")
    plt.figure(figsize=(11, 7))
    ax = plt.gca()

    # --- Colour mapping ---
    if is_num:
        numeric_vals = [float(v) for v in vals]
        vmin, vmax = min(numeric_vals), max(numeric_vals)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colours = [cmap(norm(v)) for v in numeric_vals]
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
    else:
        cats = levels
        n = max(1, len(cats) - 1)
        colour_map = {c: cmap(i / n) for i, c in enumerate(cats)}
        colours = [colour_map[str(v)] for v in vals]
        sm = None

    # --- Plot curves ---
    for rec, col in zip(scen_records, colours):
        df = rec["agg"]
        ax.plot(df["step"], df[metric_col], lw=1.8, color=col)

    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.set_title(f"1D Sweep ({title_suffix}) — coloured by {param}")

    # --- Colourbar or legend ---
    if is_num:
        plt.colorbar(sm, ax=ax, label=param)
    else:
        handles = [
            plt.Line2D([0], [0], lw=2, color=colour_map[c], label=str(c))
            for c in levels
        ]
        ax.legend(handles=handles, title=param, fontsize=8, ncol=1)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def make_1d_sweep_gif(scen_records, param, outdir: Path):
    vals = [rec["params"].get(param, "NA") for rec in scen_records]
    levels, is_num = numeric_or_cat(vals)

    # Build map param -> smoothed series
    all_steps = sorted(set(int(s) for rec in scen_records for s in rec["agg"]["step"].tolist()))
    series_map = {}

    global_min, global_max = np.inf, -np.inf
    for rec in scen_records:
        v = rec["params"].get(param, "NA")
        s = rec["agg"].set_index("step")["P_smooth_mean"].reindex(all_steps)
        series_map[v] = s
        if np.any(~np.isnan(s.values)):
            global_min = min(global_min, float(np.nanmin(s.values)))
            global_max = max(global_max, float(np.nanmax(s.values)))

    tmp = outdir / "_gif_frames_1D"
    tmp.mkdir(exist_ok=True)

    if is_num:
        xvals = [float(v) for v in levels]
        xticklabels = None
    else:
        xvals = list(range(len(levels)))
        xticklabels = levels

    frames = []

    for t in all_steps:
        y = [series_map[v].loc[t] if v in series_map else np.nan for v in levels]

        plt.figure(figsize=(8,4.5))
        plt.plot(xvals, y, "-o", lw=1.6)
        plt.ylim(global_min-1e-4, global_max+1e-4)
        plt.xlabel(param)
        plt.ylabel("Smoothed coagulation probability")
        plt.title(f"{param} sweep — t={t}")

        if xticklabels:
            plt.xticks(xvals, xticklabels, rotation=30, ha="right")

        plt.tight_layout()
        fp = tmp / f"frame_{int(t):05d}.png"
        plt.savefig(fp, dpi=120)
        plt.close()
        frames.append(imageio.imread(fp))

    imageio.mimsave(outdir / f"coagulation_1D_{param}_over_time.gif",
                    frames, duration=0.07)


# -------------------------------------------------------------
# 2D sweep (two params vary)
# -------------------------------------------------------------
def make_2d_heatmap_gif(scen_records, p1, p2, outdir: Path):
    # Build grid
    p1_vals = [rec["params"].get(p1, "NA") for rec in scen_records]
    p2_vals = [rec["params"].get(p2, "NA") for rec in scen_records]
    p1_levels, _ = numeric_or_cat(p1_vals)
    p2_levels, _ = numeric_or_cat(p2_vals)

    all_steps = sorted(set(int(s) for rec in scen_records for s in rec["agg"]["step"].tolist()))

    series_map = {}  # (p1,p2) -> smoothed series
    global_min, global_max = np.inf, -np.inf
    for rec in scen_records:
        v1 = rec["params"].get(p1, "NA")
        v2 = rec["params"].get(p2, "NA")
        s = rec["agg"].set_index("step")["P_smooth_mean"].reindex(all_steps)
        series_map[(v1, v2)] = s
        if np.any(~np.isnan(s.values)):
            global_min = min(global_min, float(np.nanmin(s.values)))
            global_max = max(global_max, float(np.nanmax(s.values)))

    tmp = outdir / "_gif_frames_2D"
    tmp.mkdir(exist_ok=True)

    frames = []

    for t in all_steps:
        grid = np.full((len(p1_levels), len(p2_levels)), np.nan)
        for i,pv1 in enumerate(p1_levels):
            for j,pv2 in enumerate(p2_levels):
                grid[i,j] = series_map.get((pv1,pv2), pd.Series(np.nan)).loc[t]

        plt.figure(figsize=(7.5,6))
        im = plt.imshow(grid, origin="lower", cmap="viridis",
                        vmin=global_min, vmax=global_max, aspect="auto")
        plt.colorbar(im, label="Smoothed coagulation probability")
        plt.xlabel(p2)
        plt.ylabel(p1)
        plt.title(f"2D Sweep — t={t}")
        plt.xticks(range(len(p2_levels)), p2_levels)
        plt.yticks(range(len(p1_levels)), p1_levels)
        plt.tight_layout()

        fp = tmp / f"frame_{t:05d}.png"
        plt.savefig(fp, dpi=120)
        plt.close()
        frames.append(imageio.imread(fp))

    imageio.mimsave(outdir / "coagulation_2D_heatmap_over_time.gif",
                    frames, duration=0.07)


# -------------------------------------------------------------
# Main per-run processing
# -------------------------------------------------------------
def process_run(run_dir: Path,
                smooth_window: int):

    sim_root = run_dir / "simulations"
    if not sim_root.exists():
        print(f"[WARN] No simulations folder in {run_dir}")
        return

    outroot = run_dir / "coagulation_analysis_tagged"
    outroot.mkdir(exist_ok=True)

    scen_repeats = []      # For global overlay, 1D/2D sweeps
    scen_final = []        # For global bar + sweep detection

    for scen_dir in sorted(sim_root.iterdir()):
        if not scen_dir.is_dir():
            continue

        name = scen_dir.name
        print(f"  Scenario: {name}")

        repeat_dirs = sorted(scen_dir.glob("repeat_*"))
        if not repeat_dirs:
            print("    No repeats.")
            continue

        per_rep = []
        for rd in repeat_dirs:
            try:
                df = per_repeat_series(rd, smooth_window)
                per_rep.append(df)
            except Exception as e:
                print("    Error:", e)

        if not per_rep:
            continue

        agg = aggregate_repeats(per_rep)
        outdir = outroot / name
        outdir.mkdir(exist_ok=True)

        # Save per-condition
        agg.to_csv(outdir / "coag_prob.csv", index=False)
        plot_curve(agg, outdir / "coag_prob.png", name)
        make_curve_gif(agg, outdir / "coag_prob.gif", name)

        params = parse_condition_parameters(name)
        scen_repeats.append({"scenario": name,
                             "params": params,
                             "agg": agg})

        scen_final.append({**params,
                           "scenario": name,
                           "final": float(agg["P_smooth_mean"].iloc[-1])})


    # ----- Global comparison -----
    if scen_final:
        df = pd.DataFrame(scen_final).sort_values("final")
        plt.figure(figsize=(11, max(4,0.25*len(df))))
        plt.barh(df["scenario"], df["final"])
        plt.xlabel("Final smoothed coagulation probability")
        plt.title("Final coagulation probability")
        plt.tight_layout()
        plt.savefig(outroot / "global_final_prob.png", dpi=200)
        plt.close()

    # ----- Sweep detection -----
    if not scen_repeats:
        return

    param_rows = []
    for r in scen_repeats:
        row = {"scenario": r["scenario"]}
        row.update(r["params"])
        param_rows.append(row)

    varying = detect_varying_params(param_rows)

    # 1-parameter sweep
    if len(varying)==1:
        p = varying[0]
        plot_1d_overlay(scen_repeats, p,
                        outroot / f"1D_sweep_overlay_{p}.png")
        make_1d_sweep_gif(scen_repeats, p, outroot)

            # --- NEW overlay for lost / n^2 ---
        plot_1d_overlay_metric(
            scen_repeats,
            p,
            outroot / f"1D_sweep_overlay_{p}_n2.png",
            metric_col="P_sq_smooth_mean",
            ylabel="Smoothed coagulation rate (lost / n^2)",
            title_suffix="lost / n^2")

        # --- NEW overlay for lost / pairs ---
        plot_1d_overlay_metric(
            scen_repeats,
            p,
            outroot / f"1D_sweep_overlay_{p}_pairs.png",
            metric_col="P_pairs_smooth_mean",
            ylabel="Smoothed coagulation rate (lost / pairs)",
            title_suffix="lost / pairs")

    # 2-parameter sweep
    if len(varying)==2:
        p1,p2 = varying
        make_2d_heatmap_gif(scen_repeats, p1, p2, outroot)


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=str, default=None)
    ap.add_argument("--run", type=str, default="all")
    ap.add_argument("--window", type=int, default=20)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_root = script_dir.parent / "ABM_sensitivity_results"
    root = Path(args.results_root).resolve() if args.results_root else default_root

    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if args.run != "all":
        run_dirs = [p for p in run_dirs if p.name==args.run]

    for r in run_dirs:
        print(f"[RUN] {r.name}")
        process_run(r, smooth_window=args.window)


if __name__=="__main__":
    main()