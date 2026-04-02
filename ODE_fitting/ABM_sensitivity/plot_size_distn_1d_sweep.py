#!/usr/bin/env python3
"""
Cluster size distributions over time for 1D sweeps (ORIGINAL logic).

This script reads per-timestep raw CSVs:
  <results_root>/<run>/simulations/<scenario>/repeat_XX/t_XXXX.csv
Each CSV must include a 'size' column.

It aggregates per timestep across repeats to build:
  - counts histogram over size
  - mass histogram over size (sum of sizes in each bin)

It then derives:
  - Probability distribution  = counts / sum(counts)
  - Raw counts                = counts
  - Mass density (cell mass)  = mass / sum(mass)

Finally, for a **1-D sweep** (one parameter varies), it plots, at n timepoints
(default 5 → ~0%, 25%, 50%, 75%, 100% of the run length), three figures per timepoint:
  1) Probability vs size (lines per parameter value)
  2) Raw count vs size (lines per parameter value)
  3) Mass density vs size (lines per parameter value)

If your run is actually 2-D (e.g., both 'a' and 'p_merge' vary), use --filter to
slice to one value of the other parameter (e.g., --filter p_merge=0.1).

Defaults are for the *original* sensitivity pipeline using global Phase-2 keys.
If you’re plotting runs from the *updated* pipeline, pass the updated --param-key
and/or use the other script you have for updated defaults.

Usage examples (ORIGINAL logic):
  # Pick latest run under ABM_sensitivity_results
  python -m ABM_sensitivity.plot_size_distn_1d_sweep \
      --results-root ABM_sensitivity_results

  # Pick a specific run folder (you can also pass the run folder as results-root)
  python -m ABM_sensitivity.plot_size_distn_1d_sweep \
      --results-root ABM_sensitivity_results --run Merge_speed_large_sweep

  # If sweep is 2-D, slice to 1-D and pick what to plot on x-axis
  python -m ABM_sensitivity.plot_size_distn_1d_sweep \
      --results-root ABM_sensitivity_results \
      --run Merge_speed_large_sweep \
      --param-key movement_v2.phase2.speed_dist.params.a \
      --filter p_merge=0.1

Options:
  --results-root : path to the parent folder that contains runs OR a run folder itself
  --run          : run folder name under results-root (or 'latest')
  --param-key    : dotted key you want on the x-axis legend (the varied parameter)
  --filter       : leaf=value to slice second parameter; repeatable
  --max-bins     : cap for number of size bins
  --n-timepoints : number of timepoints to plot (default 5; must be ≥2)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== Defaults for ORIGINAL logic =====
DEFAULT_RESULTS_ROOT = "ABM_sensitivity_results"
DEFAULT_PARAM_KEY_GUESS = "movement_v2.phase2.speed_dist.params.a"  # legacy global path

# ===== Regex helpers (parse scenario name tokens) =====
_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?([eE]-?\d+)?$")
_P_DECIMAL_RE = re.compile(r"^-?\d+p\d+([eE]-?\d+)?$")  # e.g., "0p7" -> 0.7

def parse_value(token: str):
    t = token.strip()
    if _P_DECIMAL_RE.fullmatch(t):
        t = t.replace("p", ".")
    if _NUMERIC_RE.fullmatch(t):
        try:
            return float(t)
        except Exception:
            return token
    return token

def parse_condition_tokens(condition_id: str, ignore=("scenario", "mode")) -> Dict[str, object]:
    """
    'scenario_00__mode_xxx__a_0p7__p_merge_0p1' -> {'a':0.7, 'p_merge':0.1}
    Only the last '_' splits key from value in each token part.
    """
    out: Dict[str, object] = {}
    parts = condition_id.split("__")
    for part in parts:
        if "_" not in part:
            if part not in ignore:
                out[part] = part
            continue
        key, rest = part.split("_", 1)
        if key in ignore:
            continue
        out[key] = parse_value(rest)
    return out

def identify_varying_key(param_rows: List[Dict[str, object]]) -> Optional[str]:
    df = pd.DataFrame(param_rows)
    if df.empty:
        return None
    nunq = df.nunique(dropna=True)
    varying = [k for k, n in nunq.items() if n > 1]
    if len(varying) == 1:
        return varying[0]
    return None

def resolve_run_dir(results_root: Path, run: Optional[str]) -> Path:
    """
    Accept either:
      - a parent folder that contains run folders
      - OR a direct run folder (contains 'simulations' or 'summaries')
    """
    rr = results_root.resolve()
    if (rr / "simulations").exists() or (rr / "summaries").exists():
        return rr
    if run is None or run.lower() in ("latest", "last"):
        subdirs = sorted([p for p in rr.iterdir() if p.is_dir()])
        if not subdirs:
            raise FileNotFoundError(f"No run folders under {rr}")
        return subdirs[-1]
    rd = rr / run
    if not rd.exists():
        raise FileNotFoundError(f"Run folder not found: {rd}")
    return rd

def parse_filters(repeated_flags: List[str]) -> Dict[str, float]:
    """
    Parse --filter flags like ["p_merge=0.1", "a=0.5"] into {"p_merge":0.1, "a":0.5}.
    """
    out: Dict[str, float] = {}
    for item in repeated_flags or []:
        if "=" not in item:
            raise ValueError(f"Bad --filter '{item}'. Use leaf=value, e.g., p_merge=0.1")
        k, v = item.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out

# ===== IO helpers =====
def list_conditions(sim_root: Path) -> List[Path]:
    return sorted([p for p in sim_root.iterdir() if p.is_dir()])

def list_repeats(cond_dir: Path) -> List[Path]:
    return sorted([p for p in cond_dir.glob("repeat_*") if p.is_dir()])

def list_timestep_csvs(rep_dir: Path) -> List[Path]:
    return sorted(rep_dir.glob("t_*.csv"))

# ===== Binning & aggregation =====
def compute_time_bins_and_size_bins(sim_root: Path, max_bins: int = 120) -> Tuple[List[int], np.ndarray]:
    """
    Scan all repeats across all conditions to infer:
      - maximum step index observed
      - maximum cluster size observed
    Return:
      steps     : list[int] [0..max_step]
      size_edges: np.ndarray of bin edges (int-sized, capped at max_bins)
    """
    max_step = 0
    max_size = 1
    for cond in list_conditions(sim_root):
        for rep in list_repeats(cond):
            for fp in list_timestep_csvs(rep):
                try:
                    st = int(fp.stem.split("_")[1])
                    if st > max_step:
                        max_step = st
                except Exception:
                    pass
                try:
                    df = pd.read_csv(fp, usecols=["size"])
                    if df.size:
                        smax = int(df["size"].max())
                        if smax > max_size:
                            max_size = smax
                except Exception:
                    continue
    steps = list(range(0, max_step + 1))
    if max_size > max_bins:
        edges = np.linspace(1, max_size, max_bins + 1)
    else:
        edges = np.arange(1, max_size + 2, 1)
    return steps, edges

def aggregate_counts_and_mass_for_step(cond_dir: Path, step: int, size_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate across repeats at a given timestep:
      counts_hist: histogram of 'size' (cluster counts per bin)
      mass_hist  : histogram of 'size' weighted by 'size' (sum of sizes per bin)
    """
    counts = np.zeros(len(size_edges) - 1, dtype=float)
    mass   = np.zeros(len(size_edges) - 1, dtype=float)
    for rep in list_repeats(cond_dir):
        fp = rep / f"t_{step:04d}.csv"
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp, usecols=["size"])
            if df.empty:
                continue
            sizes = df["size"].to_numpy(dtype=float)
            c, _ = np.histogram(sizes, bins=size_edges)
            m, _ = np.histogram(sizes, bins=size_edges, weights=sizes)
            counts += c
            mass   += m
        except Exception:
            continue
    return counts, mass

def build_counts_and_mass(cond_dir: Path, steps: List[int], size_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      C[t, b] = counts at step t for bin b
      M[t, b] = sum of sizes at step t for bin b
    """
    T = len(steps)
    B = len(size_edges) - 1
    C = np.zeros((T, B), dtype=float)
    M = np.zeros((T, B), dtype=float)
    for i, st in enumerate(steps):
        c, m = aggregate_counts_and_mass_for_step(cond_dir, st, size_edges)
        C[i, :] = c
        M[i, :] = m
    return C, M

# ===== Timepoint selection =====
def pick_n_indices(T: int, n: int) -> List[int]:
    """
    Pick n evenly spaced indices in [0, T-1], always including 0 and T-1.
    """
    if T <= 0 or n <= 1:
        return [max(0, T - 1)]
    idx = np.linspace(0, T - 1, num=n, dtype=int)
    # Ensure uniqueness and sorted
    out = sorted(set(int(i) for i in idx))
    # Guarantee first and last
    if out[0] != 0:
        out.insert(0, 0)
    if out[-1] != T - 1:
        out.append(T - 1)
    # Trim to <= n in case uniqueness changed length
    return out[:n]

# ===== Plotting =====
def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def plot_lines_by_param(x_centres: np.ndarray,
                        series_by_param: Dict[float, np.ndarray],
                        xlabel: str, ylabel: str, title: str,
                        out_png: Path, legend_title: str = "param"):
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    # Sort legend/colour order by parameter value
    for pv in sorted(series_by_param.keys()):
        y = series_by_param[pv]
        ax.plot(x_centres, y, lw=2, label=f"{legend_title}={pv:g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="1-D sweep: cluster size distributions over time (original logic).")
    ap.add_argument("--results-root", type=str, default=DEFAULT_RESULTS_ROOT,
                    help="Parent folder with runs, OR a direct run folder")
    ap.add_argument("--run", type=str, default="latest",
                    help="'latest' or specific run folder name under results-root")
    ap.add_argument("--param-key", type=str, default=None,
                    help="Full dotted key of the varied parameter (leaf used in folder names).")
    ap.add_argument("--filter", action="append", default=[],
                    help="Restrict to scenarios where LEAF=value (repeatable). "
                         "Examples: --filter p_merge=0.1  or  --filter a=0.5")
    ap.add_argument("--max-bins", type=int, default=120,
                    help="Max number of size bins (integer bins from 1..max_size, capped)")
    ap.add_argument(
        "--n-timepoints", type=int, default=5,
        help="How many timepoints to plot (default 5: ~0%%,25%%,50%%,75%%,100%%)"
    )
    args = ap.parse_args()

    results_root = Path(args.results_root)
    run_dir = resolve_run_dir(results_root, None if args.run in (None, "", "latest") else args.run)

    # Prefer run_dir/simulations; fall back to summaries/simulations if necessary
    if (run_dir / "simulations").exists():
        sim_root = run_dir / "simulations"
    elif (run_dir / "summaries").exists() and (run_dir / "summaries" / "simulations").exists():
        sim_root = run_dir / "summaries" / "simulations"
    else:
        raise FileNotFoundError(
            f"Could not find 'simulations' under {run_dir}. "
            f"Looked for:\n  {run_dir / 'simulations'}\n  {run_dir / 'summaries' / 'simulations'}"
        )

    # Load all scenarios, then slice with filters for 1-D
    cond_dirs_all = list_conditions(sim_root)
    filters = parse_filters(args.filter)

    cond_dirs: List[Path] = []
    param_rows: List[Dict[str, object]] = []
    for c in cond_dirs_all:
        row = parse_condition_tokens(c.name)
        ok = True
        for k, v in filters.items():
            if k not in row or not isinstance(row[k], (int, float)) or float(row[k]) != float(v):
                ok = False
                break
        if ok:
            cond_dirs.append(c)
            param_rows.append(row)

    if not cond_dirs:
        sample = cond_dirs_all[0].name if cond_dirs_all else "(none)"
        raise RuntimeError(f"No scenarios matched filters: {filters}. Example folder: {sample}")

    # Identify (or accept) the varied key
    varying_leaf = identify_varying_key(param_rows)
    if varying_leaf is None:
        # Fall back to provided or default guess
        vk = args.param_key or DEFAULT_PARAM_KEY_GUESS
        print(f"[warn] Could not auto-identify a single varying key. Using: {vk}")
        varying_leaf = vk.split(".")[-1]
    else:
        print(f"[info] Auto-identified varying key leaf: {varying_leaf}")

    # Build bins
    steps, size_edges = compute_time_bins_and_size_bins(sim_root, max_bins=args.max_bins)
    T = len(steps)
    if T == 0:
        raise RuntimeError("No timesteps found (no t_XXXX.csv files).")
    # timepoints to plot
    n_tp = max(2, int(args.n_timepoints))
    tp_idx = pick_n_indices(T, n_tp)
    tp_labels = [f"step {steps[i]}" for i in tp_idx]

    # x-axis (size)
    x_centres = 0.5 * (size_edges[:-1] + size_edges[1:])

    # Where to save plots
    out_root = run_dir / "size_distributions_1D"
    out_prob  = out_root / "probability"
    out_count = out_root / "raw_counts"
    out_mass  = out_root / "mass_density"
    for d in (out_prob, out_count, out_mass):
        _safe_mkdir(d)

    # For each timepoint, we’ll build a dict param_value -> y (one per plot type)
    for tp_i, (ti, tlabel) in enumerate(zip(tp_idx, tp_labels)):
        prob_series_by_param: Dict[float, np.ndarray]  = {}
        count_series_by_param: Dict[float, np.ndarray] = {}
        mass_series_by_param: Dict[float, np.ndarray]  = {}

        for cond in cond_dirs:
            row = parse_condition_tokens(cond.name)
            if varying_leaf not in row or not isinstance(row[varying_leaf], (int, float)):
                # Skip if this scenario doesn’t encode the chosen leaf
                continue
            pv = float(row[varying_leaf])

            # Aggregate across repeats for this condition
            C, M = build_counts_and_mass(cond, steps, size_edges)  # [T,B], [T,B]

            c = C[ti, :].copy()
            m = M[ti, :].copy()
            # Probability over size bins
            p = c / c.sum() if c.sum() > 0 else np.zeros_like(c)
            # Mass density (fraction of total cell mass in each size bin)
            md = m / m.sum() if m.sum() > 0 else np.zeros_like(m)

            prob_series_by_param[pv]  = p
            count_series_by_param[pv] = c
            mass_series_by_param[pv]  = md

        # Titles & filenames per timepoint
        leaf_label = varying_leaf
        plot_title_suffix = f"(time: {tlabel})"

        # 1) Probability
        out_png = out_prob / f"prob_{tlabel.replace(' ', '_')}.png"
        plot_lines_by_param(
            x_centres, prob_series_by_param,
            xlabel="Cluster size (cells)",
            ylabel="Probability mass",
            title=f"Size distribution — Probability {plot_title_suffix}",
            out_png=out_png, legend_title=leaf_label
        )
        print(f"[saved] {out_png}")

        # 2) Raw counts
        out_png = out_count / f"counts_{tlabel.replace(' ', '_')}.png"
        plot_lines_by_param(
            x_centres, count_series_by_param,
            xlabel="Cluster size (cells)",
            ylabel="Number of clusters (aggregated across repeats)",
            title=f"Size distribution — Raw counts {plot_title_suffix}",
            out_png=out_png, legend_title=leaf_label
        )
        print(f"[saved] {out_png}")

        # 3) Mass density
        out_png = out_mass / f"mass_{tlabel.replace(' ', '_')}.png"
        plot_lines_by_param(
            x_centres, mass_series_by_param,
            xlabel="Cluster size (cells)",
            ylabel="Mass density (fraction of total cells)",
            title=f"Size distribution — Mass density {plot_title_suffix}",
            out_png=out_png, legend_title=leaf_label
        )
        print(f"[saved] {out_png}")

if __name__ == "__main__":
    main()