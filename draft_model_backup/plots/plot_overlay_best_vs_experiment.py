
# plots/plot_overlay_best_vs_experiment.py
"""
Overlay ABM mean±CI against experimental S0/S1/S2 for the top-K scored tags.

This script:
- Loads experimental summary for the chosen condition (invasive/proliferative)
  from data/experimental/. It supports either:
    * <COND>_summary_stats.csv   (columns: S0,S1,S2 only)
    * <COND>_summary_stats_with_time.csv (columns: timepoint,hours,S0,S1,S2)
- Reads the scorer output JSON: results/<condition>/<sweep>/_scores/score_<metric>.json
  to determine the top-K tags (unless you provide --tags explicitly).
- Plots black experimental curves and overlays ABM mean curves with 95% CI shading.

Outputs:
- figures/ch4/overlay__<condition>__<sweep>__bestK.png  (or with tag list if --tags used)
- Prints: tags used, #replicates per tag, and figure save path.

Run examples:
  python plots/plot_overlay_best_vs_experiment.py --condition invasive --sweep speed_adhesion --k 3
  python plots/plot_overlay_best_vs_experiment.py --condition invasive --sweep proliferation --k 3
  # Override tags explicitly:
  python plots/plot_overlay_best_vs_experiment.py --condition invasive --sweep speed_adhesion --tags invasive_v2_adh0.7,invasive_v3_adh0.9
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _hours_from_index(df: pd.DataFrame, start_tp: int = 22) -> pd.DataFrame:
    """Attach timepoint and hours using ∆t=30 min when the CSV has only S0,S1,S2."""
    df = df.reset_index(drop=True)
    df.insert(0, "timepoint", range(start_tp, start_tp + len(df)))
    df.insert(1, "hours", df["timepoint"] * 0.5)
    return df[["hours", "S0", "S1", "S2"]]


def load_experimental(cond: str, base: str = "data/experimental") -> pd.DataFrame:
    """
    Load experimental curves for the given condition ('invasive' or 'proliferative').

    Tries enriched file first (<COND>_summary_stats_with_time.csv),
    falls back to original (<COND>_summary_stats.csv) and attaches hours.
    """
    base = Path(base)
    names = {
        "invasive": ("INV_summary_stats_with_time.csv", "INV_summary_stats.csv"),
        "proliferative": ("PRO_summary_stats_with_time.csv", "PRO_summary_stats.csv"),
    }
    enriched, plain = names[cond]
    if (base / enriched).exists():
        exp = pd.read_csv(base / enriched)
        if {"hours", "S0", "S1", "S2"}.issubset(exp.columns):
            return exp[["hours", "S0", "S1", "S2"]].reset_index(drop=True)
        # If enriched exists but lacks expected columns, fall back to plain
    if (base / plain).exists():
        exp_plain = pd.read_csv(base / plain)
        return _hours_from_index(exp_plain)
    raise FileNotFoundError(
        f"Experimental file not found. Expected one of:\n  {base/enriched}\n  {base/plain}"
    )


def load_top_tags(condition: str, sweep: str, metric: str, results_root: str, k: int) -> list[str]:
    """Load Top-K tags from the scorer JSON."""
    p = Path(results_root) / condition / sweep / "_scores" / f"score_{metric}.json"
    if not p.exists():
        raise FileNotFoundError(f"Score file not found: {p}\nRun the scorer first.")
    j = json.loads(p.read_text())
    return [rec["tag"] for rec in j.get("top", [])[:k]]


def abm_mean_ci_for_tag(tag_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Return (mean, lo, hi, n_reps) where each DataFrame has columns:
    'hours', 'S0', 'S1', 'S2'. lo/hi are 2.5%/97.5% quantiles.

    n_reps is the number of run_* folders found.
    """
    runs = sorted(tag_dir.glob("run_*/summary_S012.csv"))
    if not runs:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0
    df = pd.concat([pd.read_csv(fp) for fp in runs], ignore_index=True)
    g = df.groupby("hours")
    mean = g[["S0", "S1", "S2"]].mean()
    lo = g[["S0", "S1", "S2"]].quantile(0.025)
    hi = g[["S0", "S1", "S2"]].quantile(0.975)
    out_mean = mean.reset_index()
    out_lo = lo.reset_index()
    out_hi = hi.reset_index()
    # Ensure hours are strictly increasing
    for d in (out_mean, out_lo, out_hi):
        d.sort_values("hours", inplace=True)
    return out_mean, out_lo, out_hi, len(runs)


def main():
    ap = argparse.ArgumentParser(description="Overlay ABM mean±CI against experimental curves.")
    ap.add_argument("--condition", required=True, choices=["proliferative", "invasive"])
    ap.add_argument("--sweep", required=True, choices=["speed_adhesion", "proliferation", "fragmentation", "density"])
    ap.add_argument("--metric", default="rmse_weighted", help="Must match scorer metric name.")
    ap.add_argument("--k", type=int, default=3, help="Top-K tags to overlay (ignored if --tags provided).")
    ap.add_argument("--tags", type=str, default=None, help="Comma-separated tag names to overlay instead of Top-K.")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--outdir", default="figures/ch4")
    args = ap.parse_args()

    # Load experimental data (black curves)
    exp = load_experimental(args.condition)

    # Determine tags to overlay (explicit list or scorer Top-K)
    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    else:
        tags = load_top_tags(args.condition, args.sweep, args.metric, args.results_root, args.k)

    # Prepare plotting
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.style.use("ggplot")
    fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    labels = ["S0: number of clusters", "S1: mean cluster size", "S2: mean size²"]
    cols = ["S0", "S1", "S2"]

    # Plot experimental curves
    for ax, lab, col in zip(axs, labels, cols):
        ax.plot(exp["hours"], exp[col], color="black", lw=2.2, label=f"Experiment ({col})")
        ax.set_ylabel(lab)

    # Overlay ABM mean±CI curves for each tag
    base = Path(args.results_root) / args.condition / args.sweep
    cmap = plt.cm.get_cmap("tab10")
    used_info = []  # collect (tag, n_reps) for printing

    for i, tag in enumerate(tags):
        tag_dir = base / tag
        mean, lo, hi, n_reps = abm_mean_ci_for_tag(tag_dir)
        if n_reps == 0:
            print(f"[warn] No runs found for tag: {tag_dir}")
            continue
        used_info.append((tag, n_reps))
        for ax, col in zip(axs, cols):
            ax.plot(mean["hours"], mean[col], color=cmap(i), lw=1.8, label=f"ABM {tag}")
            ax.fill_between(mean["hours"], lo[col], hi[col], color=cmap(i), alpha=0.15)

    axs[-1].set_xlabel("Time (hours)")
    axs[0].legend(loc="best", fontsize=8, ncol=2)
    fig.suptitle(f"{args.condition.capitalize()} — {args.sweep}: Overlay vs experiment", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    # Save the figure with a descriptive name
    if args.tags:
        safe_tags = "__".join(tags)[:120]  # truncate if very long
        outpath = outdir / f"overlay__{args.condition}__{args.sweep}__tags__{safe_tags}.png"
    else:
        outpath = outdir / f"overlay__{args.condition}__{args.sweep}__best{args.k}.png"
    fig.savefig(outpath, dpi=300)

    # Print a useful summary to terminal
    print("\n[overlay] condition:", args.condition, "| sweep:", args.sweep)
    if args.tags:
        print("Tags provided explicitly:", ", ".join(tags))
    else:
        print(f"Top-{args.k} tags used (from scorer metric={args.metric}):", ", ".join(tags))
    for tag, n in used_info:
        print(f"  - {tag}: {n} replicates")
    print("[saved]", outpath)


if __name__ == "__main__":
    main()
