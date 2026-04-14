#!/usr/bin/env python3
"""
find_extreme_conditions.py

Baseline fixed: baseline__phase2_only
Find global extremes across ALL conditions, ALL times, and these metrics only:
- n_clusters
- mean_size
- var_size
- median_nnd

Outputs:
  results/fifteen_conditions/extreme_phase2only/selected_conditions.csv
  results/fifteen_conditions/extreme_phase2only/extreme_events.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ONLY the 4 metrics requested
METRICS = ["n_clusters", "mean_size", "var_size", "median_nnd"]
BASELINE = "baseline__phase2_only"
EPS = 1e-12

THIS_DIR = Path(__file__).resolve().parent
RES_DIR = THIS_DIR / "results" / "fifteen_conditions"
OUT_DIR = RES_DIR / "extreme_phase2only"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pct_diff(x, ref):
    ref2 = np.where(np.abs(ref) < EPS, np.nan, ref)
    return 100.0 * (x - ref2) / ref2


def main():
    in_csv = RES_DIR / "summary_timeseries.csv"
    if not in_csv.exists():
        raise FileNotFoundError(
            f"Missing {in_csv}. Run run_fifteen_conditions.py first."
        )

    df = pd.read_csv(in_csv)

    # mean over repeats per (condition, step)
    agg = df.groupby(["condition", "step"], as_index=False)[METRICS + ["time_min"]].mean()

    base = agg[agg["condition"] == BASELINE].copy()
    if base.empty:
        raise RuntimeError(f"Baseline {BASELINE} not found in {in_csv}")

    events = []

    for cond, sub in agg.groupby("condition"):
        if cond == BASELINE:
            continue

        merged = sub.merge(
            base[["step", "time_min"] + METRICS],
            on=["step", "time_min"],
            suffixes=("", "_base"),
            how="inner",
        )
        if merged.empty:
            continue

        for m in METRICS:
            pct = pct_diff(merged[m].to_numpy(), merged[f"{m}_base"].to_numpy())
            finite = np.where(np.isfinite(pct))[0]
            for i in finite:
                events.append(
                    dict(
                        condition=cond,
                        metric=m,
                        step=int(merged["step"].iloc[i]),
                        time_min=float(merged["time_min"].iloc[i]),
                        pct_diff=float(pct[i]),
                    )
                )

    ev = pd.DataFrame(events)
    if ev.empty:
        raise RuntimeError("No finite % differences found (baseline values may be zero everywhere).")

    pos_row = ev.loc[ev["pct_diff"].idxmax()].to_dict()
    neg_row = ev.loc[ev["pct_diff"].idxmin()].to_dict()

    selected = pd.DataFrame(
        [
            {"role": "baseline", "condition": BASELINE},
            {"role": "pos_extreme", "condition": pos_row["condition"]},
            {"role": "neg_extreme", "condition": neg_row["condition"]},
        ]
    )
    selected.to_csv(OUT_DIR / "selected_conditions.csv", index=False)

    extreme_events = pd.DataFrame(
        [
            {"role": "pos_extreme", **pos_row},
            {"role": "neg_extreme", **neg_row},
        ]
    )
    extreme_events.to_csv(OUT_DIR / "extreme_events.csv", index=False)

    print("\nSelected conditions (baseline + extremes):")
    print(selected.to_string(index=False))

    print("\nExtreme events (where the extremes occur):")
    print(extreme_events.to_string(index=False))

    print("\nSaved:")
    print(" -", OUT_DIR / "selected_conditions.csv")
    print(" -", OUT_DIR / "extreme_events.csv")


if __name__ == "__main__":
    main()