#!/usr/bin/env python3#!/es.py
"""
For each kappa condition k:
  baseline = k__phase2_only

Compare (20 comparisons total: 10 kappas × 2 two-phase models):
  k__two_phase_interact     vs baseline
  k__two_phase_no_interact  vs baseline

Find global:
  - largest positive % difference (increase)
  - largest negative % difference (decrease)
across time, metric, kappa, model.

Metrics used ONLY:
  n_clusters, mean_size, var_size, median_nnd  [2](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/plot_kappa_conditions.py)

Reads:
  results/kappa_conditions/summary_timeseries.csv  [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/run_kappa_conditions.py)

Writes:
  results/kappa_conditions/movement_effects/
    extremes.csv
    rerun_conditions.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ["n_clusters", "mean_size", "var_size", "median_nnd"]
EPS = 1e-12

THIS_DIR = Path(__file__).resolve().parent
RES_DIR = THIS_DIR / "results" / "kappa_conditions"
OUT_DIR = RES_DIR / "movement_effects"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHASE2_BASE = "phase2_only"
COMPARE_MODELS = ["two_phase_interact", "two_phase_no_interact"]


def pct_diff(x, ref):
    ref2 = np.where(np.abs(ref) < EPS, np.nan, ref)
    return 100.0 * (x - ref2) / ref2


def main():
    in_csv = RES_DIR / "summary_timeseries.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing {in_csv}. Run run_kappa_conditions.py first. [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/run_kappa_conditions.py)")

    df = pd.read_csv(in_csv)

    # mean over repeats per (condition, step)
    agg = df.groupby(["condition", "step"], as_index=False)[METRICS + ["time_min"]].mean()

    parts = agg["condition"].str.split("__", n=1, expand=True)
    agg["kappa"] = parts[0]
    agg["model"] = parts[1]

    events = []

    for kappa in sorted(agg["kappa"].unique(), key=lambda x: (x != "Default", x)):
        base_cond = f"{kappa}__{PHASE2_BASE}"
        base = agg[agg["condition"] == base_cond].copy()
        if base.empty:
            continue

        for model in COMPARE_MODELS:
            cond = f"{kappa}__{model}"
            sub = agg[agg["condition"] == cond].copy()
            if sub.empty:
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
                            kappa=kappa,
                            model=model,
                            baseline=base_cond,
                            condition=cond,
                            metric=m,
                            step=int(merged["step"].iloc[i]),
                            time_min=float(merged["time_min"].iloc[i]),
                            pct_diff=float(pct[i]),
                        )
                    )

    ev = pd.DataFrame(events)
    if ev.empty:
        raise RuntimeError("No finite % differences found. Possibly baseline values are zero across time for these metrics.")

    inc = ev.loc[ev["pct_diff"].idxmax()].to_dict()
    dec = ev.loc[ev["pct_diff"].idxmin()].to_dict()

    extremes = pd.DataFrame(
        [
            {"effect": "max_increase", **inc},
            {"effect": "max_decrease", **dec},
        ]
    )
    extremes.to_csv(OUT_DIR / "extremes.csv", index=False)

    rerun_set = {inc["baseline"], inc["condition"], dec["baseline"], dec["condition"]}
    rerun_df = pd.DataFrame([{"condition": c} for c in sorted(rerun_set)])
    rerun_df.to_csv(OUT_DIR / "rerun_conditions.csv", index=False)

    print("\nExtreme movement-model effects (κ baselines):")
    print(extremes[["effect","kappa","model","metric","step","time_min","pct_diff","baseline","condition"]].to_string(index=False))

    print("\nRerun conditions:")
    print(rerun_df.to_string(index=False))

    print("\nSaved:")
    print(" -", OUT_DIR / "extremes.csv")
    print(" -", OUT_DIR / "rerun_conditions.csv")


if __name__ == "__main__":
    main()
