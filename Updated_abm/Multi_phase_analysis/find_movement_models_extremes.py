#!/usr/bin/env python3
"""
find_movement_model_extremes.py

Within each phenotype ( phenotype's phase2_only as baseline:
  baseline_v = <variant>__phase2_only

Compare the two movement models to baseline (10 comparisons total):
  <variant>__two_phase_interact vs baseline_v
  <variant>__two_phase_no_interact vs baseline_v

Find global:
  - largest positive % difference (increase)
  - largest negative % difference (decrease)
across time, metric, variant, model.

Metrics used ONLY:
  n_clusters, mean_size, var_size, median_nnd  [2](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/forward_phase2_invasive.py)

Reads:
  results/fifteen_conditions/summary_timeseries.csv  [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/__main__.py)

Writes:
  results/fifteen_conditions/movement_effects/
    extremes.csv                 (two rows: max increase, max decrease)
    rerun_conditions.csv         (up to 4 conditions to rerun)
"""

from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ["n_clusters", "mean_size", "var_size", "median_nnd"]
EPS = 1e-12

THIS_DIR = Path(__file__).resolve().parent
RES_DIR = THIS_DIR / "results" / "fifteen_conditions"
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
        raise FileNotFoundError(f"Missing {in_csv}. Run run_fifteen_conditions.py first. [1](https://unioxfordnexus-my.sharepoint.com/personal/kebl7472_ox_ac_uk/Documents/Microsoft%20Copilot%20Chat%20Files/__main__.py)")

    df = pd.read_csv(in_csv)

    # mean over repeats per (condition, step)
    agg = df.groupby(["condition", "step"], as_index=False)[METRICS + ["time_min"]].mean()

    # split condition into variant (phenotype) and model
    parts = agg["condition"].str.split("__", n=1, expand=True)
    agg["variant"] = parts[0]
    agg["model"] = parts[1]

    events = []

    for variant in sorted(agg["variant"].unique()):
        base_cond = f"{variant}__{PHASE2_BASE}"
        base = agg[agg["condition"] == base_cond].copy()
        if base.empty:
            continue

        for model in COMPARE_MODELS:
            cond = f"{variant}__{model}"
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
                            variant=variant,
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
        raise RuntimeError("No finite % differences found. Check baseline values are non-zero for these metrics.")

    inc = ev.loc[ev["pct_diff"].idxmax()].to_dict()
    dec = ev.loc[ev["pct_diff"].idxmin()].to_dict()

    extremes = pd.DataFrame(
        [
            {"effect": "max_increase", **inc},
            {"effect": "max_decrease", **dec},
        ]
    )
    extremes.to_csv(OUT_DIR / "extremes.csv", index=False)

    # conditions to rerun (4 total: baseline+model for each extreme)
    rerun_set = {
        inc["baseline"], inc["condition"],
        dec["baseline"], dec["condition"]
    }
    rerun_df = pd.DataFrame([{"condition": c} for c in sorted(rerun_set)])
    rerun_df.to_csv(OUT_DIR / "rerun_conditions.csv", index=False)

    print("\nExtreme movement-model effects vs phenotype phase2 baseline:")
    print(extremes[["effect","variant","model","metric","step","time_min","pct_diff","baseline","condition"]].to_string(index=False))

    print("\nRerun conditions:")
    print(rerun_df.to_string(index=False))

    print("\nSaved:")
    print(" -", OUT_DIR / "extremes.csv")
    print(" -", OUT_DIR / "rerun_conditions.csv")


if __name__ == "__main__":
    main()