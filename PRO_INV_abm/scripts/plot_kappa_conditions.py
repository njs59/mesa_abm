#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------------------
# Metrics to plot
# --------------------------------------------------------------
METRICS = ["n_clusters","mean_size","var_size","median_nnd","R","z"]

# --------------------------------------------------------------
# Line styles for movement models (3 models)
# phase2_only → solid
# two_phase_interact → dashed
# two_phase_no_interact → dotted
# --------------------------------------------------------------
MODEL_STYLES = {
    "phase2_only": "-",
    "two_phase_interact": "--",
    "two_phase_no_interact": ":"
}

# --------------------------------------------------------------
# Generate 10 visually distinct colours for the 10 kappa models
# Use matplotlib tab10 so all 10 colours are distinct
# --------------------------------------------------------------
def get_kappa_colours(kappa_list):
    cmap = plt.get_cmap("tab10")
    colours = {}
    for i, k in enumerate(kappa_list):
        colours[k] = cmap(i % 10)
    return colours


def main():

    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results/results_kappa_conditions"

    print("Loading summary_timeseries.csv ...")
    df = pd.read_csv(results_dir / "summary_timeseries.csv")

    # Split condition onto (kappa, model)
    df[["kappa","model"]] = df["condition"].str.split("__", n=1, expand=True)

    print("Averaging across repeats ...")
    agg = (
        df.groupby(["kappa","model","step"], as_index=False)
          .agg({**{m:"mean" for m in METRICS}, "time_min":"mean"})
    )

    # Sort kappas with Default first, then alphabetically
    kappas = sorted(agg["kappa"].unique(), key=lambda x: (x!="Default", x))

    # Construct colour map for the 10 kappa models
    KAPPA_COLOURS = get_kappa_colours(kappas)

    # Output folders
    out_raw = results_dir / "plots_raw"
    out_pct = results_dir / "plots_pct"
    out_raw.mkdir(parents=True, exist_ok=True)
    out_pct.mkdir(parents=True, exist_ok=True)

    print("Generating raw time-series plots ...")

    # --------------------------------------------------------------
    # RAW TIME-SERIES PLOTS
    # --------------------------------------------------------------
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(12,6))

        for kappa in kappas:
            for model in MODEL_STYLES:
                sub = agg[(agg["kappa"]==kappa) & (agg["model"]==model)][["time_min", metric]]
                if sub.empty:
                    continue

                ax.plot(
                    sub["time_min"],
                    sub[metric],
                    color=KAPPA_COLOURS[kappa],     # colour = kappa condition
                    linestyle=MODEL_STYLES[model], # linestyle = movement model
                    linewidth=1.8,
                    label=f"{kappa} — {model}"
                )

        ax.set_title(f"Raw time-series: {metric}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_raw / f"{metric}__raw.png", dpi=300)
        plt.close(fig)

    print("Generating % difference plots ...")

    # --------------------------------------------------------------
    # % DIFFERENCE VS PHASE2_ONLY
    # --------------------------------------------------------------
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(12,6))

        for kappa in kappas:

            base = agg[(agg["kappa"]==kappa) & (agg["model"]=="phase2_only")][["step","time_min",metric]].set_index("step")
            if base.empty:
                continue

            for model in ["two_phase_interact", "two_phase_no_interact"]:
                comp = agg[(agg["kappa"]==kappa) & (agg["model"]==model)][["step",metric]].set_index("step")
                if comp.empty:
                    continue

                joined = base.join(comp, how="inner", lsuffix="_base", rsuffix="_comp")
                if joined.empty:
                    continue

                base_vals = joined[f"{metric}_base"].to_numpy()
                comp_vals = joined[f"{metric}_comp"].to_numpy()

                pct = np.where(base_vals==0, np.nan, 100.0*(comp_vals - base_vals)/base_vals)

                ax.plot(
                    joined["time_min"],
                    pct,
                    color=KAPPA_COLOURS[kappa],             # colour = kappa condition
                    linestyle=MODEL_STYLES[model],          # linestyle = movement model
                    linewidth=1.8,
                    label=f"{kappa} — {model}"
                )

        ax.set_title(f"% difference vs phase2_only: {metric}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("% difference")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_pct / f"{metric}__pct.png", dpi=300)
        plt.close(fig)

    print("Plots written to:")
    print(" -", out_raw)
    print(" -", out_pct)


if __name__ == "__main__":
    main()