
#!/usr/bin/env python3
"""
Plot experimental summary statistics after rescaling with MaxAbsScaler.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler


def main():
    ap = argparse.ArgumentParser(description="Plot MaxAbsScaler-rescaled experimental stats")
    ap.add_argument("--observed_ts", type=str, required=True)
    ap.add_argument("--no_gr", action="store_true")
    ap.add_argument("--out", type=str, default="results/exp_rescaled_plot.png")
    args = ap.parse_args()

    obs = pd.read_csv(args.observed_ts)

    if args.no_gr:
        stats = ["S0", "S1", "S2", "NND_med"]
    else:
        stats = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]

    obs_sorted = obs.sort_values("timestep").reset_index(drop=True)
    mat = obs_sorted[stats].to_numpy(float)

    scaler = MaxAbsScaler()
    mat_scaled = scaler.fit_transform(mat)

    # Plot
    plt.figure(figsize=(10, 6))
    for i, s in enumerate(stats):
        plt.plot(obs_sorted["timestep"], mat_scaled[:, i], label=s)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.title("Experimental summary statistics rescaled using MaxAbsScaler")
    plt.xlabel("Timestep")
    plt.ylabel("Scaled value [-1,1]")
    plt.legend()
    plt.tight_layout()

    Path("results").mkdir(exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
