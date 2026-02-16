#!/usr/bin/env python3
"""
Cluster volume distribution with multiple fitted distributions + KS ranking.

Place this script in the same folder as:
    - s073t071c2_tracking_2.csv
    - s074t071c2_tracking_2.csv

Then press "Run" in VS Code.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import (
    lognorm, gamma, weibull_min, norm, expon,
    logistic, cauchy, kstest
)

def main():
    # Directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    files = [
        "s073t071c2_tracking_2.csv",
        "s074t071c2_tracking_2.csv",
    ]

    # Load CSVs
    dfs = []
    for f in files:
        full_path = os.path.join(script_dir, f)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"CSV file '{f}' not found at:\n{full_path}")
        dfs.append(pd.read_csv(full_path))

    df = pd.concat(dfs, ignore_index=True)
    volumes = df["Cluster volume"].astype(float).values

    print("\n=== Cluster Volume Summary ===")
    print(f"Total clusters: {len(volumes)}")
    print(f"Mean:   {np.mean(volumes):.4f}")
    print(f"Median: {np.median(volumes):.4f}")

    results = {}

    # ------------------------------------------------------------
    # Fit distributions (each entry has: params, KS statistic, PDF)
    # ------------------------------------------------------------

    # Log-normal
    p = lognorm.fit(volumes, floc=0)
    results["Log-normal"] = {
        "params": p,
        "ks": kstest(volumes, "lognorm", args=p).statistic,
        "pdf": lambda x, p=p: lognorm.pdf(x, *p)
    }

    # Gamma
    p = gamma.fit(volumes, floc=0)
    results["Gamma"] = {
        "params": p,
        "ks": kstest(volumes, "gamma", args=p).statistic,
        "pdf": lambda x, p=p: gamma.pdf(x, *p)
    }

    # Weibull
    p = weibull_min.fit(volumes, floc=0)
    results["Weibull"] = {
        "params": p,
        "ks": kstest(volumes, "weibull_min", args=p).statistic,
        "pdf": lambda x, p=p: weibull_min.pdf(x, *p)
    }

    # Normal
    p = norm.fit(volumes)
    results["Normal"] = {
        "params": p,
        "ks": kstest(volumes, "norm", args=p).statistic,
        "pdf": lambda x, p=p: norm.pdf(x, *p)
    }

    # Exponential
    p = expon.fit(volumes)
    results["Exponential"] = {
        "params": p,
        "ks": kstest(volumes, "expon", args=p).statistic,
        "pdf": lambda x, p=p: expon.pdf(x, *p)
    }

    # Logistic
    p = logistic.fit(volumes)
    results["Logistic"] = {
        "params": p,
        "ks": kstest(volumes, "logistic", args=p).statistic,
        "pdf": lambda x, p=p: logistic.pdf(x, *p)
    }

    # Cauchy
    p = cauchy.fit(volumes)
    results["Cauchy"] = {
        "params": p,
        "ks": kstest(volumes, "cauchy", args=p).statistic,
        "pdf": lambda x, p=p: cauchy.pdf(x, *p)
    }


    # ------------------------------------------------------------
    # Print sorted KS results
    # ------------------------------------------------------------
    print("\n=== KS Statistics (lower = better fit) ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1]["ks"])

    for name, info in sorted_results:
        print(f"{name:15s}: KS = {info['ks']:.5f}")

    best_name = sorted_results[0][0]
    print(f"\n>>> Best fit by KS: {best_name}")

    # ------------------------------------------------------------
    # Plot top 5 PDFs + histogram
    # ------------------------------------------------------------
    x = np.linspace(min(volumes), max(volumes), 400)

    plt.figure(figsize=(10, 7))
    plt.hist(volumes, bins=40, density=True,
             alpha=0.4, color="steelblue", edgecolor="black",
             label="Data histogram")

    # Safe plotting: no crashes from bad PDFs
    plotted = 0
    for name, info in sorted_results[:5]:
        try:
            y = info["pdf"](x)
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                raise ValueError("Non-finite PDF values")
            plt.plot(x, y, lw=2, label=name)
            plotted += 1
        except Exception as e:
            print(f"Skipping {name} — PDF error: {e}")

    plt.xlabel("Cluster volume")
    plt.ylabel("Density")
    plt.title("Cluster Volume Distribution — Top 5 Fitted PDFs")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Save distribution parameters + KS metrics to JSON
    # ------------------------------------------------------------


    # Convert numpy types to floats/lists for JSON serialization
    def clean_value(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, (list, tuple, np.ndarray)):
            return [clean_value(x) for x in v]
        return v

    models_json = {}
    for name, info in results.items():
        models_json[name] = {
            "params": clean_value(info["params"]),
            "ks": clean_value(info["ks"])
        }

    # Path for JSON outputs
    json_path_all = os.path.join(script_dir, "model_fits.json")
    json_path_best = os.path.join(script_dir, "best_model.json")

    # Write all model fits
    with open(json_path_all, "w") as f:
        json.dump(models_json, f, indent=4)
    print(f"\nSaved all model parameters + KS values to:\n  {json_path_all}")

    # Best model only
    best_model_info = results[best_name]
    best_json = {
        "best_model": best_name,
        "params": clean_value(best_model_info["params"]),
        "ks": clean_value(best_model_info["ks"])
    }

    with open(json_path_best, "w") as f:
        json.dump(best_json, f, indent=4)
    print(f"Saved best model to:\n  {json_path_best}\n")

    # ------------------------------------------------------------
    # Compute mean number of clusters from the dataset
    # ------------------------------------------------------------
    # 'volumes' is already extracted from the merged CSVs
    # but we need number of clusters, not volumes.
    # For cluster count, just use length of the dataframe:
    mean_clusters = int(round(len(df)/len(files)))  # rounded integer

    mean_cluster_path = os.path.join(script_dir, "mean_initial_clusters.json")
    with open(mean_cluster_path, "w") as f:
        json.dump({"mean_initial_clusters": mean_clusters}, f, indent=4)

    print(f"Saved mean number of clusters to: {mean_cluster_path}")


if __name__ == "__main__":
    main()