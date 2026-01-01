
# plots/plot_density_threshold.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser(description="Compute time to reach S1=target across initial densities.")
    ap.add_argument("--condition", required=True, choices=["proliferative","invasive"])
    ap.add_argument("--target", type=float, default=5.0, help="Target mean size S1.")
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="figures/ch4")
    args = ap.parse_args()

    base = Path(args.results) / args.condition / "density"
    rows = []
    for tag_dir in base.glob(f"{args.condition}_n*"):
        tag = tag_dir.name
        n0 = int(tag.split("_n")[1])
        dfs = [pd.read_csv(fp) for fp in tag_dir.glob("run_*/summary_S012.csv")]
        if not dfs: continue
        df = pd.concat(dfs).groupby("hours")["S1"].mean().reset_index()
        hit = df[df["S1"] >= args.target]
        t_hit = hit["hours"].iloc[0] if not hit.empty else np.nan
        rows.append({"n0": n0, "t_hit_hours": t_hit})

    if not rows:
        raise FileNotFoundError("No density sweep data found.")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows).sort_values("n0")
    plt.figure(figsize=(6,4))
    plt.plot(res["n0"], res["t_hit_hours"], marker="o")
    plt.title(f"{args.condition.capitalize()} — Time to reach S1={args.target}")
    plt.xlabel("Initial number of clusters (n0)")
    plt.ylabel("Time (hours)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    outpath = outdir / f"density_time_to_S1__{args.condition}__S1_{args.target}.png"
    plt.savefig(outpath, dpi=300)
    print(f"[saved] {outpath}")

if __name__ == "__main__":
    main()
