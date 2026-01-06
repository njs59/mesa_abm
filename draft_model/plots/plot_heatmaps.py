
# plots/plot_heatmaps.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_speed_adhesion(base, condition):
    rows = []
    for tag_dir in (base/condition/"speed_adhesion").glob(f"{condition}_v*_adh*"):
        tag = tag_dir.name
        try:
            speed = float(tag.split("_v")[1].split("_adh")[0])
            adh   = float(tag.split("_adh")[1])
        except Exception:
            continue
        for run_csv in tag_dir.glob("run_*/summary_S012.csv"):
            df = pd.read_csv(run_csv)
            last = df.sort_values("hours").iloc[-1]
            rows.append({"speed": speed, "adh": adh, "S1_final": last["S1"]})
    return pd.DataFrame(rows)

def load_density(base, condition):
    rows = []
    for tag_dir in (base/condition/"density").glob(f"{condition}_n*"):
        tag = tag_dir.name
        n0 = int(tag.split("_n")[1])
        for run_csv in tag_dir.glob("run_*/summary_S012.csv"):
            df = pd.read_csv(run_csv)
            last = df.sort_values("hours").iloc[-1]
            rows.append({"n0": n0, "S1_final": last["S1"]})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Heatmaps for speed×adhesion or density sweeps.")
    ap.add_argument("--condition", required=True, choices=["proliferative", "invasive"])
    ap.add_argument("--sweep", required=True, choices=["speed_adhesion","density"])
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="figures/ch4")
    args = ap.parse_args()

    base = Path(args.results)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.sweep == "speed_adhesion":
        df = load_speed_adhesion(base, args.condition)
        if df.empty: raise FileNotFoundError("No data for speed_adhesion.")
        pivot = df.groupby(["speed","adh"])["S1_final"].mean().unstack("adh")
        plt.figure(figsize=(6,4))
        sns.heatmap(pivot.sort_index(), annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"{args.condition.capitalize()} — Final mean cluster size (S1) after 72 h")
        plt.xlabel("Adhesion")
        plt.ylabel("Speed")
        plt.tight_layout()
        outpath = outdir / f"heatmap_final_S1__{args.condition}__speed_adhesion.png"
        plt.savefig(outpath, dpi=300)
        print(f"[saved] {outpath}")

    elif args.sweep == "density":
        df = load_density(base, args.condition)
        if df.empty: raise FileNotFoundError("No data for density.")
        p = df.groupby("n0")["S1_final"].mean().reset_index()
        plt.figure(figsize=(6,4))
        sns.lineplot(data=p, x="n0", y="S1_final", marker="o")
        plt.title(f"{args.condition.capitalize()} — Final mean cluster size vs initial density")
        plt.xlabel("Initial number of clusters (n0)")
        plt.ylabel("Final mean cluster size (S1)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        outpath = outdir / f"line_final_S1__{args.condition}__density.png"
        plt.savefig(outpath, dpi=300)
        print(f"[saved] {outpath}")

if __name__ == "__main__":
    main()
