
# plots/plot_proliferation_overlay.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser(description="Overlay Ncells(t) for different prolif rates.")
    ap.add_argument("--condition", required=True, choices=["proliferative", "invasive"])
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="figures/ch4")
    args = ap.parse_args()

    base = Path(args.results) / args.condition / "proliferation"
    tag_dirs = sorted(base.glob(f"{args.condition}_p*"))
    if not tag_dirs:
        raise FileNotFoundError(f"No proliferation sweep data under {base}")

    plt.figure(figsize=(7,4))
    for tag_dir in tag_dirs:
        tag = tag_dir.name
        runs = list(tag_dir.glob("run_*/summary_S012.csv"))
        if not runs: continue
        dfs = [pd.read_csv(fp) for fp in runs]
        df = pd.concat(dfs)
        m = df.groupby("hours")["Ncells"].mean()
        plt.plot(m.index, m.values, label=tag)

    plt.title(f"{args.condition.capitalize()} — Proliferation sweep: Ncells(t)")
    plt.xlabel("Time (hours)")
    plt.ylabel("Total cells")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"proliferation_overlay__{args.condition}.png"
    plt.savefig(outpath, dpi=300)
    print(f"[saved] {outpath}")

if __name__ == "__main__":
    main()
