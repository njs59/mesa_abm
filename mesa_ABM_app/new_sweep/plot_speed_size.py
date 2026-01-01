
# new_sweep/plot_speed_size.py
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from new_sweep_old.plot_thesis_style import use_thesis_style

def main():
    parser = argparse.ArgumentParser(description='Speed vs size scatter from per-agent export')
    parser.add_argument('--in_csv', type=str, default='new_results/state_timeseries.csv')
    parser.add_argument('--out_png', type=str, default='new_results/figs/speed_vs_size.png')
    parser.add_argument('--sample_frac', type=float, default=0.2, help='Fraction of rows to sample for plotting')
    args = parser.parse_args()

    use_thesis_style()
    os.makedirs(os.path.dirname(args.out_png) or '.', exist_ok=True)

    df = pd.read_csv(args.in_csv)
    if 0 < args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=123)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df['size'], df['speed'], s=6, alpha=0.4)
    ax.set_xlabel('Cluster size (cells)')
    ax.set_ylabel('Speed (units/min)')
    ax.set_title('Speed–size relationship')
    fig.tight_layout()
    fig.savefig(args.out_png)
    print(f"Wrote speed–size plot -> {args.out_png}")

if __name__ == '__main__':
    main()
