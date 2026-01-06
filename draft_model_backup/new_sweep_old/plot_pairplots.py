
# new_sweep/plot_pairplots.py
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from new_sweep_old.plot_thesis_style import use_thesis_style

def main():
    parser = argparse.ArgumentParser(description='Pairwise plots of sweep metrics and parameters')
    parser.add_argument('--in_csv', type=str, default='results/sweeps_extended.csv')
    parser.add_argument('--out_png', type=str, default='figs_new_sweeps/pairs.png')
    args = parser.parse_args()

    use_thesis_style()
    os.makedirs(os.path.dirname(args.out_png) or '.', exist_ok=True)

    df = pd.read_csv(args.in_csv)
    cols = [
        'merge_prob', 'dt', 'inv_speed', 'pro_speed', 'inv_adh', 'pro_adh', 'inv_prolif', 'pro_prolif',
        'n_final', 'mean_size_final', 'total_cells_final', 'mean_speed_over_time'
    ]
    sub = df[cols]
    pd.plotting.scatter_matrix(sub, figsize=(10, 10), diagonal='kde')
    plt.suptitle('Pairs — parameters vs outcomes')
    plt.tight_layout()
    plt.savefig(args.out_png)
    print(f"Wrote pairplots -> {args.out_png}")

if __name__ == '__main__':
    main()
