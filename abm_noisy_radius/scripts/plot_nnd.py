
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_nnd(df: pd.DataFrame):
    pos = df[['x','y']].to_numpy()
    times = df['time_min'].unique()
    times.sort()
    mean_nnd = []
    for t in times:
        P = pos[df['time_min'].to_numpy()==t]
        n = P.shape[0]
        if n <= 1:
            mean_nnd.append(np.nan)
            continue
        # no wrap NND
        dmin = []
        for i in range(n):
            dx = P[i,0] - P[:,0]
            dy = P[i,1] - P[:,1]
            d = np.hypot(dx, dy)
            d[i] = np.inf
            dmin.append(d.min())
        mean_nnd.append(np.nanmean(dmin))
    return times/2.0, np.array(mean_nnd)  # hours if dt=1 -> 30 min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    t, m = compute_nnd(df)
    plt.figure(figsize=(8,5))
    plt.plot(t, m, lw=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Mean NND (pixels)')
    plt.title('Mean centre–centre NND over time (no wrap)')
    plt.tight_layout()
    plt.savefig(args.out)
    print(f'Saved {args.out}')

if __name__ == '__main__':
    main()
