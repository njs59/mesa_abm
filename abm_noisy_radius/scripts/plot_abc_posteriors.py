
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--abc_json', required=True)
    ap.add_argument('--model', choices=['constant','lognorm','gamma'], required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.abc_json,'r') as f:
        R = json.load(f)
    acc = R[args.model]['accepted_params']
    if args.model=='constant':
        speeds = [p['phenotypes']['proliferative']['speed_base'] for p in acc]
        plt.figure(figsize=(6,4))
        plt.hist(speeds, bins=20, alpha=0.7)
        plt.xlabel('speed_base')
        plt.ylabel('count')
        plt.title('ABC posterior: speed_base (constant)')
    elif args.model=='lognorm':
        s = [p['movement']['dist_params']['s'] for p in acc]
        sc = [p['movement']['dist_params']['scale'] for p in acc]
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.hist(s, bins=20, alpha=0.7); plt.xlabel('lognorm s'); plt.ylabel('count')
        plt.subplot(1,2,2); plt.hist(sc, bins=20, alpha=0.7); plt.xlabel('lognorm scale'); plt.ylabel('count')
        plt.suptitle('ABC posterior: lognorm parameters')
    else:
        a = [p['movement']['dist_params']['a'] for p in acc]
        sc = [p['movement']['dist_params']['scale'] for p in acc]
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.hist(a, bins=20, alpha=0.7); plt.xlabel('gamma a'); plt.ylabel('count')
        plt.subplot(1,2,2); plt.hist(sc, bins=20, alpha=0.7); plt.xlabel('gamma scale'); plt.ylabel('count')
        plt.suptitle('ABC posterior: gamma parameters')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f'Saved {args.out}')

if __name__ == '__main__':
    main()
