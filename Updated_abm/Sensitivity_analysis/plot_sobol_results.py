#!/usr/bin/env python3
# Sensitivity_analysis/plot_sobol_results.py
"""
Plot Sobol sensitivity results saved by sobol_runner.py.

Usage:
  python -m Sensitivity_analysis.plot_sobol_results
  python -m Sensitivity_analysis.plot_sobol_results --dir Sensitivity_analysis_results/2026-03-23_13-15-02
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def find_latest_result_dir(base="Sensitivity_analysis_results"):
    base = Path(base)
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"{base} is empty.")
    latest = sorted(subdirs)[-1]
    print(f"[info] Using newest results folder: {latest}")
    return latest

def load_data(result_dir):
    cfg_path = result_dir / "config.json"
    npz_path = result_dir / "sobol_arrays.npz"
    cfg = json.loads(cfg_path.read_text())
    arr = np.load(npz_path)
    param_names   = cfg["parameters"]
    output_labels = cfg["outputs"]
    S1 = arr["first_order"]; ST = arr["total_order"]
    S1_low = arr["S1_CI_low"]; S1_high = arr["S1_CI_high"]
    ST_low = arr["ST_CI_low"]; ST_high = arr["ST_CI_high"]
    return param_names, output_labels, S1, ST, S1_low, S1_high, ST_low, ST_high

def plot_all_outputs(result_dir, param_names, output_labels, S1, ST, S1_low, S1_high, ST_low, ST_high):
    outdir = result_dir / "plots"
    outdir.mkdir(exist_ok=True)
    x = np.arange(len(param_names))
    for i, out_label in enumerate(output_labels):
        fig, ax = plt.subplots(figsize=(10, 5))
        s1 = S1[i]; st = ST[i]
        s1_err = np.vstack([s1 - S1_low[i], S1_high[i] - s1])
        st_err = np.vstack([st - ST_low[i], ST_high[i] - st])
        ax.errorbar(x - 0.15, s1, yerr=s1_err, fmt="o", capsize=4, label="S1")
        ax.errorbar(x + 0.15, st, yerr=st_err, fmt="o", capsize=4, label="ST")
        ax.set_xticks(x); ax.set_xticklabels(param_names, rotation=30, ha="right")
        ax.set_ylabel("Sobol Index"); ax.set_title(out_label); ax.set_ylim(0, 1.05); ax.legend()
        fig.tight_layout()
        fname = outdir / f"{i:02d}_{out_label.replace(' ', '_')}.png"
        fig.savefig(fname, dpi=200); plt.close(fig)
        print(f"[saved] {fname}")

def plot_combined(result_dir, param_names, output_labels, S1, ST, S1_low, S1_high, ST_low, ST_high):
    outdir = result_dir / "plots"
    outdir.mkdir(exist_ok=True)
    n_outputs = len(output_labels)
    x = np.arange(len(param_names))
    fig, axs = plt.subplots(n_outputs, 1, figsize=(12, 3.5 * n_outputs), sharex=True)
    if n_outputs == 1: axs = [axs]
    for i, (ax, out_label) in enumerate(zip(axs, output_labels)):
        s1 = S1[i]; st = ST[i]
        s1_err = np.vstack([s1 - S1_low[i], S1_high[i] - s1])
        st_err = np.vstack([st - ST_low[i], ST_high[i] - st])
        ax.errorbar(x - 0.15, s1, yerr=s1_err, fmt="o", capsize=4, label="S1")
        ax.errorbar(x + 0.15, st, yerr=st_err, fmt="o", capsize=4, label="ST")
        ax.set_ylabel("Sobol index"); ax.set_title(out_label); ax.set_ylim(0, 1.05); ax.legend(loc="upper right")
    axs[-1].set_xticks(x); axs[-1].set_xticklabels(param_names, rotation=30, ha="right")
    fig.tight_layout()
    fname = outdir / "all_outputs_subplots.png"
    fig.savefig(fname, dpi=200); plt.close(fig)
    print(f"[saved] {fname}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None)
    args = parser.parse_args()
    result_dir = Path(args.dir).expanduser().resolve() if args.dir else find_latest_result_dir()
    loaded = load_data(result_dir)
    plot_all_outputs(result_dir, *loaded)
    plot_combined(result_dir, *loaded)

if __name__ == "__main__":
    main()