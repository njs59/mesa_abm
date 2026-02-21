#!/usr/bin/env python3
"""
Plotting for the MLE-only pipeline:

- 1D sweeps  -> line plots per model × ODE parameter
- 2D sweeps  -> heatmaps per model × ODE parameter

Inputs are the consolidated DataFrame created by MLEAggregator.
"""
from __future__ import annotations
from typing import List
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _safe_param_name(name: str) -> str:
    """Create a safe filename fragment from a parameter name."""
    return (name.replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
                .replace(":", "_")
                .replace("{", "")
                .replace("}", "")
                .replace("$", "")
                .replace("^", "")
                .replace("_", ""))


def plot_1d_sweep(
    *,
    df: pd.DataFrame,
    x_key: str,
    ode_param_names: List[str],
    out_dir: str,
    model_key: str,
):
    """
    For a single swept ABM parameter, produce one line plot per ODE parameter.
    - X-axis: ABM parameter value
    - Y-axis: fitted ODE parameter
    """
    if df.empty:
        return

    # Coerce x to numeric if possible (keeps strings if not coercible)
    try:
        df = df.copy()
        df[x_key] = pd.to_numeric(df[x_key], errors="ignore")
    except Exception:
        pass

    for pname in ode_param_names:
        col = f"ode::{pname}"
        if col not in df.columns:
            continue
        g = df[[x_key, col]].groupby(x_key, as_index=False).mean().sort_values(x_key)
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=g, x=x_key, y=col, marker="o")
        plt.title(f"{model_key} — {pname} vs {x_key}")
        plt.xlabel(x_key)
        plt.ylabel(pname)
        plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
        plt.tight_layout()
        fname = os.path.join(out_dir, f"line_{_safe_param_name(pname)}.png")
        plt.savefig(fname, dpi=150)
        plt.clf()
        plt.close("all")


def plot_2d_heatmaps(
    *,
    df: pd.DataFrame,
    x_key: str,
    y_key: str,
    ode_param_names: List[str],
    out_dir: str,
    model_key: str,
):
    """
    For two swept ABM parameters (x_key, y_key), produce a heatmap per ODE parameter.
    - X-axis: ABM param #1
    - Y-axis: ABM param #2
    - Heat:   fitted ODE parameter
    """
    if df.empty:
        return

    # Coerce keys if possible
    try:
        df = df.copy()
        df[x_key] = pd.to_numeric(df[x_key], errors="ignore")
        df[y_key] = pd.to_numeric(df[y_key], errors="ignore")
    except Exception:
        pass

    for pname in ode_param_names:
        col = f"ode::{pname}"
        if col not in df.columns:
            continue
        # Aggregate mean in case of duplicate points
        g = df[[x_key, y_key, col]].groupby([x_key, y_key], as_index=False).mean()
        # Pivot to grid
        pivot = g.pivot(index=y_key, columns=x_key, values=col)
        # Make axes sorted naturally
        try:
            pivot = pivot.sort_index(axis=0).sort_index(axis=1)
        except Exception:
            pass

        plt.figure(figsize=(8.8, 6.4))
        ax = sns.heatmap(
            pivot, cmap="viridis", cbar_kws={"label": pname},
            linewidths=0.5, linecolor="white", square=False
        )
        ax.set_title(f"{model_key} — {pname}")
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        plt.tight_layout()
        fname = os.path.join(out_dir, f"heatmap_{_safe_param_name(pname)}.png")
        plt.savefig(fname, dpi=160)
        plt.clf()
        plt.close("all")