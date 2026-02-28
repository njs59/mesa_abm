#!/usr/bin/env python3
"""
Extra sensitivity plots for 2D heatmaps:
- Sensitivity log-ratio heatmap         : log10(|dS/dx| / |dS/dy|)
- Trade-off slope (iso-effect) map      : d x / d y keeping S constant
- Gradient vector field                 : direction of ∇S; colour = dominance
- DISCRETE "compensate" maps (PAIR PNG) : steps in y to match one step in x (left),
                                          steps in x to match one step in y (right)
  (Each cell is coloured by step-count; black = impossible. Overlay shows the
   first parameter value (x or y) that meets the criterion.)

Place this file in:
    ABM_sensitivity/postprocess_extra_sensitivity_plots.py

Run from project root:
    python ABM_sensitivity/postprocess_extra_sensitivity_plots.py --run ABM_sensitivity_results/<timestamp>/ \
        [--x-key ...] [--y-key ...] [--dpi 170] [--save-csv]

All outputs are written into:
    <run>/extra_plots/
"""

import os
import argparse
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ============================ scenario parsing ============================= #
def _parse_scenario_name(scen: str) -> dict:
    """
    Robustly parse a scenario folder name like:
      scenario_02__mode_singletons_phase2_fit_all__a_0p7__p_merge_0p5
    -> { "__mode__":"singletons_phase2_fit_all", "a":0.7, "p_merge":0.5 }
    """
    parts = scen.split("__")
    out = {}
    for token in parts[1:]:
        if not token:
            continue
        if token.startswith("mode_"):
            out["__mode__"] = token[len("mode_"):]
            continue
        if "_" not in token:
            continue
        key, raw = token.rsplit("_", 1)
        val_str = raw.replace("p", ".")
        try:
            out[key] = float(val_str)
        except ValueError:
            out[key] = val_str
    return out


def _varying_keys(param_by_scen: Dict[str, dict]) -> List[str]:
    keys = set()
    for d in param_by_scen.values():
        keys |= set(d.keys())
    varying = []
    for k in sorted(keys):
        vals = {d.get(k, None) for d in param_by_scen.values()}
        if len(vals) > 1:
            varying.append(k)
    return varying


# ============================== metrics & time ============================= #
_METRIC_COLS = {
    # aggregated CSV column names
    "num_clusters":    ("mean_num_clusters", None),
    "mean_size":       ("mean_mean_size", None),
    "var_size":        ("mean_var_size", None),
    "cv_size":         ("mean_cv_size", None),
    "gini_size":       ("mean_gini_size", None),
    "median_nn":       ("mean_median_nn", None),
    "morisita":        ("mean_morisita", None),
    "coag_prob":       (None, "coag_prob"),  # point estimate (no CI)
}

def _pick_early_mid_final_indices(T: int) -> Tuple[int, int, int]:
    if T <= 0:
        return (0, 0, 0)
    early = max(0, int(round(0.10 * (T - 1))))
    mid   = max(0, int(round(0.50 * (T - 1))))
    final = max(0, T - 1)
    return (early, mid, final)

# ===================== gradient (finite differences) ====================== #
def _finite_diff_gradients(Z: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dS/dx (columns) and dS/dy (rows) on rectilinear (y,x) grid with
    central differences when available, else forward/backward. NaNs propagate.
    """
    ny, nx = Z.shape
    dZdx = np.full_like(Z, np.nan, dtype=float)
    dZdy = np.full_like(Z, np.nan, dtype=float)

    # along x
    for j in range(ny):
        for i in range(nx):
            if 0 < i < nx-1 and np.isfinite(Z[j,i-1]) and np.isfinite(Z[j,i+1]):
                dx = x[i+1] - x[i-1]
                if dx != 0:
                    dZdx[j,i] = (Z[j,i+1] - Z[j,i-1]) / dx
            elif i < nx-1 and np.isfinite(Z[j,i]) and np.isfinite(Z[j,i+1]):
                dx = x[i+1] - x[i]
                if dx != 0:
                    dZdx[j,i] = (Z[j,i+1] - Z[j,i]) / dx
            elif i > 0 and np.isfinite(Z[j,i]) and np.isfinite(Z[j,i-1]):
                dx = x[i] - x[i-1]
                if dx != 0:
                    dZdx[j,i] = (Z[j,i] - Z[j,i-1]) / dx

    # along y
    for j in range(ny):
        for i in range(nx):
            if 0 < j < ny-1 and np.isfinite(Z[j-1,i]) and np.isfinite(Z[j+1,i]):
                dy = y[j+1] - y[j-1]
                if dy != 0:
                    dZdy[j,i] = (Z[j+1,i] - Z[j-1,i]) / dy
            elif j < ny-1 and np.isfinite(Z[j,i]) and np.isfinite(Z[j+1,i]):
                dy = y[j+1] - y[j]
                if dy != 0:
                    dZdy[j,i] = (Z[j+1,i] - Z[j,i]) / dy
            elif j > 0 and np.isfinite(Z[j,i]) and np.isfinite(Z[j-1,i]):
                dy = y[j] - y[j-1]
                if dy != 0:
                    dZdy[j,i] = (Z[j,i] - Z[j-1,i]) / dy

    return dZdx, dZdy


# ============================ grid construction =========================== #
def _collect_scenarios(run_dir: str) -> List[str]:
    sim_root = os.path.join(run_dir, "simulations")
    if not os.path.isdir(sim_root):
        raise RuntimeError(f"No simulations/ folder in {run_dir}")
    return sorted(d for d in os.listdir(sim_root)
                  if os.path.isdir(os.path.join(sim_root, d)))


def _load_agg(run_dir: str, scen: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "summaries", scen, "summary_aggregated.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing aggregated summary: {path}")
    return pd.read_csv(path)


def _build_surface_for_metric(
    run_dir: str,
    scenarios: List[str],
    x_key: str,
    y_key: str,
    metric: str,
    t_index: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns Z (shape [len(y), len(x)]), x_vals, y_vals for the metric at t_index.
    """
    param_by_scen = {s: _parse_scenario_name(s) for s in scenarios}

    xs = sorted({float(param_by_scen[s][x_key]) for s in scenarios if x_key in param_by_scen[s]})
    ys = sorted({float(param_by_scen[s][y_key]) for s in scenarios if y_key in param_by_scen[s]})

    x_to_i = {v: i for i, v in enumerate(xs)}
    y_to_j = {v: j for j, v in enumerate(ys)}

    Z = np.full((len(ys), len(xs)), np.nan, dtype=float)

    mean_col, point_col = _METRIC_COLS[metric]

    for s in scenarios:
        pdict = param_by_scen[s]
        if x_key not in pdict or y_key not in pdict:
            continue
        xv = float(pdict[x_key]); yv = float(pdict[y_key])
        try:
            df = _load_agg(run_dir, s)
        except FileNotFoundError:
            continue
        if len(df) == 0:
            continue
        idx = min(max(0, t_index), len(df) - 1)
        row = df.iloc[idx]
        if point_col is not None:
            val = float(row.get(point_col, np.nan))
        else:
            val = float(row.get(mean_col, np.nan))
        Z[y_to_j[yv], x_to_i[xv]] = val

    return Z, np.asarray(xs, float), np.asarray(ys, float)


# ================================ plotting ================================ #
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True); return p

def _pretty_label(key: str) -> str:
    if key in ("a", "p_merge"):
        return key
    return key.split(".")[-1]

def _edges_from_centres(c: np.ndarray) -> np.ndarray:
    """Compute cell edges from centre coordinates (works for non-uniform spacing)."""
    c = np.asarray(c, float)
    if c.size == 1:
        step = 1.0
        return np.array([c[0] - 0.5*step, c[0] + 0.5*step], float)
    mids = 0.5*(c[1:] + c[:-1])
    first = c[0] - (mids[0] - c[0])
    last  = c[-1] + (c[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])

def _pcolormesh_grid(Z: np.ndarray, xv: np.ndarray, yv: np.ndarray, cmap: str,
                     title: str, cbar_label: str, out_path: str, dpi: int,
                     vmin=None, vmax=None):
    xe = _edges_from_centres(xv)
    ye = _edges_from_centres(yv)
    Xe, Ye = np.meshgrid(xe, ye)
    plt.figure(figsize=(8.8, 6.6))
    m = plt.pcolormesh(Xe, Ye, Z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(m)
    cbar.set_label(cbar_label)
    plt.xticks(xv, [f"{v:g}" for v in xv])
    plt.yticks(yv, [f"{v:g}" for v in yv])
    plt.title(title)
    plt.xlabel("x (parameter)")
    plt.ylabel("y (parameter)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

def _vector_field_coloured(Sx: np.ndarray, Sy: np.ndarray, Zbase: np.ndarray,
                           xv: np.ndarray, yv: np.ndarray,
                           title: str, out_path: str, dpi: int):
    """
    Gradient arrows with:
       • constant arrow length (for visibility)
       • colour = log10(|Sx|/|Sy|)   (param dominance)
       • angle = direction of ∇S
    """

    # ------ Build edges for pcolormesh ------
    xe = _edges_from_centres(xv)
    ye = _edges_from_centres(yv)
    Xe, Ye = np.meshgrid(xe, ye)
    Xc, Yc = np.meshgrid(xv, yv)  # arrow positions (centres)

    fig, ax = plt.subplots(figsize=(9, 7))

    # ------ Background ------
    bg = ax.pcolormesh(Xe, Ye, Zbase, shading="auto", cmap="viridis")
    cbar = fig.colorbar(bg, ax=ax)
    cbar.set_label("summary value")

    # ------ Arrow colour = log-ratio ------
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log10(np.abs(Sx) / np.abs(Sy))

    finite_lr = log_ratio[np.isfinite(log_ratio)]
    lim = max(np.percentile(np.abs(finite_lr), 95), 0.1) if finite_lr.size else 1.0
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    cmap = matplotlib.colormaps.get_cmap("coolwarm").copy()

    # ------ Downsample for readability ------
    ny, nx = Sx.shape
    step_x = max(1, nx // 12)
    step_y = max(1, ny // 12)

    Xs = Xc[::step_y, ::step_x]
    Ys = Yc[::step_y, ::step_x]
    Sxs = Sx[::step_y, ::step_x]
    Sys = Sy[::step_y, ::step_x]
    LRs = log_ratio[::step_y, ::step_x]

    # ------ Flatten ------
    Xf = Xs.ravel()
    Yf = Ys.ravel()
    U_raw = Sxs.ravel()
    V_raw = Sys.ravel()
    Cf = LRs.ravel()

    # ------ Constant arrow length ------
    mag = np.sqrt(U_raw**2 + V_raw**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        U_unit = np.where(mag > 0, U_raw / mag, 0.0)
        V_unit = np.where(mag > 0, V_raw / mag, 0.0)

    # choose arrow length as 30% of smallest grid spacing
    cell_x = np.median(np.diff(xv)) if len(xv) > 1 else 1.0
    cell_y = np.median(np.diff(yv)) if len(yv) > 1 else 1.0
    fixed_len = 0.30 * min(cell_x, cell_y)

    U = U_unit * fixed_len
    V = V_unit * fixed_len

    # ------ Colour the arrows ------
    arrow_colors = cmap(norm(np.clip(Cf, -lim, lim)))

    # ------ Draw arrows ------
    ax.quiver(
        Xf, Yf, U, V,
        color=arrow_colors,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.006,
        pivot="mid",
        alpha=0.9
    )

    # ------ Colourbar for dominance ------
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb2 = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label("log10(|dS/dx| / |dS/dy|)\n(warm: x dominates, cool: y dominates)")

    # ------ Axes & labels ------
    ax.set_xticks(xv)
    ax.set_xticklabels([f"{v:g}" for v in xv])
    ax.set_yticks(yv)
    ax.set_yticklabels([f"{v:g}" for v in yv])

    ax.set_xlabel("x (parameter)")
    ax.set_ylabel("y (parameter)")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

# ----------------------------- DISCRETE compensation maps -----------------------------
def _discrete_steps_y_for_x(Z: np.ndarray, xv: np.ndarray, yv: np.ndarray):
    """
    For each cell (i,k) where k < nx-1:
      Δx = |Z[i, k+1] - Z[i, k]| (one step to the right)
      Find smallest m >= 0 s.t. |Z[i+m, k] - Z[i, k]| >= Δx (moving up in y)
      steps[i,k] = m; first_val[i,k] = yv[i+m]
    If impossible, steps[i,k] = NaN.
    For k == nx-1, steps[i,k] = NaN (no right neighbour).
    """
    Z = np.asarray(Z, float)
    ny, nx = Z.shape
    steps = np.full((ny, nx), np.nan, dtype=float)
    first_val = np.full((ny, nx), np.nan, dtype=float)

    for i in range(ny):
        for k in range(nx - 1):  # last column has no right neighbour
            z0 = Z[i, k]
            zR = Z[i, k + 1]
            if not (np.isfinite(z0) and np.isfinite(zR)):
                continue
            delta_x = abs(zR - z0)
            found = False
            for m in range(0, ny - i):
                z_up = Z[i + m, k]
                if np.isfinite(z_up) and abs(z_up - z0) >= delta_x:
                    steps[i, k] = float(m)
                    first_val[i, k] = float(yv[i + m])
                    found = True
                    break
            if not found:
                pass
    return steps, first_val


def _discrete_steps_x_for_y(Z: np.ndarray, xv: np.ndarray, yv: np.ndarray):
    """
    Mirror case:
    For each cell (i,k) where i < ny-1:
      Δy = |Z[i+1, k] - Z[i, k]| (one step up)
      Find smallest m >= 0 s.t. |Z[i, k+m] - Z[i, k]| >= Δy (moving right in x)
      steps[i,k] = m; first_val[i,k] = xv[k+m]
    If impossible, steps[i,k] = NaN.
    For i == ny-1, steps[i,k] = NaN (no up neighbour).
    """
    Z = np.asarray(Z, float)
    ny, nx = Z.shape
    steps = np.full((ny, nx), np.nan, dtype=float)
    first_val = np.full((ny, nx), np.nan, dtype=float)

    for i in range(ny - 1):  # last row has no up neighbour
        for k in range(nx):
            z0 = Z[i, k]
            zU = Z[i + 1, k]
            if not (np.isfinite(z0) and np.isfinite(zU)):
                continue
            delta_y = abs(zU - z0)
            found = False
            for m in range(0, nx - k):
                z_right = Z[i, k + m]
                if np.isfinite(z_right) and abs(z_right - z0) >= delta_y:
                    steps[i, k] = float(m)
                    first_val[i, k] = float(xv[k + m])
                    found = True
                    break
            if not found:
                pass
    return steps, first_val


def _plot_discrete_compensation_pair(
    steps_y_for_x: np.ndarray, first_y_val: np.ndarray,
    steps_x_for_y: np.ndarray, first_x_val: np.ndarray,
    xv: np.ndarray, yv: np.ndarray,
    title_left: str, title_right: str,
    cbar_left: str, cbar_right: str,
    out_path: str, dpi: int = 170
):
    """
    Side-by-side figure with TWO panels in ONE PNG:
      • Left  : steps_y_for_x coloured; text = first_y_val
      • Right : steps_x_for_y coloured; text = first_x_val
    Both use the SAME discrete bounds & colormap so counts are comparable.
    NaNs (impossible) are colour-mapped to black.
    """
    from matplotlib.colors import BoundaryNorm

    # Unified bounds across both maps
    max_step_a = int(np.nanmax(steps_y_for_x)) if np.isfinite(np.nanmax(steps_y_for_x)) else 0
    max_step_b = int(np.nanmax(steps_x_for_y)) if np.isfinite(np.nanmax(steps_x_for_y)) else 0
    max_step = max(max_step_a, max_step_b, 1)
    bounds = np.arange(-0.5, max_step + 1.5, 1)
    cmap = matplotlib.colormaps.get_cmap("plasma").copy()
    cmap.set_bad("black")
    norm = BoundaryNorm(bounds, cmap.N)

    # Grid edges
    xe = _edges_from_centres(xv)
    ye = _edges_from_centres(yv)
    Xe, Ye = np.meshgrid(xe, ye)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)

    # LEFT panel
    M1 = np.ma.masked_invalid(steps_y_for_x)
    pc1 = axes[0].pcolormesh(Xe, Ye, M1, shading="auto", cmap=cmap, norm=norm)
    cb1 = fig.colorbar(pc1, ax=axes[0])
    cb1.set_label(cbar_left)
    cb1.set_ticks(np.arange(0, max_step + 1, 1))
    axes[0].set_title(title_left)
    axes[0].set_xticks(xv); axes[0].set_xticklabels([f"{v:g}" for v in xv])
    axes[0].set_yticks(yv); axes[0].set_yticklabels([f"{v:g}" for v in yv])
    axes[0].set_xlabel("x (parameter)")
    axes[0].set_ylabel("y (parameter)")
    # overlay text
    ny, nx = steps_y_for_x.shape
    for i in range(ny):
        for k in range(nx):
            if np.isfinite(steps_y_for_x[i, k]) and np.isfinite(first_y_val[i, k]):
                txt = f"{first_y_val[i, k]:.3g}"
                col = "white" if (steps_y_for_x[i, k] >= max_step/2) else "black"
                axes[0].text(xv[k], yv[i], txt, ha="center", va="center", fontsize=7.5, color=col)

    # RIGHT panel
    M2 = np.ma.masked_invalid(steps_x_for_y)
    pc2 = axes[1].pcolormesh(Xe, Ye, M2, shading="auto", cmap=cmap, norm=norm)
    cb2 = fig.colorbar(pc2, ax=axes[1])
    cb2.set_label(cbar_right)
    cb2.set_ticks(np.arange(0, max_step + 1, 1))
    axes[1].set_title(title_right)
    axes[1].set_xticks(xv); axes[1].set_xticklabels([f"{v:g}" for v in xv])
    axes[1].set_yticks(yv); axes[1].set_yticklabels([f"{v:g}" for v in yv])
    axes[1].set_xlabel("x (parameter)")
    axes[1].set_ylabel("y (parameter)")
    # overlay text
    ny2, nx2 = steps_x_for_y.shape
    for i in range(ny2):
        for k in range(nx2):
            if np.isfinite(steps_x_for_y[i, k]) and np.isfinite(first_x_val[i, k]):
                txt = f"{first_x_val[i, k]:.3g}"
                col = "white" if (steps_x_for_y[i, k] >= max_step/2) else "black"
                axes[1].text(xv[k], yv[i], txt, ha="center", va="center", fontsize=7.5, color=col)

    fig.suptitle("Discrete compensation (step counts & first satisfying parameter values)", y=1.02, fontsize=12)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ================================ driver ================================= #
def main():
    ap = argparse.ArgumentParser(description="Generate extra sensitivity plots for 2D heatmaps.")
    ap.add_argument("--run", required=True, help="Path to completed run folder (with summaries/)")
    ap.add_argument("--x-key", type=str, default=None, help="Dotted key for the X axis (auto if omitted)")
    ap.add_argument("--y-key", type=str, default=None, help="Dotted key for the Y axis (auto if omitted)")
    ap.add_argument("--dpi", type=int, default=170)
    ap.add_argument("--save-csv", action="store_true", help="Also write CSVs for all derived fields")
    args = ap.parse_args()

    # --- Root of this run ---
    run_dir = args.run.rstrip("/")

    # --- All extra sensitivity outputs in a single folder ---
    out_dir = _ensure_dir(os.path.join(run_dir, "extra_plots"))

    # scenarios + params
    scenarios = _collect_scenarios(run_dir)
    param_by_scen = {s: _parse_scenario_name(s) for s in scenarios}

    # normalise 'p' -> 'p_merge' where appropriate
    if any("p_merge" in d for d in param_by_scen.values()):
        for d in param_by_scen.values():
            if "p" in d and "p_merge" not in d:
                try:
                    float(d["p"])
                    d["p_merge"] = d.pop("p")
                except Exception:
                    pass

    varying = [k for k in _varying_keys(param_by_scen) if k != "__mode__"]
    if args.x_key and args.y_key:
        x_key, y_key = args.x_key, args.y_key
    else:
        if len(varying) < 2:
            raise RuntimeError("Need two varying parameters or specify --x-key and --y-key.")
        # heuristic: prefer familiar names
        preferred = []
        for cand in ["a", "movement_v2.phase2.speed_dist.params.a", "p_merge", "merge.p_merge"]:
            if cand in varying:
                preferred.append(cand)
        axes = []
        for k in preferred + varying:
            if k not in axes:
                axes.append(k)
            if len(axes) == 2:
                break
        x_key, y_key = axes[0], axes[1]

    print(f"Using axes:\n  X = {x_key}\n  Y = {y_key}")

    # time slices
    some_df = _load_agg(run_dir, scenarios[0])
    T = len(some_df)
    early, mid, final = _pick_early_mid_final_indices(T)
    tag_map = {"early": early, "mid": mid, "final": final}

    metric_list = list(_METRIC_COLS.keys())
    saved = []

    for metric in metric_list:
        mean_col, point_col = _METRIC_COLS[metric]
        for tag, tidx in tag_map.items():
            # build surface Z
            Z, xv, yv = _build_surface_for_metric(run_dir, scenarios, x_key, y_key, metric, tidx)
            if np.all(~np.isfinite(Z)):
                print(f"[warn] No finite values for metric={metric}, tag={tag}; skipping.")
                continue

            # gradients
            Sx, Sy = _finite_diff_gradients(Z, xv, yv)

            # ---- Sensitivity log-ratio ----
            with np.errstate(divide="ignore", invalid="ignore"):
                log_ratio = np.log10(np.abs(Sx) / np.abs(Sy))
            finite_lr = log_ratio[np.isfinite(log_ratio)]
            lim = max(np.percentile(np.abs(finite_lr), 95), 0.1) if finite_lr.size else 1.0
            logratio_path = os.path.join(out_dir, f"2D_sensitivity_logratio_{metric}_{tag}.png")
            _pcolormesh_grid(
                log_ratio, xv, yv,
                cmap="coolwarm",
                title=f"log10(|dS/d{_pretty_label(x_key)}| / |dS/d{_pretty_label(y_key)}|) — {metric} ({tag})",
                cbar_label="log10(|dS/dx| / |dS/dy|)",
                out_path=logratio_path,
                dpi=args.dpi,
                vmin=-lim, vmax=lim
            ); saved.append(logratio_path)

            # ---- Trade-off slope (iso-effect) d x / d y = -Sy/Sx ----
            with np.errstate(divide="ignore", invalid="ignore"):
                slope = - Sy / Sx
            slope_path = os.path.join(out_dir, f"2D_tradeoff_dx_dy_{metric}_{tag}.png")
            _pcolormesh_grid(
                slope, xv, yv,
                cmap="bwr",
                title=f"Iso‑effect trade‑off slope d{_pretty_label(x_key)}/d{_pretty_label(y_key)} — {metric} ({tag})",
                cbar_label=f"d{_pretty_label(x_key)}/d{_pretty_label(y_key)} (keep S constant)",
                out_path=slope_path,
                dpi=args.dpi
            ); saved.append(slope_path)

            # ---- DISCRETE compensation: compute both directions ----
            steps_y_for_x, first_y_val = _discrete_steps_y_for_x(Z, xv, yv)
            steps_x_for_y, first_x_val = _discrete_steps_x_for_y(Z, xv, yv)

            # ---- Pair PNG (both compensations side-by-side) ----
            comp_pair_path = os.path.join(out_dir, f"2D_compensate_steps_PAIR_{metric}_{tag}.png")
            _plot_discrete_compensation_pair(
                steps_y_for_x, first_y_val,
                steps_x_for_y, first_x_val,
                xv, yv,
                title_left = f"Steps in y to match one step in x — {metric} ({tag})",
                title_right= f"Steps in x to match one step in y — {metric} ({tag})",
                cbar_left  = "number of y-steps",
                cbar_right = "number of x-steps",
                out_path   = comp_pair_path,
                dpi=args.dpi
            ); saved.append(comp_pair_path)

            # ---- Vector field (colour-coded dominance) ----
            vf_path = os.path.join(out_dir, f"2D_gradient_field_{metric}_{tag}.png")
            _vector_field_coloured(
                Sx, Sy, Z, xv, yv,
                title=f"Gradient vector field — {metric} ({tag})",
                out_path=vf_path, dpi=args.dpi
            ); saved.append(vf_path)

            # Optional CSV dumps for downstream analysis
            if args.save_csv:
                def _to_df(M, xv, yv):
                    return pd.DataFrame(M, index=pd.Index(yv, name="y"), columns=pd.Index(xv, name="x"))

                _to_df(Z, xv, yv).to_csv(os.path.join(out_dir, f"2D_surface_{metric}_{tag}.csv"))
                _to_df(log_ratio, xv, yv).to_csv(os.path.join(out_dir, f"2D_sensitivity_logratio_{metric}_{tag}.csv"))
                _to_df(slope, xv, yv).to_csv(os.path.join(out_dir, f"2D_tradeoff_dx_dy_{metric}_{tag}.csv"))
                _to_df(steps_y_for_x, xv, yv).to_csv(os.path.join(out_dir, f"2D_compensate_steps_y_for_x_{metric}_{tag}.csv"))
                _to_df(first_y_val, xv, yv).to_csv(os.path.join(out_dir, f"2D_compensate_first_y_value_{metric}_{tag}.csv"))
                _to_df(steps_x_for_y, xv, yv).to_csv(os.path.join(out_dir, f"2D_compensate_steps_x_for_y_{metric}_{tag}.csv"))
                _to_df(first_x_val, xv, yv).to_csv(os.path.join(out_dir, f"2D_compensate_first_x_value_{metric}_{tag}.csv"))

    print("\nSaved:")
    for p in saved:
        print(" -", os.path.relpath(p, run_dir))


if __name__ == "__main__":
    main()