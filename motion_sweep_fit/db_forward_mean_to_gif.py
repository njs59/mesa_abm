
#!/usr/bin/env python3
"""
Create a GIF showing the actual ABM clusters over time, using the posterior *mean* parameters
from a pyABC .db results file.

- Extracts weighted-mean parameters from the last non-empty population.
- Builds ABM params (motion/speed) and runs ONE forward simulation.
- Renders spatial frames (circles at cluster positions with model radii).
- Saves a single GIF.

Requirements:
  pyabc, numpy, pandas, matplotlib, pillow, seaborn
  plus your ABM modules:
    from clusters_abm.clusters_model import ClustersModel
    from clusters_abm.utils import DEFAULTS
"""

import argparse
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from PIL import Image
import seaborn as sns
import pyabc

# --- Your ABM imports (unchanged in your repo) ---
from clusters_abm.clusters_model import ClustersModel
from clusters_abm.utils import DEFAULTS


# ---------------- Utilities ----------------
def robust_distribution(history: pyabc.History, t_target: int) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """Return (df, w, t_used) for the first non-empty distribution at or below t_target."""
    t = min(int(t_target), int(history.max_t))
    while t >= 0:
        df, w = history.get_distribution(m=0, t=t)
        if len(df) > 0:
            return df, w, t
        t -= 1
    raise RuntimeError("No non-empty populations in history.")


def weighted_mean(df: pd.DataFrame, w: np.ndarray) -> Dict[str, float]:
    """Weighted mean for all numeric columns in df (ignoring NaNs)."""
    w = np.asarray(w, dtype=float)
    w = w / (w.sum() if w.sum() > 0 else 1.0)
    out = {}
    for c in df.columns:
        x = df[c].to_numpy(dtype=float)
        m = np.isfinite(x)
        if not np.any(m):
            continue
        out[c] = float(np.sum(x[m] * w[m]) / (np.sum(w[m]) if np.sum(w[m]) > 0 else 1.0))
    return out


def fig_to_pil(fig: plt.Figure, dpi: int = 160) -> Image.Image:
    """Render a matplotlib figure to a PIL Image (in-memory PNG)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    buf.close()
    plt.close(fig)
    return im


def save_gif(frames: List[Image.Image], out_path: Path, frame_ms: int = 80, loop: int = 0):
    if not frames:
        raise RuntimeError("No frames to save.")
    first, tail = frames[0], frames[1:]
    first.save(out_path, save_all=True, append_images=tail, duration=frame_ms, loop=loop, disposal=2)


# ---------------- ABM parameter mapping (same conventions as your scripts) ----------------
def _set_nested(base: dict, dotted: str, value):
    keys = dotted.split(".")
    d = base
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def build_speed_params(speed_dist: str, particle: dict) -> dict:
    if speed_dist == "constant":
        return {}
    if speed_dist == "lognorm":
        mu = float(particle.get("speed_meanlog", 1.0))
        sd = float(particle.get("speed_sdlog", 0.7))
        return {"s": sd, "scale": float(np.exp(mu))}
    elif speed_dist == "gamma":
        shape = float(particle.get("speed_shape", 2.0))
        scale = float(particle.get("speed_scale", 1.0))
        return {"a": shape, "scale": scale}
    elif speed_dist == "weibull":
        shape = float(particle.get("speed_shape", 2.0))
        scale = float(particle.get("speed_scale", 2.0))
        return {"c": shape, "scale": scale}
    else:
        raise ValueError(f"Unknown speed_dist: {speed_dist}")


def make_params_from_particle(defaults: dict, particle: dict, motion: str, speed_dist: str, init_default: int) -> dict:
    """
    Build full params dict from 'mean' particle.
    Rounds init_n_clusters if present; otherwise uses init_default.
    """
    params = {
        "space": dict(defaults["space"]),
        "time": dict(defaults["time"]),
        "physics": dict(defaults["physics"]),
        "phenotypes": {
            "proliferative": dict(defaults["phenotypes"]["proliferative"]),
            "invasive": dict(defaults["phenotypes"]["invasive"]),
        },
        "merge": dict(defaults["merge"]),
        "init": dict(defaults["init"]),
        "movement": dict(defaults["movement"]),
    }

    # Movement
    params["movement"]["direction"] = motion
    if speed_dist == "constant":
        params["movement"]["mode"] = "constant"
        params["movement"].pop("distribution", None)
        params["movement"].pop("dist_params", None)
    else:
        params["movement"]["mode"] = "distribution"
        params["movement"]["distribution"] = speed_dist
        params["movement"]["dist_params"] = build_speed_params(speed_dist, particle)

    if motion == "persistent":
        hs = float(particle.get("heading_sigma", params["movement"].get("heading_sigma", 0.25)))
        params["movement"]["heading_sigma"] = max(0.0, hs)
    else:
        params["movement"].pop("heading_sigma", None)

    # Core scalars mapping (skip speed_* & heading_sigma which are handled above)
    mapping = {
        "prolif_rate": "phenotypes.proliferative.prolif_rate",
        "adhesion": "phenotypes.proliferative.adhesion",
        "fragment_rate": "phenotypes.proliferative.fragment_rate",
        "merge_prob": "merge.prob_contact_merge",
    }
    for k, v in particle.items():
        if k.startswith("speed_") or k == "heading_sigma" or k == "init_n_clusters":
            continue
        if ("rate" in k or "prob" in k) and v < 0:
            v = 0.0
        if k in mapping:
            _set_nested(params, mapping[k], float(v))
        else:
            # Allow passing through any extra numeric knobs if present in DF
            try:
                params[k] = float(v)
            except Exception:
                pass

    # Init
    n_init = int(max(1, round(float(particle.get("init_n_clusters", init_default)))))
    params["init"]["phenotype"] = "proliferative"
    params["init"]["n_clusters"] = n_init
    return params


# ---------------- Simulation & Frame building ----------------
def simulate_forward(params: dict, steps: int, seed: int = 0):
    """Run one forward simulation; return per-step positions and radii (for plotting)."""
    model = ClustersModel(params=params, seed=seed)
    positions: List[Optional[np.ndarray]] = []
    radii_list: List[Optional[np.ndarray]] = []
    speeds_list: List[Optional[np.ndarray]] = []
    sizes_list: List[Optional[np.ndarray]] = []

    # step 0
    ids, pos, radii, sizes, speeds = model._log_alive_snapshot()
    positions.append(pos); radii_list.append(radii)
    speeds_list.append(speeds); sizes_list.append(sizes)

    for _ in range(1, steps):
        model.step()
        ids, pos, radii, sizes, speeds = model._log_alive_snapshot()
        positions.append(pos); radii_list.append(radii)
        speeds_list.append(speeds); sizes_list.append(sizes)

    return positions, radii_list, sizes_list, speeds_list


def compute_domain(positions: List[Optional[np.ndarray]],
                   pad: float = 0.0,
                   defaults_box: Optional[Tuple[float, float, float, float]] = None):
    """
    Determine plotting extents.
    If defaults_box provided as (xmin,xmax,ymin,ymax), use that.
    Otherwise, compute from data with optional padding.
    """
    if defaults_box is not None:
        return defaults_box

    xs, ys = [], []
    for pos in positions:
        if pos is not None and len(pos) > 0:
            xs.append(pos[:, 0]); ys.append(pos[:, 1])
    if xs:
        xmin, xmax = float(np.min(np.concatenate(xs))), float(np.max(np.concatenate(xs)))
        ymin, ymax = float(np.min(np.concatenate(ys))), float(np.max(np.concatenate(ys)))
    else:
        xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0

    if pad > 0:
        dx, dy = xmax - xmin, ymax - ymin
        xmin -= pad * dx; xmax += pad * dx
        ymin -= pad * dy; ymax += pad * dy
    return xmin, xmax, ymin, ymax


def build_cluster_frames(
    positions: List[Optional[np.ndarray]],
    radii_list: List[Optional[np.ndarray]],
    sizes_list: List[Optional[np.ndarray]],
    speeds_list: List[Optional[np.ndarray]],
    title: str = "",
    frame_stride: int = 1,
    radius_scale: float = 1.0,
    color_by: str = "none",         # 'none' | 'size' | 'speed'
    cmap_name: str = "viridis",
    fixed_domain: Optional[Tuple[float, float, float, float]] = None,
    dpi: int = 160,
) -> List[Image.Image]:
    """
    Build frames showing circles for each cluster.
    - frame_stride: draw every k-th step for speed.
    - radius_scale: visual multiplier on radii for clarity.
    - color_by: colour the circles by 'size' or 'speed' (else single colour).
    - fixed_domain: (xmin, xmax, ymin, ymax) if you want stable axes; else auto.
    """
    sns.set(style="white")
    frames: List[Image.Image] = []

    # Prepare domain
    domain = fixed_domain or compute_domain(positions, pad=0.05)
    xmin, xmax, ymin, ymax = domain

    # Precompute global colour limits if colouring by a variable
    vmin, vmax = None, None
    if color_by in ("size", "speed"):
        vals = []
        seq = sizes_list if color_by == "size" else speeds_list
        for arr in seq:
            if arr is not None and len(arr) > 0 and np.any(np.isfinite(arr)):
                vals.append(arr[np.isfinite(arr)])
        if vals:
            allv = np.concatenate(vals)
            vmin, vmax = float(np.nanpercentile(allv, 2)), float(np.nanpercentile(allv, 98))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = float(np.nanmin(allv)), float(np.nanmax(allv))

    cmap = plt.get_cmap(cmap_name)

    # Build frames
    for t in range(0, len(positions), max(1, int(frame_stride))):
        pos = positions[t]; rad = radii_list[t]
        sz  = sizes_list[t]; spd = speeds_list[t]

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#f8f9fa")
        ax.grid(True, alpha=0.2, lw=0.6, ls=":")

        patches = []
        colors = []

        if pos is not None and rad is not None and len(pos) > 0:
            N = len(pos)
            # choose colour values
            if color_by == "size" and sz is not None and len(sz) == N:
                cv = np.clip((sz - vmin) / (vmax - vmin + 1e-9), 0, 1) if vmin is not None else None
            elif color_by == "speed" and spd is not None and len(spd) == N:
                cv = np.clip((spd - vmin) / (vmax - vmin + 1e-9), 0, 1) if vmin is not None else None
            else:
                cv = None

            for i in range(N):
                r = float(rad[i]) * radius_scale
                if r <= 0:
                    continue
                c = Circle((pos[i, 0], pos[i, 1]), r)
                patches.append(c)
                if cv is not None:
                    colors.append(cmap(float(cv[i])))
                else:
                    colors.append("#1f77b4")

        if patches:
            coll = PatchCollection(patches, facecolor=colors, edgecolor="none", alpha=0.8)
            ax.add_collection(coll)

        # Title and info
        infix = f" | colored by {color_by}" if color_by in ("size", "speed") else ""
        ax.set_title((title or "ABM clusters (posterior mean params)") + f" — step {t}{infix}")
        ax.set_xlabel("x"); ax.set_ylabel("y")

        frames.append(fig_to_pil(fig, dpi=dpi))

    return frames


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Export a GIF of the visual ABM (clusters) using posterior mean parameters.")
    ap.add_argument("--db", type=str, required=True, help="Path to pyABC .db")
    ap.add_argument("--motion", type=str, required=True, help="Movement type: isotropic|persistent")
    ap.add_argument("--speed_dist", type=str, required=True, help="Speed distribution: constant|lognorm|gamma|weibull")

    ap.add_argument("--steps", type=int, default=180, help="Total simulation steps")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for the forward simulation")

    ap.add_argument("--out", type=str, default=None, help="Output GIF path (default: <db_stem>_clusters.gif in same dir)")
    ap.add_argument("--frame_ms", type=int, default=80, help="GIF frame duration (ms)")
    ap.add_argument("--frame_stride", type=int, default=1, help="Use every k-th step for frames (speed/size trade-off)")
    ap.add_argument("--dpi", type=int, default=160, help="DPI used to rasterize frames")

    ap.add_argument("--radius_scale", type=float, default=1.0, help="Visual scale multiplier for cluster radii")
    ap.add_argument("--color_by", type=str, choices=["none", "size", "speed"], default="none", help="Colour circles by 'size' or 'speed'")
    ap.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap name if color_by != 'none'")

    ap.add_argument("--use_space_from_defaults", action="store_true",
                    help="Use DEFAULTS['space'] extents for fixed axes if available; else auto-fit from data")
    ap.add_argument("--init_default", type=int, default=800, help="Fallback init clusters if not in posterior")
    args = ap.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    out_path = Path(args.out) if args.out else (db_path.parent / f"{db_path.stem}_clusters.gif")

    # Load posterior mean from last non-empty population
    hist = pyabc.History(f"sqlite:///{db_path}")
    df_last, w_last, t_used = robust_distribution(hist, int(getattr(hist, "max_t", 0)))
    if len(df_last) == 0:
        raise RuntimeError("No posterior particles found in the DB.")
    p_mean = weighted_mean(df_last, w_last)

    # Build ABM params and simulate once
    params = make_params_from_particle(DEFAULTS, p_mean,
                                       motion=args.motion, speed_dist=args.speed_dist,
                                       init_default=args.init_default)

    positions, radii_list, sizes_list, speeds_list = simulate_forward(params=params,
                                                                      steps=int(args.steps),
                                                                      seed=int(args.seed))

    # Fixed domain from DEFAULTS if requested and present
    fixed_domain = None
    if args.use_space_from_defaults:
        # Try to read bounds from DEFAULTS['space'] if available (adapt this if your schema differs)
        try:
            # Example guesses—adjust if your DEFAULTS schema stores size differently
            width = float(DEFAULTS["space"].get("width", None))
            height = float(DEFAULTS["space"].get("height", None))
            if np.isfinite(width) and np.isfinite(height):
                fixed_domain = (0.0, width, 0.0, height)
        except Exception:
            fixed_domain = None

    # Build frames of visual clusters
    title = f"ABM clusters (posterior mean) — {args.motion}/{args.speed_dist}"
    frames = build_cluster_frames(
        positions=positions,
        radii_list=radii_list,
        sizes_list=sizes_list,
        speeds_list=speeds_list,
        title=title,
        frame_stride=int(args.frame_stride),
        radius_scale=float(args.radius_scale),
        color_by=args.color_by,
        cmap_name=args.cmap,
        fixed_domain=fixed_domain,
        dpi=int(args.dpi),
    )

    # Save GIF
    save_gif(frames, out_path, frame_ms=int(args.frame_ms))
    print(f"[DONE] Saved GIF → {out_path} (steps={args.steps}, stride={args.frame_stride}, t_used={t_used})")


if __name__ == "__main__":
    main()
