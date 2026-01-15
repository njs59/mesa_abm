
#!/usr/bin/env python3
"""
Create animated GIFs from a pyABC .db results file.

Outputs:
- Posterior evolution GIF for selected parameters across populations (t).
- Epsilon trajectory GIF (ε vs population) drawn progressively.

Requirements:
- pyabc, numpy, pandas, matplotlib, seaborn, Pillow (PIL)

Example:
  python db_to_gif.py \
    --db motiongrid_pkg/results/INV_persistent_lognorm_s42.db \
    --outdir gifs \
    --params prolif_rate adhesion fragment_rate merge_prob init_n_clusters \
    --pp_samples 4000 \
    --frame_ms 150

If you only want ε trajectory:
  python db_to_gif.py \
    --db motiongrid_pkg/results/INV_isotropic_lognorm_s42.db \
    --outdir gifs \
    --epsilon_only
"""
import argparse
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pyabc

# Candidate parameters (the script will filter to those present in the DB)
PARAMS_ORDER = [
    "prolif_rate", "adhesion", "fragment_rate", "merge_prob",
    "heading_sigma", "speed_meanlog", "speed_sdlog", "speed_shape", "speed_scale",
    "init_n_clusters",  # commonly discrete; we handle via histogram
]


def weighted_resample(df: pd.DataFrame, weights: np.ndarray, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Resample n rows from df with probability proportional to weights (with replacement)."""
    w = np.asarray(weights, float)
    p = w / (w.sum() if w.sum() > 0 else 1.0)
    idx = rng.choice(np.arange(len(df)), size=max(1, n), replace=True, p=p)
    return df.iloc[idx].reset_index(drop=True)


def robust_distribution(history: pyabc.History, t_target: int) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """Return (df, w, t_used) for the first non-empty distribution at or below t_target."""
    t = min(int(t_target), int(history.max_t))
    while t >= 0:
        df, w = history.get_distribution(m=0, t=t)
        if len(df) > 0:
            return df, w, t
        t -= 1
    raise RuntimeError("No non-empty populations in history.")


def fig_to_pil(fig: plt.Figure) -> Image.Image:
    """Render a matplotlib figure to a PIL Image (in-memory PNG)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    buf.close()
    plt.close(fig)
    return im


def make_posterior_frames(
    history: pyabc.History,
    params: List[str],
    pp_samples: int = 3000,
    downsample_kde: Optional[int] = None,
    max_t: Optional[int] = None,
    discrete_params: Optional[List[str]] = None,
) -> List[Image.Image]:
    """
    Build frames for the posterior evolution GIF across populations.
    Each frame shows subplots for the selected parameters at a given population t.
    """
    sns.set(style="whitegrid")
    rng = np.random.default_rng(42)

    # Get epsilon per t to show in titles
    pops = history.get_all_populations()
    t_values = pops["t"].to_numpy() if "t" in pops.columns else np.arange(len(pops))
    eps_values = pops["epsilon"].to_numpy()

    frames = []
    max_t_hist = int(history.max_t)
    t_end = max_t if max_t is not None else max_t_hist

    for t in range(0, t_end + 1):
        # Fetch distribution (if empty at t, try robust fallback)
        try:
            df_post, w_post, t_used = robust_distribution(history, t)
        except RuntimeError:
            # Skip frame if nothing available at/under t
            continue

        # For plotting order, keep only requested params that are present
        use_cols = [p for p in params if p in df_post.columns]
        if not use_cols:
            continue

        # Resample to unweighted draws
        samples = weighted_resample(df_post[use_cols], w_post, n=pp_samples, rng=rng)
        if downsample_kde and len(samples) > downsample_kde:
            samples = samples.sample(n=downsample_kde, random_state=0)

        # Create subplots grid
        n = len(use_cols)
        n_cols = min(3, n)
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.3 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        for i, p in enumerate(use_cols):
            ax = axes[i]
            data = samples[p]

            # Decide whether to use KDE or histogram
            is_discrete = (discrete_params is not None and p in discrete_params) or (p == "init_n_clusters")
            try:
                if is_discrete:
                    ax.hist(data, bins=40, density=True, alpha=0.6, color="#1f77b4")
                else:
                    sns.kdeplot(x=data, ax=ax, lw=1.8, color="#1f77b4")
            except Exception:
                ax.hist(data, bins=40, density=True, alpha=0.6, color="#1f77b4")

            ax.set_title(p)
            ax.set_xlabel(p)
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.25)

        # Remove unused axes
        for j in range(len(use_cols), len(axes)):
            axes[j].axis("off")

        # Title with epsilon and fallback t_used if any
        eps_here = eps_values[t] if t < len(eps_values) else np.nan
        subtitle = "" if t_used == t else f" (fallback from t={t} → t_used={t_used})"
        fig.suptitle(f"Posterior evolution — t={t}, ε≈{eps_here:.3f}{subtitle}", y=0.98)

        frames.append(fig_to_pil(fig))

    return frames


def make_epsilon_frames(history: pyabc.History) -> List[Image.Image]:
    """
    Build frames for the epsilon trajectory GIF, drawing the line progressively.
    """
    pops = history.get_all_populations()
    eps = pops["epsilon"].to_numpy()
    t = np.arange(len(eps))

    frames = []
    for k in range(1, len(eps) + 1):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(t[:k], eps[:k], marker="o", lw=1.8, color="#1f77b4")
        ax.set_xlabel("Population t")
        ax.set_ylabel("ε (IQR-scaled)")
        ax.set_title("Epsilon trajectory (progressive)")
        ax.grid(True, alpha=0.3)
        frames.append(fig_to_pil(fig))
    return frames


def save_gif(frames: List[Image.Image], out_path: Path, frame_ms: int = 120, loop: int = 0):
    """
    Save a list of PIL frames as an animated GIF.
    - frame_ms: duration per frame in milliseconds.
    - loop: 0 for infinite loop; >0 for number of loops.
    """
    if not frames:
        raise RuntimeError("No frames to save.")
    first, tail = frames[0], frames[1:]
    first.save(
        out_path,
        save_all=True,
        append_images=tail,
        duration=frame_ms,
        loop=loop,
        disposal=2,
    )


def main():
    ap = argparse.ArgumentParser(description="Create GIF animations from pyABC .db results.")
    ap.add_argument("--db", type=str, required=True, help="Path to pyABC SQLite DB (.db)")
    ap.add_argument("--outdir", type=str, default="gifs", help="Output directory for GIFs")
    ap.add_argument("--params", nargs="*", default=None,
                    help="Parameters to animate (default: auto-filter from candidate list)")
    ap.add_argument("--pp_samples", type=int, default=3000, help="Resample size per population")
    ap.add_argument("--downsample_kde", type=int, default=4000, help="Downsample per-frame draws before KDE")
    ap.add_argument("--max_t", type=int, default=None, help="Limit to populations t ≤ max_t")
    ap.add_argument("--frame_ms", type=int, default=120, help="GIF frame duration (ms)")
    ap.add_argument("--discrete", nargs="*", default=None,
                    help="Treat these parameters as discrete (use histogram)")
    ap.add_argument("--epsilon_only", action="store_true",
                    help="Only produce epsilon trajectory GIF (skip posterior evolution)")
    args = ap.parse_args()

    db_path = Path(args.db).resolve()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Open pyABC history
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    history = pyabc.History(f"sqlite:///{db_path}")

    # Build and save epsilon trajectory GIF
    eps_frames = make_epsilon_frames(history)
    eps_out = outdir / f"{db_path.stem}_epsilon.gif"
    save_gif(eps_frames, eps_out, frame_ms=args.frame_ms)
    print(f"[DONE] Saved epsilon GIF: {eps_out}")

    if args.epsilon_only:
        return

    # Decide parameter list
    # Probe the last non-empty population to detect available columns
    df_last, w_last, t_last = robust_distribution(history, int(history.max_t))
    if args.params:
        use_params = args.params
    else:
        # Filter candidate list to present columns, preserve order
        use_params = [p for p in PARAMS_ORDER if p in df_last.columns]

    if not use_params:
        print("[WARN] No candidate parameters found in the DB; skipping posterior evolution GIF.")
        return

    # Build and save posterior evolution GIF
    post_frames = make_posterior_frames(
        history=history,
        params=use_params,
        pp_samples=args.pp_samples,
        downsample_kde=args.downsample_kde,
        max_t=args.max_t,
        discrete_params=args.discrete,
    )
    post_out = outdir / f"{db_path.stem}_posteriors.gif"
    save_gif(post_frames, post_out, frame_ms=args.frame_ms)
    print(f"[DONE] Saved posterior evolution GIF: {post_out}")


if __name__ == "__main__":
    main()
