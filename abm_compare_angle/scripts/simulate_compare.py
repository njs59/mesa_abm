import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from abm.clusters_model import ClustersModel
from abm.utils import DEFAULTS

# ----------------------------
# Helper geometry/statistics
# ----------------------------
# --- Dynamic colour assignment ---
import hashlib
import matplotlib as mpl

def _hash_name_to_float(name: str) -> float:
    """Map a name to [0,1) deterministically using a stable hash."""
    h = hashlib.sha256(name.encode("utf-8")).hexdigest()
    # take first 8 hex chars -> int -> normalize
    return (int(h[:8], 16) % 10_000_000) / 10_000_000.0

def make_colour_map(model_names):
    """
    Return a dict {name: color} for any number of models.
    - For <= 20 entries: pick from qualitative palettes (tab20, tab20b, tab20c).
    - For > 20: map hashed names into a continuous HSV wheel (gist_rainbow).
    """
    names = list(model_names)
    n = len(names)

    # Nice qualitative colours first
    if n <= 10:
        base = mpl.cm.get_cmap("tab10").colors  # 20 tuples
        return {name: base[i % 10] for i, name in enumerate(names)}
    
    elif n <= 20:
        base = mpl.cm.get_cmap("tab20").colors  # 20 tuples
        return {name: base[i % 20] for i, name in enumerate(names)}

    elif n <= 40:
        # combine tab20 + tab20b
        base = list(mpl.cm.get_cmap("tab20").colors) + list(mpl.cm.get_cmap("tab20b").colors)
        return {name: base[i % len(base)] for i, name in enumerate(names)}

    # Many models: hash into a continuous wheel for distinct hues
    cmap = mpl.cm.get_cmap("hsv")  # full hue wheel; or "gist_rainbow"
    return {name: cmap(_hash_name_to_float(name)) for name in names}



def convex_hull_area(points):
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return 0.0

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    if hull.shape[0] < 3:
        return 0.0

    x = hull[:, 0]
    y = hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def radius_of_gyration(positions, weights=None):
    P = np.asarray(positions, dtype=float)
    if P.shape[0] == 0:
        return 0.0

    if weights is None:
        cen = P.mean(axis=0)
        return float(np.sqrt(((P - cen) ** 2).sum(axis=1).mean()))

    w = np.asarray(weights, dtype=float)
    W = w.sum()
    if W <= 0:
        cen = P.mean(axis=0)
        return float(np.sqrt(((P - cen) ** 2).sum(axis=1).mean()))

    cen = (P * w[:, None]).sum(axis=0) / W
    return float(np.sqrt(((w * ((P - cen) ** 2).sum(axis=1)).sum()) / W))


def nearest_neighbour_distances(positions):
    P = np.asarray(positions, dtype=float)
    n = P.shape[0]
    if n <= 1:
        return np.zeros((0,), float)

    dists = np.full(n, np.inf)
    for i in range(n):
        di = np.sqrt(((P[i] - P) ** 2).sum(axis=1))
        di[i] = np.inf
        dists[i] = di.min()
    return dists


# ----------------------------
# Extract metrics from model
# ----------------------------
def compute_metrics_from_model(model):
    id_log = model.id_log
    pos_log = model.pos_log
    size_log = model.size_log
    dt = float(model.dt)

    T = min(len(id_log), len(pos_log), len(size_log))

    first_pos = {}
    for t in range(T):
        ids = np.asarray(id_log[t], int)
        pos = np.asarray(pos_log[t], float)
        for i in range(min(len(ids), pos.shape[0])):
            aid = int(ids[i])
            if aid not in first_pos:
                first_pos[aid] = pos[i]

    time = np.arange(T, dtype=float) * dt

    # allocate
    mean_disp = np.zeros(T)
    msd = np.zeros(T)
    hull_area = np.zeros(T)
    rog_w = np.zeros(T)
    rog_unw = np.zeros(T)
    S0 = np.zeros(T)
    S1 = np.zeros(T)
    S2 = np.zeros(T)
    max_size = np.zeros(T)
    mean_nnd = np.zeros(T)

    for t in range(T):
        ids = np.asarray(id_log[t], int)
        pos = np.asarray(pos_log[t], float)
        sizes = np.asarray(size_log[t], float)
        n = min(len(ids), pos.shape[0], sizes.shape[0])

        if n == 0:
            continue

        ids = ids[:n]
        pos = pos[:n]
        sizes = sizes[:n]

        disp = np.zeros(n, float)
        for i in range(n):
            p0 = first_pos.get(int(ids[i]), pos[i])
            disp[i] = np.linalg.norm(pos[i] - p0)

        mean_disp[t] = disp.mean()
        msd[t] = (disp ** 2).mean()

        hull_area[t] = convex_hull_area(pos)
        rog_unw[t] = radius_of_gyration(pos)
        rog_w[t] = radius_of_gyration(pos, sizes)

        S0[t] = float(n)
        S1[t] = float(sizes.sum())
        S2[t] = float((sizes ** 2).sum())
        max_size[t] = float(sizes.max())

        nnds = nearest_neighbour_distances(pos)
        mean_nnd[t] = float(nnds.mean() if nnds.size else 0.0)

    final_pos = np.asarray(pos_log[T - 1], float) if T > 0 else np.zeros((0, 2))
    final_ids = np.asarray(id_log[T - 1], int) if T > 0 else np.array([], int)
    n_final = min(len(final_ids), final_pos.shape[0])
    nnd_final = nearest_neighbour_distances(final_pos[:n_final])

    return {
        "time": time,
        "mean_disp": mean_disp,
        "msd": msd,
        "hull_area": hull_area,
        "rog_w": rog_w,
        "rog_unw": rog_unw,
        "S0": S0,
        "S1": S1,
        "S2": S2,
        "max_size": max_size,
        "mean_nnd": mean_nnd,
    }, nnd_final


# ----------------------------
# Experiment configuration
# ----------------------------
STEPS = 145
REPLICATES = 100

MODELS = [
    {"name": "isotropic", "movement": {"direction": "isotropic"}},
    {"name": "persistent 0.25", "movement": {"direction": "persistent", "heading_sigma": 0.25}},
    {"name": "persistent 0.75", "movement": {"direction": "persistent", "heading_sigma": 0.75}},
    {"name": "von_mises isotropic", "movement": {"direction": "von_mises", "mu": 0.0, "kappa": 0.0}},
    {"name": "von_mises random 0.5", "movement": {"direction": "von_mises", "mu": 0.0, "kappa": 0.5}},
    {"name": "von_mises oscillatory 0.5", "movement": {"direction": "von_mises", "mu": np.pi, "kappa": 0.5}},
    {"name": "von_mises random 2.0", "movement": {"direction": "von_mises", "mu": 0.0, "kappa": 2.0}},
    {"name": "von_mises oscillatory 2.0", "movement": {"direction": "von_mises", "mu": np.pi, "kappa": 2.0}},
]

# MODELS = [
#     {"name": "von_mises isotropic", "movement": {"direction": "von_mises", "mu": 0.0, "kappa": 0.0}},
#     {"name": "von_mises persistent 0.5", "movement": {"direction": "von_mises", "mu": 0.0, "kappa": 0.5}},
#     {"name": "von_mises oscillatory 0.5", "movement": {"direction": "von_mises", "mu": np.pi, "kappa": 0.5}},
#     {"name": "von_mises presistent 2.0", "movement": {"direction": "von_mises", "mu": 0.0, "kappa": 2.0}},
#     {"name": "von_mises oscillatory 2.0", "movement": {"direction": "von_mises", "mu": np.pi, "kappa": 2.0}},
# ]

# MODELS = [
#     {"name": "von_mises persistent 2.0", "movement": {"direction": "von_mises", "mu": 0.0, "kappa": 2.0}},
#     {"name": "von_mises pi/2 2.0", "movement": {"direction": "von_mises", "mu": np.pi/2, "kappa": 2.0}},
#     {"name": "von_mises oscillatory 2.0", "movement": {"direction": "von_mises", "mu": np.pi, "kappa": 2.0}},
# ]

# MODELS = [
#     {"name": "isotropic", "movement": {"direction": "isotropic"}},
#     {"name": "von_mises isotropic", "movement": {"direction": "von_mises", "mu": 0.0, "kappa": 0.0}},
# ]

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(OUT_DIR, "figs")
RES_DIR = os.path.join(OUT_DIR, "results")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


# ----------------------------
# Worker for parallelisation
# ----------------------------
def run_single_job(args):
    cfg, seed = args
    params = json.loads(json.dumps(DEFAULTS))  # deep copy
    params["time"]["steps"] = STEPS

    mv = params["movement"]
    for k, v in cfg["movement"].items():
        mv[k] = v
    params["movement"] = mv

    model = ClustersModel(params=params, seed=seed, init_clusters=None)
    for _ in range(STEPS):
        model.step()

    return compute_metrics_from_model(model)


# ----------------------------
# Utility
# ----------------------------
def aggregate_time_series(stack):
    arr = np.asarray(stack, float)
    mean = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    ci = 1.96 * se
    return mean, mean - ci, mean + ci


def plot_with_ci(ax, t, mean, lo, hi, label, colour):
    ax.plot(t, mean, label=label, color=colour)
    ax.fill_between(t, lo, hi, color=colour, alpha=0.25, linewidth=0)


# ----------------------------
# Main (parallel + progress bar)
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare angular models with parallel execution.")
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Number of worker processes to use (default: all available cores).",
    )
    args = parser.parse_args()

    # Determine processes safely
    max_procs = cpu_count()
    if args.cores is None:
        processes = max_procs
    else:
        processes = max(1, min(int(args.cores), max_procs))

    print(f"Using {processes} CPU core(s) (max available: {max_procs})")

    all_series = {}
    all_nnd = {}

    total_jobs = len(MODELS) * REPLICATES
    jobs = []
    for m_idx, cfg in enumerate(MODELS):
        for r in range(REPLICATES):
            seed = int(10_000 * m_idx + r)  # deterministic per (model,rep)
            jobs.append((cfg, seed))

    # Parallel map + progress bar
    results = []
    with Pool(processes=processes) as pool:
        for out in tqdm(pool.imap(run_single_job, jobs), total=total_jobs, desc="Simulating", unit="sim"):
            results.append(out)

    # Reassemble by model
    idx = 0
    for cfg in MODELS:
        name = cfg["name"]
        ts_buffers = {k: [] for k in ["mean_disp", "msd", "hull_area", "rog_w",
                                      "rog_unw", "S0", "S1", "S2", "max_size", "mean_nnd"]}
        nnds = []
        for _ in range(REPLICATES):
            metrics, nnd_final = results[idx]
            idx += 1
            for k in ts_buffers:
                ts_buffers[k].append(metrics[k])
            if nnd_final.size:
                nnds.append(nnd_final)

        for k in ts_buffers:
            all_series[(name, k)] = np.vstack(ts_buffers[k])
        all_nnd[name] = np.concatenate(nnds) if nnds else np.zeros((0,))

    # Plotting
    # colours = {
    #     "isotropic": "#1f77b4",
    #     "persistent": "#ff7f0e",
    #     "von_mises": "#2ca02c",
    # }
    # Build colours dynamically from MODELS
    model_names = [cfg["name"] for cfg in MODELS]
    colours = make_colour_map(model_names)

    sample_name = MODELS[0]["name"]
    T = all_series[(sample_name, "mean_disp")].shape[1]
    tvec = np.arange(T, dtype=float) * DEFAULTS["time"]["dt"]

    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    ax_flat = axes.ravel()

    panels = [
        ("mean_disp", "Mean displacement"),
        ("msd", "Mean squared displacement"),
        ("rog_w", "Weighted radius of gyration"),
        ("hull_area", "Convex hull area"),
        ("S0", "S0 (# clusters)"),
        ("S1", "S1 (total cells)"),
        ("S2", "S2 (sum size²)"),
        ("max_size", "Largest cluster"),
        ("mean_nnd", "Mean NND"),
    ]

    for i, (k, label) in enumerate(panels):
        ax = ax_flat[i]
        for cfg in MODELS:
            mname = cfg["name"]
            arr = all_series[(mname, k)]
            mean, lo, hi = aggregate_time_series(arr)
            plot_with_ci(ax, tvec, mean, lo, hi, mname, colours[mname])
        ax.set_title(label)
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("Angular model comparison: mean ± 95% CI")
    plt.savefig(os.path.join(FIG_DIR, "compare_timeseries.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "compare_timeseries.png"), dpi=200)

    # NND histogram
    plt.figure(figsize=(7, 5))
    for cfg in MODELS:
        mname = cfg["name"]
        nnd = all_nnd[mname]
        if nnd.size:
            plt.hist(nnd, bins=50, density=True, alpha=0.35, label=mname, color=colours[mname])
    plt.xlabel("NND at final time")
    plt.legend()
    plt.title("Final-time NND distribution (pooled reps)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, "nnd_final_hist.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "nnd_final_hist.png"), dpi=200)

    # Save CSV
    rows = []
    for k, _ in panels:
        for cfg in MODELS:
            mname = cfg["name"]
            arr = all_series[(mname, k)]
            mean, lo, hi = aggregate_time_series(arr)
            for i, t in enumerate(tvec):
                rows.append({
                    "model": mname,
                    "metric": k,
                    "time": float(t),
                    "mean": float(mean[i]),
                    "ci_low": float(lo[i]),
                    "ci_high": float(hi[i]),
                })
    pd.DataFrame(rows).to_csv(os.path.join(RES_DIR, "summary_timeseries.csv"), index=False)

    with open(os.path.join(RES_DIR, "nnd_final.json"), "w") as f:
        json.dump({cfg["name"]: all_nnd[cfg["name"]].tolist() for cfg in MODELS}, f)

    print("Done. Results in figs/ and results/")


if __name__ == "__main__":
    main()