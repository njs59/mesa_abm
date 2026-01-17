
#!/usr/bin/env python3
"""
ABC-SMC runner with sweep mode + per-population progress bar (main process only).

- Sweep mode:
  * --motions accepts multiple values (e.g., isotropic persistent)
  * --speeds accepts multiple values (e.g., constant lognorm gamma weibull)
  * --seeds accepts multiple integers
  Runs the Cartesian product motions × speeds × seeds.

- Per-run DB naming:
  * --db_template with placeholders: {obs}, {motion}, {speed}, {seed}, {ts}
  * --timestamp optionally appends yyyymmdd_HHMMSS if {ts} not used
  * --overwrite to clobber existing DB; else auto-unique suffixes added

- Progress bar:
  Workers update a shared counter; the main process runs a tiny monitor thread
  and shows a single tqdm bar per population t: “accepted n / popsize”.

- Multicore sampler:
  Version-tolerant (tries n_procs= then processes=).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import time
import threading
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyabc

# Optional pretty progress bar in MAIN process
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# Project-specific
from abm.clusters_model import ClustersModel
from abcp.compute_summary import simulate_timeseries
from abcp.priors import load_priors
from abcp.abc_model_wrapper import particle_to_params

# Optional multicore
try:
    from pyabc.sampler import MulticoreEvalParallelSampler
except Exception:
    MulticoreEvalParallelSampler = None

# ---------------------- Progress bar acceptor (no printing here) ----------------------
from pyabc.acceptor import Acceptor, AcceptorResult

class ProgressAcceptor(Acceptor):
    """
    Acceptor that mirrors the default rule “distance <= eps(t)”, and updates
    a shared (multiprocessing) counter for accepted particles. The main process
    reads the counter and renders the progress bar.

    Parameters
    ----------
    popsize : int
        The target population size (e.g., 200).
    shared : dict
        {'t': Value('i'), 'acc': Value('i'), 'lock': RLock()}
    """
    def __init__(self, popsize: int, shared: dict | None = None):
        super().__init__()
        self.popsize = int(popsize)
        self.shared = shared  # multiprocessing.Manager() created in main

    def __call__(self, distance_function, eps, x, x_0, t, par):
        # Compute scalar distance and threshold
        d = float(distance_function(x, x_0))
        thr = float(eps(t))
        is_accepted = d <= thr

        # Update shared counters (workers only), no printing here
        if self.shared is not None:
            lock = self.shared['lock']
            with lock:
                if self.shared['t'].value != t:
                    # new population on this worker
                    self.shared['t'].value = int(t)
                    self.shared['acc'].value = 0
                if is_accepted:
                    self.shared['acc'].value += 1

        # Return version-tolerant AcceptorResult
        try:
            return AcceptorResult(accept=is_accepted, distance=d)
        except TypeError:
            # Older pyabc builds may use 'accepted='
            return AcceptorResult(accepted=is_accepted, distance=d)

# ---------------------- utilities ----------------------
def make_model_factory(seed: int = 42):
    def factory(params_dict):
        return ClustersModel(params=params_dict, seed=seed)
    return factory

def unique_path(path: Path) -> Path:
    """If path exists, append _1, _2, ... to make it unique."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    parent = path.parent
    k = 1
    while True:
        p = parent / f"{stem}_{k}{suffix}"
        if not p.exists():
            return p
        k += 1

# ---------------------- main run (single job) ----------------------
def run_one(
    db_path: Path,
    obs_df: pd.DataFrame,
    stats: list[str],
    timesteps: list[int],
    prior,
    popsize: int,
    maxgen: int,
    min_eps: float,
    total_steps: int,
    motion: str,
    speed: str,
    seed: int,
    workers: int,
):
    """Run a single ABC-SMC job and write to db_path."""
    # Observed matrix & flatten
    obs_mat = obs_df[stats].to_numpy(float)  # T x K
    obs_vec = obs_mat.flatten()              # T*K

    # Z-score normalisation per statistic (constant guard)
    stds = obs_mat.std(axis=0)
    stds = np.where(stds < 1e-12, 1.0, stds)
    stds_mat = np.tile(stds, (len(timesteps), 1))
    stds_rep = stds_mat.flatten()

    # Weights: 1.0 for S0/S1/S2/NND; 0.3 for g(r) if present (keeps order with `stats`)
    base_weights = np.array([1.0 if s in {"S0", "S1", "S2", "NND_med"} else 0.3 for s in stats], float)
    weights_rep = np.tile(base_weights, len(timesteps))

    # Model factory with per-run seed
    model_factory = make_model_factory(seed=seed)

    # pyABC model wrapper
    def abm_model(particle):
        params = particle_to_params(particle, motion=motion, speed_dist=speed)
        sim_full = simulate_timeseries(
            model_factory, params=params, total_steps=total_steps, sample_steps=tuple(timesteps)
        )
        # Keep only selected stats, in proper order
        full_order = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
        col_idx = [full_order.index(s) for s in stats]
        sim_sel = sim_full[:, col_idx]
        sim_vec = sim_sel.flatten()
        return {f"y_{i}": float(v) for i, v in enumerate(sim_vec)}

    observation = {f"y_{i}": float(v) for i, v in enumerate(obs_vec)}

    # Weighted Z-score distance
    def distance(sim, obs):
        sim_v = np.array([sim[f"y_{i}"] for i in range(len(obs_vec))], float)
        obs_v = np.array([obs[f"y_{i}"] for i in range(len(obs_vec))], float)
        z = (sim_v - obs_v) / stds_rep
        z = z * weights_rep
        return float(np.sqrt(np.sum(z * z)))

    # Shared progress state (accessible by workers)
    manager = mp.Manager()
    shared = {
        't': manager.Value('i', -1),
        'acc': manager.Value('i', 0),
        'lock': manager.RLock(),
    }

    # Optional parallel sampler (version‑tolerant)
    sampler = None
    if workers and workers > 1 and MulticoreEvalParallelSampler is not None:
        try:
            sampler = MulticoreEvalParallelSampler(n_procs=workers)
        except TypeError:
            sampler = MulticoreEvalParallelSampler(processes=workers)

    # Build ABC with progress-aware acceptor (no printing inside)
    abc = pyabc.ABCSMC(
        models=abm_model,
        parameter_priors=prior,
        distance_function=distance,
        population_size=popsize,
        sampler=sampler,
        acceptor=ProgressAcceptor(popsize=popsize, shared=shared),
    )

    db_url = f"sqlite:///{db_path}"
    abc.new(db_url, observation)

    # Progress monitor (main process only)
    stop_evt = threading.Event()
    bar_holder = {'bar': None}  # mutable holder for closure

    def start_bar(t_now: int):
        if bar_holder['bar'] is not None:
            try:
                bar_holder['bar'].close()
            except Exception:
                pass
            bar_holder['bar'] = None
        if tqdm is not None:
            bar_holder['bar'] = tqdm(total=popsize, desc=f"population t={t_now} accepted", leave=True)
        else:
            print(f"[ABCSMC] population t={t_now}: 0/{popsize} accepted", flush=True)

    def monitor():
        last_t = None
        last_acc = 0
        while not stop_evt.is_set():
            with shared['lock']:
                t_now = shared['t'].value
                acc = shared['acc'].value
            # On first detection of t or when t changes, (re)create the bar
            if t_now != -1 and t_now != last_t:
                start_bar(t_now)
                last_t = t_now
                last_acc = 0
            # Update the bar to current 'acc'
            if bar_holder['bar'] is not None and acc != last_acc:
                try:
                    bar_holder['bar'].update(acc - last_acc)
                except Exception:
                    pass
                last_acc = acc
            elif tqdm is None and t_now != -1 and acc != last_acc:
                print(f"\r[ABCSMC] population t={t_now}: {acc}/{popsize} accepted", end="", flush=True)
                last_acc = acc
            time.sleep(0.2)  # light polling

        # tidy up
        if bar_holder['bar'] is not None:
            try:
                bar_holder['bar'].close()
            except Exception:
                pass
            bar_holder['bar'] = None
        if tqdm is None and last_t is not None:
            print()  # newline after carriage-return updates

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()

    try:
        history = abc.run(max_nr_populations=maxgen, minimum_epsilon=min_eps)
    finally:
        stop_evt.set()
        mon.join(timeout=2.0)

    return history

# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run ABC‑SMC for the clustering ABM.\n"
            "Supports sweep over multiple motions and speed distributions."
        )
    )
    # DB & naming
    ap.add_argument("--db", type=str, default="results/abc_run.db",
                    help="DB path for single run (used only when NOT sweeping)")
    ap.add_argument("--db_template", type=str,
                    default="results/abc_{obs}_{motion}_{speed}_seed{seed}.db",
                    help="Template for per‑run DB when sweeping; placeholders: {obs},{motion},{speed},{seed},{ts}")
    ap.add_argument("--timestamp", action="store_true",
                    help="Append a timestamp to each DB filename (format yyyymmdd_HHMMSS)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite an existing DB if present (default: make unique suffix)")

    # ABC knobs
    ap.add_argument("--popsize", type=int, default=200)
    ap.add_argument("--maxgen", type=int, default=12)
    ap.add_argument("--min_eps", type=float, default=0.5)

    # Data & priors
    ap.add_argument("--observed_ts", type=str, default="observed/INV_ABM_ready_summary.csv")
    ap.add_argument("--t_start", type=int, default=22)  # retained for compatibility (unused directly)
    ap.add_argument("--total_steps", type=int, default=300)
    ap.add_argument("--priors_yaml", type=str, default="priors.yaml")

    # Single-run (backwards compatible)
    ap.add_argument("--motion", type=str, default="isotropic", choices=["isotropic", "persistent"],
                    help="Movement model for single run (ignored if --motions provided)")
    ap.add_argument("--speed", type=str, default="constant", choices=["constant", "lognorm", "gamma", "weibull"],
                    help="Speed distribution for single run (ignored if --speeds provided)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for single run (ignored if --seeds provided)")

    # Sweep mode (multiple)
    ap.add_argument("--motions", nargs="+", default=None, choices=["isotropic", "persistent"],
                    help="Run all these motion models")
    ap.add_argument("--speeds", nargs="+", default=None, choices=["constant", "lognorm", "gamma", "weibull"],
                    help="Run all these speed distributions")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Seeds to sweep (e.g., 41 42 43)")

    # Execution
    ap.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    ap.add_argument("--no_gr", action="store_true",
                    help="Use only S0, S1, S2, NND_med (drop g_r40, g_r80)")

    args = ap.parse_args()
    Path("results").mkdir(exist_ok=True, parents=True)

    # Load observed time‑series
    obs_df = pd.read_csv(args.observed_ts)
    timesteps = obs_df["timestep"].astype(int).to_list()
    stats = ["S0", "S1", "S2", "NND_med"] if args.no_gr else ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]

    # Priors
    prior = load_priors(args.priors_yaml if Path(args.priors_yaml).exists() else None)

    # Decide sweep lists
    motions = args.motions if args.motions else [args.motion]
    speeds = args.speeds if args.speeds else [args.speed]
    seeds = args.seeds if args.seeds else [args.seed]

    # For naming
    obs_tag = Path(args.observed_ts).stem
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S") if args.timestamp else None

    # Are we sweeping? (use template whenever any sweep list was explicitly provided)
    sweeping = (args.motions is not None) or (args.speeds is not None) or (args.seeds is not None)

    # Run all combinations
    for motion in motions:
        for speed in speeds:
            for seed in seeds:
                # DB path
                if not sweeping and (len(motions) * len(speeds) * len(seeds) == 1):
                    # Single run without sweep flags: respect --db exactly
                    db_path = Path(args.db)
                else:
                    # Sweep (even if only one combo) → use template
                    db_name = args.db_template.format(
                        obs=obs_tag, motion=motion, speed=speed, seed=seed, ts=(ts_tag or "")
                    )
                    db_path = Path(db_name)
                    # If timestamp requested but not used in template, append it
                    if args.timestamp and "{ts}" not in args.db_template:
                        db_path = db_path.with_stem(db_path.stem + f"_{ts_tag}")
                if not args.overwrite:
                    db_path = unique_path(db_path)

                print("\n" + "=" * 78)
                print(f"Starting ABC run → motion={motion} | speed={speed} | seed={seed}")
                print(f"DB: {db_path}")
                print("=" * 78)

                _ = run_one(
                    db_path=db_path,
                    obs_df=obs_df,
                    stats=stats,
                    timesteps=timesteps,
                    prior=prior,
                    popsize=args.popsize,
                    maxgen=args.maxgen,
                    min_eps=args.min_eps,
                    total_steps=args.total_steps,
                    motion=motion,
                    speed=speed,
                    seed=seed,
                    workers=args.workers,
                )

                print(f"ABC finished. DB saved to {db_path}")

if __name__ == "__main__":
    main()
