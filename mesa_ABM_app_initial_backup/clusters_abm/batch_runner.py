
#!/usr/bin/env python3
"""
Batch runner for monoculture Mesa ABM sweeps.

Assumptions / conventions
-------------------------
- Monoculture only (proliferative OR invasive), as requested.
- dt = 1.0 is treated as 30 minutes; plots/exports also include "hours = step * 0.5".
- Constant speed (we do NOT use size-dependent speed here).
- Initial quick runs use a small replicate count; final plots use larger N.

Outputs
-------
For each run:
  results/<condition>/<sweep>/<tag>/
    meta.json                    # parameters, seed, steps, condition
    state_timeseries.csv         # per-agent, per-step tidy table (from utils.export_timeseries_state)
    summary_S012.csv             # S0, S1, S2, Ncells vs time (hours)

You can aggregate across runs with your plotting scripts to produce mean±CI curves,
heatmaps, etc.

Usage examples
--------------
# Quick test: speed×adhesion for invasive
python batch_runner.py speed_adhesion --condition invasive --quick

# Full run: proliferation sweep for proliferative (24 reps)
python batch_runner.py proliferation --condition proliferative --full --reps 24

# Fragmentation sensitivity (invasive)
python batch_runner.py fragmentation --condition invasive --quick

# Initial density sweep (proliferative)
python batch_runner.py density --condition proliferative --quick
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable

# --- Make sure we can import the local package (same folder) ---
THIS = Path(__file__).resolve()
PKG_DIR = THIS.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

# --- Local imports from your codebase ---
from .clusters_model import ClustersModel
from .utils import DEFAULTS, export_timeseries_state

# ----------------------------
# Configuration dataclasses
# ----------------------------
@dataclass
class RunConfig:
    condition: str              # "proliferative" | "invasive"
    steps: int                  # number of steps (dt=1.0 -> 30 min; 144 steps ≈ 72 h)
    dt_minutes: float = 30.0    # interpret model dt=1.0 as 30 minutes (fixed)
    seed: int = 1
    out_dir: str = "results"
    sweep: str = "adhesion"
    tag: str = "default"
    params: Dict[str, Any] = None

# ----------------------------
# Helpers
# ----------------------------
def hours_from_step(step: int) -> float:
    """dt=1.0 ≡ 30 min → 0.5 h per step."""
    return step * 0.5

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def compute_S012_from_state(state_csv: Path) -> List[Dict[str, Any]]:
    """
    Compute S0, S1, S2, and Ncells from a per-agent time-series CSV.
    - S0(t) = number of clusters
    - S1(t) = mean cluster size
    - S2(t) = mean of squared cluster size
    - Ncells(t) = sum of sizes
    Returns a list of dict rows with keys: step, hours, S0, S1, S2, Ncells
    """
    import pandas as pd
    df = pd.read_csv(state_csv)
    # We rely on "step" and "size" columns exported by export_timeseries_state.
    grp = df.groupby("step")
    rows = []
    for step, g in grp:
        sizes = g["size"].values
        if sizes.size == 0:
            continue
        S0 = float(len(sizes))
        S1 = float(sizes.mean())
        S2 = float((sizes ** 2).mean())
        Ncells = float(sizes.sum())
        rows.append(
            dict(step=int(step), hours=hours_from_step(int(step)), S0=S0, S1=S1, S2=S2, Ncells=Ncells)
        )
    return rows

def export_S012(out_csv: Path, rows: List[Dict[str, Any]]) -> None:
    import pandas as pd
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("step").to_csv(out_csv, index=False)

def simulate_one(cfg: RunConfig) -> Tuple[bool, str]:
    """
    Run a single simulation and export:
      - state_timeseries.csv
      - summary_S012.csv
      - meta.json
    Returns (ok, message)
    """
    try:
        # Build params dict
        params = deepcopy(DEFAULTS if cfg.params is None else cfg.params)

        # Set dt and steps (model.dt used as a unitless tick; we interpret as 30 min)
        # We keep DEFAULTS["time"]["dt"]=1.0 and just choose steps accordingly.
        params["time"]["dt"] = 1.0
        steps = int(cfg.steps)

        # Monoculture initialisation
        # Override "init" so the requested phenotype is used
        init = params.get("init", {})
        init["phenotype"] = cfg.condition
        params["init"] = init

        # Directory setup
        out_dir = Path(cfg.out_dir) / cfg.condition / cfg.sweep / cfg.tag / f"run_{cfg.seed:03d}"
        ensure_dir(out_dir)

        # Save meta
        meta = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "condition": cfg.condition,
            "steps": steps,
            "dt_minutes": cfg.dt_minutes,
            "dt_hours_per_step": 0.5,  # 30 min
            "seed": cfg.seed,
            "sweep": cfg.sweep,
            "tag": cfg.tag,
            "params": params,
        }
        write_json(out_dir / "meta.json", meta)

        # Simulate
        model = ClustersModel(params=params, seed=cfg.seed)
        for _ in range(steps):
            model.step()

        # Export state time-series
        state_csv = out_dir / "state_timeseries.csv"
        export_timeseries_state(model, out_csv=str(state_csv))

        # Export S0/S1/S2 and Ncells
        s012_rows = compute_S012_from_state(state_csv)
        export_S012(out_dir / "summary_S012.csv", s012_rows)

        return True, f"OK: {out_dir}"
    except Exception as e:
        return False, f"FAIL (seed={cfg.seed}, tag={cfg.tag}): {e}"

# ----------------------------
# Sweep definitions
# ----------------------------
def gen_speed_adhesion_grid(condition: str,
                            speeds=(0.5, 1.0, 2.0, 3.0, 5.0),
                            adhesions=(0.3, 0.5, 0.7, 0.9),
                            prob_contact_merge=0.9) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Yield (tag, params_update) for speed×adhesion grid for the given condition.
    """
    for speed, adh in product(speeds, adhesions):
        tag = f"{condition}_v{speed:g}_adh{adh:g}"
        p = deepcopy(DEFAULTS)
        p["merge"]["prob_contact_merge"] = float(prob_contact_merge)
        p["phenotypes"][condition]["speed_base"] = float(speed)
        p["phenotypes"][condition]["adhesion"] = float(adh)
        # Keep other phenotype unused (but params exist)
        yield tag, p

def gen_prolif_grid(condition: str,
                    prolifs=(0.001, 0.003, 0.005, 0.007, 0.01)) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for pr in prolifs:
        tag = f"{condition}_p{pr:g}"
        p = deepcopy(DEFAULTS)
        p["phenotypes"][condition]["prolif_rate"] = float(pr)
        yield tag, p

def gen_fragment_grid(condition: str,
                      frags=(0.0, 1e-4, 5e-4, 1e-3, 2e-3)) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for q in frags:
        tag = f"{condition}_frag{q:g}"
        p = deepcopy(DEFAULTS)
        p["phenotypes"][condition]["fragment_rate"] = float(q)
        yield tag, p

def gen_density_grid(condition: str,
                     n_clusters=(300, 600, 1000, 1500, 2000)) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for n0 in n_clusters:
        tag = f"{condition}_n{n0}"
        p = deepcopy(DEFAULTS)
        p["init"]["n_clusters"] = int(n0)
        # Keep domain size fixed; (optionally) add a second grid that rescales the domain
        yield tag, p

# ----------------------------
# CLI / main
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Batch runner for monoculture Mesa ABM sweeps (dt=1.0 ≡ 30 min).")
    ap.add_argument("sweep",
                    choices=["speed_adhesion", "proliferation", "fragmentation", "density"],
                    help="Which sweep to run.")
    ap.add_argument("--condition", choices=["proliferative", "invasive"], required=True,
                    help="Monoculture phenotype to simulate.")
    ap.add_argument("--steps", type=int, default=144,
                    help="Number of steps (dt=1.0 ≡ 30 min → 144 steps ≈ 72 h).")
    ap.add_argument("--out-dir", type=str, default="results",
                    help="Base output directory.")
    ap.add_argument("--reps", type=int, default=None,
                    help="Replicates per grid cell. Overrides --quick/--full defaults.")
    ap.add_argument("--quick", action="store_true",
                    help="Use few replicates (e.g., 4) for rapid iteration.")
    ap.add_argument("--full", action="store_true",
                    help="Use many replicates (e.g., 24) for final plots.")
    ap.add_argument("--max-proc", type=int, default=None,
                    help="Max parallel processes (default: min(cpu_count, 8) ).")

    # Optional ranges (override defaults)
    ap.add_argument("--speeds", type=str, default=None,
                    help="Comma-separated speeds for speed_adhesion (e.g., 0.5,1,2,3,5)")
    ap.add_argument("--adhesions", type=str, default=None,
                    help="Comma-separated adhesions for speed_adhesion (e.g., 0.3,0.5,0.7,0.9)")
    ap.add_argument("--prolifs", type=str, default=None,
                    help="Comma-separated prolif rates for proliferation (e.g., 0.001,0.003,0.005)")
    ap.add_argument("--frags", type=str, default=None,
                    help="Comma-separated fragment rates (e.g., 0,0.0001,0.0005,0.001,0.002)")
    ap.add_argument("--densities", type=str, default=None,
                    help="Comma-separated n_clusters values (e.g., 300,600,1000,1500,2000)")

    return ap.parse_args()

def parse_float_series(s: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(",") if x.strip() != "")

def parse_int_series(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")

def main():
    args = parse_args()

    # Replicate policy
    if args.reps is not None:
        reps = int(args.reps)
    elif args.full:
        reps = 24
    elif args.quick:
        reps = 4
    else:
        reps = 6  # sensible default

    # Parallelism
    if args.max_proc is None:
        max_proc = min(cpu_count(), 8)
    else:
        max_proc = max(1, int(args.max_proc))

    # Build sweep grid
    sweep = args.sweep
    condition = args.condition
    steps = int(args.steps)
    out_dir = args.out_dir

    jobs: List[RunConfig] = []

    if sweep == "speed_adhesion":
        speeds = parse_float_series(args.speeds) if args.speeds else (2.0, 3.0, 5.0, 7.0, 10.0)
        adhesions = parse_float_series(args.adhesions) if args.adhesions else (0.5, 0.7, 0.9, 1.0)
        for tag, params in gen_speed_adhesion_grid(condition, speeds, adhesions):
            for r in range(reps):
                seed = (abs(hash((tag, r))) % (2**31 - 1)) + 1
                jobs.append(RunConfig(condition=condition, steps=steps, seed=seed,
                                      out_dir=out_dir, sweep="speed_adhesion", tag=tag, params=params))

    elif sweep == "proliferation":
        prolifs = parse_float_series(args.prolifs) if args.prolifs else (0.001, 0.003, 0.005, 0.007, 0.01)
        for tag, params in gen_prolif_grid(condition, prolifs):
            for r in range(reps):
                seed = (abs(hash((tag, r))) % (2**31 - 1)) + 1
                jobs.append(RunConfig(condition=condition, steps=steps, seed=seed,
                                      out_dir=out_dir, sweep="proliferation", tag=tag, params=params))

    elif sweep == "fragmentation":
        frags = parse_float_series(args.frags) if args.frags else (0.0, 1e-4, 5e-4, 1e-3, 2e-3)
        for tag, params in gen_fragment_grid(condition, frags):
            for r in range(reps):
                seed = (abs(hash((tag, r))) % (2**31 - 1)) + 1
                jobs.append(RunConfig(condition=condition, steps=steps, seed=seed,
                                      out_dir=out_dir, sweep="fragmentation", tag=tag, params=params))

    elif sweep == "density":
        densities = parse_int_series(args.densities) if args.densities else (300, 600, 1000, 1500, 2000)
        for tag, params in gen_density_grid(condition, densities):
            for r in range(reps):
                seed = (abs(hash((tag, r))) % (2**31 - 1)) + 1
                jobs.append(RunConfig(condition=condition, steps=steps, seed=seed,
                                      out_dir=out_dir, sweep="density", tag=tag, params=params))
    else:
        raise ValueError(f"Unknown sweep: {sweep}")

    # Run
    print(f"[INFO] Sweep={sweep} | condition={condition} | steps={steps} | reps/cell={reps} | jobs={len(jobs)}")
    print(f"[INFO] Using up to {max_proc} processes…")

    # Serial path for very small jobs
    if len(jobs) <= 2 or max_proc == 1:
        results = [simulate_one(j) for j in jobs]
    else:
        with Pool(processes=max_proc) as pool:
            results = pool.map(simulate_one, jobs)

    # Report
    ok = sum(1 for x, _ in results if x)
    fail = len(results) - ok
    print(f"[DONE] Successes: {ok} | Failures: {fail}")
    for x, msg in results:
        status = "OK" if x else "ERR"
        print(f"[{status}] {msg}")

if __name__ == "__main__":
    main()
