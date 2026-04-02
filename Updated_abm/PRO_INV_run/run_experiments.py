#!/usr/bin/env python3
from __future__ import annotations
import os
import math
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple
from abm.utils import DEFAULTS, export_timeseries_state
from abm.clusters_model import ClustersModel

def _force_phase2(model: ClustersModel) -> None:
    """Force all agents into movement phase 2 (and prevent switching)."""
    for a in list(model.agent_set):
        if getattr(a, 'alive', True):
            a.movement_phase = 2
            a.phase_switch_time = math.inf

def _run_one(phenotype: str, n_init: int|None, steps: int, seed: int, out_csv: Path,
             inv_prolif_scale: float = 1.0) -> None:
    """
    Run one monoculture (PRO or INV) with given IC and steps; export per-step state CSV.
    INV can optionally use a proliferation rate scaled vs PRO (inv_prolif_scale).
    """
    if n_init is None:
        return  # scenario doesn't include this phenotype

    params = deepcopy(DEFAULTS)
    params['time']['steps'] = int(steps)
    params['init'] = {'n_clusters': int(n_init), 'size': int(params['init'].get('size', 1)), 'phenotype': phenotype}

    # INV lower proliferation if requested (scale vs PRO baseline)
    if phenotype == 'invasive' and inv_prolif_scale != 1.0:
        base_pro = float(DEFAULTS['phenotypes']['proliferative']['prolif_rate'])
        params['phenotypes']['invasive']['prolif_rate'] = float(base_pro * inv_prolif_scale)

    model = ClustersModel(params=params, seed=seed)
    _force_phase2(model)

    for _ in range(int(steps)):
        model.step()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    export_timeseries_state(model, out_csv=str(out_csv))

def run_all_scenarios(*, scenarios: List[Tuple[str, int|None, int|None, float]],
                      repeats: int, steps: int|None, seed0: int,
                      workers: int|None, serial: bool, results_dir: Path) -> None:
    """
    scenarios: list of tuples (name, pro_n_init, inv_n_init, inv_scale_for_INV)
    Writes trajectories to: <results_dir>/trajectories/<scenario>/<PRO|INV>/repeat_XXX.csv
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    traj_root = results_dir / "trajectories"
    traj_root.mkdir(exist_ok=True)

    steps_final = int(steps) if steps is not None else int(DEFAULTS['time']['steps'])

    # Build task list
    # For each scenario, if pro_n_init is not None -> run PRO; if inv_n_init not None -> run INV
    tasks = []
    for s_idx, (sc_name, pro_n, inv_n, inv_scale) in enumerate(scenarios):
        # PRO tasks (only for S_PRO_ONLY in our setup)
        if pro_n is not None:
            for r in range(repeats):
                seed = int(seed0 + 10000 * (2 * s_idx) + r)  # unique seed stream for PRO of this scenario
                out_csv = traj_root / sc_name / "PRO" / f"repeat_{r:03d}.csv"
                tasks.append(('proliferative', pro_n, steps_final, seed, out_csv, 1.0))
        # INV tasks
        if inv_n is not None:
            for r in range(repeats):
                seed = int(seed0 + 10000 * (2 * s_idx + 1) + r)  # unique stream for INV
                out_csv = traj_root / sc_name / "INV" / f"repeat_{r:03d}.csv"
                tasks.append(('invasive', inv_n, steps_final, seed, out_csv, float(inv_scale)))

    # Avoid BLAS oversubscription
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

    if serial or (workers == 1):
        for i, (phenotype, n_init, steps_, seed, out_csv, inv_scale) in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] {out_csv}")
            _run_one(phenotype, n_init, steps_, seed, out_csv, inv_prolif_scale=inv_scale)
    else:
        if workers is None:
            workers = max(1, (os.cpu_count() or 2) - 1)
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(_run_one, phenotype, n_init, steps_final, seed, out_csv, inv_scale)
                for (phenotype, n_init, steps_final, seed, out_csv, inv_scale) in tasks
            ]
            for i, fut in enumerate(cf.as_completed(futs), 1):
                fut.result()  # raise on failure
                if i % max(1, len(futs)//20) == 0 or i == len(futs):
                    print(f"Completed {i}/{len(futs)} runs...")

    print("All trajectories saved under:", traj_root)