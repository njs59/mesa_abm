#!/usr/bin/env python3
import os; os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"; os.environ["NUMEXPR_NUM_THREADS"]="1"

import yaml, argparse, pandas as pd, csv, itertools
from .abm_runner import run_forward_sims
from .mle_runner import run_mle
from .mcmc_runner import run_adaptive_mcmc
from .diagnostics_runner import run_diagnostics
from .aic_tools import write_aic_table
from .utils_logging import (
    make_run_dir, save_manifest, save_run_input,
    timer, write_timings, init_run_status, update_run_status, utcnow
)

# -------------------- helpers (new) --------------------
def load_cfg(path):
    with open(path, "r") as f: return yaml.safe_load(f)

def slice_data(mean_csv, mode):
    df = pd.read_csv(mean_csv)
    if mode in ["data_t71_phase2", "singletons_phase2_fit71plus", "singletons_phase1_to_2_fit71plus"]:
        df = df[df["step"] >= 71].reset_index(drop=True)
    times = df["step"].to_numpy()
    data_values = df[["num_clusters","mean_cluster_size","mean_squared_cluster_size"]].to_numpy()
    return df, times, data_values

def _flatten(d, prefix=""):
    out={}
    for k,v in (d or {}).items():
        kk=f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict): out.update(_flatten(v, kk))
        else: out[kk]=v
    return out

def _write_abm_sweep_summary(run_dir, rows):
    if not rows: return
    fieldnames=sorted({k for r in rows for k in r.keys()})
    path=os.path.join(run_dir,"abm_sweep_summary.csv")
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    return path

def _cartesian(overrides_dict):
    """{'a':[1,2], 'b':[3,4]} -> [{'a':1,'b':3}, {'a':1,'b':4}, ...]"""
    if not overrides_dict: return [{}]
    keys=list(overrides_dict.keys())
    lists=[ vals if isinstance(vals,(list,tuple)) else [vals] for vals in overrides_dict.values() ]
    return [ {k:v for k,v in zip(keys, combo)} for combo in itertools.product(*lists) ]

def _format_val_for_name(v):
    if isinstance(v, float):
        return f"{v:.6g}".replace(".","p")
    return str(v).replace(".","p")

def _make_scenario_name(idx, mode, overrides):
    parts=[f"abm_{idx:02d}_{mode}"]
    for k in sorted(overrides.keys()):
        leaf = k.split(".")[-1]
        parts.append(f"{leaf}_{_format_val_for_name(overrides[k])}")
    return "__".join(parts)

def _expand_abm_grid(cfg):
    """
    abm_grid:
      - modes: [singletons_phase2_fit71plus]
        base_params: {initial_singleton_count: 20000}
        overrides:
          phenotypes.proliferative.fragment_rate: [0.0008, 0.01]
          merge.p_merge: [0.56, 0.9]
    """
    grid = cfg.get("abm_grid")
    if not grid: return []
    specs = grid if isinstance(grid, list) else [grid]
    out=[]
    for spec in specs:
        modes = spec.get("modes", [])
        base  = spec.get("base_params", {}) or {}
        ovmap = spec.get("overrides", {}) or {}
        combos = _cartesian(ovmap)
        for m in modes:
            for ov in combos:
                ap = dict(base)
                if ov: ap["overrides"] = ov
                out.append({"mode": m, "abm_params": ap})
    return out
# -------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "pipeline_config.yaml"))
    args = p.parse_args()

    cfg = load_cfg(args.config)
    results_root = os.path.abspath(cfg.get("results_root", "pipeline_results"))
    run_dir = make_run_dir(results_root)

    init_run_status(run_dir); update_run_status(run_dir, {"status": "running"})
    save_run_input(run_dir, cfg)

    # NEW: combine explicit abm_sweep with expanded abm_grid
    explicit = cfg.get("abm_sweep", []) or []
    expanded = _expand_abm_grid(cfg)
    sweep_list = explicit + expanded

    timings = {"scenarios": {}}
    _abm_rows = []

    with timer() as t_all:
        for i, scenario in enumerate(sweep_list):
            mode = scenario["mode"]
            abm_params = scenario.get("abm_params", {})
            abm_cfg = {**cfg["abm"], "mode": mode, "abm_params": abm_params}

            # NEW: stable, parameter-aware scenario name
            scenario_name = scenario.get("name") or _make_scenario_name(i, mode, abm_params.get("overrides", {}))
            scenario_dir  = os.path.join(run_dir, scenario_name)
            fsim_dir      = os.path.join(scenario_dir, "forward_sims")
            os.makedirs(scenario_dir, exist_ok=True)

            update_run_status(run_dir, {"sections": {scenario_name: {"abm": {"started": utcnow()}}}})
            with timer() as t_abm:
                mean_csv = run_forward_sims(abm_cfg, fsim_dir)
            timings["scenarios"].setdefault(scenario_name, {})
            timings["scenarios"][scenario_name]["abm_seconds"] = round(t_abm.seconds, 3)
            update_run_status(run_dir, {"sections": {scenario_name: {"abm": {
                "finished": utcnow(), "duration_seconds": round(t_abm.seconds, 3)
            }}}})

            # scenario metadata row (for the master CSV)
            row = {
                "scenario": scenario_name,
                "mode": mode,
                "n_runs": abm_cfg.get("n_runs", 100),
                "n_workers": abm_cfg.get("n_workers", 8),
                "seed": abm_cfg.get("seed", 42),
                "movement_phase": 2 if mode in ["data_t71_phase2","singletons_phase2_fit71plus","singletons_phase2_fit_all"] else 1,
                "start_step": 71 if mode == "data_t71_phase2" else 1,
                "initial_singleton_count": abm_params.get("initial_singleton_count", None),
                "forward_means_csv": os.path.relpath(mean_csv, run_dir),
            }
            for k, v in _flatten(abm_params.get("overrides", {})).items():
                row[f"override.{k}"] = v
            _abm_rows.append(row)

            # data slice for fitting window
            df, times, data_values = slice_data(mean_csv, mode)

            # ODE fits for this scenario
            model_results = []
            for model_key in cfg["models"]:
                model_dir = os.path.join(scenario_dir, f"model_{model_key}")
                os.makedirs(model_dir, exist_ok=True)

                update_run_status(run_dir, {"sections": {scenario_name: {model_key: {"mle": {"started": utcnow()}}}}})
                with timer() as t_mle:
                    mle_out = run_mle(model_key, times, data_values,
                                      out_dir=os.path.join(model_dir, "mle"),
                                      model_meta_overrides=cfg.get("models_meta", {}).get(model_key, {}))
                timings["scenarios"].setdefault(scenario_name, {})
                timings["scenarios"][scenario_name].setdefault(model_key, {})
                timings["scenarios"][scenario_name][model_key]["mle_seconds"] = round(t_mle.seconds, 3)
                update_run_status(run_dir, {"sections": {scenario_name: {model_key: {"mle": {
                    "finished": utcnow(), "duration_seconds": round(t_mle.seconds, 3), "AIC": mle_out["AIC"]
                }}}}})

                def _progress(b, total, rhat):
                    update_run_status(run_dir, {"sections": {scenario_name: {model_key: {
                        "mcmc": {"current_block": b, "total_iters": total, "last_rhat": rhat}
                    }}}})

                update_run_status(run_dir, {"sections": {scenario_name: {model_key: {"mcmc": {"started": utcnow()}}}}})
                with timer() as t_mcmc:
                    mcmc_out = run_adaptive_mcmc(
                        model_key, times, data_values, mcmc_cfg=cfg["mcmc"], out_dir=model_dir,
                        model_meta_overrides=cfg.get("models_meta", {}).get(model_key, {}),
                        progress_callback=_progress
                    )
                timings["scenarios"][scenario_name][model_key]["mcmc_seconds"] = round(t_mcmc.seconds, 3)

                diag_dir = os.path.join(model_dir, "diagnostics")
                update_run_status(run_dir, {"sections": {scenario_name: {model_key: {"diagnostics": {"started": utcnow()}}}}})
                with timer() as t_diag:
                    run_diagnostics(model_key, mcmc_out["chains"], mcmc_out["post_burn_flat"],
                                    times, data_values, diag_dir,
                                    nsamples_ppc=cfg["mcmc"].get("nsamples_ppc", 300),
                                    seed=cfg["mcmc"].get("random_seed", 12345))
                timings["scenarios"][scenario_name][model_key]["diagnostics_seconds"] = round(t_diag.seconds, 3)
                update_run_status(run_dir, {"sections": {scenario_name: {model_key: {"diagnostics": {
                    "finished": utcnow(), "duration_seconds": round(t_diag.seconds, 3)
                }}}}})

                model_results.append({
                    "model_key": model_key,
                    "AIC": mle_out["AIC"],
                    "max_loglik": mle_out["max_loglik"],
                    "n_params": mle_out["k_params"],
                })

            write_aic_table(model_results, os.path.join(scenario_dir, "model_comparison.csv"))

    timings["pipeline_total_seconds"] = round(t_all.seconds, 3)
    write_timings(run_dir, timings)
    save_manifest(run_dir, cfg)
    _write_abm_sweep_summary(run_dir, _abm_rows)

    update_run_status(run_dir, {"status": "finished", "pipeline_total": {
        "finished": utcnow(), "duration_seconds": timings["pipeline_total_seconds"]
    }})
    print(f"\nPipeline complete. Run folder: {run_dir}")

if __name__ == "__main__":
    main()