#!/usr/bin/env python3
import os; os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"; os.environ["NUMEXPR_NUM_THREADS"]="1"
import yaml, argparse, pandas as pd, csv, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend (safe with savefig)
import matplotlib.pyplot as plt
import seaborn as sns

from .abm_runner import run_forward_sims
from .mle_runner import run_mle
from .mcmc_runner import run_adaptive_mcmc
from .diagnostics_runner import run_diagnostics
from .aic_tools import write_aic_table
from .model_registry import get_model_meta
from .utils_logging import (
    make_run_dir, save_manifest, save_run_input,
    timer, write_timings, init_run_status, update_run_status, utcnow
)

# -------------------- config helpers --------------------
def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def slice_data(mean_csv, mode):
    df = pd.read_csv(mean_csv)
    if mode in ["data_t71_phase2", "singletons_phase2_fit71plus", "singletons_phase1_to_2_fit71plus"]:
        df = df[df["step"] >= 71].reset_index(drop=True)
    return (
        df,
        df["step"].to_numpy(),
        df[["num_clusters","mean_cluster_size","mean_squared_cluster_size"]].to_numpy()
    )

def _flatten(d, prefix=""):
    out={}
    for k,v in (d or {}).items():
        kk=f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict): out.update(_flatten(v, kk))
        else: out[kk]=v
    return out

def _write_abm_sweep_summary(run_dir, rows):
    if not rows: return
    allkeys = sorted({k for r in rows for k in r.keys()})
    path=os.path.join(run_dir,"abm_sweep_summary.csv")
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=allkeys); w.writeheader(); w.writerows(rows)
    return path

def _cartesian(overrides_dict):
    if not overrides_dict: return [{}]
    keys=list(overrides_dict.keys())
    vals=[(v if isinstance(v,(list,tuple)) else [v]) for v in overrides_dict.values()]
    return [{k:v for k,v in zip(keys, combo)} for combo in itertools.product(*vals)]

def _format_val_for_name(v):
    return str(v).replace(".","p") if isinstance(v,float) else str(v)

def _make_scenario_name(idx, mode, overrides):
    parts=[f"abm_{idx:02d}_{mode}"]
    for k in sorted(overrides.keys()):
        leaf=k.split(".")[-1]
        parts.append(f"{leaf}_{_format_val_for_name(overrides[k])}")
    return "__".join(parts)

def _expand_param_sweep(cfg, base_scenarios):
    sweep = cfg.get("abm_param_sweep")
    if not sweep: return base_scenarios
    expanded=[]
    for block in sweep:
        apply_modes = block.get("apply_to_modes", ["*"])
        overrides = block.get("overrides", {})
        combos = _cartesian(overrides)
        for base in base_scenarios:
            mode = base["mode"]
            applies = ("*" in apply_modes) or (mode in apply_modes)
            if not applies:
                expanded.append(base); continue
            for ov in combos:
                ap = dict(base.get("abm_params", {}))
                ap["overrides"] = ov
                expanded.append({"mode": mode, "abm_params": ap})
    # de-duplicate
    unique=[]; seen=set()
    def _froz(d):
        if not d: return ()
        t=[]
        for k,v in sorted(d.items()):
            if isinstance(v, dict):
                t.append((k, tuple(sorted(_flatten(v).items()))))
            else:
                t.append((k,v))
        return tuple(t)
    for sc in expanded:
        key=(sc["mode"], _froz(sc.get("abm_params")))
        if key not in seen:
            seen.add(key); unique.append(sc)
    return unique

# -------------------- sweep-level plotting (per model) --------------------
def _ensure(dirpath):
    os.makedirs(dirpath, exist_ok=True)

# ===== Helpers for compact, nicer labels (minimal-impact changes) =====
# Map movement modes to compact integers for legend/labels
MODE_MAP = {
    "data_t71_phase2": 1,
    "singletons_phase2_fit71plus": 2,
    "singletons_phase2_fit_all": 3,
    "singletons_phase1_to_2_fit71plus": 4,
    "singletons_phase1_to_2_fit_all": 5,
}

def _parse_mode_from_scenario_name(scenario_name: str):
    """From 'abm_03_data_t71_phase2__alpha_0p1' return 'data_t71_phase2'."""
    head = scenario_name.split("__")[0]
    bits = head.split("_", 2)
    return bits[2] if len(bits) >= 3 else None

def _prettify_key(k: str) -> str:
    """Replace hyphens/underscores with spaces for nicer display."""
    return k.replace("-", " ").replace("_", " ")

def _format_number_like(v_str: str) -> str:
    """
    Parse numeric string to float and format nicely:
      - integers shown as '3'
      - otherwise up to 6 significant figures
      - very small/large values use scientific notation
      - no trailing zeros/dot
    Fallback to original if parsing fails.
    """
    try:
        v = float(v_str)
        if np.isfinite(v):
            if abs(v) >= 1e-3 and abs(v) < 1e4:
                s = f"{v:.6g}"
            else:
                s = f"{v:.3e}"
            if "e" not in s and "." in s:
                s = s.rstrip("0").rstrip(".")
            return s
        else:
            return v_str
    except Exception:
        return v_str

def _overrides_from_scenario_name(scenario_name: str):
    """
    From '...__alpha_0p1__beta_2p0' return [('alpha','0.1'), ('beta','2')], prettified.
    """
    overrides = []
    for part in scenario_name.split("__")[1:]:
        if "_" in part:
            k, v = part.split("_", 1)
            v = v.replace("p", ".")
            overrides.append((_prettify_key(k), _format_number_like(v)))
    return overrides

def _label_for_scenario(scenario_name: str, include_mode: bool = False) -> str:
    """
    Build a compact label such as:
        'mode=1, alpha = 1, beta = 0.0025'      (if include_mode=True)
        'alpha = 1, beta = 0.0025'              (otherwise)
    If there are no overrides, returns 'mode=<'n'>' if include_mode else the original name.
    """
    pieces = []
    if include_mode:
        mode = _parse_mode_from_scenario_name(scenario_name)
        if mode and mode in MODE_MAP:
            pieces.append(f"mode={MODE_MAP[mode]}")
    items = sorted(_overrides_from_scenario_name(scenario_name), key=lambda kv: kv[0])
    for k, v in items:
        pieces.append(f"{k} = {v}")
    if pieces:
        return ", ".join(pieces)
    if include_mode:
        m = _parse_mode_from_scenario_name(scenario_name)
        return f"mode={MODE_MAP.get(m, '')}"
    return scenario_name
# ============================ end helpers ============================

def _plot_model_violin(out_dir, model_key, records, max_per_scenario=5000):
    """Violin of marginals across scenarios for one model."""
    if not records: return

    # Include mode in labels only if multiple modes present
    modes_present = set()
    for rec in records:
        scen = rec["scenario"]
        m = _parse_mode_from_scenario_name(scen)
        if m: modes_present.add(m)
    include_mode = (len(modes_present) > 1)

    frames=[]
    for rec in records:
        sam = rec["samples"]
        if sam.shape[0] > max_per_scenario:
            idx = np.random.default_rng(42).choice(sam.shape[0], max_per_scenario, replace=False)
            sam = sam[idx]
        df = pd.DataFrame(sam, columns=rec["param_names"]).melt(
            var_name="parameter", value_name="value")
        df["scenario"] = _label_for_scenario(rec["scenario"], include_mode=include_mode)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    plt.figure(figsize=(max(10, 2 + 1.6*all_df["parameter"].nunique()),
                        max(6, 0.45*all_df["scenario"].nunique())))
    sns.violinplot(
        data=all_df, x="value", y="scenario", hue="parameter",
        inner="quartile", density_norm="area", cut=0, bw_method="scott"
    )

    # Log scale with safe fallback
    try:
        if (all_df["value"] > 0).all():
            plt.xscale("log")
        else:
            plt.xscale("symlog", linthresh=1e-6)
    except Exception:
        pass

    plt.title(f"Across-scenario posterior marginals — {model_key}")
    plt.xlabel("Parameter value"); plt.ylabel("Scenario")
    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", title="Parameter", frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "marginals_violin.png"), dpi=150)
    plt.clf(); plt.close('all')

def _plot_model_aic_bar(out_dir, model_key, aic_rows):
    if not aic_rows: return
    df = pd.DataFrame(aic_rows).sort_values("AIC", ascending=True)

    modes_present = set()
    for s in df["scenario"]:
        m = _parse_mode_from_scenario_name(s)
        if m: modes_present.add(m)
    include_mode = (len(modes_present) > 1)

    df["scenario"] = df["scenario"].apply(lambda s: _label_for_scenario(s, include_mode=include_mode))

    plt.figure(figsize=(10, max(4, 0.35*len(df))))
    sns.barplot(data=df, x="AIC", y="scenario", orient="h", color="#4c78a8")
    plt.title(f"AIC across scenarios — {model_key}")
    plt.xlabel("AIC (lower is better)"); plt.ylabel("Scenario")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "AIC_bar.png"), dpi=150)
    plt.clf(); plt.close('all')

def _plot_model_overlaid_kde(out_dir, model_key, records, max_per_scenario=8000):
    """
    For each parameter index, draw a line KDE overlaid across scenarios.
    Saves: marginals_overlaid_param_<idx>_<name>.png
    """
    if not records: return

    # assume all records share the same param_names order
    param_names = records[0]["param_names"]

    modes_present = set()
    for rec in records:
        m = _parse_mode_from_scenario_name(rec["scenario"])
        if m: modes_present.add(m)
    include_mode = (len(modes_present) > 1)

    data_by_scenario = []
    for rec in records:
        sam = rec["samples"]
        if sam.shape[0] > max_per_scenario:
            idx = np.random.default_rng(123).choice(sam.shape[0], max_per_scenario, replace=False)
            sam = sam[idx]
        data_by_scenario.append((rec["scenario"], sam))

    palette = sns.color_palette("tab10", n_colors=len(records))
    for j, pname in enumerate(param_names):
        plt.figure(figsize=(10, 6))
        for (scen, sam), col in zip(data_by_scenario, palette):
            short = _label_for_scenario(scen, include_mode=include_mode)
            sns.kdeplot(
                x=sam[:, j], fill=False, common_norm=False,
                label=short, lw=2, color=col
            )
        plt.title(f"{model_key} — Overlaid KDE: {pname}")
        plt.xlabel("Value"); plt.ylabel("Density")
        plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", frameon=False)
        plt.tight_layout()
        safe = pname.strip("$").replace("\\", "").replace("{","").replace("}","").replace("^","").replace("_","")
        fname = f"marginals_overlaid_param_{j+1}_{safe}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.clf(); plt.close('all')

# ----------------------------- MAIN -----------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=os.path.join(os.path.dirname(__file__), "pipeline_config.yaml"))
    args = p.parse_args()

    cfg = load_cfg(args.config)
    results_root = os.path.abspath(cfg.get("results_root", "pipeline_results"))
    run_dir = make_run_dir(results_root)

    init_run_status(run_dir); update_run_status(run_dir, {"status":"running"})
    save_run_input(run_dir, cfg)

    base_scenarios = cfg.get("abm_sweep", []) or []
    sweep_list = _expand_param_sweep(cfg, base_scenarios)

    timings = {"scenarios": {}}
    summary_rows = []

    # sweep-level collectors (per model)
    post_cache = {} # { model_key: [ {scenario, samples, param_names}, ... ] }
    aic_cache = {}  # { model_key: [ {scenario, AIC}, ... ] }

    with timer() as t_all:
        for i, scenario in enumerate(sweep_list):
            mode = scenario["mode"]
            abm_params = scenario.get("abm_params", {})
            overrides = abm_params.get("overrides", {})
            scenario_name = _make_scenario_name(i, mode, overrides)
            scenario_dir = os.path.join(run_dir, scenario_name)
            fsim_dir = os.path.join(scenario_dir, "forward_sims")
            os.makedirs(scenario_dir, exist_ok=True)

            abm_cfg = {**cfg["abm"], "mode": mode, "abm_params": abm_params}

            update_run_status(run_dir, {"sections":{scenario_name:{"abm":{"started":utcnow()}}}})
            with timer() as t_abm:
                mean_csv = run_forward_sims(abm_cfg, fsim_dir)
            timings["scenarios"].setdefault(scenario_name, {})
            timings["scenarios"][scenario_name]["abm_seconds"] = round(t_abm.seconds,3)
            update_run_status(run_dir, {"sections":{scenario_name:{"abm":{
                "finished":utcnow(), "duration_seconds":round(t_abm.seconds,3)
            }}}})

            df, times, data_values = slice_data(mean_csv, mode)

            # sweep summary row
            row = {
                "scenario": scenario_name, "mode": mode,
                "init_singletons": abm_params.get("initial_singleton_count", None),
                "forward_means_csv": os.path.relpath(mean_csv, run_dir),
            }
            for k,v in _flatten(overrides).items():
                row[f"override.{k}"] = v
            summary_rows.append(row)

            # Fit each ODE model for this scenario
            model_results = []
            for model_key in cfg["models"]:
                model_dir = os.path.join(scenario_dir, f"model_{model_key}")
                os.makedirs(model_dir, exist_ok=True)

                update_run_status(run_dir, {"sections":{scenario_name:{model_key:{"mle":{"started":utcnow()}}}}})
                with timer() as t_mle:
                    mle_out = run_mle(
                        model_key, times, data_values,
                        out_dir=os.path.join(model_dir, "mle"),
                        model_meta_overrides=cfg.get("models_meta", {}).get(model_key, {})
                    )
                timings["scenarios"].setdefault(scenario_name, {})
                timings["scenarios"][scenario_name].setdefault(model_key, {})
                timings["scenarios"][scenario_name][model_key]["mle_seconds"] = round(t_mle.seconds,3)
                update_run_status(run_dir, {"sections":{scenario_name:{model_key:{"mle":{
                    "finished":utcnow(), "duration_seconds":round(t_mle.seconds,3), "AIC":mle_out["AIC"]
                }}}}})

                def _progress(b,total,rhat):
                    update_run_status(run_dir, {"sections":{scenario_name:{model_key:{
                        "mcmc":{"current_block":b, "total_iters":total, "last_rhat":rhat}
                    }}}})

                update_run_status(run_dir, {"sections":{scenario_name:{model_key:{"mcmc":{"started":utcnow()}}}}})
                with timer() as t_mcmc:
                    mcmc_out = run_adaptive_mcmc(
                        model_key, times, data_values,
                        mcmc_cfg=cfg["mcmc"], out_dir=model_dir,
                        model_meta_overrides=cfg.get("models_meta", {}).get(model_key, {}),
                        progress_callback=_progress
                    )
                timings["scenarios"][scenario_name][model_key]["mcmc_seconds"] = round(t_mcmc.seconds,3)

                # Per-scenario diagnostics (scatter + KDE pairwise written here)
                diag_dir = os.path.join(model_dir, "diagnostics")
                update_run_status(run_dir, {"sections":{scenario_name:{model_key:{"diagnostics":{"started":utcnow()}}}}})
                with timer() as t_diag:
                    run_diagnostics(
                        model_key, mcmc_out["chains"], mcmc_out["post_burn_flat"],
                        times, data_values, diag_dir,
                        nsamples_ppc=cfg["mcmc"].get("nsamples_ppc",300),
                        seed=cfg["mcmc"].get("random_seed",12345)
                    )
                timings["scenarios"][scenario_name][model_key]["diagnostics_seconds"] = round(t_diag.seconds,3)
                update_run_status(run_dir, {"sections":{scenario_name:{model_key:{"diagnostics":{
                    "finished":utcnow(), "duration_seconds":round(t_diag.seconds,3)
                }}}}})

                model_results.append({
                    "model_key": model_key,
                    "AIC": mle_out["AIC"],
                    "max_loglik": mle_out["max_loglik"],
                    "n_params": mle_out["k_params"],
                })

                # cache for summary plots per model
                meta = get_model_meta(model_key, cfg.get("models_meta", {}).get(model_key, {}))
                param_names = [f"${n}$" for n in meta["param_names"]] + [r"$\sigma_0$", r"$\sigma_1$", r"$\sigma_2$"]
                post_cache.setdefault(model_key, []).append({
                    "scenario": scenario_name,
                    "samples": mcmc_out["post_burn_flat"],
                    "param_names": param_names
                })
                aic_cache.setdefault(model_key, []).append({
                    "scenario": scenario_name,
                    "AIC": mle_out["AIC"]
                })

            # per-scenario model comparison
            write_aic_table(model_results, os.path.join(scenario_dir, "model_comparison.csv"))

    # end loop — write sweep summary
    timings["pipeline_total_seconds"] = round(t_all.seconds,3)
    write_timings(run_dir, timings)
    save_manifest(run_dir, cfg)
    _write_abm_sweep_summary(run_dir, summary_rows)

    # ------------------------ MODEL-WISE SUMMARY FOLDERS ------------------------
    for model_key, recs in post_cache.items():
        model_sum_dir = os.path.join(run_dir, f"summary_{model_key}")
        _ensure(model_sum_dir)
        # Violin (across scenarios)
        _plot_model_violin(model_sum_dir, model_key, recs, max_per_scenario=5000)
        # Overlaid KDE (one figure per parameter)
        _plot_model_overlaid_kde(model_sum_dir, model_key, recs, max_per_scenario=8000)
        # AIC bar per model
        _plot_model_aic_bar(model_sum_dir, model_key, aic_cache.get(model_key, []))

    update_run_status(run_dir, {
        "status":"finished",
        "pipeline_total":{"finished":utcnow(), "duration_seconds":timings["pipeline_total_seconds"]}
    })
    print(f"\nPipeline complete. Run folder: {run_dir}")

if __name__ == "__main__":
    main()