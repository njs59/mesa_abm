import os, sys, json, yaml
from copy import deepcopy
from scripts import run_forward_sims_phase_2 as fwd


def _as_argv(args_dict):
    argv = [fwd.__file__]
    for k, v in args_dict.items():
        if isinstance(v, bool):
            if v:
                argv.append(f"--{k}")
            continue
        if v is None:
            continue
        argv.extend([f"--{k.replace('_','-')}", str(v)])
    return argv


# ---------- override merging ----------
def _deep_set(d, dotted_key, value):
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _merge_defaults(defaults_path, overrides, out_path):
    if not overrides:
        return defaults_path
    with open(defaults_path, "r") as f:
        base = yaml.safe_load(f) or {}
    base = deepcopy(base)
    for k, v in overrides.items():
        _deep_set(base, k, v)
    with open(out_path, "w") as f:
        yaml.safe_dump(base, f, sort_keys=False)
    return out_path


# ---------- ABM mode logic (unchanged 5‑mode behaviour) ----------
def generate_initial_condition(mode, background_dir, abm_params, scenario_dir):
    best_json = os.path.join(scenario_dir, "best_model.json")
    mean_json = os.path.join(scenario_dir, "mean_initial_clusters.json")

    if mode == "data_t71_phase2":
        return (
            os.path.join(background_dir, "best_model.json"),
            os.path.join(background_dir, "mean_initial_clusters.json"),
        )

    n = int(abm_params.get("initial_singleton_count", 20000))
    ic = {"cluster_sizes": [1] * n}

    with open(best_json, "w") as f:
        json.dump({"phase": 1}, f)
    with open(mean_json, "w") as f:
        json.dump(ic, f)
    return best_json, mean_json


def get_start_step(mode):
    return 71 if mode == "data_t71_phase2" else 1


def get_movement_phase(mode):
    if mode in ["data_t71_phase2",
                "singletons_phase2_fit71plus",
                "singletons_phase2_fit_all"]:
        return 2
    return 1


# ---------- main ABM caller ----------
def run_forward_sims(abm_cfg, scenario_dir):
    os.makedirs(scenario_dir, exist_ok=True)

    mode = abm_cfg["mode"]
    abm_params = abm_cfg.get("abm_params", {})
    overrides = abm_params.get("overrides", {})
    background = abm_cfg["background_dir"]

    # merge defaults only if needed
    merged_defaults_path = _merge_defaults(
        abm_cfg["defaults_yaml"],
        overrides,
        out_path=os.path.join(scenario_dir, "merged_defaults.yaml")
    )

    best_json, mean_json = generate_initial_condition(
        mode, background, abm_params, scenario_dir
    )

    arg_map = {
        "start-step": get_start_step(mode),
        "movement-phase": get_movement_phase(mode),
        "init-singletons": abm_params.get("initial_singleton_count", None),
        "n-runs": abm_cfg.get("n_runs", 100),
        "n-workers": abm_cfg.get("n_workers", 8),
        "seed": abm_cfg.get("seed", 42),
        "defaults": merged_defaults_path,
        "best-model": best_json,
        "mean-clusters": mean_json,
        "results-dir": scenario_dir,
    }

    bak = sys.argv[:]
    try:
        sys.argv = _as_argv(arg_map)
        fwd.main()
    finally:
        sys.argv = bak

    return os.path.join(scenario_dir, "forward_means_stats.csv")