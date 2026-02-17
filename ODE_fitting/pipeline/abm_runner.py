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

# --- merge per-scenario overrides into a scenario-specific defaults file ---
def _deep_set(d, dotted_key, value):
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value

def _merge_defaults_with_overrides(defaults_path, overrides_dict, out_path):
    """Load defaults, apply dotted-key overrides, write to out_path; return path."""
    if not overrides_dict:
        return defaults_path
    with open(defaults_path, "r") as f:
        base = yaml.safe_load(f) or {}
    base = deepcopy(base)
    for k, v in overrides_dict.items():
        if "." in k:
            _deep_set(base, k, v)
        else:
            base[k] = v
    with open(out_path, "w") as f:
        yaml.safe_dump(base, f, sort_keys=False)
    return out_path
# --------------------------------------------------------------------------

def generate_initial_condition(mode, background_dir, abm_params, scenario_dir):
    best_json = os.path.join(scenario_dir, "best_model.json")
    mean_json = os.path.join(scenario_dir, "mean_initial_clusters.json")
    if mode == "data_t71_phase2":
        return (os.path.join(background_dir, "best_model.json"),
                os.path.join(background_dir, "mean_initial_clusters.json"))

    # Singleton IC for modes 2–5
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
    if mode in ["data_t71_phase2", "singletons_phase2_fit71plus", "singletons_phase2_fit_all"]:
        return 2
    return 1

def run_forward_sims(abm_cfg, scenario_dir):
    os.makedirs(scenario_dir, exist_ok=True)

    mode = abm_cfg["mode"]
    abm_params = abm_cfg.get("abm_params", {})
    background = abm_cfg["background_dir"]

    # NEW: per-scenario overrides into merged defaults
    overrides = abm_params.get("overrides", {})
    merged_defaults = _merge_defaults_with_overrides(
        defaults_path=abm_cfg["defaults_yaml"],
        overrides_dict=overrides,
        out_path=os.path.join(scenario_dir, "merged_defaults.yaml")
    )

    best_json, mean_json = generate_initial_condition(mode, background, abm_params, scenario_dir)

    arg_map = {
        "start-step":       get_start_step(mode),
        "movement-phase":   get_movement_phase(mode),
        "init-singletons":  abm_params.get("initial_singleton_count", None),
        "n-runs":           abm_cfg.get("n_runs", 100),
        "n-workers":        abm_cfg.get("n_workers", 8),
        "seed":             abm_cfg.get("seed", 42),
        "defaults":         merged_defaults,        # use per-scenario merged defaults
        "best-model":       best_json,
        "mean-clusters":    mean_json,
        "results-dir":      scenario_dir,
    }

    _bak = sys.argv[:]
    try:
        sys.argv = _as_argv(arg_map)
        fwd.main()
    finally:
        sys.argv = _bak

    return os.path.join(scenario_dir, "forward_means_stats.csv")