import argparse
from .parameter_sweep import run_parameter_sweep
from .utils_sens import load_yaml

def main():
    ap = argparse.ArgumentParser(description="Parameter_sweeps entry point")
    ap.add_argument('-c', '--config', required=True, help='Path to YAML config file')
    args = ap.parse_args()

    # Run the sweep
    run_parameter_sweep(args.config)

    # Post-processing (kept separate from the sweep module)
    cfg = load_yaml(args.config)
    results_root = cfg.get("results_root", "Parameter_sweeps_results")
    abm_cfg = cfg.get("abm", {})
    save_raw = bool(abm_cfg.get("save_raw", True))

    post_cfg = cfg.get("post", {})
    # 1) Extra plots
    extra_cfg = post_cfg.get("extra_plots", {})
    if bool(extra_cfg.get("enable", False)):
        try:
            from .make_extra_plots import run_extra_plots
            run_extra_plots(
                results_root,
                run='latest',
                metrics_per_page=int(extra_cfg.get("metrics_per_page", 10)),
                params_per_page=int(extra_cfg.get("params_per_page", 4)),
                per_metric_params_per_page=int(extra_cfg.get("per_metric_params_per_page", 6)),
            )
        except Exception as e:
            print(f"[WARN] extra plots failed: {e}")

    # 2) Coagulation analysis
    coag_cfg = post_cfg.get("coagulation", {})
    if bool(coag_cfg.get("enable", False)):
        if not save_raw:
            print("[WARN] coagulation analysis requires abm.save_raw=true; skipping.")
        else:
            try:
                from .make_coagulation_analysis import run_coagulation_analysis
                run_coagulation_analysis(
                    results_root,
                    run='latest',
                    window=int(coag_cfg.get("smooth_window", 20)),
                )
            except Exception as e:
                print(f"[WARN] coagulation analysis failed: {e}")

if __name__ == '__main__':
    main()