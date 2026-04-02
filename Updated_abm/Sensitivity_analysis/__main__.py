# Sensitivity_analysis/__main__.py
import argparse, os
from .sobol_runner import run_sobol, SAConfig, AdaptiveR
from .plot_sobol_results import main as plot_basic_main
from .sobol_extra_plots import main as plot_neat_main

def main():
    ap = argparse.ArgumentParser(description="Sobol sensitivity analysis (ABM) with dummy parameter")
    ap.add_argument("-n", "--n_power_of_two", type=int, default=256)
    ap.add_argument("--steps", type=int, default=145)
    ap.add_argument("--force-phase2", action="store_true", default=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--output-dir", type=str, default="Sensitivity_analysis_results")
    ap.add_argument("--ci-resamples", type=int, default=500)
    ap.add_argument("--rel-sem-target", type=float, default=0.07)
    ap.add_argument("--rmin", type=int, default=6)
    ap.add_argument("--rstep", type=int, default=2)
    ap.add_argument("--rmax", type=int, default=16)
    ap.add_argument("--chunksize", type=int, default=16)
    # NEW:
    ap.add_argument("--metrics-set", type=str, choices=["core4","core5"], default="core5",
                    help="Choose 'core4' (no Clark-Evans) or 'core5' (all metrics)")
    args = ap.parse_args()

    for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    cfg = SAConfig(steps=args.steps, force_phase2=args.force_phase2, replicates=10)
    adaptive = AdaptiveR(enable=True, R_min=args.rmin, R_step=args.rstep, R_max=args.rmax,
                         rel_sem_target=args.rel_sem_target)

    _, _, out_dir = run_sobol(
        n_power_of_two=args.n_power_of_two,
        cfg=cfg,
        ci_resamples=args.ci_resamples,
        output_dir=args.output_dir,
        n_workers=args.workers,
        chunksize=args.chunksize,
        adaptive=adaptive,
        metrics_set=args.metrics_set,    # NEW
    )

    # Basic and neat plots (latest results dir by default)
    plot_basic_main()
    plot_neat_main()

if __name__ == "__main__":
    main()