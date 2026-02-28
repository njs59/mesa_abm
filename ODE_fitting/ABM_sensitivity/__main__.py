"""
ABM_sensitivity.__main__
========================
CLI entrypoint with safe threading + start method.
"""

import os

# ---- Limit BLAS/OpenMP threads per process BEFORE importing NumPy/Matplotlib ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
from .sensitivity_runner import run_sensitivity


def main():
    # Use 'spawn' to avoid fork-related hangs with OpenMP/BLAS
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    here = os.path.dirname(os.path.abspath(__file__))
    default_cfg = os.path.join(here, "config.yaml")

    p = argparse.ArgumentParser(description="Run ABM sensitivity analysis.")
    p.add_argument(
        "--config", "-c",
        type=str,
        default=default_cfg,
        help=f"Path to YAML config file (default: {default_cfg})"
    )
    args = p.parse_args()

    run_sensitivity(args.config)


if __name__ == "__main__":
    main()