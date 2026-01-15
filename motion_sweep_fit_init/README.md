
# Motion Model Grid (ABC–SMC with IQR-scaled Wasserstein)

This package runs a reproducible comparison of **agent motion types** (isotropic vs persistent) and **speed laws** (constant, lognormal, gamma, Weibull) for your Mesa ABM, using **IQR-scaled Wasserstein** distance across S0, S1, S2.

## Structure
- `clusters_abm/` — ABM package (your uploaded files).
- `run_motion_grid.py` — single script to run all variants and seeds, saving DBs and a summary CSV.
- `results/` — outputs (SQLite DBs per run, `summary.csv`).

## Requirements
```
pip install pyabc mesa numpy pandas scipy matplotlib
```

## Quick start
```
python motiongrid_pkg/run_motion_grid.py --obs_csv INV_summary_stats.csv --dataset INV
# or
python motiongrid_pkg/run_motion_grid.py --obs_csv PRO_summary_stats.csv --dataset PRO
```

Optional flags:
- `--seeds 42 123 2026`
- `--popsize 200 --max_pops 10 --min_eps 0.5`
- `--start_step 20`
- `--init_cells_fixed 0` (0 = infer from S0[0])
- `--pp_samples 50`

Outputs:
- `results/{dataset}_{motion}_{speed}_{seed}.db` — pyABC SQLite DB per run
- `results/summary.csv` — per-run metrics
