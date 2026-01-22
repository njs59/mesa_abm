# ABC Pipeline for Motility-driven Clustering ABM (with Spatial Statistics)

This package runs **ABC-SMC** on your Mesa ABM and **extends the distance function** to include spatial summaries: `S0, S1, S2, NND_med, g_r40, g_r80`. It also provides rich diagnostics and posterior predictive checks.

## Folder layout

```
abc_project/
│
├── abm/                    # your Mesa ABM (unchanged)
│   ├── cluster_agent.py
│   ├── clusters_model.py
│   └── utils.py
│
├── abc/                    # ABC pipeline
│   ├── compute_summary.py  # snapshot+timeseries summaries including spatial stats
│   ├── distance.py         # Euclidean distance utilities
│   ├── abc_model_wrapper.py# map ABC particle -> model params
│   ├── priors.py           # Option 1 priors; reads priors.yaml if present
│   ├── run_abc.py          # run ABC-SMC from scratch
│   └── analyze_posterior.py# diagnostics + PPC (incl. spatial)
│
├── observed/
│   ├── INV_ABM_ready_summary.csv  # your time-series including spatial stats
│   ├── observed_final.csv         # single-row convenience file
│   └── observed_timeseries.csv    # duplicate of INV_ABM_ready_summary.csv
│
├── results/                # outputs (db, plots, csvs)
│
└── priors.yaml             # tighten priors here when needed
```

## Requirements

- Python 3.9+
- `mesa>=3.0.0`, `pyabc`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (optional), `pyyaml`

> Tip: Use a fresh virtual environment.

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install mesa pyabc numpy pandas matplotlib seaborn pyyaml
```

## Running ABC from scratch

```
python -m abcp.run_abc   --db results/abc_run.db   --popsize 200   --maxgen 12   --min_eps 0.5   --observed_ts observed/INV_ABM_ready_summary.csv   --t_start 22   --total_steps 300   --priors_yaml priors.yaml   --motion isotropic   --speed constant   --workers 1
```

- **Distance**: Straight Euclidean on the **flattened time-series** of all six statistics (most obvious, transparent choice).
- **Initial priors (Option 1)** are implemented in `abc/priors.py`. To **tighten**, edit `priors.yaml` (bounds only) and re-run.

## Diagnostics and Posterior Predictive Checks

After ABC completes:

```
python -m abcp.analyze_posterior --db results/abc_run.db   --observed_ts observed/INV_ABM_ready_summary.csv   --t_start 22 --total_steps 300 --pp 100   --motion isotropic --speed constant
```

This generates in `results/`:

- `epsilon_trajectory.png` — convergence overview
- `marginal_*.png` — parameter marginals
- `posterior_correlation.png` — pairwise correlations
- `ppc_ts_*.png` — time-series PPC for each statistic (incl. spatial)
- `ppc_ts_* .csv` — median/q05/q95 bands
- `ppc_discrepancy.png` & `ppc_discrepancy.csv` — discrepancy across time

## Notes

- The spatial statistics use **toroidal minimal-image distances** aligned with your ABM domain.
- Pair-correlation `g(r)` uses a ring half-width of **±2 units** around r=40 and r=80. Adjust in `compute_summary.py` if needed.
- If you prefer **final-time only** ABC instead of time-series, we can switch to snapshot mode easily; ask and I’ll provide a variant.
- To switch motion to **persistent** or use a **speed distribution**, add the corresponding prior bounds to `priors.yaml` (e.g., `speed_meanlog`, `speed_sdlog`, `heading_sigma`) and set `--motion persistent` / `--speed lognorm`.

## Reproducibility

- The model seed is passed via `--seed` (for ABC model factory and PPC simulations).
- ABCSMC uses its own internal RNG; set environment variable `PYABC_SEED` if you need deterministic behaviour end-to-end.

---

**Author**: Generated for Nathan Schofield — British English spelling.
