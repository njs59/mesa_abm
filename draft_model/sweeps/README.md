
# sweeps — Parameter sweep utilities for clusters ABM

This subpackage provides a simple, scriptable way to run parameter sweeps for your ABM and plot the results.

## Prerequisites
- Python 3.10+ (tested on 3.11+)
- Packages: `numpy`, `pandas`, `matplotlib`
- Your model code as a package: `clusters_abm` (with `clusters_model.py`, `utils.py`, etc.)

## Files
- `sweeps/run.py` — runs **one** simulation and returns summary metrics.
- `sweeps/grid.py` — runs a **grid** of parameter combinations with replicates; optional parallel workers.
- `sweeps/plot_sweeps.py` — creates heatmaps/lines from the sweep CSV.

## Quick start (from project root)

### Run a basic sweep
```bash
python -m sweeps.grid --steps 300 --seeds 5   --merge_prob 0.3 0.6 0.9   --invasive_speed 1.0 2.0 3.0 4.0
```
This writes `results/sweeps.csv`.

### Use multiple cores
```bash
python -m sweeps.grid --steps 300 --seeds 5 --jobs 4   --merge_prob 0.3 0.6 0.9   --invasive_speed 1.0 2.0 3.0 4.0
```

### Plot the sweep results
```bash
python -m sweeps.plot_sweeps --in results/sweeps.csv --outdir figs_sweeps
```
Outputs:
- `figs_sweeps/heatmap_mean_size_final.png`
- `figs_sweeps/lines_n_final_vs_merge.png`
- `figs_sweeps/lines_total_cells_vs_invasive_speed.png`

## Adding more parameters
You can extend the grid by adding CLI options and mapping them to dot-path overrides in `grid.py`. Examples:
- `--proliferative_speed 0.5 1.0 2.0` → `"phenotypes.proliferative.speed_base"`
- `--dt 0.5 1.0 2.0` → `"time.dt"`
- `--merge_prob 0.1 0.5 0.9` → `"merge.prob_contact_merge"`

The override system uses **dot-paths** to set nested values inside `DEFAULTS` before each run.

## Reproducibility notes
- Each replicate uses `seed = base_seed + rep` (base_seed=42). Adjust in `grid.py` if needed.
- Parallel execution uses `multiprocessing` with the `spawn` context (portable across macOS/Linux/Windows).

## Troubleshooting
- If imports fail (e.g., `ModuleNotFoundError: clusters_abm`), ensure your project root contains `clusters_abm/__init__.py` and run commands **from the root**.
- For headless servers, `plot_sweeps.py` sets Matplotlib to the `Agg` backend.

Happy sweeping!
