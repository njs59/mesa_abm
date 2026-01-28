
# ABM with Persistent Noisy Radius + Single Merge Strength + Repulsive Relaxation

This package updates your ABM to include:

1. **Persistent per‑agent radius noise**: each cluster draws a lognormal multiplier once and keeps it for life; after any size change, `radius = multiplier * radius_from_size_3d(size)`.
2. **Single merge parameter**: `merge.strength` replaces previous `adhesion` and `prob_contact_merge`.
3. **Slight repulsive relaxation** at the end of each timestep to fully remove overlaps so clusters occupy space without interpenetration.
4. **ABC scripts** for different movement models (constant speed, lognormal, gamma) and plotting utilities.

## Structure
```
abm_updated_package/
  abm/
    __init__.py
    utils.py
    cluster_agent.py
    clusters_model.py
  scripts/
    run_sim.py
    plot_nnd.py
    abc_rejection.py
    plot_abc_posteriors.py
  configs/
  data/
    observed_summary.example.json
```

## Quickstart

### 1) Run a simulation
```bash
python scripts/run_sim.py --out results/run1 --steps 300 \
  --movement constant --direction isotropic --speed 1.0 \
  --merge_strength 0.6 --sigma 0.35 --preserve area
```
This writes `results/run1/state_timeseries.csv` and `params.json`.

### 2) Plot mean NND over time (no wrap metric)
```bash
python scripts/plot_nnd.py --csv results/run1/state_timeseries.csv --out plots/nnd_run1.png
```

### 3) ABC rejection for movement models
Prepare a JSON of observed summaries (see `data/observed_summary.example.json`). Then run:
```bash
python scripts/abc_rejection.py --observed data/observed_summary.example.json --out results/abc --n 200 --keep 40 --model all
```
This saves `abc_results.json` and CSVs of accepted summaries per model. Plot simple posteriors, e.g.:
```bash
python scripts/plot_abc_posteriors.py --abc_json results/abc/abc_results.json --model lognorm --out plots/post_lognorm.png
```

## Notes
- The simulation **remains toroidal**, but all NND utilities in scripts compute distances **without wrap** (to mimic a finite FOV measurement).
- Repulsion is controlled via `repulsion` in `utils.DEFAULTS` (`max_iter`, `eps`). Increase `max_iter` for denser systems.
- Merge radius after merging uses the persistent multiplier if `radius_noise.apply_after_merge=True`; set it `False` to keep strict volume‑conserving radius at merges.
- Movement models:
  - `constant`: step = `speed_base * dt`.
  - `distribution`: choose `dist=lognorm|gamma` with parameters in `movement.dist_params`.

## Requirements
- Python 3.10+
- `mesa` v3+, `numpy`, `pandas`, `matplotlib` (for plotting scripts)

```
pip install mesa numpy pandas matplotlib pyarrow
```

