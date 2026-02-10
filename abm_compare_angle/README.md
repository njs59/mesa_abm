# ABM Angular Model Comparison

This package compares **isotropic**, **persistent**, and **von Mises** angular models in your ABM, running **100 replicates** each and plotting mean ± **95% CI** for several spatial and cluster metrics.

## Structure

```
abm_compare/
├─ abm/
│  ├─ __init__.py
│  ├─ utils.py
│  ├─ clusters_model.py
│  └─ cluster_agent.py
├─ scripts/
│  └─ simulate_compare.py
├─ results/    # CSV/JSON outputs
└─ figs/       # Figures (PDF/PNG)
```

## How to run

```bash
python -m scripts.simulate_compare
```

- Simulation steps: **145** (configurable in `scripts/simulate_compare.py` via `STEPS`).
- Replicates per angular model: **100** (change `REPLICATES`).

## Metrics

**Spatial** (time series):
- Mean displacement from each agent’s first appearance
- Mean squared displacement (MSD)
- Convex hull area (monotone-chain hull)
- Radius of gyration (weighted by cluster size; also saves unweighted)

**Cluster statistics** (time series):
- S0 = number of clusters
- S1 = total cells (sum of sizes)
- S2 = sum of size²
- Largest cluster size
- Mean nearest-neighbour distance; plus final-time NND pooled histogram

> Note: Hull area in toroidal spaces may be distorted if populations straddle boundaries. If needed I can add a de-wrapping heuristic.

## Reproducibility
We seed per (model, replicate) using a deterministic scheme. NumPy draws are independent from Mesa’s `model.random` seed. If you need strict determinism via one RNG, I can unify the sources.
