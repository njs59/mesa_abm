
# new_sweep (new_results defaults)

All CSVs and PNGs are written under `new_results/` by default.

## Basic grid
python -m new_sweep.grid   --merge_prob 0.5 0.7 0.9   --invasive_speed 2 3 4   --proliferative_speed 0.8 1.0   --dt 0.5   --init_phenotype invasive   --seeds 5 --jobs 4 --steps 300   --out_csv new_results/sweeps.csv

python -m new_sweep.plot_sweeps --in new_results/sweeps.csv --outdir new_results/figs



python -m new_sweep.grid_extended --init_phenotype invasive --merge_prob 0.5 0.6 0.7 0.8 0.9 --inv_speed 2.0 2.5 3.0 3.5 4.0 4.5 5.0 \
  --dt 0.5 \
  --seeds 10 --jobs 4 --steps 300 \
  --out_csv new_results/sweeps_invasive_min_2.csv

python -m new_sweep.plot_heatmaps \
  --in_csv new_results/sweeps_invasive_min_2.csv \
  --outdir new_results/figs


python -m new_sweep.grid_extended \
  --init_phenotype proliferative invasive \
  --merge_prob 0.6 0.9 \
  --inv_speed 2.0 4.0 \
  --pro_speed 1.0 \
  --seeds 5 --jobs 4 --steps 300 \
  --out_csv new_results/sweeps_initph_both.csv


## Extended grid
python -m new_sweep.grid_extended   --merge_prob 0.6 0.8 0.95   --dt 0.5 1.0   --inv_speed 2.0 3.0   --pro_speed 0.8 1.2   --inv_exp 0.0 0.25   --pro_exp 0.0 0.25   --inv_adh 0.3 0.6   --pro_adh 0.4 0.6   --inv_prolif 0.002 0.004   --pro_prolif 0.005 0.007   --inv_frag 0.0005 0.001   --pro_frag 0.0002 0.0005   --inv_frag_exp 0.0 1.0   --pro_frag_exp 0.0   --init_phenotype proliferative   --seeds 5 --jobs 4 --steps 300   --out_csv new_results/sweeps_extended.csv

python -m new_sweep.plot_heatmaps --in_csv new_results/sweeps_extended.csv --outdir new_results/figs
python -m new_sweep.plot_pairplots --in_csv new_results/sweeps_extended.csv --out_png new_results/figs/pairs.png

## Time-series
python -m new_sweep.run_timeseries --init_phenotype invasive --steps 200 --seed 2026 --dt 0.5 --out_csv new_results/timeseries/timeseries_demo.csv
python -m new_sweep.plot_timeseries --in_csv new_results/timeseries/timeseries_demo.csv --outdir new_results/figs

## One-shot figure render (uses new_results paths)
python -m new_sweep.make_all_figs
