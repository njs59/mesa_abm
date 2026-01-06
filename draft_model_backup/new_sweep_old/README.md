
# new_sweep: sweeps & plots for Chapter 4

Run from project root (folder containing `new_sweep/`).

## Basic grid
python new_sweep/grid.py   --merge_prob 0.5 0.7 0.9   --invasive_speed 2 3 4   --proliferative_speed 0.8 1.0   --dt 0.5   --seeds 5 --jobs 4 --steps 300   --out_csv results_new/sweeps.csv
python new_sweep/plot_sweeps.py --in results_new/sweeps.csv --outdir figs_new_sweeps

## Extended grid
python new_sweep/grid_extended.py   --merge_prob 0.6 0.8 0.95   --dt 0.5 1.0   --inv_speed 2.0 3.0   --pro_speed 0.8 1.2   --inv_exp 0.0 0.25   --pro_exp 0.0 0.25   --inv_adh 0.2 0.4 0.6   --pro_adh 0.4 0.6   --inv_prolif 0.002 0.004   --pro_prolif 0.005 0.007   --inv_frag 0.0005 0.001   --pro_frag 0.0002 0.0005   --inv_frag_exp 0.0 1.0   --pro_frag_exp 0.0 1.0   --seeds 5 --jobs 4 --steps 300   --out_csv results_new/sweeps_extended.csv
python new_sweep/plot_heatmaps.py --in_csv results_new/sweeps_extended.csv --outdir figs_new_sweeps
python new_sweep/plot_pairplots.py --in_csv results_new/sweeps_extended.csv --out_png figs_new_sweeps/pairs.png

## Time-series
python new_sweep/run_timeseries.py
python new_sweep/plot_timeseries.py --in_csv results_new/timeseries_demo.csv --outdir figs_new_sweeps

## Speed–size scatter
python new_sweep/plot_speed_size.py --in_csv results_new/state_timeseries.csv --out_png figs_new_sweeps/speed_vs_size.png

## All figures
python new_sweep/make_all_figs.py
