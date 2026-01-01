
# new_sweep/make_all_figs.py
import subprocess
import os

cmds = [
    ['python', 'new_sweep/plot_sweeps.py', '--in', 'results_new/sweeps.csv', '--outdir', 'figs_new_sweeps'],
    ['python', 'new_sweep/plot_heatmaps.py', '--in_csv', 'results_new/sweeps_extended.csv', '--outdir', 'figs_new_sweeps'],
    ['python', 'new_sweep/plot_timeseries.py', '--in_csv', 'results_new/timeseries_demo.csv', '--outdir', 'figs_new_sweeps'],
    ['python', 'new_sweep/plot_pairplots.py', '--in_csv', 'results_new/sweeps_extended.csv', '--out_png', 'figs_new_sweeps/pairs.png'],
]

if __name__ == '__main__':
    os.makedirs('figs_new_sweeps', exist_ok=True)
    for cmd in cmds:
        try:
            print('Running:', ' '.join(cmd))
            subprocess.run(cmd, check=True)
        except Exception as e:
            print('Failed:', cmd, '->', e)
