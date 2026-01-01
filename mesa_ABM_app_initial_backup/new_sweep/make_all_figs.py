
# new_sweep/make_all_figs.py
import subprocess, os
cmds = [
  ['python','-m','new_sweep.plot_sweeps','--in','new_results/sweeps.csv','--outdir','new_results/figs'],
  ['python','-m','new_sweep.plot_heatmaps','--in_csv','new_results/sweeps_extended.csv','--outdir','new_results/figs'],
  ['python','-m','new_sweep.plot_timeseries','--in_csv','new_results/timeseries/timeseries_demo.csv','--outdir','new_results/figs'],
  ['python','-m','new_sweep.plot_pairplots','--in_csv','new_results/sweeps_extended.csv','--out_png','new_results/figs/pairs.png'],
]
if __name__=='__main__':
    os.makedirs('new_results/figs', exist_ok=True)
    for cmd in cmds:
        print('Running:',' '.join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print('Failed:',cmd,'->',e)
