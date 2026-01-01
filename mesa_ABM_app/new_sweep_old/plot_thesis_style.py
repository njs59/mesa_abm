
# new_sweep/plot_thesis_style.py
import matplotlib
import matplotlib.pyplot as plt

def use_thesis_style():
    matplotlib.use('Agg')
    plt.rcParams.update({
        'figure.dpi': 200,
        'savefig.dpi': 300,
        'font.size': 11,
        'font.family': 'serif',
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'lines.linewidth': 1.6,
        'lines.markersize': 4,
    })
