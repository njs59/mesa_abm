#!/usr/bin/env python3
"""
Convenience wrapper to generate ALL plots for the Multi_phase_analysis suite.
Assumes the corresponding runs have been executed and summary_timeseries.csv exist in:
  - results/three_conditions/
  - results/fifteen_conditions/
  - results/kappa_conditions/
"""
from __future__ import annotations
from pathlib import Path
import subprocess

scripts = [
    'plot_three_conditions.py',        # RAW (3 models)
    'plot_three_conditions_pct.py',    # % DIFF (3 models)  <-- NEW
    'plot_fifteen_conditions.py',      # RAW (15 conditions)
    'plot_percentage_differences.py',  # % DIFF (15 conditions)
    'plot_kappa_conditions.py',        # RAW + % DIFF (kappa)
]

here = Path(__file__).resolve().parent
for s in scripts:
    p = here / s
    if not p.exists():
        print(f'Skipped missing {s}')
        continue
    print(f'Running {s} ...')
    try:
        subprocess.run(['python', str(p)], check=True)
    except Exception as e:
        print(f'⚠️  {s} failed: {e}')
print('All plots attempted.')