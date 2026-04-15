import os
import numpy as np
import pandas as pd


def write_aic_table(results: list, out_csv: str):
    """
    Write AIC comparison table for a list of dicts with keys:
       - model_key, AIC, max_loglik, n_params
    Returns a pandas DataFrame.
    """
    df = pd.DataFrame([
        {
            'model': r['model_key'],
            'k': r['n_params'],
            'max_loglik': r['max_loglik'],
            'AIC': r['AIC'],
        }
        for r in results
    ])
    # ΔAIC and weights
    min_aic = df['AIC'].min()
    df['delta_AIC'] = df['AIC'] - min_aic
    weights = np.exp(-0.5 * df['delta_AIC'].to_numpy())
    df['akaike_weight'] = weights / weights.sum()
    df = df.sort_values('AIC').reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df