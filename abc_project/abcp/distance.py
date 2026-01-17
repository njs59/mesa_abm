
import numpy as np


def euclidean_distance_timeseries(sim_vec: np.ndarray, obs_vec: np.ndarray) -> float:
    """
    Plain Euclidean distance in flattened time-series space.
    This is now used only as a helper — Z-score scaling is done in run_abc.py.
    """
    d = sim_vec - obs_vec
    return float(np.sqrt(np.sum(d * d)))


def zscore_distance(sim_vec: np.ndarray, obs_vec: np.ndarray, stds_rep: np.ndarray) -> float:
    """
    Z-score normalised Euclidean distance.
    (Not used directly by run_abc.py, but useful if you want to import it.)
    """
    z = (sim_vec - obs_vec) / stds_rep
    return float(np.sqrt(np.sum(z * z)))
