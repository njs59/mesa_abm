
# abc_resume.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

from pyabc import ABCSMC, History
from pyabc.sampler import MulticoreEvalParallelSampler

# --- Project hooks (same as your run scripts) ---
from abm.clusters_model import ClustersModel
from abcp.compute_summary import simulate_timeseries
from abcp.abc_model_wrapper import particle_to_params
from abcp.priors import load_priors
# ------------------------------------------------

# ---- Settings: keep these consistent with the original run ----
DB_PATH      = "sqlite:///results/abc_maxabs_nogr.db"
OBS_CSV      = "observed/INV_ABM_ready_summary.csv"
STATS        = ["S0", "S1", "S2", "NND_med"]  # --no_gr
MOTION       = "isotropic"
SPEED        = "lognorm"
TOTAL_STEPS  = 100
POPSIZE      = 500         # use a new value here if you want larger populations on resume
ADD_POPS     = 5           # how many populations to append
N_PROCS      = 8           # parallel workers
SEED_MODEL   = 42          # fixed seed per-draw (as in your run script)
# ---------------------------------------------------------------

# Build observation + scaler from experimental data (exactly like run_abc_maxabs.py)
obs_df = pd.read_csv(OBS_CSV).sort_values("timestep").reset_index(drop=True)
timesteps = obs_df["timestep"].astype(int).tolist()
obs_mat = obs_df[STATS].to_numpy(float)
T, K = obs_mat.shape

scaler = MaxAbsScaler().fit(obs_mat)
obs_scaled_vec = scaler.transform(obs_mat).flatten()
observation = {f"y_{i}": float(v) for i, v in enumerate(obs_scaled_vec)}

# Model factory
def make_model_factory(seed: int = SEED_MODEL):
    def factory(params_dict):
        return ClustersModel(params=params_dict, seed=seed)
    return factory

# pyABC model wrapper
def abm_model(particle):
    params = particle_to_params(particle, motion=MOTION, speed_dist=SPEED)
    sim_mat = simulate_timeseries(
        make_model_factory(SEED_MODEL),
        params=params,
        total_steps=TOTAL_STEPS,
        sample_steps=tuple(timesteps),
    )
    full_order = ["S0", "S1", "S2", "NND_med", "g_r40", "g_r80"]
    idx = [full_order.index(s) for s in STATS]
    sim_sel = sim_mat[:, idx]
    sim_scaled_vec = scaler.transform(sim_sel).flatten()
    return {f"y_{i}": float(v) for i, v in enumerate(sim_scaled_vec)}

# Distance: L2 in the scaled space
def l2_distance(sim, obs):
    s = np.array([sim[f"y_{i}"] for i in range(T * K)], float)
    o = np.array([obs[f"y_{i}"] for i in range(T * K)], float)
    return float(np.sqrt(((s - o) ** 2).sum()))

def main():
    # The prior must match your original priors.yaml (or defaults)
    prior = load_priors("priors.yaml" if Path("priors.yaml").exists() else None)

    # Provide components + sampler now, then load and continue
    abc = ABCSMC(
        models=abm_model,
        parameter_priors=prior,
        distance_function=l2_distance,
        population_size=POPSIZE,
        sampler=MulticoreEvalParallelSampler(n_procs=N_PROCS),
    )

    hist = History(DB_PATH)
    run_id = hist.id
    print(f"Resuming run id={run_id} from DB: {DB_PATH}")

    # IMPORTANT: run_id is positional here
    abc.load(DB_PATH, run_id)

    # Append more populations
    abc.run(max_nr_populations=ADD_POPS)
    print(f"Appended {ADD_POPS} populations to run {run_id}.")

if __name__ == "__main__":
    main()
