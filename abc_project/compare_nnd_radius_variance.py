
# compare_nnd_persistent_noise.py
# Purpose:
#   Compare Baseline vs Persistent per-agent radius noise.
#   Simulation stays toroidal (torus=True), but NND is computed WITHOUT wrap
#   (i.e., metric does not "see" the other side of the torus).
#
# Requirements:
#   abm/
#     __init__.py
#     clusters_model.py
#     cluster_agent.py
#     utils.py
#
# Outputs:
#   fig_persist_mean_nnd_vs_time.png
#   fig_persist_delta_nnd_vs_time.png
#   fig_persist_final_distribution.png
#   fig_persist_cluster_count_vs_time.png
#
# Notes:
#   - British English in labels (but matplotlib kwargs use American 'color').
#   - dt=1.0 => 30 minutes => time axis in hours.

import math
import contextlib
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# If you run headless, uncomment:
# matplotlib.use("Agg")

# ---- Import your ABM package -------------------------------------------------
from abm.clusters_model import ClustersModel
from abm import utils as abm_utils


# =============================================================================
# Configuration (tweak here)
# =============================================================================
STEPS = int(abm_utils.DEFAULTS["time"].get("steps", 300))  # ~150 h if dt=1
N_REPS = 5
SEED0 = 42  # master seed for pairing replicates

# Keep torus=True in the model, but we'll compute NND without wrap
PARAM_OVERRIDES: Dict[str, Any] = {
    # Ensure torus is ON in the simulation:
    "space.torus": True,

    # OPTIONAL: slight reduction in merging can help reveal NND↑ with noise
    # "merge.prob_contact_merge": 0.6,
    # "phenotypes.proliferative.adhesion": 0.5,

    # OPTIONAL: reduce initial density to reduce immediate heavy merging
    # "init.n_clusters": 700,
}

# Persistent per-agent radius noise settings
SIGMA = 0.35            # lognormal sigma for radius multiplier
PRESERVE = "area"       # "radius" (E[f]=1) or "area" (E[f^2]=1); "area" is often preferable for 2D packing
MERGE_MULT_COMBINE = "max"   # "max" | "weighted" | "self"
APPLY_MULT_AFTER_MERGE = True  # if True, apply multiplier to merged radius too (breaks strict volume conservation)

# Figure palette
COL_BASE = "#1f77b4"     # baseline blue
COL_PERS = "#d62728"     # persistent noise red
ALPHA_RIB = 0.22


# =============================================================================
# Small helpers
# =============================================================================
def apply_param_overrides(params: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply dot-path overrides into a nested dict copy."""
    from copy import deepcopy
    p = deepcopy(params)
    for k, v in overrides.items():
        path = k.split(".")
        d = p
        for key in path[:-1]:
            d = d[key]
        d[path[-1]] = v
    return p


def pairwise_nnd_no_wrap(positions: np.ndarray) -> np.ndarray:
    """
    Compute centre–centre nearest-neighbour distances WITHOUT torus wrapping.
    This is the requested "NND does not see the other side of the torus".
    """
    n = positions.shape[0]
    if n <= 1:
        return np.full((n,), np.nan, dtype=float)
    nnd = np.full((n,), np.inf, dtype=float)
    for i in range(n):
        dx = positions[i, 0] - positions[:, 0]
        dy = positions[i, 1] - positions[:, 1]
        d = np.hypot(dx, dy)
        d[i] = np.inf
        nnd[i] = d.min()
    return nnd


def ci95(a: np.ndarray, axis=0) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.nanpercentile(a, 2.5, axis=axis)
    hi = np.nanpercentile(a, 97.5, axis=axis)
    return lo, hi


# =============================================================================
# Persistent per-agent radius noise (monkey patches, no file edits)
# =============================================================================
@contextlib.contextmanager
def patch_persistent_radius_noise(
    sigma: float,
    preserve: str = "area",
    merge_mult_combine: str = "max",
    apply_after_merge: bool = True,
    seed: Optional[int] = None,
):
    """
    Inject a persistent per-agent radius multiplier r_mult ~ LogNormal,
    drawn once at agent creation and carried through the agent's life.

    - preserve="radius": E[f] = 1  -> factor = exp(z - 0.5*sigma^2)
    - preserve="area"  : E[f^2]=1  -> factor = exp(z - sigma^2)
    - merge_mult_combine:
        * "max"      : r_mult := max(m_self, m_other)
        * "weighted" : r_mult := (m_self*s1 + m_other*s2)/(s1+s2)  (s1,s2 are pre-merge sizes)
        * "self"     : keep m_self (ignore other's multiplier)
    - apply_after_merge:
        If True, reassign radius after merge as r = r_mult * radius_from_size_3d(size)
        (breaks strict volume conservation but enforces persistent scaling).
        If False, keep the volume-conserving radius from the raw ABM merge, but update r_mult only.
    """
    rng = np.random.default_rng(seed)

    def draw_multiplier() -> float:
        z = rng.normal(0.0, sigma)
        if preserve == "area":
            return math.exp(z - sigma**2)      # E[f^2]=1
        else:
            return math.exp(z - 0.5*sigma**2)  # E[f]=1

    import abm.cluster_agent as ca

    orig_init = ca.ClusterAgent.__init__
    orig_prolif = ca.ClusterAgent._maybe_proliferate
    orig_frag = ca.ClusterAgent._maybe_fragment
    orig_merge = ca.ClusterAgent._merge_with
    rfs = abm_utils.radius_from_size_3d  # shortcut

    # ---- patched __init__  ---------------------------------------------------
    def patched_init(self, model, size, phenotype):
        # call original
        orig_init(self, model, size, phenotype)
        # attach persistent multiplier (once)
        self.r_mult = draw_multiplier()
        # rescale radius deterministically from size using multiplier
        base = rfs(self.size)
        self.radius = float(self.r_mult * base)

    # ---- patched proliferation / fragmentation  ------------------------------
    def patched_prolif(self):
        # original will update size and set radius deterministically
        orig_prolif(self)
        if not getattr(self, "alive", True):
            return
        base = rfs(self.size)
        mult = getattr(self, "r_mult", 1.0)
        self.radius = float(mult * base)

    def patched_frag(self):
        # original will update size, set radius, and spawn a child
        # The child is created via ClusterAgent(...) -> patched_init, so it gets its own r_mult.
        orig_frag(self)
        if not getattr(self, "alive", True):
            return
        base = rfs(self.size)
        mult = getattr(self, "r_mult", 1.0)
        self.radius = float(mult * base)

    # ---- patched merge_with  -------------------------------------------------
    def patched_merge(self, other):
        # capture multipliers & sizes pre-merge
        m_self = getattr(self, "r_mult", 1.0)
        m_other = getattr(other, "r_mult", 1.0)
        s1 = int(getattr(self, "size", 1))
        s2 = int(getattr(other, "size", 1))

        # perform original merge (updates size, pos, vel, radius via volume conservation)
        orig_merge(self, other)

        # combine multiplier
        if merge_mult_combine == "weighted":
            m_new = (m_self * s1 + m_other * s2) / max(s1 + s2, 1)
        elif merge_mult_combine == "self":
            m_new = m_self
        else:  # "max"
            m_new = m_self if m_self >= m_other else m_other

        self.r_mult = float(m_new)

        if apply_after_merge:
            # Re-apply multiplier to the merged size
            base = rfs(self.size)
            self.radius = float(self.r_mult * base)
        # else: keep volume-conserving radius as computed by the original merge

    # ---- apply patches
    ca.ClusterAgent.__init__ = patched_init
    ca.ClusterAgent._maybe_proliferate = patched_prolif
    ca.ClusterAgent._maybe_fragment = patched_frag
    ca.ClusterAgent._merge_with = patched_merge

    try:
        yield
    finally:
        # restore originals
        ca.ClusterAgent.__init__ = orig_init
        ca.ClusterAgent._maybe_proliferate = orig_prolif
        ca.ClusterAgent._maybe_fragment = orig_frag
        ca.ClusterAgent._merge_with = orig_merge


# =============================================================================
# Simulation orchestration (NND computed with NO wrap)
# =============================================================================
def run_simulation(params: Dict[str, Any], steps: int, seed: int) -> Dict[str, Any]:
    """Run one replicate; return times, mean NND (no wrap), final NND, and cluster counts."""
    # Seed NumPy too (for our patch RNG reproducibility)
    np.random.seed(seed % (2**32 - 1))

    model = ClustersModel(params=params, seed=seed)
    dt = float(params["time"]["dt"])

    nnd_list: List[np.ndarray] = []
    n_agents: List[int] = []

    # t=0
    pos0 = np.asarray(model.pos_log[0], dtype=float) if len(model.pos_log) else np.empty((0, 2))
    nnd0 = pairwise_nnd_no_wrap(pos0) if pos0.size else np.array([], dtype=float)
    nnd_list.append(nnd0)
    n_agents.append(len(model.id_log[0]) if len(model.id_log) else 0)

    # time steps
    for _ in range(steps):
        model.step()
        pos = np.asarray(model.pos_log[-1], dtype=float)
        nnd = pairwise_nnd_no_wrap(pos) if pos.size else np.array([], dtype=float)
        nnd_list.append(nnd)
        n_agents.append(len(model.id_log[-1]) if len(model.id_log) else 0)

    mean_nnd = [np.nan if len(v) == 0 else float(np.nanmean(v)) for v in nnd_list]
    time_hours = np.arange(len(nnd_list)) * dt * 0.5

    return {
        "time_hours": np.array(time_hours, dtype=float),
        "mean_nnd": np.array(mean_nnd, dtype=float),
        "final_nnd": nnd_list[-1] if len(nnd_list) else np.array([], dtype=float),
        "n_agents": np.array(n_agents, dtype=int),
    }


def run_replicates(params: Dict[str, Any], steps: int, n_reps: int, seed0: int, mode: str) -> Dict[str, Any]:
    """
    mode in {"baseline", "persistent"}.
    """
    rng = np.random.default_rng(seed0)
    outs = []

    for r in range(n_reps):
        seed = int(rng.integers(1, 2**31 - 1))
        p = apply_param_overrides(abm_utils.DEFAULTS, PARAM_OVERRIDES)

        if mode == "baseline":
            out = run_simulation(p, steps, seed)

        elif mode == "persistent":
            # Use a separate seed for the patch RNG so replicates are independent
            patch_seed = int(rng.integers(1, 2**31 - 1))
            with patch_persistent_radius_noise(
                sigma=SIGMA,
                preserve=PRESERVE,
                merge_mult_combine=MERGE_MULT_COMBINE,
                apply_after_merge=APPLY_MULT_AFTER_MERGE,
                seed=patch_seed,
            ):
                out = run_simulation(p, steps, seed)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        out["rep"] = r
        outs.append(out)

    # Align lengths
    T = min(len(o["time_hours"]) for o in outs)
    times = outs[0]["time_hours"][:T]
    m = np.vstack([o["mean_nnd"][:T] for o in outs])
    n = np.vstack([o["n_agents"][:T] for o in outs])
    finals = [o["final_nnd"] for o in outs]

    return {
        "time_hours": times,
        "mean_nnd_matrix": m,
        "n_agents_matrix": n,
        "final_nnd_samples": finals,
        "n_reps": n_reps,
    }


# =============================================================================
# Plotting
# =============================================================================
def plot_mean_vs_time(times, base_m, pers_m, outpath="fig_persist_mean_nnd_vs_time.png"):
    plt.figure(figsize=(8.4, 5.2), dpi=140)

    m0 = np.nanmean(base_m, axis=0); l0, h0 = ci95(base_m, axis=0)
    m1 = np.nanmean(pers_m, axis=0); l1, h1 = ci95(pers_m, axis=0)

    plt.plot(times, m0, label="Baseline (deterministic radius)", color=COL_BASE, lw=2)
    plt.fill_between(times, l0, h0, color=COL_BASE, alpha=ALPHA_RIB)

    plt.plot(times, m1, label="Persistent radius noise", color=COL_PERS, lw=2)
    plt.fill_between(times, l1, h1, color=COL_PERS, alpha=ALPHA_RIB)

    plt.xlabel("Time (hours)")
    plt.ylabel("Mean nearest‑neighbour distance (pixels)")
    plt.title("Mean NND over time (NND computed without torus wrap)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_delta_vs_time(times, base_m, pers_m, outpath="fig_persist_delta_nnd_vs_time.png"):
    n = min(base_m.shape[0], pers_m.shape[0])
    d = pers_m[:n, :] - base_m[:n, :]
    md = np.nanmean(d, axis=0); lo, hi = ci95(d, axis=0)

    plt.figure(figsize=(8.6, 5.0), dpi=140)
    plt.axhline(0, color="#444", lw=1, ls="--")
    plt.plot(times, md, color="#2ca02c", lw=2, label="Persistent − Baseline")
    plt.fill_between(times, lo, hi, color="#2ca02c", alpha=ALPHA_RIB)
    plt.xlabel("Time (hours)")
    plt.ylabel("Δ Mean NND (pixels)")
    plt.title("Shift in mean NND (NND computed without torus wrap)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_final_distribution(b_final, p_final, outpath="fig_persist_final_distribution.png"):
    b = np.concatenate([s for s in b_final if len(s) > 0]) if b_final else np.array([])
    p = np.concatenate([s for s in p_final if len(s) > 0]) if p_final else np.array([])

    plt.figure(figsize=(8.6, 5.0), dpi=140)
    parts = plt.violinplot([b, p], positions=[1, 2], showmeans=True, showextrema=True)
    for body, color in zip(parts['bodies'], [COL_BASE, COL_PERS]):
        body.set_alpha(0.45)
        body.set_facecolor(color)
    plt.xticks([1, 2], ["Baseline", "Persistent noise"])
    plt.ylabel("Nearest‑neighbour distance (pixels)")
    plt.title("Final‑time NND distributions (no wrap in metric)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_cluster_count(times, base_n, pers_n, outpath="fig_persist_cluster_count_vs_time.png"):
    plt.figure(figsize=(8.4, 5.0), dpi=140)
    m0 = np.mean(base_n, axis=0); l0, h0 = ci95(base_n, axis=0)
    m1 = np.mean(pers_n, axis=0); l1, h1 = ci95(pers_n, axis=0)

    plt.plot(times, m0, label="Baseline", color=COL_BASE, lw=2)
    plt.fill_between(times, l0, h0, color=COL_BASE, alpha=ALPHA_RIB)

    plt.plot(times, m1, label="Persistent noise", color=COL_PERS, lw=2)
    plt.fill_between(times, l1, h1, color=COL_PERS, alpha=ALPHA_RIB)

    plt.xlabel("Time (hours)")
    plt.ylabel("Cluster count")
    plt.title("Number of clusters over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    # --- Baseline (no patch)
    base = run_replicates(
        apply_param_overrides(abm_utils.DEFAULTS, PARAM_OVERRIDES),
        steps=STEPS,
        n_reps=N_REPS,
        seed0=SEED0,
        mode="baseline",
    )

    # --- Persistent per-agent radius noise
    pers = run_replicates(
        apply_param_overrides(abm_utils.DEFAULTS, PARAM_OVERRIDES),
        steps=STEPS,
        n_reps=N_REPS,
        seed0=SEED0 + 1,
        mode="persistent",
    )

    times = base["time_hours"]

    # Plots
    plot_mean_vs_time(times, base["mean_nnd_matrix"], pers["mean_nnd_matrix"])
    plot_delta_vs_time(times, base["mean_nnd_matrix"], pers["mean_nnd_matrix"])
    plot_final_distribution(base["final_nnd_samples"], pers["final_nnd_samples"])
    plot_cluster_count(times, base["n_agents_matrix"], pers["n_agents_matrix"])

    # Console summary
    def fmean(a): return float(np.nanmean(a[:, -1]))
    m_base = fmean(base["mean_nnd_matrix"])
    m_pers = fmean(pers["mean_nnd_matrix"])
    print(f"[Summary @ final time] Mean NND (px, no wrap): baseline={m_base:.2f} | persistent={m_pers:.2f}")
    print(f"Δ(persistent−baseline)={m_pers - m_base:+.2f} px")
    print("Saved: fig_persist_mean_nnd_vs_time.png, fig_persist_delta_nnd_vs_time.png, "
          "fig_persist_final_distribution.png, fig_persist_cluster_count_vs_time.png")


if __name__ == "__main__":
    main()