from mesa import Model
from mesa.space import ContinuousSpace
import numpy as np
from abm.cluster_agent import ClusterAgent
from abm.utils import DEFAULTS

class ClustersModel(Model):
    """ABM with phenotype-specific two-phase motility and classical Gompertz (shifted) transition CDF."""

    def __init__(self, params=None, seed=42, init_clusters=None):
        super().__init__()
        self.random.seed(seed)
        self.np_rng = np.random.default_rng(seed)
        self.params = params or DEFAULTS

        # Space
        W = float(self.params["space"]["width"])
        H = float(self.params["space"]["height"])
        self.space = ContinuousSpace(W, H, torus=bool(self.params["space"]["torus"]))

        self.dt = float(self.params["time"]["dt"])
        self.time = 0.0

        # keep Mesa‑3.x safe agent container
        self.agent_set = set()

        # --- BUILD TRANSITION DISTRIBUTION (UPDATED) ---
        self._build_transition_lookup()

        # Spawn initial clusters
        if init_clusters is None:
            icfg = self.params.get("init", {})
            n0 = int(icfg.get("n_clusters", 1000))
            sz = int(icfg.get("size", 1))
            ph = str(icfg.get("phenotype", "proliferative"))
            init_clusters = [{"size": sz, "phenotype": ph} for _ in range(n0)]

        for ic in init_clusters:
            self.spawn_cluster(ic["size"], ic["phenotype"])

        # Logging buffers
        self.size_log, self.speed_log, self.pos_log = [], [], []
        self.radius_log, self.id_log = [], []
        self._log_state()

    # ----------------------------------------------------------------------
    # UPDATED: Classical shifted Gompertz CDF lookup
    # ----------------------------------------------------------------------
    def _build_transition_lookup(self):
        """
        Replace ABM's original Gompertz-hazard PDF with the classical
        shifted Gompertz CDF used in the empirical fitting script:

            F(t) = L * (1 - exp(-(b/a)*(exp(a*(t - t0)) - 1)))

        NOTE:
        - We KEEP PARAMETER NAMES "b" and "c" as requested.
        - "c" now plays the role of classical "a" (growth rate).
        - "b" stays "b" (shape).
        - "shift" remains the time shift (t0).

        All phenotypes use the same interface as before.
        """
        self.transition_t = {}
        self.transition_cdf = {}

        phenos = list(self.params.get("phenotypes", {}).keys())
        mv2 = self.params.get("movement_v2", {})

        mv2_is_global = (
            isinstance(mv2, dict)
            and ("phase1" in mv2)
            and ("phase2" in mv2)
            and ("transition" in mv2)
        )

        for ph in phenos:
            cfg = mv2 if mv2_is_global else mv2[ph]
            trans = cfg["transition"]

            # Classical parameters — KEEP "b" and "c"
            L = float(trans.get("p_max", 1.0))  # use p_max as plateau for backward compatibility
            t0 = float(trans["shift"])         # same as before
            b = float(trans["b"])              # stays b
            a = float(trans["c"])              # "c" now acts as "a" in classical formula

            tmax = float(trans.get("t_max", 400.0))
            npts = int(trans.get("n_points", 3000))

            # Grid
            t = np.linspace(t0, tmax, npts)
            u = np.maximum(t - t0, 0.0)

            # Classical shifted Gompertz CDF
            term = (b / a) * (np.exp(a * u) - 1.0)
            cdf = L * (1.0 - np.exp(-term))

            # force exact zero below t0
            cdf[t < t0] = 0.0

            # normalize to 1
            if cdf[-1] > 0:
                cdf = cdf / cdf[-1]

            # store
            self.transition_t[ph] = t
            self.transition_cdf[ph] = cdf

    def sample_transition_time(self, phenotype: str) -> float:
        """Sample a Phase‑1 → Phase‑2 switch time from the stored CDF."""
        if phenotype not in self.transition_t:
            phenotype = next(iter(self.transition_t.keys()))
        t = self.transition_t[phenotype]
        cdf = self.transition_cdf[phenotype]

        U = float(self.np_rng.uniform(0.0, 1.0))
        idx = int(np.searchsorted(cdf, U, side="left"))
        idx = min(idx, len(t) - 1)
        return float(t[idx])

    # ----------------------------------------------------------------------
    # SPAWN CLUSTER
    # ----------------------------------------------------------------------
    def spawn_cluster(self, size, phenotype, pos=None, jitter=False, phase_switch_time=None):
        a = ClusterAgent(
            self,
            size=size,
            phenotype=phenotype,
            phase_switch_time=phase_switch_time,
        )
        self.agent_set.add(a)

        if pos is None:
            pos = np.array(
                [
                    self.random.uniform(0, self.params["space"]["width"]),
                    self.random.uniform(0, self.params["space"]["height"]),
                ],
                dtype=float,
            )
        else:
            pos = np.array(pos, dtype=float)

        if jitter:
            pos += np.array(
                [
                    self.random.normalvariate(0, 1),
                    self.random.normalvariate(0, 1),
                ]
            )

        x, y = self.space.torus_adj(tuple(pos))
        self.space.place_agent(a, (float(x), float(y)))
        return a

    # ----------------------------------------------------------------------
    # Neighbours / lifecycle
    # ----------------------------------------------------------------------
    def get_neighbors(self, agent, r):
        if agent.pos is None:
            return []
        cand = self.space.get_neighbors(
            pos=agent.pos,
            radius=float(r),
            include_center=False
        )
        alive = [a for a in cand if getattr(a, "alive", True)]

        allow_cross = bool(self.params.get("interactions", {}).get(
            "allow_cross_phase_interactions", True
        ))
        if not allow_cross:
            same_phase = []
            am = getattr(agent, "movement_phase", None)
            for a in alive:
                try:
                    if getattr(a, "movement_phase", None) == am:
                        same_phase.append(a)
                except Exception:
                    pass
            return same_phase

        return alive

    def remove_agent(self, agent):
        self.agent_set.discard(agent)
        try:
            self.space.remove_agent(agent)
        except Exception:
            pass

    # ----------------------------------------------------------------------
    # Step & logging
    # ----------------------------------------------------------------------
    def step(self):
        L = list(self.agent_set)
        self.random.shuffle(L)
        for a in L:
            if getattr(a, "alive", True):
                a.step()
        self.time += self.dt
        self._log_append()

    def _log_append(self):
        ids, pos, radii, sizes, speeds = self._snapshot_alive()
        self.id_log.append(ids)
        self.pos_log.append(pos)
        self.radius_log.append(radii)
        self.size_log.append(sizes)
        self.speed_log.append(speeds)

    def _log_state(self):
        ids, pos, radii, sizes, speeds = self._snapshot_alive()
        self.id_log.append(ids)
        self.pos_log.append(pos)
        self.radius_log.append(radii)
        self.size_log.append(sizes)
        self.speed_log.append(speeds)

    def _snapshot_alive(self):
        xs, ys, ids, radii, sizes, speeds = [], [], [], [], [], []
        for a in list(self.agent_set):
            if not getattr(a, "alive", True):
                continue
            p = getattr(a, "pos", None)
            if p is None:
                continue
            try:
                x, y = float(p[0]), float(p[1])
                r = float(a.radius)
                s = float(a.size)
                v = float(np.linalg.norm(a.vel))
                i = int(a.unique_id)
            except Exception:
                continue

            xs.append(x)
            ys.append(y)
            radii.append(r)
            sizes.append(s)
            speeds.append(v)
            ids.append(i)

        if len(xs) == 0:
            import numpy as _np
            return (
                _np.array([], dtype=int),
                _np.empty((0, 2), dtype=float),
                _np.array([], dtype=float),
                _np.array([], dtype=float),
                _np.array([], dtype=float),
            )

        pos = np.column_stack([np.array(xs), np.array(ys)])
        return (
            np.array(ids, dtype=int),
            pos,
            np.array(radii),
            np.array(sizes),
            np.array(speeds),
        )