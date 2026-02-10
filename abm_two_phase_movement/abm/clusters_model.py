from mesa import Model
from mesa.space import ContinuousSpace
import numpy as np

from abm.cluster_agent import ClusterAgent
from abm.utils import DEFAULTS

class ClustersModel(Model):
    """ABM with two‑phase motility and shifted‑Gompertz transition.

    Logs (per step): id_log, pos_log, radius_log, size_log, speed_log
    """
    def __init__(self, params=None, seed=42, init_clusters=None):
        super().__init__()
        self.random.seed(seed)
        self.np_rng = np.random.default_rng(seed)
        self.params = params or DEFAULTS

        # space
        W = float(self.params["space"]["width"])
        H = float(self.params["space"]["height"])
        self.space = ContinuousSpace(W, H, torus=bool(self.params["space"]["torus"]))

        self.dt = float(self.params["time"]["dt"])
        self.time = 0.0

        # Agents container (simple set; manual stepping)
        self.agents = set()

        # Build transition lookup table once
        self._build_transition_lookup()

        # Spawn initial population
        if init_clusters is None:
            icfg = self.params.get("init", {})
            n0 = int(icfg.get("n_clusters", 1000))
            sz = int(icfg.get("size", 1))
            ph = str(icfg.get("phenotype", "proliferative"))
            init_clusters = [{"size": sz, "phenotype": ph} for _ in range(n0)]
        for ic in init_clusters:
            self.spawn_cluster(ic["size"], ic["phenotype"])

        # Logs
        self.size_log, self.speed_log, self.pos_log = [], [], []
        self.radius_log, self.id_log = [], []
        self._log_state()

    # ------------------------------------------------------------------
    def _build_transition_lookup(self):
        trans = self.params["movement_v2"]["transition"]
        pmax = float(trans["p_max"])  # final CDF height
        shift = float(trans["shift"])  # time offset
        b = float(trans["b"])         # Gompertz b
        c = float(trans["c"])         # Gompertz c
        tmax = float(trans.get("t_max", 400.0))
        npts = int(trans.get("n_points", 3000))
        t = np.linspace(shift, tmax, npts)
        u = t - shift
        # PDF of shifted Gompertz (scaled by pmax)
        pdf = pmax * b * c * np.exp(c*u) * np.exp(-b*(np.exp(c*u) - 1))
        cdf = np.cumsum(pdf) * (t[1] - t[0])
        # normalise to 1 to make it a proper CDF
        if cdf[-1] > 0:
            cdf /= cdf[-1]
        self.transition_t = t
        self.transition_cdf = cdf

    def sample_transition_time(self) -> float:
        U = float(self.np_rng.uniform(0.0, 1.0))
        idx = int(np.searchsorted(self.transition_cdf, U, side='left'))
        idx = min(idx, len(self.transition_t)-1)
        return float(self.transition_t[idx])

    # ------------------------------------------------------------------
    def get_neighbors(self, agent, r):
        if agent.pos is None:
            return []
        cand = self.space.get_neighbors(pos=agent.pos, radius=float(r), include_center=False)
        return [a for a in cand if getattr(a, 'alive', True)]

    def remove_agent(self, agent):
        try:
            self.agents.discard(agent)
        except Exception:
            pass
        try:
            self.space.remove_agent(agent)
        except Exception:
            pass

    def spawn_cluster(self, size, phenotype, pos=None, jitter=False):
        a = ClusterAgent(self, size=size, phenotype=phenotype)
        self.agents.add(a)
        if pos is None:
            pos = np.array([
                self.random.uniform(0, self.params["space"]["width"]),
                self.random.uniform(0, self.params["space"]["height"]),
            ], dtype=float)
        else:
            pos = np.array(pos, dtype=float)
        if jitter:
            pos += np.array([self.random.normalvariate(0,1), self.random.normalvariate(0,1)])
        x, y = self.space.torus_adj(tuple(pos))
        self.space.place_agent(a, (float(x), float(y)))
        return a

    # ------------------------------------------------------------------
    def step(self):
        # randomised stepping
        L = list(self.agents)
        self.random.shuffle(L)
        for a in L:
            if getattr(a, 'alive', True):
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
        ids, xs, ys, radii, sizes, speeds = [], [], [], [], [], []
        for a in list(self.agents):
            if not getattr(a, 'alive', True):
                continue
            p = getattr(a, 'pos', None)
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
            xs.append(x); ys.append(y)
            radii.append(r); sizes.append(s); speeds.append(v); ids.append(i)
        if len(xs) == 0:
            import numpy as _np
            return (
                _np.array([], dtype=int),
                _np.empty((0,2), dtype=float),
                _np.array([], dtype=float),
                _np.array([], dtype=float),
                _np.array([], dtype=float),
            )
        pos = np.column_stack([np.array(xs), np.array(ys)])
        return (np.array(ids, dtype=int), pos, np.array(radii), np.array(sizes), np.array(speeds))
