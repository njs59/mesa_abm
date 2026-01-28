
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector  # optional (not used explicitly)
import numpy as np
from .cluster_agent import ClusterAgent
from .utils import DEFAULTS

class ClustersModel(Model):
    """
    ABM for motility-driven clustering with merging, proliferation, fragmentation,
    persistent per-agent radius noise, a single merge strength, and
    post-step repulsive relaxation to remove overlaps.
    Logs per timestep: size_log, speed_log, pos_log, radius_log, id_log.
    """
    def __init__(self, params=None, seed=42, init_clusters=None):
        super().__init__()
        self.random.seed(seed)
        self.params = params or DEFAULTS

        # Space
        W = float(self.params["space"]["width"])
        H = float(self.params["space"]["height"])
        self.space = ContinuousSpace(x_max=W, y_max=H, torus=bool(self.params["space"]["torus"]))

        # Time
        self.dt = float(self.params["time"]["dt"])
        self.time = 0.0

        # Initial population
        if init_clusters is None:
            init_cfg = self.params.get("init", {})
            n0 = int(init_cfg.get("n_clusters", 800))
            sz0 = int(init_cfg.get("size", 1))
            ph0 = str(init_cfg.get("phenotype", "proliferative"))
            init_clusters = [{"size": sz0, "phenotype": ph0} for _ in range(n0)]
        for ic in init_clusters:
            self.spawn_cluster(size=ic["size"], phenotype=ic["phenotype"])

        # Logs
        self.size_log = []
        self.speed_log = []
        self.pos_log = []
        self.radius_log = []
        self.id_log = []
        self._log_state()  # t=0 snapshot

    # ---------------- Geometry helpers ----------------
    def min_image_delta(self, p_self, p_other):
        """Return (dx, dy) with torus minimum-image convention if torus enabled."""
        x1, y1 = float(p_self[0]), float(p_self[1])
        x2, y2 = float(p_other[0]), float(p_other[1])
        dx = x2 - x1
        dy = y2 - y1
        if bool(self.params["space"]["torus"]):
            W = float(self.params["space"]["width"])
            H = float(self.params["space"]["height"])
            dx = (dx + W/2.0) % W - W/2.0
            dy = (dy + H/2.0) % H - H/2.0
        return dx, dy

    def get_neighbors(self, agent, r):
        if agent.pos is None:
            return []
        candidates = self.space.get_neighbors(pos=agent.pos, radius=float(r), include_center=False)
        return [a for a in candidates if getattr(a, "alive", True)]

    def remove_agent(self, agent):
        # Remove from space; Mesa AgentSet handles membership
        try:
            self.space.remove_agent(agent)
        except Exception:
            pass
        try:
            self.agents.remove(agent)
        except Exception:
            pass

    def spawn_cluster(self, size, phenotype, pos=None, jitter=False):
        a = ClusterAgent(model=self, size=size, phenotype=phenotype)
        self.agents.add(a)
        if pos is None:
            pos = np.array([
                self.random.uniform(0, self.params["space"]["width"]),
                self.random.uniform(0, self.params["space"]["height"]),
            ], dtype=float)
        else:
            pos = np.array(pos, dtype=float)
        if jitter:
            jx = self.random.normalvariate(0, 1)
            jy = self.random.normalvariate(0, 1)
            pos = pos + np.array([jx, jy], dtype=float)
        x, y = self.space.torus_adj(tuple(pos))
        self.space.place_agent(a, (float(x), float(y)))
        return a

    # ---------------- Logging ----------------
    def _alive_snapshot(self):
        ids = []
        xs, ys = [], []
        radii = []
        sizes = []
        speeds = []
        for a in list(self.agents):
            if not getattr(a, "alive", True):
                continue
            p = getattr(a, "pos", None)
            if p is None:
                continue
            try:
                x = float(p[0]); y = float(p[1])
                r = float(a.radius); s = float(a.size)
                v = float(np.linalg.norm(a.vel)); i = int(a.unique_id)
            except Exception:
                continue
            xs.append(x); ys.append(y)
            radii.append(r); sizes.append(s); speeds.append(v); ids.append(i)
        if len(xs) == 0:
            return (
                np.array([], dtype=int),
                np.empty((0, 2), dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
            )
        pos = np.column_stack([np.array(xs, float), np.array(ys, float)])
        return (
            np.array(ids, dtype=int),
            pos.astype(float, copy=False),
            np.array(radii, float),
            np.array(sizes, float),
            np.array(speeds, float),
        )

    def _log_state(self):
        ids, pos, radii, sizes, speeds = self._alive_snapshot()
        self.id_log.append(ids)
        self.pos_log.append(pos)
        self.radius_log.append(radii)
        self.size_log.append(sizes)
        self.speed_log.append(speeds)

    # ---------------- Repulsion (overlap removal) ----------------
    def _resolve_overlaps(self):
        rp = self.params.get("repulsion", {})
        if not bool(rp.get("enable", True)):
            return
        max_iter = int(rp.get("max_iter", 6))
        eps = float(rp.get("eps", 1e-6))
        W = float(self.params["space"]["width"])
        H = float(self.params["space"]["height"])
        tor = bool(self.params["space"]["torus"])

        for _ in range(max_iter):
            moved = False
            agents = [a for a in self.agents if getattr(a, "alive", True) and a.pos is not None]
            n = len(agents)
            for i in range(n):
                ai = agents[i]
                pi = np.array(ai.pos, dtype=float)
                ri = float(ai.radius)
                for j in range(i+1, n):
                    aj = agents[j]
                    pj = np.array(aj.pos, dtype=float)
                    rj = float(aj.radius)
                    dx, dy = self.min_image_delta(pi, pj)
                    dist = np.hypot(dx, dy)
                    min_sep = ri + rj + eps
                    if dist < min_sep and dist > 0:
                        # overlap amount
                        overlap = (min_sep - dist) / 2.0
                        # normalised direction from i->j
                        ux, uy = dx / dist, dy / dist
                        # displace equally in opposite directions
                        pi_new = pi - overlap * np.array([ux, uy])
                        pj_new = pj + overlap * np.array([ux, uy])
                        # wrap if torus
                        if tor:
                            pi_new[0] = (pi_new[0] + W) % W
                            pi_new[1] = (pi_new[1] + H) % H
                            pj_new[0] = (pj_new[0] + W) % W
                            pj_new[1] = (pj_new[1] + H) % H
                        self.space.move_agent(ai, (float(pi_new[0]), float(pi_new[1])))
                        self.space.move_agent(aj, (float(pj_new[0]), float(pj_new[1])))
                        moved = True
            if not moved:
                break

    # ---------------- Step ----------------
    def step(self):
        # Agents act
        self.agents.shuffle_do("step")  # Mesa 3 AgentSet API
        # Repulsion pass to eliminate overlaps by end of step
        self._resolve_overlaps()
        # Advance time and log
        self.time += self.dt
        self._log_state()
