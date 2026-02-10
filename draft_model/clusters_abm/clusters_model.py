from mesa import Model
from mesa.space import ContinuousSpace
import numpy as np
from .cluster_agent import ClusterAgent
from .utils import DEFAULTS

class ClustersModel(Model):
    """
    ABM for motility-driven clustering with merging, proliferation, and fragmentation.

    Logs recorded each timestep (aligned numpy arrays):
    - size_log : list of (N,) arrays (cluster sizes for alive agents)
    - speed_log : list of (N,) arrays (speed magnitudes for alive agents)
    - pos_log : list of (N,2) arrays (x,y for alive agents)
    - radius_log : list of (N,) arrays (r for alive agents)
    - id_log : list of (N,) integer arrays (agent unique_id for alive agents)

    N varies over time due to merges and fragmentation.
    """

    def __init__(self, params=None, seed=42, init_clusters=None):
        super().__init__()

        # Seed the model RNG
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
            n0 = int(init_cfg.get("n_clusters", 1000))
            sz0 = int(init_cfg.get("size", 1))
            ph0 = str(init_cfg.get("phenotype", "proliferative"))
            # uniform initial population of one phenotype
            init_clusters = [{"size": sz0, "phenotype": ph0} for _ in range(n0)]

        # Place agents
        for ic in init_clusters:
            self.spawn_cluster(size=ic["size"], phenotype=ic["phenotype"])

        # Logs
        self.size_log = []
        self.speed_log = []
        self.pos_log = []
        self.radius_log = []
        self.id_log = []

        # Snapshot at t = 0 (aligned logs)
        self._log_state()

    def get_neighbors(self, agent, r):
        """
        Neighbour query using ContinuousSpace indexing.
        Returns alive neighbours within radius r (excludes self).
        Toroidal wrapping handled by the space.
        """
        if agent.pos is None:
            return []
        candidates = self.space.get_neighbors(pos=agent.pos, radius=float(r), include_center=False)
        return [a for a in candidates if getattr(a, "alive", True)]

    def remove_agent(self, agent):
        """Remove agent from the model AgentSet and space (robust to double-calls)."""
        try:
            self.agents.remove(agent)
        except Exception:
            pass
        try:
            self.space.remove_agent(agent)
        except Exception:
            pass

    def spawn_cluster(self, size, phenotype, pos=None, jitter=False):
        # Construct agent and register first so unique_id exists before placement
        uid = self.next_id()
        a = ClusterAgent(uid, self, size=size, phenotype=phenotype)
        self.agents.add(a)

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
            jx = self.random.normalvariate(0, 1)
            jy = self.random.normalvariate(0, 1)
            pos = pos + np.array([jx, jy], dtype=float)

        # Ensure torus adjustment BEFORE placement (place_agent does not wrap)
        x, y = self.space.torus_adj(tuple(pos))
        self.space.place_agent(a, (float(x), float(y)))  # ContinuousSpace sets a.pos
        return a

    def _log_alive_snapshot(self):
        """
        Build aligned, float-typed arrays for all alive agents with valid positions.
        Returns: ids, pos(N,2), radii, sizes, speeds (all numpy arrays)
        """
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
                continue  # not yet placed / just removed
            try:
                x = float(p[0])
                y = float(p[1])
                r = float(a.radius)
                s = float(a.size)
                v = float(np.linalg.norm(a.vel))
                i = int(a.unique_id)
            except Exception:
                # Skip anything that doesn't parse cleanly this tick
                continue
            xs.append(x)
            ys.append(y)
            radii.append(r)
            sizes.append(s)
            speeds.append(v)
            ids.append(i)

        if len(xs) == 0:
            # Return empty, correctly-shaped arrays
            return (
                np.array([], dtype=int),
                np.empty((0, 2), dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
            )

        pos = np.column_stack([np.array(xs, dtype=float), np.array(ys, dtype=float)])
        return (
            np.array(ids, dtype=int),
            pos.astype(float, copy=False),
            np.array(radii, dtype=float),
            np.array(sizes, dtype=float),
            np.array(speeds, dtype=float),
        )

    def step(self):
        # Activate each agent once in random order (Mesa AgentSet API)
        self.agents.shuffle_do("step")  # calls a.step() for each agent in random order

        # Advance time
        self.time += self.dt

        # Robust per-timestep logs (aligned arrays)
        ids, pos, radii, sizes, speeds = self._log_alive_snapshot()
        self.id_log.append(ids)
        self.pos_log.append(pos)
        self.radius_log.append(radii)
        self.size_log.append(sizes)
        self.speed_log.append(speeds)

    def _log_state(self):
        """Record a one-off snapshot of the initial state at t = 0 (aligned)."""
        ids, pos, radii, sizes, speeds = self._log_alive_snapshot()
        self.id_log.append(ids)
        self.pos_log.append(pos)
        self.radius_log.append(radii)
        self.size_log.append(sizes)
        self.speed_log.append(speeds)  # ensure t=0 includes speed as well