from mesa import Agent
import numpy as np

from abm.utils import (
    radius_from_size_3d,
    volume_conserving_radius,
    mass_from_size,
    momentum_merge,
)

class ClusterAgent(Agent):
    """Two‑phase motile agent with one‑way Phase1→Phase2 transition.

    Phase 1: speed~lognormal, turning~VonMises(mu=pi,kappa≈0.242)
    Phase 2: speed~gamma,     turning~VonMises(mu=0,kappa≈0.147)
    Transition time is drawn once at spawn from a shifted‑Gompertz CDF
    tabulated by the model and sampled via inverse lookup.
    """


    def __init__(self, model, size, phenotype):
        # --- Backward/forward compatible super() call ---
        try:
            # Mesa ≤2.x signature: (unique_id, model)
            uid = getattr(model, "next_id", None)
            if callable(uid):               # Mesa 1/2 provide next_id() method
                uid = uid()
            elif isinstance(uid, int):      # or an int counter
                # if model.next_id is an int, use and increment it
                uid, model.next_id = uid, uid + 1
            else:
                # fallback local counter on the model
                if not hasattr(model, "_uid_counter"):
                    model._uid_counter = 0
                uid = model._uid_counter
                model._uid_counter += 1
            super().__init__(uid, model)
        except TypeError:
            # Mesa 3.x signature: (model) only
            super().__init__(model)

        self.size = int(size)
        self.phenotype = phenotype
        self.radius = radius_from_size_3d(self.size)
        self.vel = np.zeros(2, dtype=float)
        self.alive = True
        self.event_log = []

        self.movement_phase = 1
        self._theta = None

        # One-way switch time from model's pre-tabulated shifted-Gompertz CDF
        self.phase_switch_time = float(self.model.sample_transition_time())


    # ------------------------------------------------------------------
    def step(self):
        # Switch to Phase 2 once, irreversibly
        if self.movement_phase == 1 and float(self.model.time) >= self.phase_switch_time:
            self.movement_phase = 2
            self.event_log.append(("phase_switch", self.model.time))
        # Movement
        self._move_two_phase()
        # Biology
        self._try_merge()
        self._maybe_proliferate()
        self._maybe_fragment()

    # ------------------------------------------------------------------
    def _move_two_phase(self):
        if self.pos is None:
            return
        cfg = self.model.params["movement_v2"]
        phase_block = cfg["phase1"] if self.movement_phase == 1 else cfg["phase2"]
        sp = phase_block["speed_dist"]
        trn = phase_block["turning"]

        # phenotype base speed with optional size scaling
        ph = self.model.params["phenotypes"][self.phenotype]
        speed_base = float(ph["speed_base"])
        beta = float(ph.get("speed_size_exp", 0.0))
        if beta != 0.0:
            speed_base *= (self.size ** (-beta))

        rng = self.model.np_rng
        # speed sampling
        name = sp["name"].lower()
        dp = sp["params"]
        if name == "lognorm":
            s = float(dp["s"]); scale = float(dp["scale"])
            speed = scale * np.exp(rng.normal(0.0, s))
        elif name == "gamma":
            a = float(dp["a"]); scale = float(dp["scale"])
            speed = rng.gamma(a, scale)
        else:
            speed = speed_base

        dt = float(self.model.dt)
        step_mag = speed * dt

        # turning (relative Von Mises)
        if self._theta is None:
            self._theta = self.model.random.uniform(-np.pi, np.pi)
        mu = float(trn.get("mu", 0.0))
        kappa = float(trn.get("kappa", 0.0))
        dtheta = float(rng.vonmises(mu=mu, kappa=max(kappa, 0.0)))
        theta = float(np.arctan2(np.sin(self._theta + dtheta), np.cos(self._theta + dtheta)))
        self._theta = theta

        dir_vec = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        self.vel = speed * dir_vec
        newp = np.asarray(self.pos, dtype=float) + dir_vec * step_mag
        self.model.space.move_agent(self, (float(newp[0]), float(newp[1])))

        # short‑range soft separation
        self._soft_separate()

    # ------------------------------------------------------------------
    def _soft_separate(self):
        if not self.model.params["physics"].get("soft_separate", True):
            return
        if self.pos is None:
            return
        r_self = float(self.radius)
        neighs = self.model.get_neighbors(self, r=2.4*r_self)
        pos_self = np.asarray(self.pos, dtype=float)
        disp = np.zeros(2, dtype=float)
        softness = float(self.model.params["physics"].get("softness", 0.15))
        for other in neighs:
            if (other is self) or (not getattr(other, "alive", True)) or (other.pos is None):
                continue
            rij = pos_self - np.asarray(other.pos, dtype=float)
            d = float(np.linalg.norm(rij))
            r_sum = r_self + float(other.radius)
            if d < r_sum:
                if d < 1e-12:
                    phi = self.model.random.uniform(-np.pi, np.pi)
                    rij_unit = np.array([np.cos(phi), np.sin(phi)], dtype=float)
                else:
                    rij_unit = rij / d
                disp += (r_sum - d) * softness * rij_unit
        if np.any(disp):
            newp = pos_self + disp
            self.model.space.move_agent(self, (float(newp[0]), float(newp[1])))

    # ------------------------------------------------------------------
    def _try_merge(self):
        if self.pos is None:
            return
        p_merge = float(self.model.params["merge"].get("p_merge", 0.9))
        neighbors = self.model.get_neighbors(self, 2* self.radius)
        pos_self = np.asarray(self.pos, dtype=float)
        contacts = []
        for other in neighbors:
            if other is self or not other.alive or other.pos is None:
                continue
            d2 = float(np.sum((np.asarray(other.pos, dtype=float) - pos_self)**2))
            if d2 <= (self.radius + other.radius)**2:
                contacts.append((d2, other))
        if not contacts:
            return
        d2_min = min(d2 for d2,_ in contacts)
        tied = [o for d2,o in contacts if abs(d2 - d2_min) <= 1e-12]
        target = self.model.random.choice(tied)
        if self.model.random.random() < p_merge:
            self._merge_with(target)

    def _merge_with(self, other):
        p_self = np.asarray(self.pos, dtype=float)
        p_other = np.asarray(other.pos, dtype=float)
        m1, m2 = mass_from_size(self.size), mass_from_size(other.size)
        size_new = self.size + other.size
        r_new = volume_conserving_radius(self.radius, other.radius)
        v_new = momentum_merge(m1, self.vel, m2, other.vel)
        pos_new = (m1*p_self + m2*p_other)/(m1+m2)

        other.alive = False
        self.model.remove_agent(other)

        self.size = int(size_new)
        self.radius = float(r_new)
        self.vel = v_new
        self.model.space.move_agent(self, (float(pos_new[0]), float(pos_new[1])))
        self.event_log.append(("merge", other.unique_id, self.model.time))

    # ------------------------------------------------------------------
    def _maybe_proliferate(self):
        ph = self.model.params["phenotypes"][self.phenotype]
        lam = ph["prolif_rate"] * self.size * self.model.dt
        if self.model.random.random() < lam:
            self.size += 1
            self.radius = radius_from_size_3d(self.size)
            self.event_log.append(("proliferate", 1, self.model.time))

    def _maybe_fragment(self):
        ph = self.model.params["phenotypes"][self.phenotype]
        lam = ph["fragment_rate"] * self.model.dt
        if self.size > 1 and self.model.random.random() < lam:
            self.size -= 1
            self.radius = radius_from_size_3d(self.size)
            if self.pos is None:
                return
            p_self = np.asarray(self.pos, dtype=float)
            r_child = float(radius_from_size_3d(1))
            factor = float(self.model.params["physics"].get("fragment_minsep_factor", 1.1))
            min_sep = factor * (self.radius + r_child)
            theta = self.model.random.uniform(-np.pi, np.pi)
            offset = min_sep * np.array([np.cos(theta), np.sin(theta)], dtype=float)
            child_pos = tuple(p_self + offset)
            child = self.model.spawn_cluster(1, self.phenotype, pos=child_pos, jitter=False)
            if self.movement_phase == 2:
                child.movement_phase = 2
                child.phase_switch_time = np.inf
            self.event_log.append(("fragment", child.unique_id, self.model.time))
