
from mesa import Agent
import numpy as np
import math
from .utils import (
    radius_from_size_3d,
    volume_conserving_radius,
    mass_from_size,
    momentum_merge,
)

class ClusterAgent(Agent):
    def __init__(self, model, size, phenotype):
        # Mesa 3: pass only the model; unique_id auto-assigned
        super().__init__(model)
        self.size = int(size)
        self.phenotype = phenotype
        self.vel = np.zeros(2, dtype=float)
        self.alive = True
        self.event_log = []
        # Persistent per-agent radius multiplier
        rn = self.model.params.get("radius_noise", {})
        if bool(rn.get("enable", True)):
            sigma = float(rn.get("sigma", 0.35))
            z = self.model.random.normalvariate(0.0, sigma)
            if str(rn.get("preserve", "area")).lower() == "area":
                self.r_mult = math.exp(z - sigma**2)      # E[f^2] = 1
            else:
                self.r_mult = math.exp(z - 0.5*sigma**2)  # E[f] = 1
        else:
            self.r_mult = 1.0
        self.radius = float(self.r_mult * radius_from_size_3d(self.size))
        # self.pos is set when placed in space

    def step(self):
        self._move()
        self._try_merge()
        self._maybe_proliferate()
        self._maybe_fragment()

    # ---------------- Movement ----------------
    def _move(self):
        if self.pos is None:
            return
        ph = self.model.params["phenotypes"][self.phenotype]
        speed_base = float(ph["speed_base"])
        dt = float(self.model.dt)
        mv = self.model.params.get("movement", {})
        mode = mv.get("mode", "constant")
        direction_mode = mv.get("direction", "isotropic")

        # 1) Step magnitude
        if mode == "constant":
            step_mag = speed_base * dt
        elif mode == "distribution":
            dist_name = str(mv.get("distribution", "lognorm")).lower()
            dp = mv.get("dist_params", {})
            eps = 1e-12
            if dist_name == "lognorm":
                s = float(dp.get("s", 0.6))
                scale = float(dp.get("scale", 2.0))
                z = self.model.random.normalvariate(0.0, s)
                step_mag = max(scale * np.exp(z), eps)
            elif dist_name == "gamma":
                a = float(dp.get("a", 2.0))
                sc = float(dp.get("scale", 1.0))
                step_mag = max(np.random.default_rng().gamma(shape=a, scale=sc), eps)
            elif dist_name == "weibull":
                c = float(dp.get("c", 1.5))
                sc = float(dp.get("scale", 2.0))
                U = self.model.random.random()
                step_mag = max(sc * (-np.log(max(1.0 - U, eps))) ** (1.0 / c), eps)
            elif dist_name == "rayleigh":
                sc = float(dp.get("scale", 2.0))
                U = self.model.random.random()
                step_mag = max(sc * np.sqrt(-2.0 * np.log(max(1.0 - U, eps))), eps)
            elif dist_name == "expon":
                sc = float(dp.get("scale", 1.0))
                U = self.model.random.random()
                step_mag = max(-sc * np.log(max(1.0 - U, eps)), eps)
            elif dist_name == "invgauss":
                mu = float(dp.get("mu", 1.0))
                sc = float(dp.get("scale", 1.0))
                step_mag = max(np.random.default_rng().wald(mean=mu, scale=sc), eps)
            else:
                step_mag = speed_base * dt
        else:
            step_mag = speed_base * dt

        # 2) Direction
        if direction_mode == "isotropic":
            theta = self.model.random.uniform(-np.pi, np.pi)
        elif direction_mode == "persistent":
            theta_prev = getattr(self, "_theta", None)
            if theta_prev is None:
                theta_prev = self.model.random.uniform(-np.pi, np.pi)
            sigma = float(mv.get("heading_sigma", 0.25))
            theta = float(theta_prev + self.model.random.normalvariate(0.0, sigma))
            self._theta = theta
        else:
            theta = self.model.random.uniform(-np.pi, np.pi)

        dir_vec = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        self.vel = (step_mag / max(dt, 1e-12)) * dir_vec
        pos_curr = np.asarray(self.pos, dtype=float)
        new_pos = pos_curr + dir_vec * step_mag
        x, y = float(new_pos[0]), float(new_pos[1])
        self.model.space.move_agent(self, (x, y))

    # ---------------- Interactions ----------------
    def _try_merge(self):
        if self.pos is None or not self.alive:
            return
        merge_strength = float(self.model.params["merge"]["strength"])  # single parameter
        neighbors = self.model.get_neighbors(self, 2 * self.radius)
        pos_self = np.asarray(self.pos, dtype=float)
        contacts = []  # (d2, other)
        for other in neighbors:
            if other is self or not other.alive or other.pos is None:
                continue
            pos_other = np.asarray(other.pos, dtype=float)
            dx, dy = self.model.min_image_delta(pos_self, pos_other)
            d2 = dx*dx + dy*dy
            r_sum = self.radius + other.radius
            if d2 <= r_sum * r_sum:
                contacts.append((d2, other))
        if not contacts:
            return
        # choose closest contact
        d2_min = min(d2 for d2, _ in contacts)
        tol = 1e-12
        tied = [other for d2, other in contacts if abs(d2 - d2_min) <= tol]
        target = self.model.random.choice(tied)
        if self.model.random.random() < merge_strength:
            self._merge_with(target)

    def _merge_with(self, other):
        p_self = np.array(self.pos, dtype=float)
        p_other = np.array(other.pos, dtype=float)
        m1 = mass_from_size(self.size)
        m2 = mass_from_size(other.size)
        size_new = self.size + other.size
        r_new_vc = volume_conserving_radius(self.radius, other.radius)
        v_new = momentum_merge(m1, self.vel, m2, other.vel)
        pos_new = (m1 * p_self + m2 * p_other) / (m1 + m2)

        # retire other
        other.alive = False
        self.model.remove_agent(other)

        # combine multipliers
        rn = self.model.params.get("radius_noise", {})
        method = str(rn.get("merge_combine", "max")).lower()
        m_self = getattr(self, "r_mult", 1.0)
        m_other = getattr(other, "r_mult", 1.0)
        if method == "weighted":
            m_new = (m_self * self.size + m_other * other.size) / max(self.size + other.size, 1)
        elif method == "self":
            m_new = m_self
        else:
            m_new = m_self if m_self >= m_other else m_other
        self.r_mult = float(m_new)

        # update size, radius, velocity and position
        self.size = int(size_new)
        if bool(rn.get("apply_after_merge", True)):
            base = radius_from_size_3d(self.size)
            self.radius = float(self.r_mult * base)
        else:
            self.radius = float(r_new_vc)
        self.vel = v_new
        x, y = float(pos_new[0]), float(pos_new[1])
        self.model.space.move_agent(self, (x, y))
        self.event_log.append(("merge", other.unique_id, self.model.time))

    # ---------------- Lifecycle ----------------
    def _maybe_proliferate(self):
        ph = self.model.params["phenotypes"][self.phenotype]
        lam = float(ph["prolif_rate"]) * self.size * self.model.dt
        if self.model.random.random() < lam:
            self.size += 1
            base = radius_from_size_3d(self.size)
            self.radius = float(self.r_mult * base)
            self.event_log.append(("proliferate", 1, self.model.time))

    def _maybe_fragment(self):
        ph = self.model.params["phenotypes"][self.phenotype]
        lam = float(ph["fragment_rate"]) * self.model.dt
        if self.size > 1 and self.model.random.random() < lam:
            self.size -= 1
            base = radius_from_size_3d(self.size)
            self.radius = float(self.r_mult * base)
            if self.pos is None:
                return
            p_self = np.array(self.pos, dtype=float)
            jitter = self.model.random.normalvariate(0, 3)
            candidate = p_self + np.array([jitter, -jitter], dtype=float)
            child_pos = self.model.space.torus_adj(tuple(candidate))
            child = self.model.spawn_cluster(size=1, phenotype=self.phenotype, pos=child_pos, jitter=False)
            self.event_log.append(("fragment", child.unique_id, self.model.time))
