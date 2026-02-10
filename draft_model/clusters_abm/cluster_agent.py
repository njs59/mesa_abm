from mesa import Agent
import numpy as np
from .utils import (
    radius_from_size_3d,
    volume_conserving_radius,
    mass_from_size,
    momentum_merge,
)

class ClusterAgent(Agent):
    def __init__(self, unique_id, model, size, phenotype):
        # Mesa classic signature: (unique_id, model)
        super().__init__(unique_id, model)

        # Agent state
        self.size = int(size)
        self.phenotype = phenotype
        self.radius = radius_from_size_3d(self.size)
        self.vel = np.zeros(2, dtype=float)
        self.alive = True
        self.event_log = []
        # self.pos is set when placed into space

    def step(self):
        self._move()
        self._try_merge()
        self._maybe_proliferate()
        self._maybe_fragment()

    def _move(self):
        """
        Update agent position. Supports either:
        - constant step magnitude = speed_base * dt (mode="constant"),
        - or sampled step magnitude from a configured distribution (mode="distribution").

        Direction handling:
        - "isotropic": draw a new random angle each step (default),
        - "persistent": keep previous heading, with small Gaussian angular jitter.
        """
        # Guard: require a valid position
        if self.pos is None:
            return

        ph = self.model.params["phenotypes"][self.phenotype]
        speed_base = float(ph["speed_base"])
        dt = float(self.model.dt)

        mv = self.model.params.get("movement", {})
        mode = mv.get("mode", "constant")
        direction_mode = mv.get("direction", "isotropic")

        # --- 1) Choose step magnitude ---
        if mode == "constant":
            step_mag = speed_base * dt
        elif mode == "distribution":
            # Draw step magnitude directly in model distance units per step.
            # Use the model's seeded RNGs for reproducibility.
            dist_name = str(mv.get("distribution", "lognorm")).lower()
            dp = mv.get("dist_params", {})
            eps = 1e-12

            # Prefer model.rng (NumPy Generator) or fall back to stdlib model.random
            rng = getattr(self.model, "rng", None)

            if dist_name == "lognorm":
                # ln X ~ N(mu, sigma^2); emulate via stdlib normal plus exp
                s = float(dp.get("s", 0.6))       # sigma
                scale = float(dp.get("scale", 2.0))  # exp(mu)
                z = (rng.normal(0.0, s) if rng is not None
                     else np.random.normal(0.0, s))
                step_mag = max(scale * np.exp(z), eps)

            elif dist_name == "gamma":
                a = float(dp.get("a", 2.0))
                sc = float(dp.get("scale", 1.0))
                step_mag = max((rng.gamma(a, sc) if rng is not None
                                else np.random.gamma(a, sc)), eps)

            elif dist_name == "weibull":
                c = float(dp.get("c", 1.5))
                sc = float(dp.get("scale", 2.0))
                U = (self.model.random.random())
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
                step_mag = max((rng.wald(mu, sc) if rng is not None
                                else np.random.wald(mu, sc)), eps)
            else:
                step_mag = speed_base * dt
        else:
            step_mag = speed_base * dt

        # --- 2) Choose direction ---
        pos_curr = np.asarray(self.pos, dtype=float)
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

        # Store a per-unit-time velocity vector for logging
        self.vel = (step_mag / max(dt, 1e-12)) * dir_vec

        # Advance position (ContinuousSpace handles wrapping if torus=True)
        new_pos = pos_curr + dir_vec * step_mag
        x, y = float(new_pos[0]), float(new_pos[1])
        self.model.space.move_agent(self, (x, y))

    def _try_merge(self):
        if self.pos is None:
            return

        merge_prob = self.model.params["merge"]["prob_contact_merge"]
        ph_self = self.model.params["phenotypes"][self.phenotype]

        neighbors = self.model.get_neighbors(self, 2 * self.radius)
        pos_self = np.asarray(self.pos, dtype=float)

        contacts = []  # list of (d2, other, adh_avg)
        for other in neighbors:
            if other is self or not other.alive or other.pos is None:
                continue
            pos_other = np.asarray(other.pos, dtype=float)
            d2 = np.sum((pos_other - pos_self) ** 2)
            r_sum = self.radius + other.radius
            if d2 <= r_sum * r_sum:
                ph_other = self.model.params["phenotypes"][other.phenotype]
                adh_avg = 0.5 * (ph_self["adhesion"] + ph_other["adhesion"])
                contacts.append((d2, other, adh_avg))

        if not contacts:
            return

        d2_min = min(d2 for d2, _, _ in contacts)
        tol = 1e-12
        tied = [(other, adh) for d2, other, adh in contacts if abs(d2 - d2_min) <= tol]
        target, adh_avg = self.model.random.choice(tied)

        if self.model.random.random() < merge_prob * adh_avg:
            self._merge_with(target)
            return  # one merge per step

    def _merge_with(self, other):
        p_self = np.array(self.pos, dtype=float)
        p_other = np.array(other.pos, dtype=float)
        m1 = mass_from_size(self.size)
        m2 = mass_from_size(other.size)

        size_new = self.size + other.size
        r_new = volume_conserving_radius(self.radius, other.radius)
        v_new = momentum_merge(m1, self.vel, m2, other.vel)
        pos_new = (m1 * p_self + m2 * p_other) / (m1 + m2)

        # Remove 'other' from the model
        other.alive = False
        self.model.remove_agent(other)

        # Update 'self'
        self.size = int(size_new)
        self.radius = float(r_new)
        self.vel = v_new
        x, y = float(pos_new[0]), float(pos_new[1])
        self.model.space.move_agent(self, (x, y))
        self.event_log.append(("merge", other.unique_id, self.model.time))

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
            p_self = np.array(self.pos, dtype=float)
            jitter = self.model.random.normalvariate(0, 3)
            candidate_pos = p_self + np.array([jitter, -jitter], dtype=float)

            # Wrap child position onto the torus BEFORE placement
            child_pos = self.model.space.torus_adj(tuple(candidate_pos))
            child = self.model.spawn_cluster(size=1, phenotype=self.phenotype, pos=child_pos, jitter=False)
            self.event_log.append(("fragment", child.unique_id, self.model.time))