
from mesa import Agent
import numpy as np

from .utils import (
    radius_from_size_3d,
    volume_conserving_radius,
    mass_from_size,
    momentum_merge,
)

class ClusterAgent(Agent):
    def __init__(self, model, size, phenotype):
        # Mesa 3: pass only the model; unique_id is auto-assigned
        super().__init__(model)
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
        ph = self.model.params["phenotypes"][self.phenotype]
        speed = ph["speed_base"]  # size-independent
        if self.pos is None:
            return
        pos_curr = np.array(self.pos, dtype=float)
        theta = self.model.random.uniform(-np.pi, np.pi)
        self.vel = speed * np.array([np.cos(theta), np.sin(theta)], dtype=float)
        new_pos = pos_curr + self.vel * self.model.dt
        x, y = float(new_pos[0]), float(new_pos[1])
        # Move; torus wrapping handled internally by ContinuousSpace
        self.model.space.move_agent(self, (x, y))
        # Do NOT overwrite self.pos: ContinuousSpace maintains the wrapped position.

    def _try_merge(self):
        if self.pos is None:
            return
        merge_prob = self.model.params["merge"]["prob_contact_merge"]
        neighbors = self.model.get_neighbors(self, 2 * self.radius)
        pos_self = np.array(self.pos, dtype=float)
        for other in neighbors:
            if other is self or not other.alive or other.pos is None:
                continue
            pos_other = np.array(other.pos, dtype=float)
            d = np.linalg.norm(pos_other - pos_self)
            if d <= (self.radius + other.radius):
                adh = 0.5 * (
                    self.model.params["phenotypes"][self.phenotype]["adhesion"]
                    + self.model.params["phenotypes"][other.phenotype]["adhesion"]
                )
                if self.model.random.random() < merge_prob * adh:
                    self._merge_with(other)
                    break  # merge once per step

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
        self.model.space.move_agent(self, (x, y))  # wraps if torus=True
        # Do NOT overwrite self.pos.

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
