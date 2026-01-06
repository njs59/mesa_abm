
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

        # Parameters
        merge_prob = self.model.params["merge"]["prob_contact_merge"]
        ph_self = self.model.params["phenotypes"][self.phenotype]

        # Neighbour query within a reasonable radius bucket
        neighbors = self.model.get_neighbors(self, 2 * self.radius)
        pos_self = np.asarray(self.pos, dtype=float)

        # Gather all contacted neighbours with their squared distances and adhesion averages
        contacts = []  # list of (d2, other, adh_avg)
        for other in neighbors:
            if other is self or not other.alive or other.pos is None:
                continue
            pos_other = np.asarray(other.pos, dtype=float)

            # Squared distance; compare to squared sum of radii
            d2 = np.sum((pos_other - pos_self) ** 2)
            r_sum = self.radius + other.radius
            if d2 <= r_sum * r_sum:
                ph_other = self.model.params["phenotypes"][other.phenotype]
                adh_avg = 0.5 * (ph_self["adhesion"] + ph_other["adhesion"])
                contacts.append((d2, other, adh_avg))

        if not contacts:
            return

        # ---- Active path: closest neighbour with RANDOM tie-breaking ----
        # Find the minimum squared distance among contacts
        d2_min = min(d2 for d2, _, _ in contacts)

        # Small tolerance to treat numerically-equal distances as ties
        # If your domain coordinates are very large, you can relax this (e.g., 1e-9)
        tol = 1e-12

        # All neighbours whose distance is tied with the minimum (within tol)
        tied = [(other, adh_avg) for d2, other, adh_avg in contacts if abs(d2 - d2_min) <= tol]

        # Randomly pick ONE neighbour among exact-distance ties (reproducible via model.random)
        target, adh_avg = self.model.random.choice(tied)

        # Bernoulli merge trial; at most one merge per step
        if self.model.random.random() < merge_prob * adh_avg:
            self._merge_with(target)
            return  # one merge per step

        # ----------------------------------------------------------------
        # OPTIONAL (commented): probabilistic selection among ALL contacts
        # ----------------------------------------------------------------
        # Enable this block if you prefer a single stochastic pick among all contacted neighbours
        # weighted by adhesion and inverse distance (rather than pure closest-distance with tie-break).
        #
        # Idea: score_i ∝ (adh_i ** alpha) / ((d2_i + eps) ** beta)
        # where alpha, beta ≥ 0, eps is a small stabiliser.
        # Example: alpha = 1.0, beta = 1.0 (adhesion upweights; closer neighbours upweight)
        #
        # alpha = 1.0
        # beta = 1.0
        # eps = 1e-12
        #
        # # Compute unnormalised scores
        # scores = []
        # for d2, other, adh in contacts:
        #     # If you prefer plain distance (not squared), you could use sqrt(d2) here,
        #     # but keeping d2 avoids an extra sqrt and is consistent with the contact test.
        #     s = (adh ** alpha) / ((d2 + eps) ** beta)
        #     scores.append(s)
        #
        # total = sum(scores)
        # if total <= 0.0:
        #     # Degenerate case: fall back to uniform random pick among contacts
        #     idx = self.model.random.randrange(len(contacts))
        #     target = contacts[idx][1]
        #     adh_avg = contacts[idx][2]
        # else:
        #     # Convert to cumulative probabilities and sample using model RNG (reproducible)
        #     cum = []
        #     acc = 0.0
        #     for s in scores:
        #         acc += s / total
        #         cum.append(acc)
        #     u = self.model.random.random()
        #     # find first index where cum[idx] >= u
        #     idx = 0
        #     while idx < len(cum) and cum[idx] < u:
        #         idx += 1
        #     if idx >= len(cum):
        #         idx = len(cum) - 1
        #     target = contacts[idx][1]
        #     adh_avg = contacts[idx][2]
        #
        # # Now attempt merge with the probabilistically selected target
        # if self.model.random.random() < merge_prob * adh_avg:
        #     self._merge_with(target)
        #     return  # one merge per step


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
