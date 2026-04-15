from mesa import Agent
import numpy as np
from abm.utils import (
    radius_from_size_3d,
    volume_conserving_radius,
    mass_from_size,
    momentum_merge,
)

class ClusterAgent(Agent):
    """
    Two-phase motile agent with one-way Phase1→Phase2 transition.
    Movement is controlled ONLY by movement_v2[phenotype]:
      - Phase 1 speed distribution + turning
      - Phase 2 speed distribution + turning
      - Transition time distribution (sampled once at spawn)
    """

    def __init__(self, model, size, phenotype, phase_switch_time=None):
        # Mesa compatibility for unique ID (unchanged)
        try:
            uid = getattr(model, "next_id", None)
            if callable(uid):
                uid = uid()
            elif isinstance(uid, int):
                uid, model.next_id = uid, uid + 1
            else:
                if not hasattr(model, "_uid_counter"):
                    model._uid_counter = 0
                uid = model._uid_counter
                model._uid_counter += 1
            super().__init__(uid, model)
        except TypeError:
            super().__init__(model)

        self.size = int(size)
        self.phenotype = phenotype
        self.radius = radius_from_size_3d(self.size)
        self.vel = np.zeros(2, dtype=float)
        self.alive = True
        self.event_log = []
        self.movement_phase = 1
        self._theta = None

        # Transition-time logic (unchanged)
        if phase_switch_time is not None:
            self.phase_switch_time = float(phase_switch_time)
        else:
            self.phase_switch_time = float(
                self.model.sample_transition_time(self.phenotype)
            )

    # ------------------------------------------------------------------
    def step(self):
        # Phase 1 → Phase 2 transition (unchanged)
        if self.movement_phase == 1 and float(self.model.time) >= self.phase_switch_time:
            self.movement_phase = 2
            self.event_log.append(("phase_switch", self.model.time))

        # Order preserved from your old code:
        self._move_two_phase()      # 1) Move
        self._try_merge()           # 2) Merge (single-target)
        self._maybe_proliferate()   # 3) Proliferate
        self._maybe_fragment()      # 4) Fragment

        # NEW: deterministic push-apart at the end of the step
        self._apply_repulsion()     # 5) Repulsion (instantaneous)

    # ------------------------------------------------------------------
    # Movement code — ORIGINAL (kept), with soft-separate call removed
    # ------------------------------------------------------------------
    def _move_two_phase(self):
        if self.pos is None:
            return

        mv2 = self.model.params["movement_v2"]
        mv2_is_global = isinstance(mv2, dict) and ("phase1" in mv2) and ("phase2" in mv2)
        cfg = mv2 if mv2_is_global else mv2[self.phenotype]

        phase_block = cfg["phase1"] if self.movement_phase == 1 else cfg["phase2"]
        sp = phase_block["speed_dist"]
        trn = phase_block["turning"]
        rng = self.model.np_rng

        # Speed sampling (unchanged)
        name = str(sp.get("name", "")).lower()
        dp = sp.get("params", {})
        if name == "lognorm":
            s = float(dp["s"])
            scale = float(dp["scale"])
            speed = scale * np.exp(rng.normal(0.0, s))
        elif name == "gamma":
            a = float(dp["a"])
            scale = float(dp["scale"])
            speed = rng.gamma(a, scale)
        else:
            speed = float(dp.get("speed", 1.0))

        dt = float(self.model.dt)
        step_mag = speed * dt

        # Turning (unchanged)
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

        # OLD: self._soft_separate()  <-- REMOVED to avoid in-move “lingering” pushes (new rule)

    # ------------------------------------------------------------------
    # NEW: deterministic repulsion (instant push-apart to just touching)
    # ------------------------------------------------------------------
    def _apply_repulsion(self):
        if self.pos is None:
            return
        pos_self = np.asarray(self.pos, dtype=float)
        r_self = float(self.radius)

        # Use a conservative search radius (same idea as Fix B)
        r_max = float(getattr(self.model, "max_radius", r_self))
        neighbors = self.model.get_neighbors(self, r_self + r_max + 1e-9)

        total_disp = np.zeros(2, dtype=float)
        for other in neighbors:
            if other is self or not getattr(other, "alive", True) or other.pos is None:
                continue

            rij = pos_self + total_disp - np.asarray(other.pos, dtype=float)
            d = float(np.linalg.norm(rij))
            r_sum = r_self + float(other.radius)

            if d < r_sum:
                if d < 1e-12:
                    # Perfectly coincident -> pick a random gentle direction
                    phi = self.model.random.uniform(-np.pi, np.pi)
                    unit = np.array([np.cos(phi), np.sin(phi)], dtype=float)
                else:
                    unit = rij / d
                # Push self just enough to be tangent to 'other'
                total_disp += (r_sum - d) * unit

        if np.any(total_disp):
            newp = pos_self + total_disp
            self.model.space.move_agent(self, (float(newp[0]), float(newp[1])))

    # ------------------------------------------------------------------
    # (Legacy) soft separation kept for compatibility but no longer used
    # ------------------------------------------------------------------
    def _soft_separate(self):
        # Kept as a no-op unless explicitly re-enabled in movement
        if not self.model.params["physics"].get("soft_separate", True):
            return
        if self.pos is None:
            return
        r_self = float(self.radius)
        neighs = self.model.get_neighbors(self, r=2.4 * r_self)
        pos_self = np.asarray(self.pos, dtype=float)
        disp = np.zeros(2, dtype=float)
        softness = float(self.model.params["physics"].get("softness", 0.15))
        for other in neighs:
            if other is self or not getattr(other, "alive", True) or other.pos is None:
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
    # Merge logic — ORIGINAL behaviour with Fix B in pre-filter radius
    # ------------------------------------------------------------------
    def _try_merge(self):
        if self.pos is None:
            return

        # Your old default here is 0.9; leaving it unchanged
        p_merge = float(self.model.params["merge"].get("p_merge", 0.9))  # old default kept 

        # Fix B: candidate search uses size-aware upper bound to avoid missing larger neighbours
        r_self = float(self.radius)
        r_max = float(getattr(self.model, "max_radius", r_self))
        search_r = r_self + r_max + 1e-9
        neighbors = self.model.get_neighbors(self, search_r)

        pos_self = np.asarray(self.pos, dtype=float)
        contacts = []
        for other in neighbors:
            if other is self or not other.alive or other.pos is None:
                continue
            d2 = float(np.sum((np.asarray(other.pos) - pos_self) ** 2))
            if d2 <= (self.radius + other.radius) ** 2:
                contacts.append((d2, other))

        if not contacts:
            return

        # Single-target: choose the closest (ties random) — same as old code
        d2_min = min(d2 for d2, _ in contacts)
        tied = [o for d2, o in contacts if abs(d2 - d2_min) <= 1e-12]
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
        pos_new = (m1 * p_self + m2 * p_other) / (m1 + m2)

        other.alive = False
        self.model.remove_agent(other)

        self.size = int(size_new)
        self.radius = float(r_new)
        # Keep model's max_radius current for Fix B
        if hasattr(self.model, "update_max_radius"):
            self.model.update_max_radius(self.radius)

        self.vel = v_new
        self.model.space.move_agent(self, (float(pos_new[0]), float(pos_new[1])))
        self.event_log.append(("merge", other.unique_id, self.model.time))

    # ------------------------------------------------------------------
    # Proliferation — ORIGINAL PRESERVED (plus max_radius ping)
    # ------------------------------------------------------------------
    def _maybe_proliferate(self):
        ph = self.model.params["phenotypes"][self.phenotype]
        lam = ph["prolif_rate"] * self.size * self.model.dt
        if self.model.random.random() < lam:
            self.size += 1
            self.radius = radius_from_size_3d(self.size)
            if hasattr(self.model, "update_max_radius"):
                self.model.update_max_radius(self.radius)
            self.event_log.append(("proliferate", 1, self.model.time))

    # ------------------------------------------------------------------
    # Fragmentation — ORIGINAL (child inherits phase time); spawn updates max
    # ------------------------------------------------------------------
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

            # CHILD INHERITS parent's transition time (unchanged)
            child = self.model.spawn_cluster(
                1,
                self.phenotype,
                pos=child_pos,
                jitter=False,
                phase_switch_time=self.phase_switch_time,
            )
            if self.movement_phase == 2:
                child.movement_phase = 2
                child.phase_switch_time = np.inf

            self.event_log.append(("fragment", child.unique_id, self.model.time))