from mesa import Agent
import numpy as np
from .utils import (
    radius_from_size_3d,
    volume_conserving_radius,
    mass_from_size,
    momentum_merge,
)

class ClusterAgent(Agent):
    def __init__(self, model, size, phenotype, movement_phases=None):
        # Mesa 3: pass only the model; unique_id is auto-assigned
        super().__init__(model)
        self.size = int(size)
        self.phenotype = phenotype
        self.radius = radius_from_size_3d(self.size)
        self.vel = np.zeros(2, dtype=float)
        self.alive = True
        self.event_log = []

        # self.pos is set when placed into space

        # Optional per-agent movement schedule (list of dicts with t_start)
        # If None, we will fall back to model.params["movement"] (legacy global).
        self.movement_phases = None
        if movement_phases is None:
            phases = self.model.params.get("movement_phases", None)
        else:
            phases = movement_phases

        if phases:
            # Shallow-copy & sort by t_start for deterministic selection
            def _tstart(ph):
                return float(ph.get("t_start", 0.0))

            self.movement_phases = sorted([dict(p) for p in phases], key=_tstart)

        # Current heading memory for "persistent" mode (and now Von Mises-relative)
        self._theta = None

    def step(self):
        self._move()
        self._try_merge()
        self._maybe_proliferate()
        self._maybe_fragment()

    # -------------------------
    # Movement (updated)
    # -------------------------
    def _active_motion_spec(self):
        """
        Returns a dict describing the active motion controls at the current time.
        Priority:
          1) Per-agent movement_phases if present (choose last with t_start <= time)
          2) Fallback to model.params["movement"]

        The spec provides:
          speed_mode: "constant"|"distribution"
          speed_value: float (if constant)
          speed_distribution: str (if distribution)
          speed_dist_params: dict (if distribution)
          direction: "isotropic"|"persistent"|"von_mises"
          heading_sigma: float (for persistent)
          mu: float, kappa: float (for von_mises; here μ is the MEAN TURN (Δθ), not absolute)
        """
        # Default: legacy movement block
        mv = self.model.params.get("movement", {})
        spec = {
            "speed_mode": mv.get("mode", "constant"),
            "speed_value": None,  # will default to phenotype base if None
            "speed_distribution": mv.get("distribution", "lognorm"),
            "speed_dist_params": mv.get("dist_params", {}),
            "direction": mv.get("direction", "isotropic"),
            "heading_sigma": float(mv.get("heading_sigma", 0.25)),
            "mu": float(mv.get("mu", 0.0)),
            "kappa": float(mv.get("kappa", 2.0)),
        }

        # Phases override if present
        if self.movement_phases:
            t = float(self.model.time)
            # Pick the last phase with t_start <= t
            active = None
            for ph in self.movement_phases:
                if float(ph.get("t_start", 0.0)) <= t:
                    active = ph
                else:
                    break
            if active is None:
                active = self.movement_phases[0]  # before first start -> first phase

            spec["speed_mode"] = active.get("speed_mode", spec["speed_mode"])
            spec["speed_value"] = active.get("speed_value", spec["speed_value"])
            spec["speed_distribution"] = active.get(
                "speed_distribution", spec["speed_distribution"]
            )
            spec["speed_dist_params"] = active.get(
                "speed_dist_params", spec["speed_dist_params"]
            )
            spec["direction"] = active.get("direction", spec["direction"])
            spec["heading_sigma"] = float(
                active.get("heading_sigma", spec["heading_sigma"])
            )
            spec["mu"] = float(active.get("mu", spec["mu"]))
            spec["kappa"] = float(active.get("kappa", spec["kappa"]))
        return spec

    def _move(self):
        """
        Update agent position by moving with either:
          - constant step magnitude = speed * dt (speed_mode == "constant"),
          - or sampled step magnitude from a distribution (speed_mode == "distribution").

        Direction handling:
          - "isotropic": draw a new random angle each step (uniform on [-π, π]).
          - "persistent": keep previous heading, with small Gaussian angular jitter (delta update).
          - "von_mises": draw an angular CHANGE Δθ ~ VonMises(μ, κ) and set
                         θ_new = wrap(θ_prev + Δθ) to (-π, π]; i.e., RELATIVE turning.

        After moving, apply a gentle short-range separation that nudges overlapping
        neighbours apart by a small fraction of the overlap depth.
        """
        # Guard: require a valid position
        if self.pos is None:
            return

        # Phenotype base speed (with optional size scaling, as before)
        ph = self.model.params["phenotypes"][self.phenotype]
        speed_base = float(ph["speed_base"])
        beta = float(ph.get("speed_size_exp", 0.0))
        if beta != 0.0:
            speed_base = speed_base * (self.size ** (-beta))

        dt = float(self.model.dt)
        spec = self._active_motion_spec()

        # --- 1) Choose step magnitude ---
        if spec["speed_mode"] == "constant":
            # Use phase-provided speed_value if present, else phenotype base
            speed = float(spec["speed_value"]) if (spec["speed_value"] is not None) else speed_base
            step_mag = speed * dt
        else:  # "distribution"
            dist_name = str(spec["speed_distribution"]).lower()
            dp = spec["speed_dist_params"] or {}
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
                mu_ig = float(dp.get("mu", 1.0))
                sc = float(dp.get("scale", 1.0))
                step_mag = max(np.random.default_rng().wald(mean=mu_ig, scale=sc), eps)
            else:
                step_mag = speed_base * dt

        # --- 2) Choose direction ---
        pos_curr = np.asarray(self.pos, dtype=float)
        direction_mode = str(spec["direction"]).lower()

        if direction_mode == "isotropic":
            theta = self.model.random.uniform(-np.pi, np.pi)

        elif direction_mode == "persistent":
            theta_prev = self._theta if (self._theta is not None) else self.model.random.uniform(-np.pi, np.pi)
            sigma = float(spec.get("heading_sigma", 0.25))
            theta = float(theta_prev + self.model.random.normalvariate(0.0, sigma))
            self._theta = theta  # store updated heading

        elif direction_mode == "von_mises":
            # --- Relative Von Mises turning ---
            # Ensure we have a previous heading; if not, initialise uniformly.
            if self._theta is None:
                self._theta = self.model.random.uniform(-np.pi, np.pi)

            mu = float(spec.get("mu", 0.0))       # mean TURN (Δθ), radians
            kappa = float(spec.get("kappa", 2.0)) # concentration (>= 0)

            # Use a shared NumPy RNG if the model provides one; else make a local generator.
            rng = getattr(self.model, "np_rng", None)
            if rng is None:
                rng = np.random.default_rng()

            # Sample the angular CHANGE from Von Mises, then add to previous heading.
            dtheta = float(rng.vonmises(mu=mu, kappa=max(kappa, 0.0)))
            theta_new = self._theta + dtheta

            # Wrap to (-π, π] using arctan2(sin, cos), which is numerically robust on the circle.
            theta = float(np.arctan2(np.sin(theta_new), np.cos(theta_new)))

            # Store updated heading
            self._theta = theta

        else:
            theta = self.model.random.uniform(-np.pi, np.pi)

        dir_vec = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        self.vel = (step_mag / max(dt, 1e-12)) * dir_vec  # velocity magnitude = step_mag / dt
        new_pos = pos_curr + dir_vec * step_mag
        x, y = float(new_pos[0]), float(new_pos[1])
        self.model.space.move_agent(self, (x, y))

        # --- Gentle post-move soft separation (short-range repulsion) ---
        params_physics = self.model.params.get("physics", {})
        if params_physics.get("soft_separate", True):
            r_self = float(self.radius)
            # Limit neighbour search range to keep it cheap
            neighs = self.model.get_neighbors(self, r=2.4 * r_self)
            pos_self = np.asarray(self.pos, dtype=float)
            disp = np.zeros(2, dtype=float)
            softness = float(params_physics.get("softness", 0.15))
            for other in neighs:
                if (other is self) or (other.pos is None) or (not other.alive):
                    continue
                rij = pos_self - np.asarray(other.pos, dtype=float)
                d = float(np.linalg.norm(rij))
                r_sum = r_self + float(other.radius)
                if d < r_sum:
                    # Compute a small push proportional to overlap depth
                    if d < 1e-12:
                        # Random unit vector to avoid NaNs on coincident locations
                        phi = self.model.random.uniform(-np.pi, np.pi)
                        rij_unit = np.array([np.cos(phi), np.sin(phi)], dtype=float)
                    else:
                        rij_unit = rij / d
                    delta = (r_sum - d) * softness
                    disp += delta * rij_unit
            if np.any(disp):
                newp = pos_self + disp
                self.model.space.move_agent(self, (float(newp[0]), float(newp[1])))

    # -------------------------
    # Merge, proliferate, fragment (unchanged)
    # -------------------------
    def _try_merge(self):
        """
        Merge logic using ONE global probability parameter (no phenotypic adhesion, no product).
        """
        if self.pos is None:
            return
        merge_cfg = self.model.params["merge"]
        # Single parameter that IS the probability to merge upon contact
        p_merge = float(merge_cfg.get("p_merge", 0.9))
        p_merge = min(max(p_merge, 0.0), 1.0)  # clamp to [0,1]

        neighbors = self.model.get_neighbors(self, 2 * self.radius)
        pos_self = np.asarray(self.pos, dtype=float)

        contacts = []  # list of (d2, other)
        for other in neighbors:
            if other is self or not other.alive or other.pos is None:
                continue
            pos_other = np.asarray(other.pos, dtype=float)
            d2 = np.sum((pos_other - pos_self) ** 2)
            r_sum = self.radius + other.radius
            if d2 <= r_sum * r_sum:
                contacts.append((d2, other))
        if not contacts:
            return

        d2_min = min(d2 for d2, _ in contacts)
        tol = 1e-12
        tied = [other for d2, other in contacts if abs(d2 - d2_min) <= tol]
        target = self.model.random.choice(tied)

        if self.model.random.random() < p_merge:
            self._merge_with(target)
        return

    def _merge_with(self, other):
        p_self = np.array(self.pos, dtype=float)
        p_other = np.array(other.pos, dtype=float)
        m1 = mass_from_size(self.size)
        m2 = mass_from_size(other.size)

        size_new = self.size + other.size
        r_new = volume_conserving_radius(self.radius, other.radius)
        v_new = momentum_merge(m1, self.vel, m2, other.vel)
        pos_new = (m1 * p_self + m2 * p_other) / (m1 + m2)

        other.alive = False
        self.model.remove_agent(other)

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
        """
        With rate 'fragment_rate', reduce size by 1 and spawn a new child cluster.
        Place the fragment child at a minimum separation from the parent
        (>= factor * (r_parent + r_child)), along a random direction,
        to avoid seeding near-zero NND pairs.
        """
        ph = self.model.params["phenotypes"][self.phenotype]
        lam = ph["fragment_rate"] * self.model.dt
        if self.size > 1 and self.model.random.random() < lam:
            self.size -= 1
            self.radius = radius_from_size_3d(self.size)
            if self.pos is None:
                return

            p_self = np.array(self.pos, dtype=float)
            params_physics = self.model.params.get("physics", {})
            r_child = float(radius_from_size_3d(1))
            factor = float(params_physics.get("fragment_minsep_factor", 1.1))
            min_sep = factor * (self.radius + r_child)

            theta = self.model.random.uniform(-np.pi, np.pi)
            offset = min_sep * np.array([np.cos(theta), np.sin(theta)], dtype=float)
            candidate_pos = p_self + offset
            # Respect torus if enabled; ContinuousSpace will handle wrapping
            child_pos = self.model.space.torus_adj(tuple(candidate_pos))

            child = self.model.spawn_cluster(
                size=1, phenotype=self.phenotype, pos=child_pos, jitter=False
            )
            self.event_log.append(("fragment", child.unique_id, self.model.time))