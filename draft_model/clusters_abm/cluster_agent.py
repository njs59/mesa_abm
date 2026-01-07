
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
        """
        Update agent position by moving with either:
        - constant step magnitude = speed_base * dt  (mode="constant"),
        - or sampled step magnitude from a configured distribution (mode="distribution").

        Direction handling:
        - "isotropic": draw a new random angle each step (default; your current behaviour),
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
            # Your current behaviour: fixed magnitude
            step_mag = speed_base * dt

        elif mode == "distribution":
            # Draw step magnitude in *model distance units per step* directly.
            # This avoids mixing dt twice. Interpret dist_params as per distribution docs.
            # If you prefer magnitude-per-unit-time, multiply by dt here.

            dist_name = str(mv.get("distribution", "lognorm")).lower()
            dp = mv.get("dist_params", {})  # dict of parameters; see notes above

            # Draw from supported families using stdlib RNG (reproducible with model.seed)
            # NOTE: We clamp negative draws to a tiny positive epsilon.
            eps = 1e-12

            if dist_name == "lognorm":
                # SciPy-like: lognorm(s, loc=0, scale) ⇒ we emulate by normal draw on log-space
                # Let ln X ~ N(mu, sigma^2). If only s and scale are given, we set:
                s = float(dp.get("s", 0.6))      # sigma
                scale = float(dp.get("scale", 2.0))  # exp(mu)
                # Draw Z ~ N(0, s^2), then X = scale * exp(Z)
                z = self.model.random.normalvariate(0.0, s)
                step_mag = max(scale * np.exp(z), eps)

            elif dist_name == "gamma":
                # Gamma(a, scale): shape a > 0, scale > 0
                a = float(dp.get("a", 2.0))
                sc = float(dp.get("scale", 1.0))
                # Use numpy's gamma via deterministic seed path if available,
                # else approximate with stdlib: we fallback to numpy here safely.
                step_mag = max(np.random.default_rng().gamma(shape=a, scale=sc), eps)

            elif dist_name == "weibull":
                # Weibull_min(c, scale): c > 0
                c = float(dp.get("c", 1.5))
                sc = float(dp.get("scale", 2.0))
                # Draw U ~ Uniform(0,1); X = sc * (-ln(1-U))^(1/c)
                U = self.model.random.random()
                step_mag = max(sc * (-np.log(max(1.0 - U, eps))) ** (1.0 / c), eps)

            elif dist_name == "rayleigh":
                # Rayleigh(scale): scale > 0
                sc = float(dp.get("scale", 2.0))
                # If Z ~ N(0, sc^2) in each component and speed = sqrt(Zx^2 + Zy^2), Rayleigh applies
                # Draw via inverse CDF: X = sc * sqrt(-2 ln(1-U))
                U = self.model.random.random()
                step_mag = max(sc * np.sqrt(-2.0 * np.log(max(1.0 - U, eps))), eps)

            elif dist_name == "expon":
                # Exponential(scale): scale > 0
                sc = float(dp.get("scale", 1.0))
                U = self.model.random.random()
                step_mag = max(-sc * np.log(max(1.0 - U, eps)), eps)

            elif dist_name == "invgauss":
                # Inverse-Gaussian (Wald): parameters mu (>0), scale (>0)
                mu = float(dp.get("mu", 1.0))
                sc = float(dp.get("scale", 1.0))
                # Draw via numpy for numerical stability
                step_mag = max(np.random.default_rng().wald(mean=mu, scale=sc), eps)

            else:
                # Fallback: constant
                step_mag = speed_base * dt

        else:
            # Unknown mode -> fallback to constant
            step_mag = speed_base * dt

        # --- 2) Choose direction ---
        pos_curr = np.asarray(self.pos, dtype=float)

        if direction_mode == "isotropic":
            # Draw a fresh heading uniformly
            theta = self.model.random.uniform(-np.pi, np.pi)

        elif direction_mode == "persistent":
            # Maintain previous heading with small Gaussian jitter
            # Keep track of self._theta (create if missing)
            theta_prev = getattr(self, "_theta", None)
            if theta_prev is None:
                theta_prev = self.model.random.uniform(-np.pi, np.pi)
            sigma = float(mv.get("heading_sigma", 0.25))
            theta = float(theta_prev + self.model.random.normalvariate(0.0, sigma))
            # Store back for next step
            self._theta = theta

        else:
            # Fallback to isotropic
            theta = self.model.random.uniform(-np.pi, np.pi)

        # Direction unit vector and velocity
        dir_vec = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        # Your self.vel remains a 'per-step' vector consistent with previous usage
        # If you prefer 'per-unit-time' velocity, divide by dt accordingly.
        self.vel = (step_mag / max(dt, 1e-12)) * dir_vec  # velocity magnitude = step_mag / dt

        # Advance position
        new_pos = pos_curr + dir_vec * step_mag
        x, y = float(new_pos[0]), float(new_pos[1])

        # Apply move (ContinuousSpace handles wrapping if torus=True)
        self.model.space.move_agent(self, (x, y))

        
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
