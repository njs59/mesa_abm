import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises, gamma


def simulate_correlated_walk(
        mu_turn: float,
        kappa_turn: float,
        n_particles: int = 1000,
        n_steps: int = 300,
        speed_model: dict = None,
        initial_heading: float | None = None,
        rng=None,
):
    """
    Simulate a 2D correlated random walk where the TURNING ANGLE is drawn from:

        dθ ~ VonMises(mu_turn, kappa_turn)

    and the step SPEED is either:

    (A) constant:
        speed_model = {"type": "constant", "speed": s}

    (B) gamma distributed:
        speed_model = {"type": "gamma", "shape": a, "scale": b}

    Parameters
    ----------
    mu_turn : float
        Mean turning angle (0 = continue straight)
    kappa_turn : float
        Turning concentration (0 = random, high = persistent)
    n_particles : int
    n_steps : int
    speed_model : dict
        Model for step length
    initial_heading : float or None
        If None: random heading in [-pi, pi]
    rng : np.random.Generator
        Optional, for reproducibility

    Returns
    -------
    distances : array (n_steps+1, n_particles)
    """

    if rng is None:
        rng = np.random.default_rng()

    # --- 1. Initialise heading ---
    if initial_heading is None:
        headings = rng.uniform(-np.pi, np.pi, size=n_particles)
    else:
        headings = np.full(n_particles, float(initial_heading))

    # --- 2. Initialise trajectories ---
    x = np.zeros((n_steps + 1, n_particles))
    y = np.zeros((n_steps + 1, n_particles))

    # Check speed model
    if speed_model is None:
        speed_model = {"type": "constant", "speed": 1.0}

    for t in range(n_steps):

        # --- Turning angle update ---
        dtheta = vonmises(kappa_turn, loc=mu_turn).rvs(size=n_particles, random_state=rng)
        headings = headings + dtheta

        # --- Speed update ---
        if speed_model["type"] == "constant":
            step_len = speed_model["speed"]
            speeds = np.full(n_particles, step_len)

        elif speed_model["type"] == "gamma":
            a = speed_model["shape"]
            b = speed_model["scale"]
            speeds = gamma(a, scale=b).rvs(size=n_particles, random_state=rng)

        else:
            raise ValueError("speed_model must be 'constant' or 'gamma'.")

        # --- Motion update ---
        x[t + 1] = x[t] + speeds * np.cos(headings)
        y[t + 1] = y[t] + speeds * np.sin(headings)

    distances = np.sqrt(x**2 + y**2)
    return distances


def mean_ci(distances, alpha=0.05):
    """Mean ± 95% CI."""
    mean = np.mean(distances, axis=1)
    sd = np.std(distances, axis=1, ddof=1)
    n = distances.shape[1]
    z = 1.96
    margin = z * sd / np.sqrt(n)
    return mean, mean - margin, mean + margin


def simulate_and_plot(motion_types,
                      n_particles=1000,
                      n_steps=300,
                      seed=123,
                      title="Correlated random walks: distance from origin"):
    """
    motion_types = [
        {
            "name": "...",
            "mu_turn": ...,
            "kappa_turn": ...,
            "speed_model": {"type":"constant","speed":...}
                          or
                           {"type":"gamma","shape":..., "scale":...}
        },
        ...
    ]
    """
    rng = np.random.default_rng(seed)

    plt.figure(figsize=(10, 6))
    colours = plt.cm.tab10(np.linspace(0, 1, len(motion_types)))

    t = np.arange(n_steps + 1)

    for i, spec in enumerate(motion_types):
        mu = spec["mu_turn"]
        kappa = spec["kappa_turn"]
        speed_model = spec["speed_model"]
        label = spec["name"]

        dists = simulate_correlated_walk(
            mu_turn=mu,
            kappa_turn=kappa,
            n_particles=n_particles,
            n_steps=n_steps,
            speed_model=speed_model,
            rng=rng,
        )

        mean, lo, hi = mean_ci(dists)

        c = colours[i]
        plt.plot(t, mean, lw=2, color=c, label=label)
        plt.fill_between(t, lo, hi, color=c, alpha=0.2)

    plt.xlabel("Time steps")
    plt.ylabel("Distance from origin")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    MOTION_TYPES = [
        {
            "name": "Persistent (constant speed)",
            "mu_turn": 0.0,
            "kappa_turn": 0.5,
            "speed_model": {"type": "constant", "speed": 1.0},
        },

        {
            "name": "Persistent (gamma speed)",
            "mu_turn": 0.0,
            "kappa_turn": 0.5,
            "speed_model": {"type": "gamma", "shape": 2.0, "scale": 0.5},
        },

        {
            "name": "Random walk (constant speed)",
            "mu_turn": 0.0,
            "kappa_turn": 0,
            "speed_model": {"type": "constant", "speed": 1.0},
        },

        {
            "name": "Random walk (gamma speed)",
            "mu_turn": 0.0,
            "kappa_turn": 0,
            "speed_model": {"type": "gamma", "shape": 2.0, "scale": 1.0},
        },

        {
            "name": "Persistent oscillatory (constant speed)",
            "mu_turn": np.pi,
            "kappa_turn": 0.5,
            "speed_model": {"type": "constant", "speed": 1.0},
        },

        {
            "name": "Persistent oscillatory (gamma speed)",
            "mu_turn": np.pi,
            "kappa_turn": 0.5,
            "speed_model": {"type": "gamma", "shape": 2.0, "scale": 0.5},
        },
    ]

    simulate_and_plot(MOTION_TYPES)