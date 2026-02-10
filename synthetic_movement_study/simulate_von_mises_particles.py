import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises

def simulate_correlated_walk(
        mu_turn: float,
        kappa_turn: float,
        n_particles: int = 1000,
        n_steps: int = 300,
        step_len: float = 1.0,
        initial_heading: float | None = None,
        rng=None,
):
    """
    Simulate a correlated random walk where each step's *turning angle*
    is drawn from a von Mises distribution:
        Δθ ~ vonMises(mu_turn, kappa_turn)

    That means:
        new_heading = old_heading + Δθ

    And each particle then moves one step of length step_len.

    Returns:
        distances: shape (n_steps+1, n_particles)
    """

    if rng is None:
        rng = np.random.default_rng()

    # --- 1. Initialise heading of every particle
    if initial_heading is None:
        # random initial direction in [-π,π]
        headings = rng.uniform(-np.pi, np.pi, size=n_particles)
    else:
        headings = np.full(n_particles, float(initial_heading))

    # Track positions
    x = np.zeros((n_steps+1, n_particles))
    y = np.zeros((n_steps+1, n_particles))

    # --- 2. Simulate Turning-Angle Process ---
    for t in range(n_steps):
        dtheta = vonmises(kappa_turn, loc=mu_turn).rvs(size=n_particles, random_state=rng)

        # Update headings
        headings = headings + dtheta

        # Move forward one step
        x[t+1] = x[t] + step_len * np.cos(headings)
        y[t+1] = y[t] + step_len * np.sin(headings)

    distances = np.sqrt(x**2 + y**2)
    return distances


def mean_ci(distances, alpha=0.05):
    """Compute mean ± 95% CI for the mean over particles."""
    mean = np.mean(distances, axis=1)
    sd = np.std(distances, axis=1, ddof=1)
    n = distances.shape[1]

    z = 1.96
    margin = z * sd / np.sqrt(n)
    return mean, mean - margin, mean + margin


def simulate_and_plot(motion_types,
                      n_particles=1000,
                      n_steps=300,
                      step_len=1.0,
                      seed=123,
                      title="Correlated random walks: distance from origin"):
    """
    motion_types = list of dicts:
        {"name":..., "mu_turn":..., "kappa_turn":...}
    """
    rng = np.random.default_rng(seed)

    plt.figure(figsize=(10,6))
    # colours = plt.cm.tab10(np.linspace(0,1,len(motion_types)))
    colours = plt.cm.tab10(np.linspace(0,1,10))

    t = np.arange(n_steps+1)

    for i, spec in enumerate(motion_types):
        mu = spec["mu_turn"]
        kappa = spec["kappa_turn"]
        label = spec.get("name", f"mu={mu:.2f}, kappa={kappa:.2f}")

        dists = simulate_correlated_walk(
            mu_turn=mu,
            kappa_turn=kappa,
            n_particles=n_particles,
            n_steps=n_steps,
            step_len=step_len,
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
    # MOTION_TYPES = [
    #     {"name": "Straight (kappa large)", "mu_turn": 0.0, "kappa_turn": 50},
    #     {"name": "Wiggly",                 "mu_turn": 0.0, "kappa_turn": 2},
    #     {"name": "Random",                 "mu_turn": 0.0, "kappa_turn": 0},
    #     {"name": "Left-biased",            "mu_turn": +np.pi/8, "kappa_turn": 5},
    #     {"name": "Right-biased",           "mu_turn": -np.pi/8, "kappa_turn": 5},
    # ]

    MOTION_TYPES = [
        {"name": "Persistent 0.5", "mu_turn": 0, "kappa_turn": 0.5},
        {"name": "Persistent 0.2",             "mu_turn": 0.0, "kappa_turn": 0.2},
        {"name": "Random",                 "mu_turn": 0.0, "kappa_turn": 0},
        {"name": "Persistent oscillatory 0.2", "mu_turn": +np.pi, "kappa_turn": 0.2},
        {"name": "Persistent oscillatory 0.5", "mu_turn": +np.pi, "kappa_turn": 0.5},
    ]

    simulate_and_plot(MOTION_TYPES)