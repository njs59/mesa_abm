# global_overlap_vs_a.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist
from scipy.stats import vonmises
from scipy.integrate import quad

# --- p_e(delta; a) machinery (exact geometry) ---
def phi_from_s_delta_R(s, delta, R):
    d = R - delta
    denom = 2.0 * d * s
    if denom == 0:
        return 0.0
    U = (2.0 * d * delta + delta**2 - s**2) / denom
    U = np.clip(U, -1.0, 1.0)
    return float(np.arccos(U))

def prob_window_vonmises(phi, kappa):
    if kappa == 0:
        return phi / np.pi
    phi = max(0.0, min(np.pi, phi))
    cdf_phi = vonmises.cdf(phi, kappa, loc=0)
    cdf_negphi = vonmises.cdf(-phi, kappa, loc=0)
    return float(cdf_phi - cdf_negphi)

def prob_eligible_given_s_delta(s, delta, R, kappa):
    if s < delta:
        return 1.0
    phi = phi_from_s_delta_R(s, delta, R)
    return 1.0 - prob_window_vonmises(phi, kappa)

def pe_delta_quadrature(delta, a, theta, dt, kappa, R, rtol=1e-6):
    scale = dt * theta
    def f_S(s):
        if s <= 0: return 0.0
        return gamma_dist.pdf(s, a, scale=scale)
    def integrand(s):
        return prob_eligible_given_s_delta(s, delta, R, kappa) * f_S(s)
    val, _ = quad(integrand, 0.0, np.inf, epsabs=0, epsrel=rtol, limit=200)
    return float(val)

# --- Average over f_Delta to get e(a) ---
def e_of_a_from_samples(a, theta, dt, kappa, R, delta_samples, method='quad'):
    vals = []
    for delta in delta_samples:
        vals.append(pe_delta_quadrature(delta, a, theta, dt, kappa, R))
    return float(np.mean(vals))

def e_of_a_from_parametric(a, theta, dt, kappa, R,
                           fdelta_shape=2.0, fdelta_scale=1.0,  # Gamma(shape, scale)
                           n_mc=20000, rng=None):
    if rng is None: rng = np.random.default_rng(123)
    deltas = rng.gamma(shape=fdelta_shape, scale=fdelta_scale, size=n_mc)
    return e_of_a_from_samples(a, theta, dt, kappa, R, deltas)

# --- Renewal metrics ---
def T_ep(a, theta, dt, kappa, R, p_merge, e_a):
    q = 1.0 - p_merge
    tau_sub = 0.5 * dt
    return tau_sub * (1.0 + q / (1.0 - e_a * q))

def D_eff(a, theta, D_theta):
    return (a * theta)**2 / (2.0 * D_theta)

def T_between(a, theta, dt, D_theta, rho, R, L,
              alpha_bg, alpha_loc, c, delta_sep, eta_eff):
    De = D_eff(a, theta, D_theta)
    k_bg  = alpha_bg  * De * rho * np.log(L / R)
    r_star = R + c * dt * a * theta
    r0     = R + delta_sep
    k_loc = alpha_loc * De * np.log(r_star / r0)
    lam = eta_eff * (k_bg + k_loc)
    return 1.0 / lam

def overlap_fraction_Phi(a, theta, dt, kappa, R, p_merge,
                         D_theta, rho, L, alpha_bg, alpha_loc, c, delta_sep, eta_eff,
                         e_a):
    Tep = T_ep(a, theta, dt, kappa, R, p_merge, e_a)
    Tbt = T_between(a, theta, dt, D_theta, rho, R, L, alpha_bg, alpha_loc, c, delta_sep, eta_eff)
    return Tep / (Tep + Tbt), Tep, Tbt

# --- Demo: plot Phi(a) vs a ---
def demo_plot_phi_vs_a(output_png='phi_vs_a.png',
                       use_empirical=False, delta_samples=None):
    # --- Choose your ABM / kinetic params here ---
    R = 10.0
    dt = 1.0
    theta = 1.0
    kappa = 1.0
    p_merge = 0.2
    D_theta = 1.0
    rho = 0.01
    L = 1.0 / np.sqrt(np.pi * rho)  # typical spacing scale
    alpha_bg = 4.0 * np.pi  # these two alphas absorb constants; tune from logs if available
    alpha_loc = 4.0 * np.pi
    c = 1.0
    delta_sep = 0.2
    eta_eff = 1.0

    a_values = np.linspace(0.5, 8.0, 24)
    Phi_vals, Tep_vals, Tbt_vals = [], [], []

    for a in a_values:
        if use_empirical and (delta_samples is not None):
            e_a = e_of_a_from_samples(a, theta, dt, kappa, R, delta_samples)
        else:
            e_a = e_of_a_from_parametric(a, theta, dt, kappa, R,
                                         fdelta_shape=2.0, fdelta_scale=1.0)
        Phi, Tep, Tbt = overlap_fraction_Phi(
            a, theta, dt, kappa, R, p_merge,
            D_theta, rho, L, alpha_bg, alpha_loc, c, delta_sep, eta_eff,
            e_a
        )
        Phi_vals.append(Phi); Tep_vals.append(Tep); Tbt_vals.append(Tbt)

    # Plot Phi(a) and optionally Teps/Tbt
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.2))
    ax.plot(a_values, Phi_vals, '-o', label=r'$\Phi(a)$: fraction of time overlapped')
    ax.set_xlabel(r'Shape $a$ (controls mean speed via $m=a\theta$)')
    ax.set_ylabel(r'Overlap fraction $\Phi$')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    return output_png

if __name__ == '__main__':
    out = demo_plot_phi_vs_a()
    print(f"Saved plot to {out}")