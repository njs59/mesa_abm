#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
try:
    from scipy.integrate import cumulative_trapezoid
    USE_SCIPY = True
except Exception:
    USE_SCIPY = False

# ==========================================================
# 1) ABM distribution: Gompertz PDF with a hard time shift
#    pdf_ABM(t) = b*c * exp(c*(t-shift)) * exp(-b*(exp(c*(t-shift))-1)), t >= shift
#    CDF is obtained numerically (to mirror your ABM lookup).
# ==========================================================
def gompertz_shifted_pdf_abm(t, shift, b, c):
    t = np.asarray(t, dtype=float)
    u = t - shift
    pdf = np.zeros_like(t)
    mask = u >= 0
    u_valid = u[mask]
    pdf[mask] = b * c * np.exp(c * u_valid) * np.exp(-b * (np.exp(c * u_valid) - 1))
    return pdf

def gompertz_shifted_cdf_abm(t, shift, b, c):
    t = np.asarray(t, dtype=float)
    pdf = gompertz_shifted_pdf_abm(t, shift, b, c)
    if USE_SCIPY:
        cdf = cumulative_trapezoid(pdf, t, initial=0.0)
    else:
        dt = t[1] - t[0]
        cdf = np.cumsum(pdf) * dt
    # normalise to 1 (as in your ABM)
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    return cdf

# ==========================================================
# 2) Classical shifted Gompertz (your fitter):
#    CDF_classic(t) = L * (1 - exp(-(b/a)*(exp(a*(t - t0)) - 1)))
#    We'll set L=1 for comparability (a plateau at 1).
#    Its PDF is the derivative of the CDF (we compute numerically).
# ==========================================================
def gompertz_shifted_cdf_classic(t, L, t0, b, a):
    t = np.asarray(t, dtype=float)
    a = max(float(a), 1e-12)
    b = max(float(b), 1e-12)
    x = np.maximum(t - t0, 0.0)
    term = (b / a) * (np.exp(a * x) - 1.0)
    F = L * (1.0 - np.exp(-term))
    # Ensure support starts at t0 (optional clipping)
    F = np.where(t < t0, 0.0, F)
    # Clip to [0, L]
    return np.clip(F, 0.0, L)

def numerical_pdf_from_cdf(t, F):
    """Centre-difference derivative to visualise the classical PDF fairly."""
    t = np.asarray(t, dtype=float)
    F = np.asarray(F, dtype=float)
    pdf = np.zeros_like(F)
    dt = t[1] - t[0]
    # central differences for interior, forward/backward at ends
    pdf[1:-1] = (F[2:] - F[:-2]) / (2.0 * dt)
    pdf[0] = (F[1] - F[0]) / dt
    pdf[-1] = (F[-1] - F[-2]) / dt
    pdf = np.maximum(pdf, 0.0)  # small numerical negatives to zero
    # normalise just in case discretisation error accumulates
    area = np.trapz(pdf, t)
    if area > 0:
        pdf /= area
    return pdf

# ==========================================================
# 3) Parameter sets to compare
#    Mapping used for comparison: shift ↔ t0, c ↔ a, b ↔ b, and L=1
#    Feel free to add/edit rows here.
# ==========================================================
param_sets = [
    {"shift": 13.182589764562103, "b": 0.027843583207496258,"c": 0.03085478819397927, "label": "Fitted params"},
    {"shift": 13.182589764562103, "c": 0.027843583207496258,"b": 0.03085478819397927, "label": "Fitted params 2"},
]

# ==========================================================
# 4) Time grid
# ==========================================================
t = np.linspace(0.0, 200.0, 4000)

# ==========================================================
# 5) Plot PDFs: ABM pdf vs classic pdf (derived from classic CDF)
# ==========================================================
plt.figure(figsize=(10, 5))
for ps in param_sets:
    label = ps["label"]

    # ABM pdf
    pdf_abm = gompertz_shifted_pdf_abm(t, ps["shift"], ps["b"], ps["c"])

    # Classic CDF (with mapped parameters), then numerical pdf for fair comparison
    F_class = gompertz_shifted_cdf_classic(
        t,
        L=1.0,
        t0=ps["shift"],   # map shift -> t0
        b=ps["b"],        # map b -> b
        a=ps["c"],        # map c -> a
    )
    pdf_class = numerical_pdf_from_cdf(t, F_class)

    plt.plot(t, pdf_abm,    lw=2.0, label=f"{label} — ABM pdf")
    plt.plot(t, pdf_class,  lw=1.8, ls="--", label=f"{label} — classic pdf")

plt.title("PDF comparison: ABM Gompertz (shifted) vs Classical shifted Gompertz")
plt.xlabel("Time")
plt.ylabel("PDF")
plt.grid(True, alpha=0.3)
plt.legend(ncols=2, fontsize=8)
plt.tight_layout()

# ==========================================================
# 6) Plot CDFs: ABM numeric CDF vs classic analytic CDF
# ==========================================================
plt.figure(figsize=(10, 5))
for ps in param_sets:
    label = ps["label"]

    # ABM numeric CDF (normalised)
    F_abm = gompertz_shifted_cdf_abm(t, ps["shift"], ps["b"], ps["c"])

    # Classic analytic CDF (with mapped parameters)
    F_class = gompertz_shifted_cdf_classic(
        t,
        L=1.0,
        t0=ps["shift"],   # map shift -> t0
        b=ps["b"],        # map b -> b
        a=ps["c"],        # map c -> a
    )

    plt.plot(t, F_abm,   lw=2.0, label=f"{label} — ABM CDF")
    plt.plot(t, F_class, lw=1.8, ls="--", label=f"{label} — classic CDF")

plt.title("CDF comparison: ABM Gompertz (shifted) vs Classical shifted Gompertz")
plt.xlabel("Time")
plt.ylabel("CDF")
plt.ylim(-0.02, 1.02)
plt.grid(True, alpha=0.3)
plt.legend(ncols=2, fontsize=8)
plt.tight_layout()

plt.show()