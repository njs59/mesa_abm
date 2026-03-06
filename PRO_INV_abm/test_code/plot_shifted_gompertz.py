import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------
# Shifted Gompertz distribution (same formula your ABM uses)
# -----------------------------------------------------------
def shifted_gompertz_pdf(t, shift, b, c, p_max=1.0):
    """
    PDF used in your ABM:
        pdf(u) = p_max * b * c * exp(c*u) * exp(-b*(exp(c*u)-1))
    where u = t - shift, and pdf = 0 for t < shift.
    """
    t = np.asarray(t)
    u = t - shift

    pdf = np.zeros_like(t)
    mask = u >= 0
    u_valid = u[mask]

    pdf[mask] = (
        p_max * b * c *
        np.exp(c * u_valid) *
        np.exp(-b * (np.exp(c * u_valid) - 1))
    )

    return pdf


def shifted_gompertz_cdf(t, shift, b, c, p_max=1.0):
    """
    Numerical CDF of the shifted Gompertz PDF.
    """
    pdf = shifted_gompertz_pdf(t, shift, b, c, p_max)
    dt = t[1] - t[0]
    cdf = np.cumsum(pdf) * dt

    # Normalise to 1
    if cdf[-1] > 0:
        cdf /= cdf[-1]

    return cdf


# -----------------------------------------------------------
# Parameter sets to plot
# (You can edit or add as many as you want)
# -----------------------------------------------------------
params_list = [
    {"shift": 5,  "b": 0.02, "c": 0.03, "label": "Set 1 (baseline)"},
    {"shift": 10, "b": 0.02, "c": 0.03, "label": "Set 2 (larger shift)"},
    {"shift": 5,  "b": 0.05, "c": 0.03, "label": "Set 3 (larger b)"},
    {"shift": 5,  "b": 0.02, "c": 0.08, "label": "Set 4 (larger c)"},
    {"shift": 13.182589764562103, "b": 0.027843583207496258,"c": 0.03085478819397927, "label": "Fitted params"},
    {"shift": 13.182589764562103, "c": 0.027843583207496258,"b": 0.03085478819397927, "label": "Fitted params 2"},
]

# Time grid
t = np.linspace(0, 200, 4000)


# -----------------------------------------------------------
# Plot PDFs
# -----------------------------------------------------------
plt.figure(figsize=(10, 5))
for p in params_list:
    pdf = shifted_gompertz_pdf(
        t,
        shift=p["shift"],
        b=p["b"],
        c=p["c"],
        p_max=p.get("p_max", 1.0),
    )
    plt.plot(t, pdf, label=p["label"])

plt.title("Shifted Gompertz PDF for different parameter sets")
plt.xlabel("Time")
plt.ylabel("PDF")
plt.legend()
plt.grid(True)
plt.tight_layout()


# -----------------------------------------------------------
# Plot CDFs
# -----------------------------------------------------------
plt.figure(figsize=(10, 5))
for p in params_list:
    cdf = shifted_gompertz_cdf(
        t,
        shift=p["shift"],
        b=p["b"],
        c=p["c"],
        p_max=p.get("p_max", 1.0),
    )
    plt.plot(t, cdf, label=p["label"])

plt.title("Shifted Gompertz CDF for different parameter sets")
plt.xlabel("Time")
plt.ylabel("CDF")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()