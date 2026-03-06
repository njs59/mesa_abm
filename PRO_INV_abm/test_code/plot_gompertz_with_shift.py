#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Gompertz PDF used in your ABM (with time shift)
# ----------------------------------------------------------
def gompertz_shifted_pdf(t, shift, b, c):
    """
    pdf(t) = b*c * exp(c*(t-shift)) * exp(-b*(exp(c*(t-shift)) - 1)),  t >= shift
    pdf(t) = 0 otherwise
    """
    t = np.asarray(t, dtype=float)
    u = t - shift
    pdf = np.zeros_like(t)
    mask = u >= 0
    u_valid = u[mask]
    pdf[mask] = b * c * np.exp(c * u_valid) * np.exp(-b * (np.exp(c * u_valid) - 1))
    return pdf

# ----------------------------------------------------------
# CDF via simple rectangular rule (ABM-like: cumsum(pdf)*dt)
# ----------------------------------------------------------
def gompertz_shifted_cdf(t, shift, b, c):
    t = np.asarray(t, dtype=float)
    pdf = gompertz_shifted_pdf(t, shift, b, c)
    # assume uniform grid
    dt = t[1] - t[0]
    cdf = np.cumsum(pdf) * dt
    # normalise to 1 (same as in ABM)
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    return cdf

# ----------------------------------------------------------
# Example parameter sets
# ----------------------------------------------------------
params_list = [
    {"shift": 5,  "b": 0.02, "c": 0.03, "label": "Set 1"},
    {"shift": 10, "b": 0.02, "c": 0.03, "label": "Set 2 (larger shift)"},
    {"shift": 5,  "b": 0.05, "c": 0.03, "label": "Set 3 (larger b)"},
    {"shift": 5,  "b": 0.02, "c": 0.08, "label": "Set 4 (larger c)"},
    {"shift": 13.182589764562103, "b": 0.027843583207496258,"c": 0.03085478819397927, "label": "Fitted params"},
    {"shift": 13.182589764562103, "c": 0.027843583207496258,"b": 0.03085478819397927, "label": "Fitted params 2"},
]

# Time grid
t = np.linspace(0, 200, 4000)

# ----------------------------------------------------------
# Plot PDFs
# ----------------------------------------------------------
plt.figure(figsize=(10, 5))
for p in params_list:
    pdf = gompertz_shifted_pdf(t, p["shift"], p["b"], p["c"])
    plt.plot(t, pdf, label=p["label"])
plt.title("Gompertz (with time shift) — PDF")
plt.xlabel("Time")
plt.ylabel("PDF")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# ----------------------------------------------------------
# Plot CDFs
# ----------------------------------------------------------
plt.figure(figsize=(10, 5))
for p in params_list:
    cdf = gompertz_shifted_cdf(t, p["shift"], p["b"], p["c"])
    plt.plot(t, cdf, label=p["label"])
plt.title("Gompertz (with time shift) — CDF (ABM-style integration)")
plt.xlabel("Time")
plt.ylabel("CDF")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()