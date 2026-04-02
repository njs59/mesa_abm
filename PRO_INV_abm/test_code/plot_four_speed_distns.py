#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math   # for lgamma


# ----------------------------
# PDF helpers
# ----------------------------
def pdf_lognorm(x, s, scale):
    """Lognormal with parameters:
       s = sigma (log-space std)
       scale = exp(mu)
    """
    x = np.asarray(x, float)
    pdf = np.zeros_like(x)
    mask = x > 0
    xs = x[mask]

    sigma = float(s)
    mu = np.log(float(scale))

    denom = xs * sigma * np.sqrt(2*np.pi)
    z = (np.log(xs) - mu) / sigma
    pdf[mask] = np.exp(-0.5 * z * z) / denom
    return pdf


def pdf_gamma(x, a, scale):
    """Gamma(k=a, theta=scale) using math.lgamma."""
    x = np.asarray(x, float)
    pdf = np.zeros_like(x)
    mask = x > 0
    xs = x[mask]

    a = float(a)
    theta = float(scale)

    log_pdf = (
        (a - 1.0) * np.log(xs)
        - (xs / theta)
        - math.lgamma(a)
        - a * np.log(theta)
    )
    pdf[mask] = np.exp(log_pdf)
    return pdf


# ----------------------------------------
# Parameters
# ----------------------------------------
lognorm_prolif = {"s": 1.0725282304382302, "scale": 6.883996177701719}
gamma_prolif   = {"a": 1.872995576819194,  "scale": 1.421743119834589}

lognorm_inv    = {"s": 0.9702953903470796, "scale": 4.517258693059166}
gamma_inv      = {"a": 2.08195513401392,   "scale": 3.548763926965852}


# ---------------------------
# Compute means for vertical lines
# ---------------------------
mean_prolif_p1 = lognorm_prolif["scale"] * np.exp(lognorm_prolif["s"]**2 / 2)
mean_prolif_p2 = gamma_prolif["a"] * gamma_prolif["scale"]

mean_inv_p1 = lognorm_inv["scale"] * np.exp(lognorm_inv["s"]**2 / 2)
mean_inv_p2 = gamma_inv["a"] * gamma_inv["scale"]


# ---------------------------
# Speed range
# ---------------------------
x_max = 50
x = np.linspace(0, x_max, 2000)
x[0] = 1e-9


# ---------------------------
# PDFs
# ---------------------------
pdf_prolif_p1 = pdf_lognorm(x, **lognorm_prolif)
pdf_prolif_p2 = pdf_gamma(x,   **gamma_prolif)

pdf_inv_p1    = pdf_lognorm(x, **lognorm_inv)
pdf_inv_p2    = pdf_gamma(x,   **gamma_inv)


# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(10, 6))

# PDFs
plt.plot(x, pdf_prolif_p1, lw=2.2, color="#ba1dba",
         label="Proliferative — Phase 1 (lognormal)")
plt.plot(x, pdf_prolif_p2, lw=2.2, color="#ba1dba", ls="--",
         label="Proliferative — Phase 2 (gamma)")

plt.plot(x, pdf_inv_p1, lw=2.2, color="#469e2c",
         label="Invasive — Phase 1 (lognormal)")
plt.plot(x, pdf_inv_p2, lw=2.2, color="#469e2c", ls="--",
         label="Invasive — Phase 2 (gamma)")


# ---------------------------
# Vertical mean lines
# ---------------------------
plt.axvline(mean_prolif_p1, color="#ba1dba",  lw=2,
            label=f"Mean Prolif P1 = {mean_prolif_p1:.2f}")
plt.axvline(mean_prolif_p2, color="#ba1dba", ls="--", lw=2,
            label=f"Mean Prolif P2 = {mean_prolif_p2:.2f}")

plt.axvline(mean_inv_p1, color="#469e2c",  lw=2,
            label=f"Mean Invasive P1 = {mean_inv_p1:.2f}")
plt.axvline(mean_inv_p2, color="#469e2c", ls="--", lw=2,
            label=f"Mean Invasive P2 = {mean_inv_p2:.2f}")


# ---------------------------
# Finishing touches
# ---------------------------
plt.title("Speed distributions (with mean values) for two phenotypes and two phases")
plt.xlabel("Speed")
plt.ylabel("Probability density (PDF)")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()