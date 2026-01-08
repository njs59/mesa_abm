
# plot_lognorm_posterior.py
# Plots a lognormal speed distribution with given meanlog (mu) and sdlog (sigma),
# and overlays a Monte-Carlo sample histogram for intuition.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from pathlib import Path

# Parameters from your posterior medians
mu = 1.36051     # mean of underlying normal
sigma = 1.66391  # std.dev of underlying normal

# SciPy's lognorm parameterisation:
# shape (s) = sigma, loc = 0, scale = exp(mu)
s = sigma
scale = np.exp(mu)

# X-axis range: cover a wide range of speeds (positive support)
# Use quantiles to pick a visually meaningful range
dist = lognorm(s=s, loc=0.0, scale=scale)
x_min = dist.ppf(1e-4)  # very low quantile
x_max = dist.ppf(1 - 0.1)  # very high quantile
x = np.linspace(x_min, x_max, 800)

# Theoretical PDF
pdf = dist.pdf(x)

# Draw a Monte-Carlo sample for a histogram overlay (optional)
rng = np.random.default_rng(42)
sample = dist.rvs(size=50000, random_state=rng)

# Build figure
fig, ax = plt.subplots(figsize=(8, 5.2))

# Histogram of samples (density=True so it’s comparable to the PDF)
ax.hist(sample, bins=120, range = (x_min,x_max), density=True, alpha=0.28, color="#4e79a7", edgecolor="white", label="Sample histogram")
ax.set_xlim(x_min, x_max)

# Plot the theoretical PDF
ax.plot(x, pdf, color="#d62728", lw=2.4, label=f"Lognormal PDF (μ={mu:.3f}, σ={sigma:.3f})")

# Add a few reference lines: mean, median, mode
mean = dist.mean()       # = exp(mu + 0.5*sigma^2)
median = dist.median()   # = exp(mu)
mode = scale * np.exp(-sigma**2)  # for loc=0
for val, name, style in [(mean, "Mean", ":"), (median, "Median", "--"), (mode, "Mode", "-.")]:
    ax.axvline(val, color="#2ca02c", ls=style, lw=1.6, alpha=0.8)
    ax.text(val, ax.get_ylim()[1]*0.95, name, rotation=90, va="top", ha="right", color="#2ca02c")

ax.set_title("Posterior lognormal speed distribution (PDF + sample)")
ax.set_xlabel("Speed (units)")
ax.set_ylabel("Density")
ax.legend()
fig.tight_layout()

# Save next to your results folder
out_dir = Path("clusters_abm/results")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "lognorm_posterior_speed.png"
fig.savefig(out_path, dpi=220, bbox_inches="tight")
print(f"Saved plot → {out_path.resolve()}")
