import numpy as np
from scipy.stats import vonmises
import matplotlib.pyplot as plt

# --------------------------
# Parameters
# --------------------------
kappa = 4.0             # concentration of the movement distribution
N = 50000               # number of samples
normal_angle = np.pi/3  # the "normal" direction between overlapping agents (60 degrees)

# --------------------------
# Generate movement angles
# --------------------------
# Movement angle sampled in global coordinates
theta_global = vonmises.rvs(kappa, loc=0, size=N)

# Convert to coordinates relative to the normal
theta_relative = theta_global - normal_angle

# Wrap angles to [-pi, pi]
theta_relative = np.angle(np.exp(1j * theta_relative))

# --------------------------
# Plot results
# --------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(theta_global, bins=80, density=True, alpha=0.7, color='steelblue')
plt.title("Movement in GLOBAL frame\nθ ~ vM(0, κ)")

plt.subplot(1,2,2)
plt.hist(theta_relative, bins=80, density=True, alpha=0.7, color='darkorange')
plt.title("Movement RELATIVE to normal\nθ_relative = θ_global − φ")

plt.tight_layout()
plt.show()