import numpy as np
import matplotlib.pyplot as plt


PLANCK = 6.626e-34     
BOLTZ = 1.38e-23        
MASS = 9.31e-31          


def power_density(nu, T):
    x = PLANCK * nu / (BOLTZ * T)
    velocity_term = np.sqrt((2 * BOLTZ * T) / MASS)
    return np.exp(-x) * (0.6 * x**2 + 2.2 * x + 2.2) / velocity_term


nu_values = np.logspace(-2, 25, 1000)


temperatures = [300, 1e6]
labels = [r"$T = 300\,\mathrm{K}$", r"$T = 10^6\,\mathrm{K}$"]

plt.figure(figsize=(8, 5))
for T, lbl in zip(temperatures, labels):
    plt.plot(nu_values, power_density(nu_values, T), label=lbl)


plt.xscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.title("Emitted Power Density per Unit Frequency")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.legend()
plt.tight_layout()
plt.show()
