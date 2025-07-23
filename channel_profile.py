import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# ==============================================================================
# PALM's measurement
#
data = np.loadtxt("./data/rodi1997_channel_data.csv", skiprows=0, delimiter=',')
y = data[:,1]
u = data[:,0]

# Interpolate the original data
u_interp = interp1d(y, u, kind="cubic", fill_value="extrapolate")

# Create a fine grid
y_fine = np.linspace(0, 2, 500)
# Enforce symmetry: average u(y) and u(2 - y)
u_sym = 0.5 * (u_interp(y_fine) + u_interp(2 - y_fine))
# Final symmetric interpolant
sym_interp = interp1d(y_fine, u_sym, kind="cubic")

# ==============================================================================
# MKM data
#
data = np.loadtxt("./data/MKM_chan590.mean", skiprows=25)
mkm_y = data[:,0]
mkm_u = data[:,2]

# ==============================================================================
# Plotting
#
plt.figure(1); plt.clf(); plt.show(block=False)
plt.plot(y, u, 'o', label="Rodi's data", alpha=0.6)
plt.plot(y_fine, u_interp(y_fine), '--', label="Rodi's interpolation", alpha=0.5)
plt.plot(y_fine, u_sym, label="Rodi's symmetrized", linewidth=2)

plt.plot(mkm_y, mkm_u, '+', label="MKM's data")

plt.xlabel(r"$y$")
plt.ylabel(r"$u(y)$")
plt.legend()
plt.grid(True)
plt.tight_layout()
