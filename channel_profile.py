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

# check bulk velocity and scale
u_bulk = np.trapezoid(u_sym, y_fine) / 2
u_sym /= u_bulk  # scale to bulk velocity

# ==============================================================================
# Moser, Kim & Mansour data: Re_tau = 590
#
data = np.loadtxt("./data/MKM_chan590.mean", skiprows=25)
MKM_y = data[:,0]
MKM_u = data[:,2]
MKM_y = np.concatenate((MKM_y, 2 - MKM_y[::-1]), axis=0)
MKM_u = np.concatenate((MKM_u,     MKM_u[::-1]), axis=0)

# check bulk velocity and scale
u_bulk = np.trapezoid(MKM_u, MKM_y) / MKM_y[-1]
MKM_u /= u_bulk  # scale to bulk velocity

# ==============================================================================
# Lee & Moser data: Re_tau = 5200
#
data = np.loadtxt("./data/LeeMoser_chan5200.mean", skiprows=72)
LM_y = data[:,0]
LM_u = data[:,2] * 4.14872e-02 # <U> * u_tau (that's how it's normalized in the file)

# check bulk velocity (already scaled)
u_bulk = np.trapezoid(LM_u, LM_y) # = 0.9988998265065104

# ==============================================================================
# Plotting
#
plt.figure(1); plt.clf(); plt.show(block=False)
plt.plot(y, u, 'o', label="Rodi's data", alpha=0.6)
plt.plot(y_fine, u_interp(y_fine), '--', label="Rodi's interpolation", alpha=0.5)
plt.plot(y_fine, u_sym, label=r"Rodi's symmetrized and scaled to $U_b=1$, $Re_b = 40000$", linewidth=2)

plt.plot(MKM_y, MKM_u, '-+b', label="Moser, Kim and Mansour's data, $Re_b = 10779$")
plt.plot(LM_y, LM_u,   '-r', label="Lee and Moser's data, $Re_b = 128127$")

plt.xlabel(r"$y$")
plt.ylabel(r"$u(y)$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.draw()
