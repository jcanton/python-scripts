import numpy as np
import matplotlib.pyplot as plt



# constants
kappa    = 0.4 # Von Karman constant
H        = 100.0 # building height
lambda_f = 0.16 # frontal area index

# lambda_f - dependent
u_H    = 0.5       # free-stream velocity at building height (table 1)
a      = 1.32      # attenuation coefficient (table 1)
z_w    = H * 1.5   # wake height (table 2)
u_star = u_H / 4.4 # friction velocity (tables 1 and 2)
d      = 0.32  * H # displacement height (table 2)
z0     = 0.084 * H # roughness height (table 2)
l_c    = 0.14  * H # mixing length scale in the canopy (table 2)

# computed
A = l_c - H / (z_w - H) * (kappa * (z_w - d) - l_c) # eq 23
B = 1 / (z_w - H) * (kappa * (z_w - d) - l_c) # eq 24

# equations
eq07 = lambda z: u_H * np.exp(a * (z/H - 1))
eq26 = lambda z: u_star / B     * np.log( (A + B*z) / (A + B*H)) + u_H
eq01 = lambda z: u_star / kappa * np.log((z - d) / z0)



# plot
z = np.linspace(0, 500, 100)

z07 = z[np.where(z<=H)[0]]
z26 = z[np.where(z>=z07[-1])[0]]; z26 = z26[np.where(z26<=z_w)[0]]
z01 = z[np.where(z>=z26[-1])[0]]

plt.figure(1); plt.clf(); plt.show(block=False)
plt.plot(eq07(z07), z07, '-k')
plt.plot(eq26(z26), z26, '-k')
plt.plot(eq01(z01), z01, '-k')
plt.draw()


# mac_data=np.loadtxt('macdonald_2000_cube_arrays.csv', delimiter=',', skiprows=2)
# plt.plot(
#     mac_data[:,4],
#     mac_data[:,5],
#     color='blue',
#     linestyle='',
#     marker='+',
#     markevery=1,
#     ms=4,
# )
# plt.draw()
