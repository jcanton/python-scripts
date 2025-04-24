import pickle
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


#-------------------------------------------------------------------------------

# Define symbols
x, y, k, Z = sp.symbols('x y k Z')
hx, hy, hw, hh = sp.symbols('hx hy hw hh')
s1, s2, n, Zt = sp.symbols('s1 s2 n Zt')

# Define functions
hill = hh * sp.exp( -( (x - hx)**2 + (y - hy)**2 ) / hw**2 )

h1 = hill
h2 = 0*x*y

# src/atm_dyn_iconam/mo_init_vgrid.f90   mo_init_vgrid 󰊕 init_vert_coord
b1 = sp.sinh( (Zt/s1)**n - (Z/s1)**n ) / sp.sinh( (Zt/s1)**n )
b2 = sp.sinh( (Zt/s2)**n - (Z/s2)**n ) / sp.sinh( (Zt/s2)**n )
z = Z + h1 * b1 + h2 * b2

# Compute the derivative dz/dZ
dz_dZ = sp.diff(z, Z)

# Convert symbolic functions to discrete lambdas
z_func =     sp.lambdify((x, y, Z, Zt, s1, s2, n, hx, hy, hh, hw), z)
dz_dZ_func = sp.lambdify((x, y, Z, Zt, s1, s2, n, hx, hy, hh, hw), dz_dZ)

#-------------------------------------------------------------------------------
# Discrete evaluation
#

# domain
x0=0; x1=1000;
y0=0; y1=1000;

# hill
hh = 100; hw = 100;
hx = (x0 + x1)/2; hy = (y0 + y1)/2;

# vertical coordinates
s1 = 150; s2 = 150;
n = 1.2; Zt = 600; flat_height = 500
min_lt = 3 # min layer thickness
s = 1      # stretch factor
nlev = 100

# vertical levels
# src/atm_dyn_iconam/mo_init_vgrid.f90  󰊕 init_sleve_coord  mo_init_vgrid
d = np.log( min_lt / Zt) / np.log( 2.0 / np.pi * np.arccos( float(nlev - 1) ** s / float(nlev) ** s))
vct_a = Zt * ( 2.0 / np.pi * np.arccos( np.arange(nlev + 1, dtype=float) ** s / float(nlev) ** s)) ** d

#-------------------------------------------------------------------------------
# plot
#

if False:
    # uniformly spaced
    nx = 100
    px = np.linspace(x0,x1,nx)
    py = hy*np.ones(nx)
else:
    # load icon data
    with open("data/coordinates_section.pkl", "rb") as f:
        px, py, ddqz_z_half = pickle.load(f)
        nx = px.shape[0]

pX = np.tile(px, (nlev+1,1)).T

z_ifc = np.zeros((nx,nlev+1))
pJacobian = np.zeros((nx,nlev+1))

z_mc = np.zeros((nx,nlev))
ddqz_z = np.zeros((nx,nlev))

# Evaluate and plot
plt.figure(1); plt.clf(); plt.show(block=False)
for k in range(nlev+1):
    Z = vct_a[k]
    if Z >= flat_height:
        z_ifc[:,k] = Z
        pJacobian[:,k] = 1
    else:
        z_ifc[:,k] = z_func(px, py, Z, Zt, s1, s2, 1.2,  hx, hy, hh, hw)
        pJacobian[:,k] = dz_dZ_func(px, py, Z, Zt, s1, s2, 1.2,  hx, hy, hh, hw)
    #
    plt.plot(px,z_ifc[:,k], color="black")
    plt.plot(px,y_coords_i[:,k], color="blue")
plt.xlim([x0,x1])
plt.ylim([0,Zt])
plt.draw()

for k in range(nlev):
    z_mc[:,k] = 0.5*(z_ifc[:,k] + z_ifc[:,k+1])
    ddqz_z[:,k] = z_ifc[:,k] - z_ifc[:,k+1]


norm = lambda cmin, cmax: colors.Normalize(vmin=min(-1e-9, cmin), vmax=max(1e-9,cmax))

plt.figure(1); plt.clf(); plt.show(block=False)
data = pJacobian
im = plt.pcolormesh(pX,z_ifc,data, cmap="YlOrRd") #, norm=norm(data.min(), data.max()))
cbar = plt.colorbar(im)
cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, 5))
cbar.set_ticklabels([f"{c:.2f}" for c in np.linspace(cbar.vmin, cbar.vmax, 5)])
plt.axis("equal")
plt.xlim([x0,x1])
plt.ylim([0,Zt])
plt.draw()

plt.figure(1); plt.clf(); plt.show(block=False)
data = ddqz_z
im = plt.pcolormesh(pX[:,:-1],z_mc,data, cmap="YlOrRd", norm=norm(data.min(), data.max()))
cbar = plt.colorbar(im)
cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, 5))
cbar.set_ticklabels([f"{c:.2f}" for c in np.linspace(cbar.vmin, cbar.vmax, 5)])
plt.axis("equal")
plt.xlim([x0,x1])
plt.ylim([0,Zt])
plt.draw()

plt.figure(1); plt.clf(); plt.show(block=False)
data = ddqz_z_half
im = plt.pcolormesh(pX,z_ifc,data, cmap="YlOrRd", norm=norm(data.min(), data.max()))
cbar = plt.colorbar(im)
cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, 5))
cbar.set_ticklabels([f"{c:.2f}" for c in np.linspace(cbar.vmin, cbar.vmax, 5)])
plt.axis("equal")
plt.xlim([x0,x1])
plt.ylim([0,Zt])
plt.draw()
