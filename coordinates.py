# SLEVE_NML
# MIN_LAY_THCKN =    3.000000000000000     ,
# MAX_LAY_THCKN =    25000.00000000000     ,
# HTOP_THCKNLIMIT =    15000.00000000000     ,
# TOP_HEIGHT =    600.0000000000000     ,
# DECAY_SCALE_1 =    150.0000000000000     ,
# DECAY_SCALE_2 =    150.0000000000000     ,
# DECAY_EXP =    1.200000000000000     ,
# FLAT_HEIGHT =    500.0000000000000     ,
# STRETCH_FAC =    1.000000000000000     ,
# ELSE IF (ivctype == 2 ) THEN ! SLEVE coordinate (Leuenberger et al. MWR 2010)
# DO jk = 1, nflat
#   jk1 = jk + nshift
#   z3d_i(1:nlen,jk,jb) = vct_a(jk1)
# ENDDO
# DO jk = nflat + 1, nlev
#   jk1 = jk + nshift
#   ! Scaling factors for large-scale and small-scale topography
#   z_fac1 = SINH((top_height/decay_scale_1)**decay_exp - (vct_a(jk1)/decay_scale_1)**decay_exp) / SINH((top_height/decay_scale_1)**decay_exp)
#   z_fac2 = SINH((top_height/decay_scale_2)**decay_exp - (vct_a(jk1)/decay_scale_2)**decay_exp) / SINH((top_height/decay_scale_2)**decay_exp)
#
#   ! Small-scale topography (i.e. full topo - smooth topo)
#   z_topo_dev(1:nlen) = topo(1:nlen,jb) - topo_smt(1:nlen,jb)
#
#   z3d_i(1:nlen,jk,jb) = vct_a(jk1) + topo_smt(1:nlen,jb)*z_fac1 + z_topo_dev(1:nlen)*z_fac2
# ENDDO


import sympy as sp

# Define symbols
x, y, Z = sp.symbols('x y Z')
hx, hy, hw, hh = sp.symbols('hx hy hw hh')
s1, s2, n, Zt = sp.symbols('s1 s2 n Zt')

# Define functions
hill = hh * sp.exp( -( (x - hx)**2 + (y - hy)**2 ) / hw**2 )
h1 = hill
h2 = 0*x*y
b1 = sp.sinh( (Zt/s1)**n - (Z/s1)**n ) / sp.sinh( (Zt/s1)**n )
b2 = sp.sinh( (Zt/s2)**n - (Z/s2)**n ) / sp.sinh( (Zt/s2)**n )

# Define z
z = Z + h1 * b1 + h2 * b2

# Compute the derivative dz/dZ
dz_dZ = sp.diff(z, Z)
print(dz_dZ)

#-------------------------------------------------------------------------------
# Discrete
#
import numpy as np
import matplotlib.pyplot as plt

# domain
x0=0; x1=1000;
y0=0; y1=1000;

# hill
hh = 100; hw = 100;
hx = (x0 + x1)/2; hy = (y0 + y1)/2;

# coordinates
s1 = 150; s2 = 150;
n = 1.2; Zt = 500; zt = 600;

# functions
hill = lambda x,y: hh * np.exp( -( (x - hx)**2 + (y - hy)**2 ) / hw**2 )
h1 = lambda x,y: hill(x,y);
h2 = lambda x,y: 0*x*y;
b1 = lambda Z: np.sinh( (Zt/s1)**n - (Z/s1)**n ) / np.sinh( (Zt/s1)**n )
b2 = lambda Z: np.sinh( (Zt/s2)**n - (Z/s2)**n ) / np.sinh( (Zt/s2)**n )

z = lambda x,y,Z:  Z + h1(x,y) * b1(Z)+ h2(x,y) * b2(Z);

# plot
px = np.linspace(x0,x1,100)
pz = np.linspace(0,Zt, 100)
plt.figure(1); # plt.clf(); plt.show(block=False)
#PX, PZ = np.meshgrid(px, pz)
#plt.contourf(PX, PZ, z(PX,hy, PZ), 50, edgecolors='black')
#plt.plot(px,hill(px,hy))
for zhat in range(0,zt,50):
    plt.plot(px,z(px,hy, zhat), color="blue")
plt.draw()

#-------------------------------------------------------------------------------
# Jacobian
#
Jm1 = lambda x,y,Z: 1 - hh*n*(Z/s1)**n*np.exp((-(-hx + x)**2 - (-hy + y)**2)/hw**2)*np.cosh((Zt/s1)**n - (Z/s1)**n)/(Z*np.sinh((Zt/s1)**n))
plt.figure(1); # plt.clf(); plt.show(block=False)
for zhat in range(0,zt,50):
    plt.plot(px,Jm1(px,hy, zhat), color="blue")
plt.draw()
