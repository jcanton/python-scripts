# from sympy import symbols, sinh, exp, diff
#
# # Define symbols
# x, y, Z = symbols('x y Z')
# hx, hy, hw, hh = symbols('hx hy hw hh')
# s1, s2, H = symbols('s1 s2 H')
#
# # Define functions
# hill = hh * exp(- ((x-hx)**2 + (y-hy)**2)/hw**2)
# h1 = hill
# h2 = 0*x*y
# b1 = sinh((H-Z)/s1) / sinh(H/s1)
# b2 = sinh((H-Z)/s2) / sinh(H/s2)
#
# # Define z
# z = Z + h1 * b1 + h2 * b2
#
# # Compute the derivative dz/dZ
# dz_dZ = diff(z, Z)
# print(dz_dZ)

import numpy as np
import matplotlib.pyplot as plt

# domain
x0=0; x1=1000;
y0=0; y1=1000;

# hill
hh = 100; hw = 100;
hx = (x0+x1)/2; hy = (y0+y1)/2;

# coordinates
s1 = 150; s2 = 150;
H = 500;

# functions
hill = lambda x,y: hh * np.exp(- ((x-hx)**2 + (y-hy)**2)/hw**2)
h1 = lambda x,y: hill(x,y);
h2 = lambda x,y: 0*x*y;
b1 = lambda Z: np.sinh((H-Z)/s1) / np.sinh(H/s1)
b2 = lambda Z: np.sinh((H-Z)/s2) / np.sinh(H/s2)

z = lambda x,y,Z:  Z + h1(x,y) * b1(Z)+ h2(x,y) * b2(Z);

# plot
px = np.linspace(x0,x1,100)
pz = np.linspace(0,H, 100)
plt.figure(1); plt.clf(); plt.show(block=False)
#PX, PZ = np.meshgrid(px, pz)
#plt.contourf(PX, PZ, z(PX,hy, PZ), 50, edgecolors='black')
#plt.plot(px,hill(px,hy))
for zhat in range(0,H,50):
    plt.plot(px,z(px,hy, zhat))
plt.draw()

#-------------------------------------------------------------------------------
# Jacobian
#
Jm1 = lambda x,y,Z: -hh*np.exp((-(-hx + x)**2 - (-hy + y)**2)/hw**2)*np.cosh((H - Z)/s1)/(s1*np.sinh(H/s1)) + 1
plt.figure(1); plt.clf(); plt.show(block=False)
for zhat in range(0,H,50):
    plt.plot(px,Jm1(px,hy, zhat))
plt.draw()
