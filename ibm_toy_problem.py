import numpy as np
import matplotlib.pyplot as plt

#===============================================================================
# Data
#
N  = 100
Nt = 10
dz = 1e-1
dt = 1e-1

w_top = 0
w_bot = 0
w_ibm = 0

# extra boundary points
ibm = [1,2,3,4,5]

w = 0.0*np.ones((N)); w[0] = w_top; w[N-1] = w_bot; w[[ibm]] = w_ibm

#p = lambda t: 0.1*np.ones(N)
p = lambda t: 0.1*np.linspace(0,10,N)
#p = lambda t: 0.1*np.cos(np.linspace(0,2*np.pi,N))
#p = lambda t: 0.01np.cos(np.linspace(0,2*np.pi,N) + t*np.pi/10)

M = np.zeros((N,N))
rhs = np.zeros((N))

#===============================================================================
# Linear system
#
def build_M(w):
    for k in range(1,N-1):
        # a: sub-diagonal
        M[k,k-1] = + w[k]/(2*dz)
        # b: diagonal
        M[k,k] = 1/dt
        # c: sup-diagonal
        M[k,k+1] = - w[k]/(2*dz)
    k = 0;                 M[k,k] = 1; M[k,k+1] = 0
    k = N-1; M[k,k-1] = 0; M[k,k] = 1

def build_rhs(w, p):
    for k in range(1,N-1):
        rhs[k] = w[k]/dt - (p[k-1] - p[k+1])/(2*dz)
    k = 0;   rhs[k] = w_top
    k = N-1; rhs[k] = w_bot


#===============================================================================
# Solve
#
plt.figure(1); plt.clf(); plt.show(block=False)
#plt.plot(w, "-o", label=f"t = {t}")

for t in range(1,Nt+1):

    # Assemble linear system
    build_M(w)
    build_rhs(w, p(t))

    # Option 2: modify linear system
    for k in ibm:
        M[k,   k-1] = 0
        M[k,   k  ] = 1
        M[k,   k+1] = 0
        rhs[k] = w_ibm

    # Solve
    w = np.linalg.solve(M, rhs)

    # # Option 1: force after solve
    # w[[ibm]] = w_ibm

    plt.plot(w, "-o", label=f"t = {t}")

plt.legend(fontsize="small")
plt.draw()
