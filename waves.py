import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# load data
with open("data/waves.pkl", "rb") as f:
    px, py, icon_ddqz_z_half, z_ifc = pickle.load(f)

ix = 65

# waves
#
c = 2
dt = 0.1
u0 = lambda x:


# x \in [0, 600]
x_analytic = z_ifc[ix, :]
x_approxim = icon_ddqz_z_half[ix, :]
