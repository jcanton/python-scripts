import os, pickle
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import gt4py.next as gtx
from icon4py.model.common.io import plots

main_dir = os.getcwd() + "/"
state_fname = 'testdata/prognostic_state_initial.pkl'
savepoint_path = 'testdata/ser_icondata/mpitask1/gauss3d_torus/ser_data'
grid_file_path = "testdata/grids/gauss3d_torus/Torus_Triangles_2000m_x_2000m_res250m.nc"

_plot = plots.Plot(
    savepoint_path = main_dir + savepoint_path,
    grid_file_path = main_dir + grid_file_path,
    backend = gtx.gtfn_cpu,
    )

fname = os.getcwd() + "/testdata/w_matrix_0.pkl"
with open(fname, "rb") as ifile:
    (a, b, c, d) = pickle.load(ifile)
fname = os.getcwd() + "/testdata/w_matrix_313.pkl"
with open(fname, "rb") as ifile:
    (a3, b3, c3, d3) = pickle.load(ifile)

# # get only the lower levels
# nlevs = 4
# a = a[:, -nlevs:]
# b = b[:, -nlevs:]
# c = c[:, -nlevs:]
# d = d[:, -nlevs:]

a = a.flatten()
b = b.flatten()
c = c.flatten()
d = d.flatten()

M = sp.sparse.diags([a, b, c], [-1, 0, 1], shape=(len(b), len(b)))
w = sp.sparse.linalg.spsolve(M, d)

w = w.reshape(-1, nlevs)
axs = _plot.plot_data(w, nlevs, label=f"w")
plt.show(block=False)