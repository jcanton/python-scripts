import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# -------------------------------------------------------------------------------
# Some serialized data
#
dx = 1.25

# icon4py_dir = os.path.join(os.getcwd(), "../icon4py")
# ICON4PY_SAVEPOINT_PATH="ser_data/exclaim_gauss3d_250x250x250.uniform200_flat/ser_data"
# ICON4PY_GRID_FILE_PATH="testdata/grids/gauss3d_torus/Torus_Triangles_250m_x_250m_res1.25m.nc"
# grid_file_path = os.path.join(icon4py_dir, ICON4PY_GRID_FILE_PATH)
# savepoint_path = os.path.join(icon4py_dir, ICON4PY_SAVEPOINT_PATH)
# import gt4py.next as gtx
# from icon4py.model.common.io import plots
# plot = plots.Plot(
#     savepoint_path=savepoint_path,
#     grid_file_path=grid_file_path,
#     backend=gtx.gtfn_cpu,
# )
# tri = plot.tri

with open("../data/plotting_250x250x250_1.25.pkl", "rb") as f:
    plotting = pickle.load(f)

#-------------------------------------------------------------------------------
# Load data
#

fname = os.path.join("../data/runyb_test_wiggles_cube_adv_term/advection_corrector.pkl")
with open(fname, "rb") as ifile:
    state = pickle.load(ifile)
    w_dvndz = state["w_dvndz"]
    vn_eh = state["vn_eh"]
    w_wcc_cf = state["w_wcc_cf"]

#fname = os.path.join("../data/runyb_test_wiggles_cube_adv_term/end_of_dyn_timestep.pkl")
fname = os.path.join("../data/runyb_test_wiggles_cube_adv_term/end_of_timestep_000000.pkl")
#fname = os.path.join("../data/runyb_test_wiggles_cube_adv_term/end_of_timestep_002500.pkl")
with open(fname, "rb") as ifile:
    state = pickle.load(ifile)
    vn = state["vn"]
    w = state["w"]
    rho = state["rho"]
    exner = state["exner"]
    theta_v = state["theta_v"]

#-------------------------------------------------------------------------------
# Vertical profiles (b)
#
x0 = [170, 185]
y0 = 124

# pick edge indexes
e_dist = ((tri.edge_y-y0)**2 )**0.5
e_idxs = np.where(e_dist < dx/2)[0]
e_idxs = e_idxs[np.where(tri.edge_x[e_idxs] > x0[0])[0]]
e_idxs = e_idxs[np.where(tri.edge_x[e_idxs] < x0[1])[0]]
n_edges = len(e_idxs)
# pick cell indexes
c_dist = ((tri.cell_y-y0)**2 )**0.5
c_idxs = np.where(c_dist < dx/2)[0]
c_idxs = c_idxs[np.where(tri.cell_x[c_idxs] > x0[0])[0]]
c_idxs = c_idxs[np.where(tri.cell_x[c_idxs] < x0[1])[0]]
n_cells = len(c_idxs)

if n_edges > n_cells:
    e_idxs = e_idxs[:n_cells]
    n_edges = len(e_idxs)
if n_cells > n_edges:
    c_idxs = c_idxs[:n_edges]
    n_cells = len(c_idxs)
n_points = n_edges

# profiles locations
fig = plt.figure(1); plt.clf(); plt.show(block=False)
ax = fig.subplots(nrows=1, ncols=1)
cmap = plt.get_cmap("Greys"); cmap.set_under("white", alpha=0)
im = ax.tripcolor(tri, ibm_mask_cf[:, -1].astype(float), edgecolor="k", shading="flat", cmap=cmap, vmin=0.5, alpha=0.2)
ax.plot(tri.edge_x[e_idxs], tri.edge_y[e_idxs], '+r')
ax.plot(tri.cell_x[c_idxs], tri.cell_y[c_idxs], 'ob')
for i in range(n_points):
    ax.text(tri.edge_x[e_idxs[i]], tri.edge_y[e_idxs[i]], str(i+1), color='red',  fontsize=8, ha='left', va='bottom')
    ax.text(tri.cell_x[c_idxs[i]], tri.cell_y[c_idxs[i]], str(i+1), color='blue', fontsize=8, ha='left', va='bottom')
ax.set_xlim(x0[0]-3*dx, x0[1]+3*dx)
ax.set_ylim(y0-3*dx, y0+3*dx)
plt.draw()

# vertical profiles
fig = plt.figure(2); plt.clf(); plt.show(block=False)
axs = fig.subplots(nrows=2, ncols=n_points, sharex=False, sharey=True)
for i in range(n_points):

    axs[0][i].set_title(f"E {i+1}")
    axs[0][i].plot(-vn[e_idxs[i],::-1], range(200), '-o', ms=2)
    #axs[0][i].plot(-vn_eh[e_idxs[i],::-1], range(201), '-o', ms=2)

    axs[1][i].set_title(f"C {i+1}")
    axs[1][i].plot(w[c_idxs[i],::-1], range(201), '-o', ms=2)

axs[0][0].set_ylabel(r"$v_n$ [m/s]")
axs[1][0].set_ylabel(r"$w$ [m/s]")

axs[0][0].set_ylim([75, 95])

plt.draw()


# #-------------------------------------------------------------------------------
# # Horizontal section
# #
# lev = 120
# fig = plt.figure(3); plt.clf(); plt.show(block=False)
# ax = fig.subplots(nrows=1, ncols=1)
# axs = [ax]
# caxs = [make_axes_locatable(ax).append_axes('right', size='3%', pad=0.02) for ax in axs]
# cax = caxs[0]
#
# # ibm mask
# cmap = plt.get_cmap("Greys"); cmap.set_under("white", alpha=0)
# im = ax.tripcolor(tri, ibm_mask_cf[:, -1].astype(float), edgecolor="k", shading="flat", cmap=cmap, vmin=0.5, alpha=0.2)
#
# # mark selected edge
# ax.plot(tri.edge_x[e_idx], tri.edge_y[e_idx], '+r')
# # mark selected cell
# ax.plot(tri.cell_x[c_idx], tri.cell_y[c_idx], 'xr')
#
# # # vn
# # cmap = plt.get_cmap("gist_ncar_r"); cmap.set_under("white", alpha=0); cmap.set_over("white", alpha=0)
# # im = ax.scatter(tri.edge_x, tri.edge_y, c=vn[:, lev], s=6**2, cmap=cmap) #, vmax=-0.6)
# # cbar = fig.colorbar(im, cax=cax, orientation='vertical')
#
# # # w_dvndz
# # cmap = plt.get_cmap("gist_ncar_r"); cmap.set_under("white", alpha=0); cmap.set_over("white", alpha=0)
# # im = ax.scatter(tri.edge_x, tri.edge_y, c=w_dvndz[:, lev], s=6**2, cmap=cmap)
# # cbar = fig.colorbar(im, cax=cax, orientation='vertical')
#
# #ax.set_aspect("equal")
# #ax.set_xlim([160, 190])
# #ax.set_ylim([ 75, 180])
# plt.draw()

# #-------------------------------------------------------------------------------
# # Vertical profiles
# #
# x0, y0 = (174.5, 125.5)
#
# # pick edge index
# e_dist = ( (tri.edge_x-x0)**2 + (tri.edge_y-y0)**2 )**0.5
# e_idx = np.argmin(e_dist)
#
# # pick cell index
# c_dist = ( (tri.cell_x-x0)**2 + (tri.cell_y-y0)**2 )**0.5
# c_idx = np.argmin(c_dist)
#
# fig = plt.figure(1); plt.clf(); plt.show(block=False)
# #plt.plot(vn[e_num,:], range(200), '-+')
# plt.plot(w[c_idx,:], range(201), '-+')
# plt.draw()
