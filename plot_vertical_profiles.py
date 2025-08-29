import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------------------
# Some serialized data
#
dx = 5

#with open("./data/plotting_250x250x1000_2.5.pkl", "rb") as f:
#with open("./data/plotting_250x250x250_1.25.pkl", "rb") as f:
with open("./data/plotting_channel_950x350x100_5m_nlev20.pkl", "rb") as f:
    plotting = pickle.load(f)
    tri = plotting["tri"]
    full_level_heights = plotting["full_level_heights"]
    half_level_heights = plotting["half_level_heights"]
    full_cell_mask = plotting["full_cell_mask"]
    half_cell_mask = plotting["half_cell_mask"]
    full_edge_mask = plotting["full_edge_mask"]
    half_edge_mask = plotting["half_edge_mask"]
full_levels = full_level_heights[0,:]
half_levels = half_level_heights[0,:]

num_cells = len(tri.cell_x)
num_levels = len(full_levels)

#-------------------------------------------------------------------------------
# Load data
#
main_dir = "../runs/icon4py/"
run_name = "channel_950m_x_350m_res5m_nlev20"

#fname = os.path.join(main_dir, run_name, "initial_condition.pkl")
#fname = os.path.join(main_dir, run_name, "000000_initial_condition.pkl")
fname = os.path.join(main_dir, run_name, "000001_initial_condition_ibm.pkl")
with open(fname, "rb") as ifile:
    state = pickle.load(ifile)
    vn0 = state["vn"]
    w0 = state["w"]
    rho0 = state["rho"]
    exner0 = state["exner"]
    theta_v0 = state["theta_v"]

#fname = os.path.join(main_dir, run_name, "000001_initial_condition_ibm.pkl")
#fname = os.path.join(main_dir, run_name, "000002_end_of_predictor.pkl")
#fname = glob.glob(os.path.join(main_dir, run_name, "000121_end_of_timestep_??????.pkl"))[0]
fname = glob.glob(os.path.join(main_dir, run_name, "000400_*.pkl"))[0]
#fname = os.path.join(main_dir, run_name, "000000_debug_channel_fields.pkl")
#fname = os.path.join(main_dir, run_name, "initial_condition.pkl")
#fname = os.path.join(main_dir, run_name, "end_of_timestep_000000175.pkl")
#fname = os.path.join(main_dir, run_name, "avgs/avg_hour020.pkl")
with open(fname, "rb") as ifile:
    state = pickle.load(ifile)
    vn = state["vn"]
    w = state["w"]
    rho = state["rho"]
    exner = state["exner"]
    theta_v = state["theta_v"]
    # sponge_full_cell = state["sponge_full_cell"]
    # c_vn = vn.copy()
    # c_w  = w.copy()

#-------------------------------------------------------------------------------
# Vertical profiles (b)
#
x0 = [0, 50] # beginning of channel
#x0 = [130, 180] # cube leading edge
#x0 = [180, 230] # cube trailing edge
#x0 = [330, 380] # middle of nowhere
x0 = [850, 950] # end of channel
y0 = 175

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
im = ax.tripcolor(tri, full_cell_mask[:, -1].astype(float), edgecolor="k", shading="flat", cmap=cmap, vmin=0.5, alpha=0.2)
im = ax.tripcolor(tri, full_level_heights[:, -1].astype(float), edgecolor="k", shading="flat", cmap=cmap, alpha=0.2)
ax.plot(tri.edge_x[e_idxs], tri.edge_y[e_idxs], '+r')
ax.plot(tri.cell_x[c_idxs], tri.cell_y[c_idxs], 'ob')
for i in range(n_points):
    ax.text(tri.edge_x[e_idxs[i]], tri.edge_y[e_idxs[i]], str(i+1), color='red',  fontsize=8, ha='left', va='bottom')
    ax.text(tri.cell_x[c_idxs[i]], tri.cell_y[c_idxs[i]], str(i+1), color='blue', fontsize=8, ha='left', va='bottom')
ax.set_xlim(x0[0]-5*dx, x0[1]+5*dx)
ax.set_ylim(y0-5*dx, y0+5*dx)
ax.set_aspect('equal') #, adjustable='box')
plt.draw()

# vertical profiles
fig = plt.figure(2); plt.clf(); plt.show(block=False)
fig.suptitle(fname)
axs = fig.subplots(nrows=5, ncols=n_points, sharex=False, sharey=True)
for i in range(n_points):

    axs[0][i].set_title(f"{i+1}")
    axs[0][i].plot(-vn [e_idxs[i],:], full_levels, '-o',  ms=4)

    axs[1][i].plot(w [c_idxs[i],:], half_levels, '-o',  ms=4)

    axs[2][i].plot(rho[c_idxs[i],:], full_levels, '-o', ms=4)
    #axs[2][i].plot(rho[c_idxs[i],:] - rho0[c_idxs[i],:], full_levels, '--o', ms=4)

    axs[3][i].plot(exner[c_idxs[i],:], full_levels, '-o', ms=4)
    #axs[3][i].plot(exner[c_idxs[i],:] - exner0[c_idxs[i],:], full_levels, '--o', ms=4)

    axs[4][i].plot(theta_v[c_idxs[i],:], full_levels, '-o', ms=4)
    #axs[4][i].plot(theta_v[c_idxs[i],:] - theta_v0[c_idxs[i],:], full_levels, '--o', ms=4)

    for iax, ax in enumerate(axs):
        # ibm masks
        if iax == 0:
            ax[i].plot(0 * np.ones(np.sum(half_edge_mask[e_idxs[i], :].astype(int))), half_levels[half_edge_mask[e_idxs[i], :].astype(bool)], '+k')
            ax[i].plot(0 * np.ones(np.sum(full_edge_mask[e_idxs[i], :].astype(int))), full_levels[full_edge_mask[e_idxs[i], :].astype(bool)], 'xk')
        else:
            ax[i].plot(0 * np.ones(np.sum(half_cell_mask[c_idxs[i], :].astype(int))), half_levels[half_cell_mask[c_idxs[i], :].astype(bool)], '+k')
            ax[i].plot(0 * np.ones(np.sum(full_cell_mask[c_idxs[i], :].astype(int))), full_levels[full_cell_mask[c_idxs[i], :].astype(bool)], 'xk')
        # grid (full and half levels)
        ax[i].set_yticks(full_levels, minor=False)
        ax[i].set_yticks(half_levels, minor=True)
        ax[i].yaxis.grid(which='major', color='#DDDDDD', linewidth=0.8)
        ax[i].yaxis.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)

axs[0][0].set_ylabel(r"$v_n$")
axs[1][0].set_ylabel(r"$w$")
axs[2][0].set_ylabel(r"$\rho$")
axs[3][0].set_ylabel(r"$\pi$")
axs[4][0].set_ylabel(r"$\theta_v$")
#axs[0][0].set_ylim([90, 115])
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.draw()

# # ==============================================================================
# # spatio-temporal average
# fig = plt.figure(3); plt.clf(); plt.show(block=False)
# fig.suptitle(fname)
# axs = fig.subplots(nrows=5, ncols=1, sharex=False, sharey=True)
# 
# axs[0].plot(-np.average(vn     , axis=0),                                full_levels, '-o',  ms=4)
# axs[1].plot( np.average(w      , axis=0),                                half_levels, '-o',  ms=4)
# #
# #axs[2].plot( np.average(rho    , axis=0),                                full_levels, '-o', ms=4)
# axs[2].plot( np.average(rho    , axis=0) - np.average(rho0,     axis=0), full_levels, '--o', ms=4)
# #
# #axs[3].plot( np.average(exner  , axis=0),                                full_levels, '-o', ms=4)
# axs[3].plot( np.average(exner  , axis=0) - np.average(exner0,   axis=0), full_levels, '--o', ms=4)
# #
# #axs[4].plot( np.average(theta_v, axis=0),                                full_levels, '-o', ms=4)
# axs[4].plot( np.average(theta_v, axis=0) - np.average(theta_v0, axis=0), full_levels, '--o', ms=4)
# 
# for iax, ax in enumerate(axs):
#     # grid (full and half levels)
#     ax.set_yticks(full_levels, minor=False)
#     ax.set_yticks(half_levels, minor=True)
#     ax.yaxis.grid(which='major', color='#DDDDDD', linewidth=0.8)
#     ax.yaxis.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
# 
# axs[0].set_ylabel(r"$v_n$")
# axs[1].set_ylabel(r"$w$")
# axs[2].set_ylabel(r"$\rho$")
# axs[3].set_ylabel(r"$\pi$")
# axs[4].set_ylabel(r"$\theta_v$")
# #axs[0][0].set_ylim([90, 115])
# plt.subplots_adjust(hspace=0.5, wspace=0.3)
# plt.draw()

# ==============================================================================
# PALM verification
#
fnames = glob.glob(os.path.join(main_dir, run_name, "avgs/avg_hour???.pkl"))
fnames.sort()
u_cf = np.zeros((num_cells, num_levels))
for fname in fnames[2:]:
    with open(fname, "rb") as ifile:
        state = pickle.load(ifile)
        wind_cf = state["wind_cf"]
    u_cf += wind_cf[:, :, 0]
    #v_cf = wind_cf[:, :, 1]
    #w_cf = wind_cf[:, :, 2]
u_cf /= (len(fnames) - 2)

H   = 50
xH  = 3*H
x0s = [xH - H, xH + 0.5*H, xH + H, xH + 1.5*H, xH + 2.5*H, xH + 4*H]
y0  = 175

# pick cell indexes
c_idxs = []
for x0 in x0s:
    c_dist = ((tri.cell_x - x0)**2 + (tri.cell_y - y0)**2 )**0.5
    c_idx = np.argmin(c_dist)
    c_idxs.append(c_idx)

fig = plt.figure(4); plt.clf(); plt.show(block=False)
axs = fig.subplots(nrows=1, ncols=len(x0s), sharex=True, sharey=True)
for ix, x0 in enumerate(x0s):
    axs[ix].plot(u_cf[c_idxs[ix],:], full_levels/H, '-o',  ms=4)
    axs[ix].plot([0,0],[0,2], '-k')
    axs[ix].set_xlabel(r"$U / U_B$")
    axs[ix].set_title(fr"$x / H = {(x0-xH)/H:.1f}$")
    axs[ix].set_aspect('equal')

axs[0].set_ylabel(r"$y / H$")
axs[0].set_xlim([-0.7, 1.5])
axs[0].set_ylim([ 0.0, 2.0])
axs[0].set_xticks(np.arange(-0.5, 2.0, 0.5))
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.draw()
