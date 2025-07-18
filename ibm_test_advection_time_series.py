import os, pickle, glob
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------
# Some serialized data
#
dx = 1.25

with open("data/plotting_250x250x250_1.25.pkl", "rb") as f:
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

#-------------------------------------------------------------------------------
# Load data
#
filenames = glob.glob(os.path.join("../icon4py/runyb_test_wiggles_cube_adv_term", "??????_vn_tendency_*.pkl"))
filenames.sort()

for ifile, fname in enumerate(filenames):

    print(f"Processing {fname}")

    with open(fname, "rb") as infile:
        state = pickle.load(infile)
        vn_c = state["vn_curr"]
        vn_n = state["vn_next"]
        vn_adv = state["dt"] * state["vn_adv"]
        vn_pgr = state["dt"] * state["cpd"] * state["theta_v_ef"] *  state["gradh_exner"]

    #---------------------------------------------------------------------------
    # Vertical profiles
    #
    if ifile == 0:
        x0 = [170, 181.3]
        y0 = 124
        # pick edge indexes
        e_dist = ((tri.edge_y-y0)**2 )**0.5
        e_idxs = np.where(e_dist < dx/2)[0]
        e_idxs = e_idxs[np.where(tri.edge_x[e_idxs] > x0[0])[0]]
        e_idxs = e_idxs[np.where(tri.edge_x[e_idxs] < x0[1])[0]]
        n_edges = len(e_idxs)
        n_points = n_edges

    # vertical profiles
    fig = plt.figure(2); plt.clf(); # plt.show(block=False)
    axs = fig.subplots(nrows=2, ncols=n_points, sharex=False, sharey=True)
    for i in range(n_points):

        axs[0][i].set_title(f"E {i+1}")
        axs[0][i].plot(-vn_c[e_idxs[i],:], full_levels, '-o',  ms=4)
        axs[0][i].plot(-vn_n[e_idxs[i],:], full_levels, '--d', ms=4)

        axs[1][i].plot(vn_adv[e_idxs[i],:], full_levels, '-4',  ms=4)
        axs[1][i].plot(vn_pgr[e_idxs[i],:], full_levels, '--2', ms=4)

        # ibm masks
        axs[0][i].plot(0 * np.ones(np.sum(half_edge_mask[e_idxs[i], :].astype(int))), half_levels[half_edge_mask[e_idxs[i], :].astype(bool)], '+k')
        axs[0][i].plot(0 * np.ones(np.sum(full_edge_mask[e_idxs[i], :].astype(int))), full_levels[full_edge_mask[e_idxs[i], :].astype(bool)], 'xk')
        axs[1][i].plot(0 * np.ones(np.sum(half_edge_mask[e_idxs[i], :].astype(int))), half_levels[half_edge_mask[e_idxs[i], :].astype(bool)], '+k')
        axs[1][i].plot(0 * np.ones(np.sum(full_edge_mask[e_idxs[i], :].astype(int))), full_levels[full_edge_mask[e_idxs[i], :].astype(bool)], 'xk')
        # grid (full and half levels)
        axs[0][i].set_yticks(full_levels, minor=False)
        axs[0][i].set_yticks(half_levels, minor=True)
        axs[0][i].yaxis.grid(which='major', color='#DDDDDD', linewidth=0.8)
        axs[0][i].yaxis.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
        axs[1][i].set_yticks(full_levels, minor=False)
        axs[1][i].set_yticks(half_levels, minor=True)
        axs[1][i].yaxis.grid(which='major', color='#DDDDDD', linewidth=0.8)
        axs[1][i].yaxis.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)

    axs[0][0].set_ylabel(r"$v_n$ [m/s]")
    axs[1][0].set_ylabel(r"tendencies [m/s]")
    axs[0][0].set_ylim([90, 110])
    plt.draw()

    plt.savefig(f"imgs/{ifile:06d}_tendencies")
