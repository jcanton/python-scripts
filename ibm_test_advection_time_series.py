import glob
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np


def export_figure(args):
    ifile, fname = args
    print(f"Processing {fname}")

    with open(fname, "rb") as infile:
        state = pickle.load(infile)
        vn = state["vn"]
        w = state["w"]
        rho = state["rho"]
        exner = state["exner"]
        theta_v = state["theta_v"]

    #---------------------------------------------------------------------------
    # Vertical profiles
    #
    x0 = [170, 185]
    y0 = 124
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

    # vertical profiles
    fig = plt.figure(2, figsize=(25.6, 13.36)); plt.clf(); # plt.show(block=False)
    axs = fig.subplots(nrows=5, ncols=n_points, sharex=False, sharey=True)
    for i in range(n_points):
        axs[0][i].set_title(f"{i+1}")

        axs[0][i].plot(-vn[e_idxs[i],:], full_levels, '--+',  ms=4)
        axs[1][i].plot(w[c_idxs[i],:], half_levels, '--+',  ms=4)
        axs[2][i].plot(rho[c_idxs[i],:] - rho0[c_idxs[i],:], full_levels, '--+', ms=4)
        axs[3][i].plot(exner[c_idxs[i],:] - exner0[c_idxs[i],:], full_levels, '--+', ms=4)
        axs[4][i].plot(theta_v[c_idxs[i],:] - theta_v0[c_idxs[i],:], full_levels, '--+', ms=4)

        for ax in axs:
            # ibm masks
            ax[i].plot(0 * np.ones(np.sum(half_edge_mask[e_idxs[i], :].astype(int))), half_levels[half_edge_mask[e_idxs[i], :].astype(bool)], '+k')
            ax[i].plot(0 * np.ones(np.sum(full_edge_mask[e_idxs[i], :].astype(int))), full_levels[full_edge_mask[e_idxs[i], :].astype(bool)], 'xk')
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
    axs[0][0].set_ylim([90, 115])
    plt.draw()

    plt.savefig(f"imgs/{ifile:06d}_timestep.png", dpi=600, bbox_inches='tight')

# ===============================================================================
if __name__ == "__main__":

    if len(sys.argv) < 1:
        print("Usage: python ibm_test_advection_time_series.py <num_workers>")
        sys.exit(1)
    num_workers = int(sys.argv[1])

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
    main_dir = "../icon4py"
    run_name = "runyf_test_wiggles"
    
    fname = os.path.join(main_dir, run_name, "000001_initial_condition_ibm.pkl")
    with open(fname, "rb") as ifile:
        state = pickle.load(ifile)
        vn0 = state["vn"]
        w0 = state["w"]
        rho0 = state["rho"]
        exner0 = state["exner"]
        theta_v0 = state["theta_v"]
    
    filenames = glob.glob(os.path.join(main_dir, run_name, "??????_end_of_timestep_??????.pkl"))
    filenames.sort()

    # Prepare arguments for each file
    arguments = []
    for ifile, file_path in enumerate(filenames):
        arguments.append((ifile, file_path))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(export_figure, arguments))

