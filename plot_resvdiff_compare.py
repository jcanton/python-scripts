import glob
import os
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

main_dir = "../runs/icon4py/"
runs = [
    #{"dx": 5.00, "y0": 175.0, "name": "channel_950m_x_350m_res5m_nlev20"},
    #
    #{"dx": 2.50, "y0": 176.0, "name": "channel_950m_x_350m_res2.5m_nlev40"},
    #{"dx": 2.50, "y0": 176.0, "name": "channel_950m_x_350m_res2.5m_nlev40_vdiff0015"},
    #{"dx": 2.50, "y0": 176.0, "name": "channel_950m_x_350m_res2.5m_nlev40_vdiff0010"},
    #{"dx": 2.50, "y0": 176.0, "name": "channel_950m_x_350m_res2.5m_nlev40_vdiff0005"},
    #{"dx": 2.50, "y0": 176.0, "name": "channel_950m_x_350m_res2.5m_nlev40_vdiff00015"},
    #{"dx": 2.50, "y0": 176.0, "name": "channel_950m_x_350m_res2.5m_nlev40_vdiff0001"},
    #{"dx": 2.50, "y0": 176.0, "name": "channel_950m_x_350m_res2.5m_nlev40_vdiff000015"},
    #
    #{"dx": 1.50, "y0": 174.6, "name": "channel_950m_x_350m_res1.5m_nlev64_vdiff0010"},
    #{"dx": 1.50, "y0": 174.6, "name": "channel_950m_x_350m_res1.5m_nlev64_vdiff0005"},
    #{"dx": 1.50, "y0": 174.6, "name": "channel_950m_x_350m_res1.5m_nlev64_vdiff0001"},
    #{"dx": 1.50, "y0": 174.6, "name": "channel_950m_x_350m_res1.5m_nlev64_vdiff00005"},
    #
    #{"dx": 1.25, "y0": 175.2, "name": "channel_950m_x_350m_res1.25m_nlev80_vdiff00150"},
    #{"dx": 1.25, "y0": 175.2, "name": "channel_950m_x_350m_res1.25m_nlev80_vdiff00100"},
    #{"dx": 1.25, "y0": 175.2, "name": "channel_950m_x_350m_res1.25m_nlev80_vdiff00050"},
    #{"dx": 1.25, "y0": 175.2, "name": "channel_950m_x_350m_res1.25m_nlev80_vdiff00010"},
    #{"dx": 1.25, "y0": 175.2, "name": "channel_950m_x_350m_res1.25m_nlev80_vdiff00005"},
    #{"dx": 1.25, "y0": 175.2, "name": "channel_950m_x_350m_res1.25m_nlev80_vdiff00001"},
    #
    {"dx": 1.00, "y0": 174.4, "name": "channel_950m_x_350m_res1m_nlev100_vdiff00150"},
    {"dx": 1.00, "y0": 174.4, "name": "channel_950m_x_350m_res1m_nlev100_vdiff00100"},
    {"dx": 1.00, "y0": 174.4, "name": "channel_950m_x_350m_res1m_nlev100_vdiff00050"},
    {"dx": 1.00, "y0": 174.4, "name": "channel_950m_x_350m_res1m_nlev100_vdiff00010"},
    {"dx": 1.00, "y0": 174.4, "name": "channel_950m_x_350m_res1m_nlev100_vdiff00005"},
    {"dx": 1.00, "y0": 174.4, "name": "channel_950m_x_350m_res1m_nlev100_vdiff00001"},
]

for run in runs:
    dx = run["dx"]
    run_name = run["name"]
    y0 = run["y0"]

    # -------------------------------------------------------------------------------
    # Plotting data
    #

    match dx:
        case 5.0:
            plotting_data_file = "plotting_channel_950x350x100_5m_nlev20.pkl"
        case 2.5:
            plotting_data_file = "plotting_channel_950x350x100_2.5m_nlev40.pkl"
        case 1.5:
            plotting_data_file = "plotting_channel_950x350x100_1.5m_nlev64.pkl"
        case 1.25:
            plotting_data_file = "plotting_channel_950x350x100_1.25m_nlev80.pkl"
        case 1.0:
            plotting_data_file = "plotting_channel_950x350x100_1m_nlev100.pkl"

    with open(os.path.join("data", plotting_data_file), "rb") as f:
        plotting = pickle.load(f)
        tri = plotting["tri"]
        full_level_heights = plotting["full_level_heights"]
        half_level_heights = plotting["half_level_heights"]
        full_cell_mask = plotting["full_cell_mask"]
        half_cell_mask = plotting["half_cell_mask"]
        full_edge_mask = plotting["full_edge_mask"]
        half_edge_mask = plotting["half_edge_mask"]
    full_levels = full_level_heights[0, :]
    half_levels = half_level_heights[0, :]

    num_cells = len(tri.cell_x)
    num_levels = len(full_levels)

    # ==============================================================================
    # PALM verification
    #
    fnames = glob.glob(os.path.join(main_dir, run_name, "avgs/avg_hour???.pkl"))
    fnames.sort()
    #print(f"Found {len(fnames) - 2} files for temporal averaging")
    u_cf = np.zeros((num_cells, num_levels))
    for fname in fnames[2:]:
        with open(fname, "rb") as ifile:
            state = pickle.load(ifile)
            wind_cf = state["wind_cf"]
        u_cf += wind_cf[:, :, 0]
    u_cf /= len(fnames) - 2

    H = 50
    xH = 3 * H
    x0s = [xH - H, xH + 0.5 * H, xH + H, xH + 1.5 * H, xH + 2.5 * H, xH + 4 * H]

    vali_data_path = os.path.join("data", "palm_validation_plots")
    vali_labels = [
        "xH_m10",
        "xH_p05",
        "xH_p10",
        "xH_p15",
        "xH_p25",
        "xH_p40",
    ]

    # pick cell indexes
    c_idxs = []
    for x0 in x0s:
        c_dist = ((tri.cell_x - x0) ** 2 + (tri.cell_y - y0) ** 2) ** 0.5
        c_idx = np.argmin(c_dist)
        c_idxs.append(c_idx)

    fig = plt.figure(figsize=(24, 8))
    #plt.clf()
    #plt.show(block=False)
    axs = fig.subplots(nrows=1, ncols=len(x0s), sharex=True, sharey=True)

    deltas = np.zeros(len(x0s))

    for ix, x0 in enumerate(x0s):
        vali_img = mpimg.imread(os.path.join(vali_data_path, vali_labels[ix] + ".png"))
        vali_data = np.loadtxt(os.path.join(vali_data_path, vali_labels[ix] + ".csv"), delimiter=",")

        interp_data = np.interp(vali_data[:, 1], full_levels[::-1]/H, u_cf[c_idxs[ix], :][::-1])

        axs[ix].imshow(vali_img, extent=[-0.7, 1.5, 0.0, 2.0], aspect="equal")

        axs[ix].plot(u_cf[c_idxs[ix], :], full_levels / H, "-", ms=4)

        axs[ix].plot(vali_data[:, 0], vali_data[:, 1], "+", ms=4)
        #axs[ix].plot(interp_data,     vali_data[:, 1], "o", ms=4)

        axs[ix].plot([0, 0], [0, 2], "-k")
        axs[ix].set_xlabel(r"$U / U_B$")
        axs[ix].set_title(rf"$x / H = {(x0 - xH) / H:.1f}$")
        # axs[ix].set_aspect('equal')
        #
        deltas[ix] = np.linalg.norm(interp_data - vali_data[:, 0])
        axs[ix].text(0.05, 0.8, rf"$\Delta$ = {deltas[ix]:.2e}", fontsize="small", transform=axs[ix].transAxes, bbox=dict(facecolor='white', edgecolor='white', alpha=0.8))

    axs[0].set_ylabel(r"$y / H$")
    axs[0].set_xlim([-0.7, 1.5])
    axs[0].set_ylim([0.0, 2.0])
    axs[0].set_xticks(np.arange(-0.5, 2.0, 0.5))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.suptitle(f"{run_name}, avg_delta: {np.mean(deltas):.3f}")
    plt.draw()
    plt.savefig(f"imgs/resvdiff_compare_{run_name}.png")

    print(f"{run_name}, n_hours: {len(fnames)-2} delta: {deltas}, avg_delta: {np.mean(deltas)}")
    #continue_prompt = input("Press Enter to continue, x to exit ...")
