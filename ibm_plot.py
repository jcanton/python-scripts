import os
import pickle
import xarray as xr

import gt4py.next as gtx
import matplotlib.pyplot as plt
import numpy as np


from icon4py.model.common.io import plots

QSCALE = 50
PEVERY = 1

f_or_p = 'f'

hill_x = 500.0
hill_y = 500.0
hill_height = 100.0
hill_width = 100.0
compute_distance_from_hill = lambda x, y: ((x - hill_x) ** 2 + (y - hill_y) ** 2) ** 0.5
compute_hill_elevation = lambda x, y: hill_height * np.exp(
    -((compute_distance_from_hill(x, y) / hill_width) ** 2)
)
x = np.linspace(0, 2*hill_x, 500)
y = hill_y
hill = compute_hill_elevation(x, y)

main_dir = os.getcwd() + "/../icon4py/"
grid_file_path = main_dir + "testdata/grids/gauss3d_torus/Torus_Triangles_1000m_x_1000m_res10m.nc"

# -------------------------------------------------------------------------------
# Serialized data
#
if f_or_p == 'f':
    savepoint_path = "/capstor/scratch/cscs/jcanton/ser_data/exclaim_gauss3d.uniform100_hill100x100/ser_data/"
elif f_or_p == 'p':
    savepoint_path = "/capstor/scratch/cscs/jcanton/ser_data/exclaim_gauss3d.uniform100_flat/ser_data/"

plot = plots.Plot(
    savepoint_path=savepoint_path,
    grid_file_path=grid_file_path,
    backend=gtx.gtfn_cpu,
)

# -------------------------------------------------------------------------------
# Load plot data

if f_or_p == 'f':
    fortran_files_dir = "/capstor/scratch/cscs/jcanton/plot_data/exclaim_gauss3d.uniform100_hill100x100/"
    output_files = os.listdir(fortran_files_dir)
    output_files.sort()
elif f_or_p == 'p':
    python_files_dir = main_dir + "imgs/"
    output_files = os.listdir(python_files_dir)
    output_files.sort()

# -------------------------------------------------------------------------------
# Plot
#

for filename in output_files:

    if f_or_p == 'f':
        if not filename.startswith("exclaim_gauss3d_sb_insta"):
            continue
        ds = xr.open_dataset(fortran_files_dir + filename)
        data = ds.w.values[0,:,:].T

    elif f_or_p == 'p':
        if not filename.endswith(".pkl"):
            continue
        with open(python_files_dir + filename, "rb") as f:
            state = pickle.load(f)
        data = state['w']


    print(f"Plotting {filename}")

    axs, x_coords_i, y_coords_i, u_i, w_i, idxs = plot.plot_sections(
        data=data,
        sections_x=[],
        sections_y=[hill_y],
        label="w",
        plot_every=PEVERY,
        qscale=QSCALE,
    )
    axs[0].plot(x, hill, "--", color="black")
    axs[0].set_aspect("equal")
    # axs[0].set_xlim([150,850])
    #axs[0].set_ylim([-1, 500])
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("z [m]")
    plt.draw()
    #plt.show(block=False)
    plt.savefig(f"figures_fortran/{filename[:-3]}.png", dpi=600)
