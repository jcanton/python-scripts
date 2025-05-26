import os
import pickle
import xarray as xr

import gt4py.next as gtx
import matplotlib.pyplot as plt
import numpy as np

from icon4py.model.common.io import plots
from icon4py.model.atmosphere.dycore import ibm

# -------------------------------------------------------------------------------
#
main_dir = os.getcwd() + "/../icon4py/"
grid_file_path = main_dir + "testdata/grids/gauss3d_torus/Torus_Triangles_1000m_x_1000m_res10m.nc"

savepoint_path = "/capstor/scratch/cscs/jcanton/ser_data/exclaim_gauss3d.uniform200_flat/ser_data/"
imgs_dir = "runxx_ibm_check_slip_noslip"


if not os.path.exists(imgs_dir):
    os.makedirs(imgs_dir)

plot = plots.Plot(
    savepoint_path=savepoint_path,
    grid_file_path=grid_file_path,
    backend=gtx.gtfn_cpu,
)
tri = plot.tri
full_level_heights = plot.full_level_heights
half_level_heights = plot.half_level_heights

cases = [
    main_dir + "run01_hill100x100_nlev200/",
    main_dir + "run01_hill100x100_nlev200_noSlip/",
]
output_files = [
    "end_of_timestep_002900.pkl"
]

# -------------------------------------------------------------------------------
# vert profiles
#
#markers = ["o", "s", "d"]
markers = ["+", "x", "d"]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
profiles = [
    #[450, 500],
    [500, 500],
    [550, 500]
]
# compute distances and find coordinates
for profile in profiles:
    cell_dist = np.sqrt( (tri.cell_x - profile[0])**2 + (tri.cell_y - profile[1])**2 )
    edge_dist = np.sqrt( (tri.edge_x - profile[0])**2 + (tri.edge_y - profile[1])**2 )
    cell_id = cell_dist.argmin()
    edge_id = edge_dist.argmin()
    profile.append(cell_id)
    profile.append(edge_id)

fig = plt.figure(1, figsize=(8, 12))
plt.clf()
plt.show(block=False)
axs = fig.subplots(nrows=2, ncols=len(profiles), sharex=False, sharey=True)

for i, profile in enumerate(profiles):

    cell_id = profile[2]
    edge_id = profile[3]

    axs[0, i].set_title(f"(x,y) = ({tri.cell_x[cell_id]:.1f}, {tri.cell_y[cell_id]:.1f})m")
    axs[1, i].set_title(f"(x,y) = ({tri.edge_x[edge_id]:.1f}, {tri.edge_y[edge_id]:.1f})m")

    for j, case in enumerate(cases):

        with open(case + output_files[0], "rb") as f:
            state = pickle.load(f)
        data_vn = state['vn']
        data_w  = state['w']

        axs[0, i].plot(
            data_w[cell_id, :],
            half_level_heights[cell_id, :],
            color=colors[j],
            marker=markers[j],
            markevery=1,
        )
        axs[1, i].plot(
            data_vn[edge_id, :],
            full_level_heights[edge_id, :],
            color=colors[j],
            marker=markers[j],
            markevery=1,
        )

    axs[0, i].set_xlabel("w   [m/s]")
    axs[1, i].set_xlabel("v_n [m/s]")

axs[0, 0].set_ylabel("z [m]")
axs[1, 0].set_ylabel("z [m]")
# axs[0,0].set_ylim([-1,Z_TOP])
axs[0, 0].set_ylim([70, 110])
# axs[0].legend(fontsize="small")
plt.draw()
plt.savefig(imgs_dir + "/compare.png", dpi=600)
