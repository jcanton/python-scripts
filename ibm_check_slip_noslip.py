import os
import pickle

import gt4py.next as gtx
import matplotlib.pyplot as plt
import numpy as np

from icon4py.model.common.io import plots
from icon4py.model.atmosphere.dycore import ibm
from scipy.interpolate import griddata

# -------------------------------------------------------------------------------
#
main_dir = os.getcwd() + "/../icon4py/"
grid_file_path = main_dir + "testdata/grids/gauss3d_torus/Torus_Triangles_1000m_x_1000m_res10m.nc"

#savepoint_path = "/capstor/scratch/cscs/jcanton/ser_data/exclaim_gauss3d.uniform200_flat/ser_data/"
savepoint_path = "/scratch/mch/jcanton/ser_data/exclaim_gauss3d.uniform800_flat/ser_data/"
imgs_dir = "run60_barray_2x2_nlev800"
#savepoint_path = "/scratch/mch/jcanton/ser_data/exclaim_gauss3d.uniform200_flat/ser_data/"
#imgs_dir = "runxx_ibm_check_slip_noslip"


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

cell_centres = np.column_stack((tri.cell_x, tri.cell_y))
edges = np.column_stack((tri.edge_x, tri.edge_y))
full_level_heights_edges = np.zeros((len(tri.edge_x), full_level_heights.shape[1]))
for k in range(full_level_heights.shape[1]):
    z_cells = full_level_heights[:,k]  # shape (num_cells,)
    z_edges = griddata(cell_centres, z_cells, edges, method='linear')
    z_edges_fill = griddata(cell_centres, z_cells, edges, method='nearest')
    full_level_heights_edges[:,k] = np.where(np.isnan(z_edges), z_edges_fill, z_edges)

cases = [
    #main_dir + "run69_barray_1x0_nlev200/",
    #main_dir + "run69_barray_1x0_nlev200_noSlip/",
    #main_dir + "run69_barray_1x0_nlev200_dirich3/",
    #main_dir + "run69_barray_1x0_nlev200_wholeDomain/",
    main_dir + "run60_barray_2x2_nlev800/",
]
output_files = [
    #"end_of_timestep_002400.pkl"
    "end_of_timestep_013600.pkl",
    "end_of_timestep_013700.pkl",
    "end_of_timestep_013800.pkl",
    "end_of_timestep_013900.pkl",
    "end_of_timestep_014000.pkl",
    "end_of_timestep_014100.pkl",
]

# -------------------------------------------------------------------------------
# vert profiles
#
markers = ["+", "x", "d", "o"]
linestyles = ["-", "--", ":", "-."]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
profiles = [
    #[450, 500],
    #[500, 500],
    #[550, 500],
    [350, 500],
    #[500, 500],
    #[560, 500],
]
# compute distances and find coordinates
for profile in profiles:
    cell_dist = np.sqrt( (tri.cell_x - profile[0])**2 + (tri.cell_y - profile[1])**2 )
    edge_dist = np.sqrt( (tri.edge_x - profile[0])**2 + (tri.edge_y - profile[1])**2 )
    cell_id = cell_dist.argmin()
    edge_id = edge_dist.argmin()
    profile.append(cell_id)
    profile.append(edge_id)


for k, output_file in enumerate(output_files):

    fig = plt.figure(1, figsize=(8, 12)); plt.clf()
    plt.show(block=False)
    axs = fig.subplots(nrows=len(profiles), ncols=2, sharex='col', sharey=True, squeeze=False)
    
    for i, profile in enumerate(profiles):
    
        cell_id = profile[2]
        edge_id = profile[3]
    
        axs[i, 0].set_title(f"(x,y) = ({tri.cell_x[cell_id]:.1f}, {tri.cell_y[cell_id]:.1f})m")
        axs[i, 1].set_title(f"(x,y) = ({tri.edge_x[edge_id]:.1f}, {tri.edge_y[edge_id]:.1f})m")
    
        for j, case in enumerate(cases):
    
            with open(case + output_file, "rb") as f:
                state = pickle.load(f)
            data_vn = state['vn']
            data_w  = state['w']
    
            axs[i, 0].plot(
                data_w[cell_id, :],
                half_level_heights[cell_id, :],
                color=colors[j],
                linestyle=linestyles[j],
                marker=markers[j],
                markevery=1,
                ms=1,
            )
            axs[i, 1].plot(
                - data_vn[edge_id, :],
                full_level_heights_edges[edge_id, :],
                color=colors[j],
                linestyle=linestyles[j],
                marker=markers[j],
                markevery=1,
                ms=1,
            )
    
        axs[i, 0].set_ylabel("z [m]")
    
    axs[-1, 0].set_xlabel("w   [m/s]")
    axs[-1, 1].set_xlabel("v_n [m/s]")
    # axs[0,0].set_ylim([-1,Z_TOP])
    #axs[0, 0].set_ylim([0, 50])
    axs[0, 0].set_ylim([0, 120])
    # axs[0].legend(fontsize="small")
    plt.draw()
    plt.savefig(imgs_dir + f"/v_profiles_{output_file.split(sep='.')[0]}.png", dpi=600)

