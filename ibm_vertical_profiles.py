import os
import pickle

import gt4py.next as gtx
import matplotlib.pyplot as plt
import numpy as np

from icon4py.model.common.io import plots
from scipy.interpolate import griddata

# -------------------------------------------------------------------------------
#
main_dir = os.getcwd() + "/../icon4py/"
grid_file_path = main_dir + "testdata/grids/gauss3d_torus/Torus_Triangles_1000m_x_1000m_res10m.nc"

#savepoint_path = "/capstor/scratch/cscs/jcanton/ser_data/exclaim_gauss3d.uniform200_flat/ser_data/"
savepoint_path = "/scratch/mch/jcanton/ser_data/exclaim_gauss3d.uniform200_flat/ser_data/"
#imgs_dir = "run60_barray_2x2_nlev800"
#imgs_dir = "runxx_ibm_check_slip_noslip"
#savepoint_path = "/scratch/mch/jcanton/ser_data/exclaim_gauss3d.uniform200_flat/ser_data/"
#imgs_dir = "runxx_ibm_check_slip_noslip"

plot = plots.Plot(
    savepoint_path=savepoint_path,
    grid_file_path=grid_file_path,
    backend=gtx.gtfn_cpu,
)
tri = plot.tri
full_level_heights = plot.full_level_heights
half_level_heights = plot.half_level_heights
# compute z coordinates for edge centres
cell_centres = np.column_stack((tri.cell_x, tri.cell_y))
edges = np.column_stack((tri.edge_x, tri.edge_y))
full_level_heights_edges = np.zeros((len(tri.edge_x), full_level_heights.shape[1]))
for k in range(full_level_heights.shape[1]):
    z_cells = full_level_heights[:,k]  # shape (num_cells,)
    z_edges = griddata(cell_centres, z_cells, edges, method='linear')
    z_edges_fill = griddata(cell_centres, z_cells, edges, method='nearest')
    full_level_heights_edges[:,k] = np.where(np.isnan(z_edges), z_edges_fill, z_edges)

cases = [
    #main_dir + "run61_barray_2x2_nlev800_flatFaces/",
    main_dir + "run62_barray_4x4_nlev200_flatFaces/",
]
imgs_dir=cases[0].split('/')[-2]
output_files = [
    #"end_of_timestep_013600.pkl",
    #"end_of_timestep_013700.pkl",
    #"end_of_timestep_013800.pkl",
    #"end_of_timestep_013900.pkl",
    #"end_of_timestep_014000.pkl",
    #"end_of_timestep_014100.pkl",
    #
    #"end_of_timestep_033600.pkl"
    "avg_state_end_of_timestep_036000-end_of_timestep_180000.pkl",
    "avg_state_end_of_timestep_180000-end_of_timestep_324000.pkl",
]
mac_data=np.loadtxt('macdonald_2000_cube_arrays.csv', delimiter=',', skiprows=2)

if not os.path.exists(imgs_dir):
    os.makedirs(imgs_dir)

# -------------------------------------------------------------------------------
# vert profiles
#
markers = ["+", "x", "d", "o"]
linestyles = ["-", "--", ":", "-."]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
if "2x2" in cases[0]:
    exp_data = [mac_data[:,0:2]]
    profiles = [ [350, 500] ]
elif "4x4" in cases[0]:
    exp_data = [mac_data[:,2:4],mac_data[:,4:]]
    profiles = [ [225, 250] ]
else:
    exp_data = []
    profiles = []
    print("no automatic case selected")

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
        #axs[i, 1].set_title(f"(x,y) = ({tri.edge_x[edge_id]:.1f}, {tri.edge_y[edge_id]:.1f})m")
        #axs[i, 2].set_title(f"(x,y) = ({tri.edge_x[edge_id]:.1f}, {tri.edge_y[edge_id]:.1f})m")
        #axs[i, 3].set_title(f"(x,y) = ({tri.cell_x[cell_id]:.1f}, {tri.cell_y[cell_id]:.1f})m")
        axs[i, 1].set_title(f"(x,y) = ({tri.cell_x[cell_id]:.1f}, {tri.cell_y[cell_id]:.1f})m")
    
        for j, case in enumerate(cases):
    
            with open(case + output_file, "rb") as f:
                state = pickle.load(f)
            data_vn = state['vn']
            data_w  = state['w']
            data_vt = plot._compute_vt(data_vn)
            data_u_cf, data_v_cf = plot._vec_interpolate_to_cell_center(data_vn)
            #data_u = data_vn[:, -1-i]*plot.primal_normal[0] + data_vt[:, -1-i]*plot.primal_tangent[0]
    
            axs[i, 0].plot(
                data_w[cell_id, :],
                half_level_heights[cell_id, :],
                color=colors[j],
                linestyle=linestyles[j],
                marker=markers[j],
                markevery=1,
                ms=1,
            )
            #axs[i, 1].plot(
            #    data_vn[edge_id, :],
            #    full_level_heights_edges[edge_id, :],
            #    color=colors[j],
            #    linestyle=linestyles[j],
            #    marker=markers[j],
            #    markevery=1,
            #    ms=1,
            #)
            #axs[i, 2].plot(
            #    data_vn[edge_id, :]*plot.primal_normal[0][edge_id] + data_vt[edge_id, :]*plot.primal_normal[0][edge_id],
            #    full_level_heights_edges[edge_id, :],
            #    color=colors[j],
            #    linestyle=linestyles[j],
            #    marker=markers[j],
            #    markevery=1,
            #    ms=1,
            #)
            #axs[i, 3].plot(
            #    data_u_cf[cell_id, :],
            #    full_level_heights[cell_id, :],
            #    color=colors[j],
            #    linestyle=linestyles[j],
            #    marker=markers[j],
            #    markevery=1,
            #    ms=1,
            #)
            axs[i, 1].plot(
                (data_u_cf[cell_id, :-1] + data_u_cf[cell_id, 1:])/2,
                half_level_heights[cell_id, 1:-1],
                color=colors[j],
                linestyle=linestyles[j],
                marker=markers[j],
                markevery=1,
                ms=1,
            )
            for iexp, expp in enumerate(exp_data):
                axs[i, 1].plot(
                    expp[:,0] / 2,
                    expp[:,1] * 100,
                    color='black',
                    linestyle='',
                    marker=markers[iexp],
                    markevery=1,
                    ms=4,
                )
    
        axs[i, 0].set_ylabel("z [m]")
    
    axs[-1, 0].set_xlabel("w   [m/s]")
    #axs[-1, 1].set_xlabel("v_n [m/s]")
    #axs[-1, 2].set_xlabel("u_e [m/s]")
    #axs[-1, 3].set_xlabel("u_c [m/s]")
    axs[-1, 1].set_xlabel("u_c [m/s]")
    # axs[0,0].set_ylim([-1,Z_TOP])
    #axs[0, 0].set_ylim([0, 50])
    # axs[0].legend(fontsize="small")
    axs[0, 0].set_ylim([0, 500]); plt.draw()
    plt.savefig(imgs_dir + f"/v_profiles_{output_file.split(sep='.')[0]}_500m_exp.png", dpi=600)
    axs[0, 0].set_ylim([0, 300]); plt.draw()
    plt.savefig(imgs_dir + f"/v_profiles_{output_file.split(sep='.')[0]}_300m_exp.png", dpi=600)
    axs[0, 0].set_ylim([0, 150]); plt.draw()
    plt.savefig(imgs_dir + f"/v_profiles_{output_file.split(sep='.')[0]}_150m_exp.png", dpi=600)

