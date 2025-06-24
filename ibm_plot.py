import os, glob
import pickle
import importlib
import xarray as xr

import gt4py.next as gtx
import matplotlib.pyplot as plt
import numpy as np

from plot_vtk import export_vtk

from icon4py.model.common.io import plots
from icon4py.model.atmosphere.dycore import ibm

QSCALE = 50
PEVERY = 1

F_OR_P = 'p'
EXPORT_VTKS = True

hill_x = 500.0
hill_y = 500.0
hill_height = 100.0
hill_width  = 100.0
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
# Data
#
if F_OR_P == 'f':
    #savepoint_path = "/capstor/scratch/cscs/jcanton/ser_data/exclaim_gauss3d.uniform100_hill100x100/ser_data/"
    #savepoint_path = "/capstor/scratch/cscs/jcanton/ser_data/exclaim_gauss3d.grid10_uniform100_hill100x100/ser_data/"
    #fortran_files_dir = "/capstor/scratch/cscs/jcanton/plot_data/exclaim_gauss3d.uniform100_hill100x100_Fr022/"
    #imgs_dir="runf3_hill100x100_nlev100_Fr022"
    savepoint_path = "/scratch/mch/jcanton/ser_data/exclaim_gauss3d.uniform100_hill100x100/ser_data/"
    fortran_files_dir = "/scratch/mch/jcanton/icon-exclaim/icon-exclaim/build_acc/experiments/exclaim_gauss3d.uniform100_hill100x100/"
    imgs_dir="runf1_hill100x100_nlev100"
elif F_OR_P == 'p':
    savepoint_path = os.path.join(os.environ["SCRATCH"], "ser_data/exclaim_gauss3d.uniform100_flat/ser_data/")
    run_dir = os.path.join(os.environ["SCRATCH"], "icon4py/")
    #
    #run_name = "run03_hill100x100_nlev800/"
    run_name = "run62_barray_4x4_nlev800_pert/"
    #
    imgs_dir=run_name

if not os.path.exists(imgs_dir):
    os.makedirs(imgs_dir)

plot = plots.Plot(
    savepoint_path=savepoint_path,
    grid_file_path=grid_file_path,
    backend=gtx.gtfn_cpu,
)
importlib.reload(ibm)
_ibm = ibm.ImmersedBoundaryMethod(
    grid=plot.grid,
    savepoint_path=savepoint_path,
    grid_file_path=grid_file_path,
    backend=gtx.gtfn_cpu,
)

# -------------------------------------------------------------------------------
# Load plot data

if F_OR_P == 'f':
    output_files = glob.glob(os.path.join(fortran_files_dir, 'exclaim_gauss3d_sb_insta*.nc'))
    output_files.sort()
elif F_OR_P == 'p':
    python_files_dir = run_dir + run_name
    output_files = glob.glob(os.path.join(python_files_dir, 'end_of_timestep_*.pkl'))
    output_files.sort()

fileLabel = lambda file_path: file_path.split('/')[-1].split('.')[0]

# -------------------------------------------------------------------------------
# Plot
#

for file_path in output_files:

    filename = fileLabel(file_path)

    if F_OR_P == 'f':
        ds = xr.open_dataset(file_path)
        data_rho     = ds.rho.values[0,:,:].T
        data_theta_v = ds.theta_v.values[0,:,:].T
        data_exner   = ds.pres.values[0,:,:].T
        data_w       = ds.w.values[0,:,:].T
        data_vn      = ds.vn.values[0,:,:].T

    elif F_OR_P == 'p':
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        data_rho     = state["rho"]
        data_theta_v = state["theta_v"]
        data_exner   = state["exner"]
        data_w       = state['w']
        data_vn      = state['vn']

    print(f"Plotting {filename}")

    if EXPORT_VTKS:
        u_cf, v_cf = plot._vec_interpolate_to_cell_center(data_vn)
        w_cf = plot._scal_interpolate_to_full_levels(data_w)
        export_vtk(
            tri=plot.tri,
            half_level_heights=plot.half_level_heights,
            filename=f"{imgs_dir}/{filename}.vtu",
            data={
                "rho":     data_rho,
                "theta_v": data_theta_v,
                "exner":   data_exner,
                #"vn": data_vn,
                "w": data_w,
                "wind_cf": np.stack([u_cf, v_cf, w_cf], axis=-1),
                "cell_mask": _ibm.full_cell_mask.asnumpy().astype(float),
            }
        )

    axs, x_coords_i, y_coords_i, u_i, w_i, idxs = plot.plot_sections(
        data=data_w,
        sections_x=[],
        sections_y=[500],
        label="w",
        plot_every=PEVERY,
        qscale=QSCALE,
    )
    #axs[0].plot(x, hill, "--", color="black")
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("z [m]")
    plt.draw()
    plt.savefig(f"{imgs_dir}/{filename}_section_w.png", dpi=600)

    #axs, edge_x, edge_y, vn, vt = plot.plot_levels(data_vn, 10, label=f"vvec_edge")
    #axs[0].set_aspect("equal")
    #plt.draw()
    #plt.savefig(f"{imgs_dir}/{filename}_levels_uv.png", dpi=600)
