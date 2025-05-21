import os
import pickle
import xarray as xr

import gt4py.next as gtx
import matplotlib.pyplot as plt
import numpy as np


from icon4py.model.common.io import plots

QSCALE = 50
PEVERY = 1

f_or_p = 'p'

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
run_dir  = "/scratch/l_jcanton/run_data/"
run_name = "run00_hill100x100_nlev100/"
#run_name = "run10_cube100x100x100_nlev400/"

imgs_dir=run_name

if not os.path.exists(imgs_dir):
    os.makedirs(imgs_dir)


# -------------------------------------------------------------------------------
# Serialized data
#
if f_or_p == 'f':
    savepoint_path = "/capstor/scratch/cscs/jcanton/ser_data/exclaim_gauss3d.uniform100_hill100x100/ser_data/"
elif f_or_p == 'p':
    #savepoint_path = "/capstor/scratch/cscs/jcanton/ser_data/exclaim_gauss3d.uniform100_flat/ser_data/"
    savepoint_path = "/scratch/l_jcanton/ser_data/exclaim_gauss3d.uniform100_flat/ser_data/"

plot = plots.Plot(
    savepoint_path=savepoint_path,
    grid_file_path=grid_file_path,
    backend=gtx.gtfn_cpu,
)

# -------------------------------------------------------------------------------
#
def export_vtk(tri, half_level_heights: np.ndarray, filename: str, data: dict):
    """
    Export data to a VTK UnstructuredGrid (.vtu, binary) file for ParaView/VisIt.
    Assumes a triangular grid extruded in z to form VTK_WEDGE cells.
    """
    import meshio
    num_vertices = len(tri.x)
    num_cells = len(tri.cell_x)
    num_half_levels = half_level_heights.shape[1]

    # --- Build 3D points array ---
    # tri.x, tri.y are (num_vertices,)
    points = []
    for k in range(num_half_levels):
        z = half_level_heights[:, k]  # shape: (num_vertices,)
        for i in range(num_vertices):
            points.append([tri.x[i], tri.y[i], z[i]])
    points = np.array(points)

    # --- Build wedge cells ---
    # Each wedge is formed by connecting a triangle at level k and k+1
    wedges = []
    unmasked_cell_indices = []
    mask = getattr(tri, 'mask', None)
    for cell_id, (v0, v1, v2) in enumerate(tri.triangles):  # tri.triangles: (num_cells, 3)
        if mask is not None and mask[cell_id]:
            continue  # skip masked triangles
        unmasked_cell_indices.append(cell_id)
        for k in range(num_half_levels - 1):
            # Compute global vertex indices for bottom and top triangles
            v0b = v0 + k * num_vertices
            v1b = v1 + k * num_vertices
            v2b = v2 + k * num_vertices
            v0t = v0 + (k + 1) * num_vertices
            v1t = v1 + (k + 1) * num_vertices
            v2t = v2 + (k + 1) * num_vertices
            # VTK_WEDGE: [v0b, v1b, v2b, v0t, v1t, v2t]
            wedges.append([v0b, v1b, v2b, v0t, v1t, v2t])
    cells = [("wedge", np.array(wedges))]

    # --- Prepare data arrays ---
    point_data = {}
    cell_data = {}
    for name, arr in data.items():
        arr = np.ascontiguousarray(arr)
        if arr.shape[0] == num_vertices:  # vertex data
            # Flatten to (num_vertices * num_half_levels,)
            point_data[name] = arr.flatten()
        elif arr.shape[0] == num_cells:  # cell-centered data
            # Only include unmasked cells
            arr_unmasked = arr[unmasked_cell_indices, :]
            arr3d = arr_unmasked[:, :-1]  # shape: (num_unmasked_cells, num_half_levels-1)
            cell_data[name] = [arr3d.flatten()]
        else:
            raise ValueError(f"Unsupported data shape for '{name}': {arr.shape}")

    # --- Write to VTU ---
    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
    )
    meshio.write(filename, mesh, file_format="vtu")

# -------------------------------------------------------------------------------
# Load plot data

if f_or_p == 'f':
    fortran_files_dir = "/capstor/scratch/cscs/jcanton/plot_data/exclaim_gauss3d.uniform100_hill100x100/"
    output_files = os.listdir(fortran_files_dir)
    output_files.sort()
elif f_or_p == 'p':
    python_files_dir = run_dir + run_name
    output_files = os.listdir(python_files_dir)
    output_files.sort()


# -------------------------------------------------------------------------------
# Plot
#

for filename in output_files[:1]:

    if f_or_p == 'f':
        if not filename.startswith("exclaim_gauss3d_sb_insta"):
            continue
        ds = xr.open_dataset(fortran_files_dir + filename)
        data_w  = ds.w.values[0,:,:].T
        data_vn = ds.vn.values[0,:,:].T

    elif f_or_p == 'p':
        if not filename.endswith(".pkl"):
            continue
        with open(python_files_dir + filename, "rb") as f:
            state = pickle.load(f)
        data_w  = state['w']
        data_vn = state['vn']


    print(f"Plotting {filename}")

    export_vtk(
        tri=plot.tri,
        half_level_heights=plot.half_level_heights,
        filename=f"{filename[:-4]}.vtu",
        data={
            #"vn": data_vn,
            "w": data_w,
        }
    )
    #axs, x_coords_i, y_coords_i, u_i, w_i, idxs = plot.plot_sections(
    #    data=data_w,
    #    sections_x=[],
    #    sections_y=[500],
    #    label="w",
    #    plot_every=PEVERY,
    #    qscale=QSCALE,
    #)
    ##axs[0].plot(x, hill, "--", color="black")
    #axs[0].set_aspect("equal")
    #axs[0].set_xlabel("x [m]")
    #axs[0].set_ylabel("z [m]")
    #plt.draw()
    #plt.savefig(f"{imgs_dir}/{filename[:-4]}_section_w.png", dpi=600)

    #axs, edge_x, edge_y, vn, vt = plot.plot_levels(data_vn, 10, label=f"vvec_edge")
    #axs[0].set_aspect("equal")
    #plt.draw()
    #plt.savefig(f"{imgs_dir}/{filename[:-4]}_levels_uv.png", dpi=600)
