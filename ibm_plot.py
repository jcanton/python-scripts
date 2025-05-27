import os
import pickle
import xarray as xr

import gt4py.next as gtx
import matplotlib.pyplot as plt
import numpy as np


from icon4py.model.common.io import plots
from icon4py.model.atmosphere.dycore import ibm

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



# -------------------------------------------------------------------------------
# Data
#
if f_or_p == 'f':
    savepoint_path = "/capstor/scratch/cscs/jcanton/ser_data/exclaim_gauss3d.uniform100_hill100x100/ser_data/"
    #savepoint_path = "/capstor/scratch/cscs/jcanton/ser_data/exclaim_gauss3d.grid10_uniform100_hill100x100/ser_data/"
    fortran_files_dir = "/capstor/scratch/cscs/jcanton/plot_data/exclaim_gauss3d.uniform100_hill100x100_Fr022/"
    imgs_dir="runf3_hill100x100_nlev100_Fr022"
elif f_or_p == 'p':
    #savepoint_path = "/capstor/scratch/cscs/jcanton/ser_data/exclaim_gauss3d.uniform800_flat/ser_data/"
    savepoint_path = "/scratch/mch/jcanton/ser_data/exclaim_gauss3d.uniform800_flat/ser_data/"
    #savepoint_path = "/scratch/l_jcanton/ser_data/exclaim_gauss3d.uniform200_flat/ser_data/"
    #
    #run_dir = "/capstor/scratch/cscs/jcanton/icon4py/"
    run_dir = "/scratch/mch/jcanton/icon4py/"
    #run_dir  = "/scratch/l_jcanton/run_data/"
    #
    run_name = "run61_barray_2x2_nlev800_noSlip/"
    #run_name = "run69_barray_1x0_nlev200_wholeDomain/"
    #
    imgs_dir=run_name

if not os.path.exists(imgs_dir):
    os.makedirs(imgs_dir)

plot = plots.Plot(
    savepoint_path=savepoint_path,
    grid_file_path=grid_file_path,
    backend=gtx.gtfn_cpu,
)
_ibm = ibm.ImmersedBoundaryMethod(
    grid=plot.grid,
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
    from scipy.interpolate import griddata

    num_vertices = len(tri.x)
    num_cells = len(tri.cell_x)
    num_half_levels = half_level_heights.shape[1]

    # --- Build 3D points array ---
    # tri.x, tri.y are (num_vertices,)
    points = []
    cell_centres = np.column_stack((tri.cell_x, tri.cell_y))
    vertices = np.column_stack((tri.x, tri.y))  # shape (num_vertices, 2)
    for k in range(num_half_levels):
        z_cells = half_level_heights[:,k]  # shape (num_cells,)
        z_verts = griddata(cell_centres, z_cells, vertices, method='linear')
        z_verts_fill = griddata(cell_centres, z_cells, vertices, method='nearest')
        z_verts = np.where(np.isnan(z_verts), z_verts_fill, z_verts)
        for i in range(num_vertices):
            points.append([tri.x[i], tri.y[i], z_verts[i]])
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
            # Handle vector fields (shape: (num_cells, num_levels, 3))
            if arr_unmasked.ndim == 3 and arr_unmasked.shape[2] == 3:
                # Data defined on full levels (wedge centers)
                arr3d = arr_unmasked[:, :, :]
                # Flatten to (num_cells * num_levels, 3)
                cell_data[name] = [arr3d.reshape(-1, 3)]
            elif arr_unmasked.ndim == 2:
                if arr_unmasked.shape[1] == num_half_levels-1:
                    # Data defined on full levels (wedge centers)
                    arr3d = arr_unmasked[:, :]
                else:
                    # Data defined on half levels (wedge triangular faces)
                    # For now, we remove the last (ground) half level
                    # and plot it at wedge centers
                    arr3d = arr_unmasked[:, :-1]
                cell_data[name] = [arr3d.flatten()]
            else:
                raise ValueError(f"Unsupported cell data shape for '{name}': {arr.shape}")
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
    output_files = os.listdir(fortran_files_dir)
    output_files.sort()
elif f_or_p == 'p':
    python_files_dir = run_dir + run_name
    output_files = os.listdir(python_files_dir)
    output_files.sort()


# -------------------------------------------------------------------------------
# Plot
#

for filename in output_files[10:20]:

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

    u_cf, v_cf = plot._vec_interpolate_to_cell_center(data_vn)
    w_cf = plot._scal_interpolate_to_full_levels(data_w)
    export_vtk(
        tri=plot.tri,
        half_level_heights=plot.half_level_heights,
        filename=f"{imgs_dir}/{filename.split(sep='.')[0]}.vtu",
        data={
            #"vn": data_vn,
            "w": data_w,
            "wind_cf": np.stack([u_cf, v_cf, w_cf], axis=-1),
            "cell_mask": _ibm.full_cell_mask.asnumpy().astype(float),
        }
    )

    axs, x_coords_i, y_coords_i, u_i, w_i, idxs = plot.plot_sections(
        data=data_w,
        sections_x=[],
        sections_y=[250],
        label="w",
        plot_every=PEVERY,
        qscale=QSCALE,
    )
    #axs[0].plot(x, hill, "--", color="black")
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("z [m]")
    plt.draw()
    plt.savefig(f"{imgs_dir}/{filename.split(sep='.')[0]}_section_w.png", dpi=600)

    #axs, edge_x, edge_y, vn, vt = plot.plot_levels(data_vn, 10, label=f"vvec_edge")
    #axs[0].set_aspect("equal")
    #plt.draw()
    #plt.savefig(f"{imgs_dir}/{filename[:-4]}_levels_uv.png", dpi=600)
