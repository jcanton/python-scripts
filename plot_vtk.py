import glob
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import gt4py.next as gtx
import meshio
import numpy as np
from icon4py.model.common.io import plots
from scipy.interpolate import griddata

try:
    from icon4py.model.atmosphere.dycore import ibm

    do_ibm = True
except ImportError:
    do_ibm = False


# -------------------------------------------------------------------------------
#
def export_vtk(tri, half_level_heights: np.ndarray, filename: str, data: dict):
    """
    Export data to a VTK UnstructuredGrid (.vtu, binary) file for ParaView/VisIt.
    Assumes a triangular grid extruded in z to form VTK_WEDGE cells.
    """

    num_vertices = len(tri.x)
    num_cells = len(tri.cell_x)
    num_half_levels = half_level_heights.shape[1]

    # --- Build 3D points array ---
    # tri.x, tri.y are (num_vertices,)
    points = []
    cell_centres = np.column_stack((tri.cell_x, tri.cell_y))
    vertices = np.column_stack((tri.x, tri.y))
    for k in range(num_half_levels):
        z_cells = half_level_heights[:, k]
        z_verts = griddata(cell_centres, z_cells, vertices, method="linear")
        z_verts_fill = griddata(cell_centres, z_cells, vertices, method="nearest")
        z_verts = np.where(np.isnan(z_verts), z_verts_fill, z_verts)
        for i in range(num_vertices):
            points.append([tri.x[i], tri.y[i], z_verts[i]])
    points = np.array(points)

    # --- Build wedge cells ---
    # Each wedge is formed by connecting a triangle at level k and k+1
    wedges = []
    unmasked_cell_indices = []
    mask = getattr(tri, "mask", None)
    for cell_id, (v0, v1, v2) in enumerate(tri.triangles):
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
                if arr_unmasked.shape[1] == num_half_levels - 1:
                    # Data defined on full levels (wedge centers)
                    arr3d = arr_unmasked[:, :]
                else:
                    # Data defined on half levels (wedge triangular faces)
                    # For now, we remove the last (ground) half level
                    # and plot it at wedge centers
                    arr3d = arr_unmasked[:, :-1]
                cell_data[name] = [arr3d.flatten()]
            else:
                raise ValueError(
                    f"Unsupported cell data shape for '{name}': {arr.shape}"
                )
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


def process_file(args):
    # NOTE: plot and _ibm must be created in each process, as they are not picklable.
    file_path, out_path, savepoint_path, grid_file_path = args
    if os.path.exists(out_path):
        return  # Skip if already processed

    with open(file_path, "rb") as f:
        state = pickle.load(f)
    data_rho = state["rho"]
    data_theta_v = state["theta_v"]
    data_exner = state["exner"]
    data_w = state["w"]
    data_vn = state["vn"]
    if "sponge_full_cell" in state:
        data_sponge_fc = state["sponge_full_cell"]
    else:
        data_sponge_fc = None

    plot = plots.Plot(
        savepoint_path=savepoint_path,
        grid_file_path=grid_file_path,
        backend=gtx.gtfn_cpu,
    )
    if do_ibm:
        _ibm = ibm.ImmersedBoundaryMethod(
            grid=plot.grid,
            savepoint_path=savepoint_path,
            grid_file_path=grid_file_path,
            backend=gtx.gtfn_cpu,
        )

    filename = os.path.basename(file_path).split(".")[0]
    print(f"Plotting {filename}")

    u_cf, v_cf = plot._vec_interpolate_to_cell_center(data_vn)
    w_cf = plot._scal_interpolate_to_full_levels(data_w)
    vort_cf = plot._compute_vorticity(data_vn)
    output_dict = {
        "rho": data_rho,
        "theta_v": data_theta_v,
        "exner": data_exner,
        "wind_cf": np.stack([u_cf, v_cf, w_cf], axis=-1),
        "vort_cf": vort_cf,
    }
    if do_ibm:
        output_dict["cell_mask"] = _ibm.full_cell_mask.asnumpy().astype(float)
    if data_sponge_fc is not None:
        output_dict["sponge_fc"] = data_sponge_fc

    export_vtk(
        tri=plot.tri,
        half_level_heights=plot.half_level_heights,
        filename=out_path,
        data=output_dict,
    )


# ===============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(
            "Usage: python plot_vtk.py <num_workers> <python_files_dir> <savepoint_path> <grid_file_path>"
        )
        sys.exit(1)
    num_workers = int(sys.argv[1])
    output_files_dir = sys.argv[2]
    savepoint_path = sys.argv[3]
    grid_file_path = sys.argv[4]

    vtks_dir = os.path.join(output_files_dir, "vtks")
    if not os.path.exists(vtks_dir):
        os.makedirs(vtks_dir)

    output_files = glob.glob(
        os.path.join(output_files_dir, "??????_end_of_timestep_*pkl")
    )
    output_files += glob.glob(
        os.path.join(output_files_dir, "?????_initial_condition*.pkl")
    )
    output_files += glob.glob(os.path.join(output_files_dir, "?????_debug_*.pkl"))
    if len(output_files) == 0:
        output_files = glob.glob(
            os.path.join(output_files_dir, "end_of_timestep_*.pkl")
        )
    output_files.sort()

    print("")
    print(f"Using {num_workers} workers")
    print(f"Output files directory: {output_files_dir}")
    print(f"Savepoint path: {savepoint_path}")
    print(f"Grid file path: {grid_file_path}")
    print(f"Found {len(output_files)} output files in {output_files_dir}")
    print("")

    fileLabel = lambda file_path: file_path.split("/")[-1].split(".")[0][7:]

    # Prepare arguments for each file
    tasks = []
    for file_path in output_files:
        filename = fileLabel(file_path)
        out_path = os.path.join(vtks_dir, f"{filename}.vtu")
        tasks.append((file_path, out_path, savepoint_path, grid_file_path))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_file, tasks))
