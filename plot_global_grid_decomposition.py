import pickle

import numpy as np
import pyvista as pv
import xarray as xr

import helpers

# -------------------------------------------------------------------------------
# Configuration
#
FONT_SIZE = 30

# grid_filename = "./data/global_grid_R02B02.nc"
grid_filename = "./testdata/grids/r01b01_global/icon_grid_R01B01.nc"
# grid_filename = "./testdata/grids/r02b04_global/icon_grid_0013_R02B04_R.nc"

# -------------------------------------------------------------------------------
# Load grid file
#
with open(grid_filename, "rb") as f:
    grid = xr.open_dataset(f)

# -------------------------------------------------------------------------------
# Create PyVista mesh
#
try:
    v_x = grid.cartesian_x_vertices.values
    v_y = grid.cartesian_y_vertices.values
    v_z = grid.cartesian_z_vertices.values
except AttributeError:
    v_x, v_y, v_z = helpers.lonlat2cart(grid.vlon, grid.vlat)

c_x, c_y, c_z = helpers.lonlat2cart(grid.clon, grid.clat)

c2v = grid.vertex_of_cell.values.T - 1  # transpose and convert to 0-based

# Stack vertices into Nx3 array
vertices = np.column_stack([v_x, v_y, v_z])

# Convert triangles to PyVista format: [n_pts, idx0, idx1, idx2, ...]
faces = np.column_stack([np.full(len(c2v), 3), c2v]).flatten()

# Create base mesh
base_mesh = pv.PolyData(vertices, faces)

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# Plot global grid (separate window)
#
global_plotter = pv.Plotter(title="Global Grid")
global_plotter.add_mesh(
    base_mesh,
    color="lightgrey",
    edge_color="black",
    line_width=0.2,
    show_edges=True,
)

# global_plotter.add_points(
#     np.column_stack([c_x, c_y, c_z]),
#     color="red",
#     point_size=20,
# )

# # Add labels
# for i, (x, y, z) in enumerate(zip(c_x, c_y, c_z)):
#     global_plotter.add_point_labels(
#         [(x, y, z)],
#         [str(i)],
#         font_size=FONT_SIZE,
#         text_color="black",
#         always_visible=False,
#     )

#global_plotter.show()

# -------------------------------------------------------------------------------
# Create plotter with 2x2 subplots for rank decompositions
#
plotter = pv.Plotter(shape=(2, 2), title="Rank Decomposition")

# -------------------------------------------------------------------------------
# Partitioned grid
#
ranked_dumps = [
    "../icon4py/rank0_debug.pkl",
    "../icon4py/rank1_debug.pkl",
    "../icon4py/rank2_debug.pkl",
    "../icon4py/rank3_debug.pkl",
]

for idx, rank_file in enumerate(ranked_dumps):
    # Load partition data
    with open(rank_file, "rb") as f:
        partition = pickle.load(f)

    owned_cells = partition["owned_cells"]
    first_halo_cells = partition["first_halo_cells"]
    second_halo_cells = partition["second_halo_cells"]

    rank_num = int(rank_file.split("rank")[1].split("_")[0])

    # Create a scalar array for coloring: 0=unowned, 1=owned, 2=halo
    cell_colors = np.zeros(len(c2v), dtype=int)
    cell_colors[owned_cells] = 1
    cell_colors[first_halo_cells] = 2
    cell_colors[second_halo_cells] = 3

    # Create the mesh
    mesh = pv.PolyData(vertices, faces)
    mesh["cell_type"] = cell_colors

    # Plot in appropriate subplot (2x2 grid)
    row = idx // 2
    col = idx % 2
    plotter.subplot(row, col)
    plotter.add_title(f"Rank {rank_num}", font_size=FONT_SIZE)

    plotter.add_mesh(
        mesh,
        scalars="cell_type",
        cmap=["lightgrey", "blue", "orange", "green"],
        edge_color="black",
        line_width=0.2,
        show_edges=True,
        scalar_bar_args={"title": "Cell Type", "label_font_size": FONT_SIZE},
    )

    # Add cell center labels for owned and halo cells
    for cell_idx in owned_cells:
        plotter.add_point_labels(
            [(c_x[cell_idx], c_y[cell_idx], c_z[cell_idx])],
            [str(cell_idx)],
            font_size=FONT_SIZE,
            text_color="black",
            always_visible=False,
        )

    for cell_idx in first_halo_cells:
        plotter.add_point_labels(
            [(c_x[cell_idx], c_y[cell_idx], c_z[cell_idx])],
            [str(cell_idx)],
            font_size=FONT_SIZE,
            text_color="black",
            always_visible=False,
        )

# Link cameras across all subplots to sync rotation and zoom
plotter.link_views()

plotter.show()
