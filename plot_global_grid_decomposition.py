import numpy as np
import pyvista as pv
import xarray as xr

import helpers

# grid_filename = "./data/global_grid_R02B02.nc"
grid_filename = "./testdata/grids/r01b01_global/icon_grid_R01B01.nc"

# -------------------------------------------------------------------------------
# Load grid file
#
with open(grid_filename, "rb") as f:
    grid = xr.open_dataset(f)

# -------------------------------------------------------------------------------
# Create PyVista mesh
#
v_x = grid.cartesian_x_vertices.values
v_y = grid.cartesian_y_vertices.values
v_z = grid.cartesian_z_vertices.values

c_x, c_y, c_z = helpers.lonlat2cart(grid.clon, grid.clat)

c2v = grid.vertex_of_cell.values.T - 1  # transpose and convert to 0-based

# Stack vertices into Nx3 array
vertices = np.column_stack([v_x, v_y, v_z])

# Convert triangles to PyVista format: [n_pts, idx0, idx1, idx2, ...]
faces = np.column_stack([np.full(len(c2v), 3), c2v]).flatten()

# Create mesh
mesh = pv.PolyData(vertices, faces)

# -------------------------------------------------------------------------------
# Plot with PyVista
#
plotter = pv.Plotter()
plotter.add_mesh(
    mesh,
    color="lightgrey",
    edge_color="black",
    line_width=0.2,
    show_edges=True,
)

plotter.add_points(
    np.column_stack([c_x, c_y, c_z]),
    color="red",
    point_size=20,
)

# Add labels
for i, (x, y, z) in enumerate(zip(c_x, c_y, c_z)):
    plotter.add_point_labels(
        [(x, y, z)],
        [str(i)],
        font_size=30,
        text_color="red",
    )

plotter.show()
