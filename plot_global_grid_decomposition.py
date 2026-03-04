import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
import uxarray as ux
import panel as pn

# grid_filename = "./data/global_grid_R02B02.nc"
grid_filename = "./testdata/grids/r01b01_global/icon_grid_R01B01.nc"

# -------------------------------------------------------------------------------
# Load grid file
#
with open(grid_filename, "rb") as f:
    uxgrid = xr.open_dataset(f)

# -------------------------------------------------------------------------------
# Create matplotlib triangulation
#
v_x = uxgrid.cartesian_x_vertices.values
v_y = uxgrid.cartesian_y_vertices.values
v_z = uxgrid.cartesian_z_vertices.values

c2v = uxgrid.vertex_of_cell.values.T - 1  # transpose and convert to 0-based
v2v = uxgrid.vertices_of_vertex

tri = mpl.tri.Triangulation(
    v_x,
    v_y,
    triangles=c2v,
)

# -------------------------------------------------------------------------------
# Plot 3D
#
fig = plt.figure(2)
plt.clf()
plt.show(block=False)
ax = plt.axes(projection="3d", computed_zorder=False)
ax.plot_trisurf(
    v_x,
    v_y,
    v_z,
    triangles=tri.triangles,
    color="lightgrey",
    linewidth=0.2,
    edgecolor="black",
    alpha=0.99,
    zorder=1,
)
ax.set_aspect("equal")
plt.draw()

# -------------------------------------------------------------------------------
# UXarray
#
uxgrid = ux.open_grid(grid_filename)
hvplot = (
    uxgrid.plot.edges(periodic_elements="ignore", color="black")
    * uxgrid.plot.nodes(marker="o", size=150).relabel("Corner Nodes")
    * uxgrid.plot.face_centers(marker="s", size=150).relabel("Face Centers")
    * uxgrid.plot.edge_centers(marker="^", size=150).relabel("Edge Centers")
).opts(width=1400, height=700, title="Grid Coordinates", legend_position="top_right")

server = pn.panel(hvplot).show()
