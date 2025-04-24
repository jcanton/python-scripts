import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

grid_filename = "./data/global_grid_R02B02.nc"

#-------------------------------------------------------------------------------
# Load grid file
#
with open(grid_filename, "rb") as f:
    grid = xr.open_dataset(f)

#-------------------------------------------------------------------------------
# Create matplotlib triangulation
#
v_x = grid.cartesian_x_vertices.values
v_y = grid.cartesian_y_vertices.values
v_z = grid.cartesian_z_vertices.values

c2v = grid.vertex_of_cell.values.T - 1 # transpose and convert to 0-based
v2v = grid.vertices_of_vertex

tri = mpl.tri.Triangulation(
    v_x,
    v_y,
    triangles = c2v,
)

#-------------------------------------------------------------------------------
# Identify pentagons
#
eov = [i for x, i in zip(v2v[5,:], range(len(v2v[5,:]))) if x == -1]

#-------------------------------------------------------------------------------
# Plot
#

# # 2D, annoying projection and "wrapping triangles"
# fig = plt.figure(1); plt.clf(); plt.show(block=False)
# ax = fig.subplots(nrows=1, ncols=1)
# ax.triplot(tri, color='k', linewidth=0.25)
# ax.plot(tri.x,      tri.y,      'vr')
# ax.set_aspect("equal")
# plt.draw()

# 3D
fig = plt.figure(2); plt.clf(); plt.show(block=False)
ax = plt.axes(projection ='3d', computed_zorder=False)
ax.plot_trisurf(v_x, v_y, v_z,
                triangles=tri.triangles,
                color="lightgrey",
                linewidth=.2,
                edgecolor="black",
                alpha=0.99,
                zorder=1,
                )
ax.scatter(v_x[eov], v_y[eov], v_z[eov], marker='o', color='red', s=100, zorder=10)
ax.set_aspect("equal")
plt.draw()
