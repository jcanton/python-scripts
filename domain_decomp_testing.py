import numpy as np
import xarray as xr
from icon4py.model.common.grid import simple as simple_grid

import helpers


#grid = simple_grid.simple_grid() # only connectivities, no coordinates

grid_fpath = "./testdata/grids/torus_1000x1000_res250/Torus_Triangles_1000m_x_1000m_res250m.nc"
       #"./testdata/grids/r02b04_global/icon_grid_0013_R02B04_R.nc"

tri = helpers.create_triangulation(grid_fpath)
helpers.plot_grid(tri)

grid = xr.open_dataset(grid_fpath)

cells = grid.cell_index.values - 1
edges = grid.edge_index.values - 1

num_cells = cells.size
num_edges = edges.size

c2v = grid.vertex_of_cell.values.T -1
c2e = grid.edge_of_cell.values.T -1
e2c = grid.adjacent_cell_of_edge.values.T -1

adj_cells = e2c[c2e, :] # shape (num_cells, 3, 2)
c2e2c = np.where(adj_cells[:,:,0] == cells[:,None], adj_cells[:,:,1], adj_cells[:,:,0]) # remove starting cells
