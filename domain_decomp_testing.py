import numpy as np
import pymetis
import xarray as xr

import helpers

num_partitions = 4


grid_fpath = "./testdata/grids/torus_1000x1000_res250/Torus_Triangles_1000m_x_1000m_res250m.nc"
       #"./testdata/grids/r02b04_global/icon_grid_0013_R02B04_R.nc"

grid_data = helpers.read_gridfile(grid_fpath)
tri = helpers.create_triangulation(**grid_data)
helpers.plot_grid(tri)

grid = xr.open_dataset(grid_fpath)

# Build required connectivity
cells = grid.cell_index.values - 1
c2v = grid.vertex_of_cell.values.T -1
c2e = grid.edge_of_cell.values.T -1
e2c = grid.adjacent_cell_of_edge.values.T -1
adj_cells = e2c[c2e, :] # shape (num_cells, 3, 2)
c2e2c = np.where(adj_cells[:,:,0] == cells[:,None], adj_cells[:,:,1], adj_cells[:,:,0]) # remove starting cells

# Partition grid
part_count, c2rank_mapping = pymetis.part_graph(nparts=num_partitions, adjacency=c2e2c)
c2rank_mapping = np.array(c2rank_mapping)


# Plot partitioned grid
helpers.plot_partitioned_grid(tri, c2rank_mapping, label_ranks=True, show_centers=True)