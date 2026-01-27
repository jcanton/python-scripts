import matplotlib.pyplot as plt
import numpy as np
import pymetis
import xarray as xr

import helpers

num_mpi_ranks = 4
num_halo_levels = 2


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
part_count, c2rank_map = pymetis.part_graph(nparts=num_mpi_ranks, adjacency=c2e2c)
c2rank_map = np.array(c2rank_map)

# Plot partitioned grid
helpers.plot_partitioned_grid(tri, c2rank_map, label_ranks=True, show_centers=True)

# Construct halo regions
def _find_halo_cells(boundary_cells, c2e2c, c2rank_map, mpi_rank, exclude):
    """Find cells neighboring boundary_cells that belong to a different rank."""
    neighbor_cells = c2e2c[boundary_cells]  # shape (num_boundary, 3)
    neighbors_flat = neighbor_cells.flatten()
    halo_mask = c2rank_map[neighbors_flat] != mpi_rank
    halo_cells = neighbors_flat[halo_mask]
    halo_cells = halo_cells[~np.isin(halo_cells, exclude)]
    return np.unique(halo_cells)

subdomains = []

for mpi_rank in range(num_mpi_ranks):
    owned_cells = np.where(c2rank_map == mpi_rank)[0]
    halo_levels = []
    boundary = owned_cells.copy()

    for level in range(num_halo_levels):
        exclude = np.concatenate([owned_cells] + halo_levels) if halo_levels else owned_cells
        new_halo = _find_halo_cells(boundary, c2e2c, c2rank_map, mpi_rank, exclude)
        halo_levels.append(new_halo)
        boundary = new_halo

    all_cells = np.concatenate([owned_cells] + halo_levels) if halo_levels else owned_cells
    subdomains.append({
        "rank": mpi_rank,
        "owned_cells": owned_cells,
        "halo_levels": halo_levels,
        "all_cells": all_cells,
    })

# Plot each subdomain with owned/halo coloring
helpers.plot_subdomains(tri, subdomains, cols=2, show_edges=True, show_masked=True, show_points=True, print_indexes=True)

plt.show()