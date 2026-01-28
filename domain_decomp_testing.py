import matplotlib.pyplot as plt
import numpy as np
import pymetis
import xarray as xr

import helpers

num_mpi_ranks = 4
num_halo_rows = 1  # Each row contains 2 levels: c2e2c and c2v2c


grid_fpath = "./testdata/grids/torus_1000x1000_res250/Torus_Triangles_1000m_x_1000m_res250m.nc"
#grid_fpath = "./testdata/grids/r02b04_global/icon_grid_0013_R02B04_R.nc"

grid_data = helpers.read_gridfile(grid_fpath)
tri = helpers.create_triangulation(**grid_data)
helpers.plot_grid(tri)

grid = xr.open_dataset(grid_fpath)

# Build required connectivity
cells = grid.cell_index.values -1
c2v = grid.vertex_of_cell.values.T -1
v2c = grid.cells_of_vertex.values.T -1
c2e = grid.edge_of_cell.values.T -1
e2c = grid.adjacent_cell_of_edge.values.T -1
# build c2e2c
adj_cells = e2c[c2e, :] # shape (num_cells, 3, 2)
c2e2c = np.where(adj_cells[:,:,0] == cells[:,None], adj_cells[:,:,1], adj_cells[:,:,0]) # remove starting cells

# Partition grid
part_count, c2rank_map = pymetis.part_graph(nparts=num_mpi_ranks, adjacency=c2e2c)
c2rank_map = np.array(c2rank_map)

# Plot partitioned grid
helpers.plot_partitioned_grid(tri, c2rank_map, label_ranks=True, show_centers=True)

# Construct halo regions
def _find_halo_cells_edge(boundary_cells, c2e2c, c2rank_map, mpi_rank, exclude):
    """Find edge-adjacent cells (via c2e2c) belonging to a different rank."""
    neighbor_cells = c2e2c[boundary_cells]
    neighbors_flat = neighbor_cells.flatten()
    halo_mask = c2rank_map[neighbors_flat] != mpi_rank
    halo_cells = neighbors_flat[halo_mask]
    halo_cells = halo_cells[~np.isin(halo_cells, exclude)]
    return np.unique(halo_cells)

def _find_halo_cells_vertex(boundary_cells, c2v, v2c, c2rank_map, mpi_rank, exclude):
    """Find vertex-adjacent cells by finding boundary vertices and their cells."""
    # Find all vertices of boundary cells
    boundary_vertices = c2v[boundary_cells].flatten()
    boundary_vertices = np.unique(boundary_vertices[boundary_vertices >= 0])
    
    # Find all cells sharing those vertices
    neighbor_cells = v2c[boundary_vertices].flatten()
    neighbor_cells = neighbor_cells[neighbor_cells >= 0]  # filter out invalid (-1)
    
    # Filter by rank and exclusion
    halo_mask = c2rank_map[neighbor_cells] != mpi_rank
    halo_cells = neighbor_cells[halo_mask]
    halo_cells = halo_cells[~np.isin(halo_cells, exclude)]
    return np.unique(halo_cells)

subdomains = []

for mpi_rank in range(num_mpi_ranks):
    owned_cells = np.where(c2rank_map == mpi_rank)[0]
    halo_levels = []
    boundary = owned_cells.copy()

    for row in range(num_halo_rows):
        exclude = np.concatenate([owned_cells] + halo_levels) if halo_levels else owned_cells
        
        # Level 1 of this row: edge-adjacent neighbors
        level1 = _find_halo_cells_edge(boundary, c2e2c, c2rank_map, mpi_rank, exclude)
        halo_levels.append(level1)
        
        # Level 2 of this row: vertex-adjacent neighbors, minus level1
        exclude_with_level1 = np.concatenate([exclude, level1]) if len(level1) > 0 else exclude
        level2 = _find_halo_cells_vertex(boundary, c2v, v2c, c2rank_map, mpi_rank, exclude_with_level1)
        halo_levels.append(level2)
        
        # Next row's boundary is this row's complete halo (level1 + level2)
        boundary = np.concatenate([level1, level2]) if len(level1) > 0 or len(level2) > 0 else np.array([], dtype=int)

    all_cells = np.concatenate([owned_cells] + halo_levels) if halo_levels else owned_cells
    subdomains.append({
        "rank": mpi_rank,
        "owned_cells": owned_cells,
        "halo_levels": halo_levels,
        "all_cells": all_cells,
    })

# Plot each subdomain with owned/halo coloring
helpers.plot_subdomains(tri, subdomains, cols=2, show_edges=True, show_masked=True, show_points=True, print_indexes=True)

plt.show(block=False)