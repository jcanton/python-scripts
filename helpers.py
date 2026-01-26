import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from icon4py.model.common.grid.base import GeometryType


def plot_grid(tri: mpl.tri.Triangulation, ax=None, print_indexes: bool = True, plot_masked: bool = True) -> None:
    if ax is None:
        fig = plt.figure(1)
        plt.show(block=False)
        ax = fig.subplots(nrows=1, ncols=1)
    ax.triplot(tri, color="k", linewidth=0.25)
    ax.plot(tri.x, tri.y, "vr")
    ax.plot(tri.edge_x, tri.edge_y, "sg")
    ax.plot(tri.cell_x, tri.cell_y, "ob")
    ax.set_aspect("equal")
    if print_indexes:
        for i, (x, y) in enumerate(zip(tri.x*1.05, tri.y)):
            plt.text(x, y, str(i), color="r", fontsize=10)
        for i, (x, y) in enumerate(zip(tri.edge_x*1.05, tri.edge_y)):
            plt.text(x, y, str(i), color="g", fontsize=10)
        for i, (x, y) in enumerate(zip(tri.cell_x*1.05, tri.cell_y)):
            plt.text(x, y, str(i), color="b", fontsize=10)
    
    # Plot masked (boundary) triangles with dashed lines
    if plot_masked and tri.mask is not None:
        masked_indices = np.where(tri.mask)[0]
        for idx in masked_indices:
            triangle = tri.triangles[idx]
            x_coords = tri.x[triangle]
            y_coords = tri.y[triangle]
            
            # Get the triangle center to determine which side to plot on
            center_x = tri.cell_x[idx]
            center_y = tri.cell_y[idx]
            
            # Duplicate vertices and shift them to be on the same side as the center
            x_shifted = x_coords.copy()
            y_shifted = y_coords.copy()
            
            # Check if triangle wraps in x direction
            if np.max(x_coords) - np.min(x_coords) > tri.domain_length / 2:
                if center_x < tri.domain_length / 2:
                    # Center is on the left, shift high x values down
                    x_shifted = np.where(x_coords > tri.domain_length / 2, 
                                        x_coords - tri.domain_length, 
                                        x_coords)
                else:
                    # Center is on the right, shift low x values up
                    x_shifted = np.where(x_coords < tri.domain_length / 2, 
                                        x_coords + tri.domain_length, 
                                        x_coords)
            
            # Check if triangle wraps in y direction  
            if np.max(y_coords) - np.min(y_coords) > tri.domain_height / 2:
                if center_y < tri.domain_height / 2:
                    # Center is on the bottom, shift high y values down
                    y_shifted = np.where(y_coords > tri.domain_height / 2, 
                                        y_coords - tri.domain_height, 
                                        y_coords)
                else:
                    # Center is on the top, shift low y values up
                    y_shifted = np.where(y_coords < tri.domain_height / 2, 
                                        y_coords + tri.domain_height, 
                                        y_coords)
            
            # Close the triangle loop and plot with dashed lines
            x_plot = np.append(x_shifted, x_shifted[0])
            y_plot = np.append(y_shifted, y_shifted[0])
            ax.plot(x_plot, y_plot, 'k--', linewidth=0.25, dashes=(5, 5))

    plt.draw()


def read_gridfile(grid_file_name: str) -> dict:
    """
    Read a grid file and extract coordinate and geometry information.

    Args:
        grid_file_name: The path to the grid file.

    Returns:
        A dictionary containing:
        - vert_x, vert_y: Vertex coordinates
        - edge_x, edge_y: Edge midpoint coordinates
        - cell_x, cell_y: Cell circumcenter coordinates
        - triangles: Cell-to-vertex connectivity array
        - mean_cell_area: Mean cell area
        - domain_length: Domain length (for torus grids)
        - domain_height: Domain height (for torus grids)
        - geometry_type: GeometryType enum (TORUS or ICOSAHEDRON)
    """
    grid_file = xr.open_dataset(grid_file_name)

    if hasattr(grid_file, "grid_geometry"):
        geometry_type = GeometryType.TORUS
    else:
        geometry_type = GeometryType.ICOSAHEDRON

    # Extract coordinates
    vert_x = grid_file.cartesian_x_vertices.values
    vert_y = grid_file.cartesian_y_vertices.values
    edge_x = grid_file.edge_middle_cartesian_x.values
    edge_y = grid_file.edge_middle_cartesian_y.values
    cell_x = grid_file.cell_circumcenter_cartesian_x.values
    cell_y = grid_file.cell_circumcenter_cartesian_y.values
    triangles = grid_file.vertex_of_cell.values.T - 1
    mean_cell_area = grid_file.cell_area.values.mean()

    # Get domain dimensions and clean up coordinates for torus grids
    domain_length = None
    domain_height = None
    if geometry_type == GeometryType.TORUS:
        domain_length = float(grid_file.domain_length)
        domain_height = float(grid_file.domain_height)
        # Adjust x values to coincide with the periodic boundary
        vert_x = np.where(np.abs(vert_x - domain_length) < 1e-9, 0, vert_x)
        edge_x = np.where(np.abs(edge_x - domain_length) < 1e-9, 0, edge_x)
        cell_x = np.where(np.abs(cell_x - domain_length) < 1e-9, 0, cell_x)

    return {
        "vert_x": vert_x,
        "vert_y": vert_y,
        "edge_x": edge_x,
        "edge_y": edge_y,
        "cell_x": cell_x,
        "cell_y": cell_y,
        "triangles": triangles,
        "mean_cell_area": mean_cell_area,
        "domain_length": domain_length,
        "domain_height": domain_height,
        "geometry_type": geometry_type,
    }


def create_triangulation(
    vert_x: np.ndarray,
    vert_y: np.ndarray,
    triangles: np.ndarray,
    edge_x: np.ndarray,
    edge_y: np.ndarray,
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    mean_cell_area: float,
    domain_length: float = None,
    domain_height: float = None,
) -> mpl.tri.Triangulation:
    """
    Create a triangulation from coordinate and connectivity arrays.
    For torus grids, mask boundary triangles and set up wrapping information.

    Args:
        vert_x: X coordinates of vertices.
        vert_y: Y coordinates of vertices.
        triangles: Cell-to-vertex connectivity array (n_cells x 3).
        edge_x: X coordinates of edge midpoints.
        edge_y: Y coordinates of edge midpoints.
        cell_x: X coordinates of cell circumcenters.
        cell_y: Y coordinates of cell circumcenters.
        mean_cell_area: Mean area of cells.
        domain_length: Domain length for torus grids.
        domain_height: Domain height for torus grids.

    Returns:
        A triangulation object
    """
    tri = mpl.tri.Triangulation(vert_x, vert_y, triangles=triangles)

    # Add elements
    tri.edge_x = edge_x
    tri.edge_y = edge_y
    tri.cell_x = cell_x
    tri.cell_y = cell_y
    tri.mean_cell_area = mean_cell_area
    tri.mean_edge_length = np.sqrt(mean_cell_area * 2)

    # Set up torus-specific attributes
    if domain_length is not None and domain_height is not None:
        tri.domain_length = domain_length
        tri.domain_height = domain_height
        tri = mask_boundary_triangles(tri)
    else:
        tri.domain_length = None
        tri.domain_height = None

    return tri


def mask_boundary_triangles(
    tri: mpl.tri.Triangulation,
    mask_edges: bool = False,
    ratio: float = 1.5,
) -> mpl.tri.Triangulation:
    """
    Mask boundary triangles from a triangulation.

    This function examines each triangle in the provided triangulation and
    determines if it is elongated based on the ratio of its longest edge to the
    grid average edge length. If the ratio exceeds `ratio`, the triangle is
    considered elongated and is masked out.
    Also save the original set of edges

    Args:
        tri: The input triangulation to be processed.

    Returns:
        The modified triangulation with elongated triangles masked out.
    """

    tri.all_edges = tri.edges.copy()
    tri.n_all_triangles = tri.triangles.shape[0]
    tri.n_all_edges = tri.edges.shape[0]

    boundary_triangles_mask = []

    # Mask triangles that have too long edges
    for triangle in tri.triangles:
        node_x_diff = tri.x[triangle] - np.roll(tri.x[triangle], 1)
        node_y_diff = tri.y[triangle] - np.roll(tri.y[triangle], 1)
        edges = np.sqrt(node_x_diff**2 + node_y_diff**2)
        if np.max(edges) > ratio * tri.mean_edge_length:
            boundary_triangles_mask.append(True)
        else:
            boundary_triangles_mask.append(False)

    tri.set_mask(boundary_triangles_mask)

    if mask_edges:
        # Mask out edges that are part of boundary triangles
        edges_mask = np.ones(tri.all_edges.shape[0], dtype=bool)
        for i, edge in enumerate(tri.all_edges):
            if any(np.array_equal(edge, filtered_edge) for filtered_edge in tri.edges):
                edges_mask[i] = False
        tri.edges_mask = edges_mask
    else:
        tri.edges_mask = None

    return tri
