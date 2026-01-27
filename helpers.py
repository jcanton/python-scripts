import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from icon4py.model.common.grid.base import GeometryType
from matplotlib.collections import PolyCollection


def plot_grid(tri: mpl.tri.Triangulation, ax=None, print_indexes: bool = True, plot_masked: bool = True) -> None:
    if ax is None:
        fig = plt.figure(1)
        plt.show(block=False)
        ax = fig.subplots(nrows=1, ncols=1)
    ax.triplot(tri, color="k", linewidth=0.8)
    _plot_points_and_indices(ax, tri, show_points=True, print_indexes=print_indexes)
    ax.set_aspect("equal")
    
    # Plot masked (boundary) triangles with dashed lines
    if plot_masked and tri.mask is not None:
        _plot_masked_triangles(ax, tri)

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


def _triangle_plot_coords(tri: mpl.tri.Triangulation, idx: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return triangle coordinates shifted so wrapped cells plot on the correct side of a torus.
    """
    triangle = tri.triangles[idx]
    x_coords = tri.x[triangle]
    y_coords = tri.y[triangle]

    x_shifted = x_coords.copy()
    y_shifted = y_coords.copy()

    domain_length = getattr(tri, "domain_length", None)
    domain_height = getattr(tri, "domain_height", None)

    if domain_length is not None and domain_height is not None:
        span_x = np.max(x_coords) - np.min(x_coords)
        span_y = np.max(y_coords) - np.min(y_coords)
        center_x = tri.cell_x[idx]
        center_y = tri.cell_y[idx]

        if span_x > domain_length / 2:
            if center_x < domain_length / 2:
                x_shifted = np.where(x_coords > domain_length / 2, x_coords - domain_length, x_coords)
            else:
                x_shifted = np.where(x_coords < domain_length / 2, x_coords + domain_length, x_coords)

        if span_y > domain_height / 2:
            if center_y < domain_height / 2:
                y_shifted = np.where(y_coords > domain_height / 2, y_coords - domain_height, y_coords)
            else:
                y_shifted = np.where(y_coords < domain_height / 2, y_coords + domain_height, y_coords)

    x_plot = np.append(x_shifted, x_shifted[0])
    y_plot = np.append(y_shifted, y_shifted[0])
    return x_plot, y_plot


def _plot_masked_triangles(ax, tri: mpl.tri.Triangulation, **kwargs) -> None:
    masked_indices = np.where(tri.mask)[0]
    for idx in masked_indices:
        x_plot, y_plot = _triangle_plot_coords(tri, idx)
        ax.plot(x_plot, y_plot, color="k", linewidth=0.5, linestyle="--", dashes=(5, 5), **kwargs)


def _plot_points_and_indices(
    ax,
    tri: mpl.tri.Triangulation,
    show_points: bool = True,
    print_indexes: bool = False,
) -> None:
    if show_points:
        ax.plot(tri.x, tri.y, "vr")
        ax.plot(tri.edge_x, tri.edge_y, "sg")
        ax.plot(tri.cell_x, tri.cell_y, "ob")

    if print_indexes:
        for i, (x, y) in enumerate(zip(tri.x*1.05, tri.y)):
            ax.text(x, y, str(i), color="r", fontsize=10)
        for i, (x, y) in enumerate(zip(tri.edge_x*1.05, tri.edge_y)):
            ax.text(x, y, str(i), color="g", fontsize=10)
        for i, (x, y) in enumerate(zip(tri.cell_x*1.05, tri.cell_y)):
            ax.text(x, y, str(i), color="b", fontsize=10)


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
    geometry_type: GeometryType = GeometryType.ICOSAHEDRON,
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
    if geometry_type == GeometryType.TORUS:
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


def plot_partitioned_grid(
    tri: mpl.tri.Triangulation,
    c2rank_mapping: np.ndarray,
    ax=None,
    cmap: str = "tab20",
    alpha: float = 0.45,
    show_edges: bool = True,
    label_ranks: bool = False,
    show_centers: bool = False,
    plot_masked: bool = True,
    show_points: bool = True,
    print_indexes: bool = False,
) -> None:
    """
    Plot a partitioned grid, coloring each cell by its assigned rank.

    Args:
        tri: Triangulation with geometry and mask info.
        c2rank_mapping: Array mapping each cell to a rank (length = n_cells).
        ax: Matplotlib axes. If None, a new figure/axes is created.
        cmap: Matplotlib colormap name for rank colors.
        alpha: Face color alpha for cell fills.
        show_edges: Whether to draw grid edges on top of fills.
        label_ranks: Whether to label cell centers with their rank.
        show_centers: Whether to plot cell centers as markers.
        plot_masked: Whether to draw masked boundary triangles as dashed lines.
        show_points: Whether to plot vertex/edge/cell markers (same as plot_grid).
        print_indexes: Whether to print global vertex/edge/cell indices (same as plot_grid).
    """

    if c2rank_mapping.shape[0] != tri.triangles.shape[0]:
        raise ValueError("c2rank_mapping must have one entry per cell")

    if ax is None:
        fig = plt.figure(2)
        plt.show(block=False)
        ax = fig.subplots(nrows=1, ncols=1)

    cmap_obj = plt.get_cmap(cmap)
    polys = []
    facecolors = []

    for idx, rank in enumerate(c2rank_mapping):
        x_plot, y_plot = _triangle_plot_coords(tri, idx)
        polys.append(np.column_stack([x_plot[:-1], y_plot[:-1]]))
        facecolors.append(cmap_obj(int(rank) % cmap_obj.N))

    collection = PolyCollection(
        polys,
        facecolors=facecolors,
        edgecolors="none",
        alpha=alpha,
    )
    ax.add_collection(collection)

    if show_edges:
        ax.triplot(tri, color="k", linewidth=0.8)

    if plot_masked and tri.mask is not None:
        _plot_masked_triangles(ax, tri)

    if show_centers:
        ax.plot(tri.cell_x, tri.cell_y, "o", markersize=2.5, color="k", alpha=0.6)

    if label_ranks:
        for idx, rank in enumerate(c2rank_mapping):
            ax.text(tri.cell_x[idx]*1.05, tri.cell_y[idx], str(int(rank)), color="k", fontsize=10, ha="center", va="center")

    _plot_points_and_indices(ax, tri, show_points=show_points, print_indexes=print_indexes)

    ax.set_aspect("equal")
    plt.draw()


def plot_subdomains(
    tri: mpl.tri.Triangulation,
    subdomains: list,
    cols: int = 2,
    owned_cmap: str = "tab10",
    halo_cmap: str = "OrRd",
    alpha: float = 0.55,
    show_edges: bool = True,
    show_masked: bool = True,
    show_points: bool = False,
    print_indexes: bool = False,
) -> None:
    """
    Plot each subdomain (owned + halo levels) in its own subplot.

    Args:
        tri: Triangulation with geometry and mask info.
        subdomains: List of dicts with keys: rank, owned_cells, halo_levels, all_cells.
        cols: Number of subplot columns.
        owned_cmap: Colormap name for owned cells (indexed by rank).
        halo_cmap: Colormap name for halo levels (progressively darker).
        alpha: Face alpha for fills.
        show_edges: Whether to draw full-grid edges.
        show_masked: Whether to draw masked boundary triangles dashed.
        show_points: Whether to plot vertex/edge/cell markers.
        print_indexes: Whether to print global vertex/edge/cell indices.
    """

    n = len(subdomains)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.atleast_1d(axes).ravel()

    owned_cm = plt.get_cmap(owned_cmap)
    halo_cm = plt.get_cmap(halo_cmap)

    # Same owned color for every subdomain; consistent halo colors by level across ranks
    owned_color = owned_cm(0)
    max_levels = max(len(sd.get("halo_levels", [])) for sd in subdomains)
    halo_levels_colors = [halo_cm((i + 1) / (max_levels + 1 or 1)) for i in range(max_levels)]

    for ax, sd in zip(axes, subdomains):
        polys = []
        facecolors = []

        for cell in sd["owned_cells"]:
            x_plot, y_plot = _triangle_plot_coords(tri, cell)
            polys.append(np.column_stack([x_plot[:-1], y_plot[:-1]]))
            facecolors.append(owned_color)

        for lvl, cells in enumerate(sd.get("halo_levels", [])):
            halo_color = halo_levels_colors[lvl % len(halo_levels_colors)] if halo_levels_colors else (0.7, 0.7, 0.7, 1)
            for cell in cells:
                x_plot, y_plot = _triangle_plot_coords(tri, cell)
                polys.append(np.column_stack([x_plot[:-1], y_plot[:-1]]))
                facecolors.append(halo_color)

        if polys:
            collection = PolyCollection(
                polys,
                facecolors=facecolors,
                edgecolors="none",
                alpha=alpha,
            )
            ax.add_collection(collection)

        if show_edges:
            ax.triplot(tri, color="k", linewidth=0.8)

        if show_masked and tri.mask is not None:
            _plot_masked_triangles(ax, tri)

        _plot_points_and_indices(ax, tri, show_points=show_points, print_indexes=print_indexes)

        ax.set_title(f"Rank {sd['rank']}")
        ax.set_aspect("equal")

    # Hide any unused axes
    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.draw()
    plt.show(block=False)
