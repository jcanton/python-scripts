import numpy as np

# Parameters for setting up the decay function of the topographic signal for
# SLEVE. Default values from mo_sleve_nml.
SLEVE_decay_exponent = 1.2
#: minimum absolute layer thickness 1 for SLEVE coordinates
SLEVE_minimum_layer_thickness_1 = 100.0
#: minimum absolute layer thickness 2 for SLEVE coordinates
SLEVE_minimum_layer_thickness_2 = 500.0
#: minimum relative layer thickness for nominal thicknesses <= SLEVE_minimum_layer_thickness_1
SLEVE_minimum_relative_layer_thickness_1 = 1.0 / 3.0
#: minimum relative layer thickness for a nominal thickness of SLEVE_minimum_layer_thickness_2
SLEVE_minimum_relative_layer_thickness_2 = 0.5


#-------------------------------------------------------------------------------
def smooth_topography(x_coords, topography):
    def _compute_ddx(x_coords, psi):
        dx = x_coords[1:] - x_coords[:-1]
        ddx_psi = (psi[:-2] - 2 * psi[1:-1] + psi[2:]) / (dx[:-1] * dx[1:])
        return np.concatenate(([0], ddx_psi, [0]))

    dx = x_coords[1:] - x_coords[:-1]
    dx = np.concatenate((dx, [dx[-1]]))
    smoothed_topography = topography.copy()
    for _ in range(25):
        nabla2_topo = _compute_ddx(x_coords,smoothed_topography)
        smoothed_topography += 0.125 * nabla2_topo * dx
    return smoothed_topography


#-------------------------------------------------------------------------------
def compute_vct_a(
    lowest_layer_thickness,
    maximal_layer_thickness,
    top_height_limit_for_maximal_layer_thickness,
    model_top,
    stretch_factor,
    num_levels,
):
    if lowest_layer_thickness > 0.0:
        # src/atm_dyn_iconam/mo_init_vgrid.f90  󰊕 init_sleve_coord  mo_init_vgrid
        d = np.log( lowest_layer_thickness / model_top) / np.log( 2.0 / np.pi * np.arccos( float(num_levels - 1) ** stretch_factor / float(num_levels) ** stretch_factor))
        vct_a = model_top * ( 2.0 / np.pi * np.arccos( np.arange(num_levels + 1, dtype=float) ** stretch_factor / float(num_levels) ** stretch_factor)) ** d
        # limiter
        if (2 * lowest_layer_thickness
            < maximal_layer_thickness
            < 0.5 * top_height_limit_for_maximal_layer_thickness
                ):
            layer_thickness = vct_a[:num_levels] - vct_a[1:]
            lowest_level_exceeding_limit = np.max(np.where(layer_thickness > maximal_layer_thickness))
            modified_vct_a = np.zeros(num_levels+1, dtype=float)
            lowest_level_unmodified_thickness = 0
            shifted_levels = 0
            for k in range(num_levels - 1, -1, -1):
                if ( modified_vct_a[k + 1] < top_height_limit_for_maximal_layer_thickness):
                    modified_vct_a[k] = modified_vct_a[k + 1] + np.minimum(maximal_layer_thickness, layer_thickness[k])
                elif lowest_level_unmodified_thickness == 0:
                    lowest_level_unmodified_thickness = k + 1
                    shifted_levels = max(
                        0, lowest_level_exceeding_limit - lowest_level_unmodified_thickness
                    )
                    modified_vct_a[k] = modified_vct_a[k + 1] + layer_thickness[k + shifted_levels]
                else:
                    modified_vct_a[k] = modified_vct_a[k + 1] + layer_thickness[k + shifted_levels]

            stretchfac = (
                1.0
                if shifted_levels == 0
                else (
                    vct_a[0]
                    - modified_vct_a[lowest_level_unmodified_thickness]
                    - float(lowest_level_unmodified_thickness)
                    * maximal_layer_thickness
                )
                / (
                    modified_vct_a[0]
                    - modified_vct_a[lowest_level_unmodified_thickness]
                    - float(lowest_level_unmodified_thickness)
                    * maximal_layer_thickness
                )
            )

            for k in range(num_levels - 1, -1, -1):
                if vct_a[k + 1] < top_height_limit_for_maximal_layer_thickness:
                    vct_a[k] = vct_a[k + 1] + np.minimum(
                        maximal_layer_thickness, layer_thickness[k]
                    )
                else:
                    vct_a[k] = (
                        vct_a[k + 1]
                        + maximal_layer_thickness
                        + (
                            layer_thickness[k + shifted_levels]
                            - maximal_layer_thickness
                        )
                        * stretchfac
                    )

            # Try to apply additional smoothing on the stretching factor above the constant-thickness layer
            if stretchfac != 1.0 and lowest_level_exceeding_limit < num_levels - 4:
                for k in range(num_levels - 1, -1, -1):
                    if (
                        modified_vct_a[k + 1]
                        < top_height_limit_for_maximal_layer_thickness
                    ):
                        modified_vct_a[k] = vct_a[k]
                    else:
                        modified_layer_thickness = np.minimum(
                            1.025 * (vct_a[k] - vct_a[k + 1]),
                            1.025
                            * (
                                modified_vct_a[lowest_level_exceeding_limit + 1]
                                - modified_vct_a[lowest_level_exceeding_limit + 2]
                            )
                            / (
                                modified_vct_a[lowest_level_exceeding_limit + 2]
                                - modified_vct_a[lowest_level_exceeding_limit + 3]
                            )
                            * (modified_vct_a[k + 1] - modified_vct_a[k + 2]),
                        )
                        modified_vct_a[k] = np.minimum(
                            vct_a[k], modified_vct_a[k + 1] + modified_layer_thickness
                        )
                if modified_vct_a[0] == vct_a[0]:
                    vct_a[0:2] = modified_vct_a[0:2]
                    vct_a[
                        lowest_level_unmodified_thickness + 1 : num_levels
                    ] = modified_vct_a[
                        lowest_level_unmodified_thickness + 1 : num_levels
                    ]
                    vct_a[2 : lowest_level_unmodified_thickness + 1] = 0.5 * (
                        modified_vct_a[1:lowest_level_unmodified_thickness]
                        + modified_vct_a[3 : lowest_level_unmodified_thickness + 2]
                    )

    else:
        # uniform spacing
        vct_a = np.linspace(0, model_top, num_levels + 1)[::-1]

    return vct_a


#-------------------------------------------------------------------------------
def compute_SLEVE_coordinate(
    x_coords,
    vct_a,
    topography,
    flat_height,
    model_top,
    decay_scale_1,
    decay_scale_2,
    num_levels,
):
    nx = topography.shape[0]
    nflatlev = np.max(np.where(vct_a >= flat_height)[0])

    smoothed_topography = smooth_topography(x_coords=x_coords, topography=topography)
    small_scale_topography = topography - smoothed_topography

    # src/atm_dyn_iconam/mo_init_vgrid.f90   mo_init_vgrid 󰊕 init_vert_coord
    decay_func = lambda Z, Zt, s12, n: np.sinh( (Zt/s12)**n - (Z/s12)**n ) / np.sinh( (Zt/s12)**n )

    vertical_coordinate = np.zeros((nx, num_levels + 1))
    vertical_coordinate[:, num_levels] = topography


    k = range(nflatlev + 1)
    vertical_coordinate[:, k] = vct_a[k]

    k = range(nflatlev + 1, num_levels)
    # Scaling factors for large-scale and small-scale topography
    z_fac1 = decay_func( vct_a[k], model_top, decay_scale_1, SLEVE_decay_exponent)
    z_fac2 = decay_func( vct_a[k], model_top, decay_scale_2, SLEVE_decay_exponent)
    vertical_coordinate[:, k] = (
        vct_a[k][np.newaxis, :]
        + smoothed_topography[:, np.newaxis] * z_fac1
        + small_scale_topography[:, np.newaxis] * z_fac2
    )
    return vertical_coordinate


#-------------------------------------------------------------------------------
def check_and_correct_layer_thickness(
    vct_a,
    vertical_coordinate,
    lowest_layer_thickness,
):

    nx = vertical_coordinate.shape[0]
    num_levels = vertical_coordinate.shape[1] - 1

    ktop_thicklimit = np.asarray(nx * [num_levels], dtype=int)

    # Ensure that layer thicknesses are not too small; this would potentially
    # cause instabilities in vertical advection
    for k in reversed(range(num_levels)):
        delta_vct_a = vct_a[k] - vct_a[k + 1]
        if delta_vct_a < SLEVE_minimum_layer_thickness_1:
            # limit layer thickness to SLEVE_minimum_relative_layer_thickness_1 times its nominal value
            minimum_layer_thickness = (
                SLEVE_minimum_relative_layer_thickness_1 * delta_vct_a
            )
        elif delta_vct_a < SLEVE_minimum_layer_thickness_2:
            # limitation factor changes from SLEVE_minimum_relative_layer_thickness_1 to SLEVE_minimum_relative_layer_thickness_2
            layer_thickness_adjustment_factor = (
                (SLEVE_minimum_layer_thickness_2 - delta_vct_a)
                / (
                    SLEVE_minimum_layer_thickness_2
                    - SLEVE_minimum_layer_thickness_1
                )
            ) ** 2
            minimum_layer_thickness = (
                SLEVE_minimum_relative_layer_thickness_1
                * layer_thickness_adjustment_factor
                + SLEVE_minimum_relative_layer_thickness_2
                * (1.0 - layer_thickness_adjustment_factor)
            ) * delta_vct_a
        else:
            # limitation factor decreases again
            minimum_layer_thickness = (
                SLEVE_minimum_relative_layer_thickness_2
                * SLEVE_minimum_layer_thickness_2
                * (delta_vct_a / SLEVE_minimum_layer_thickness_2) ** (1.0 / 3.0)
            )

        minimum_layer_thickness = max(
            minimum_layer_thickness, min(50, lowest_layer_thickness)
        )

        # Ensure that the layer thickness is not too small, if so fix it and
        # save the layer index
        cell_ids = np.argwhere(
            vertical_coordinate[:, k + 1] + minimum_layer_thickness > vertical_coordinate[:, k]
        )
        vertical_coordinate[cell_ids, k] = (
            vertical_coordinate[cell_ids, k + 1] + minimum_layer_thickness
        )
        ktop_thicklimit[cell_ids] = k

    # Smooth layer thickness ratios in the transition layer of columns where the
    # thickness limiter has been active (exclude lowest and highest layers)
    cell_ids = np.argwhere((ktop_thicklimit <= num_levels - 3) & (ktop_thicklimit >= 3)).flatten()
    if cell_ids.size > 0:
        delta_z1 = (
            vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] + 1]
            - vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] + 2]
        )
        delta_z2 = (
            vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] - 3]
            - vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] - 2]
        )
        stretching_factor = (delta_z2 / delta_z1) ** 0.25
        delta_z3 = (
            vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] - 2]
            - vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] + 1]
        ) / (stretching_factor * (1.0 + stretching_factor * (1.0 + stretching_factor)))
        vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids]] = np.maximum(
            vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids]],
            vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] + 1]
            + delta_z3 * stretching_factor,
        )
        vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] - 1] = np.maximum(
            vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] - 1],
            vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids]]
            + delta_z3 * stretching_factor**2,
        )

    # Check if ktop_thicklimit is sufficiently far away from the model top
    if not np.all(ktop_thicklimit > 2):
        if num_levels > 6:
            print(
                f"Model top is too low and num_levels, {num_levels}, > 6."
            )
        else:
            print(
                f"Model top is too low. But num_levels, {num_levels}, <= 6. "
            )

    return vertical_coordinate


#-------------------------------------------------------------------------------
def compute_mc(
    vertical_coordinate,
):
    return 0.5 * (vertical_coordinate[:,:-1] + vertical_coordinate[:,1:])


#-------------------------------------------------------------------------------
def compute_ddqz(
    vertical_coordinate,
):
    return vertical_coordinate[:,:-1] - vertical_coordinate[:,1:]
