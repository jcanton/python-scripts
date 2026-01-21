from math import acos, cos, exp, log, radians, sin, sqrt

earth_radius = 6371e3  # in m


# -------------------------------------------------------------------------------
# Functions
#
def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))


def gc2cc(lon, lat):
    lon_rad = radians(lon)
    lat_rad = radians(lat)

    x = cos(lon_rad) * cos(lat_rad)
    y = sin(lon_rad) * cos(lat_rad)
    z = sin(lat_rad)

    return (x, y, z)


def arc_length_cartesian(P, Q):
    norm1 = sqrt(dot_product(P, P))
    norm2 = sqrt(dot_product(Q, Q))
    cc = dot_product(P, Q) / (norm1 * norm2)

    return acos(cc)


def rbf_gaussian_kernel(r, scale):  # default for cell and vertex
    return exp(-((r / scale) ** 2))


def rbf_inv_multiquadric_kernel(r, scale):  # default for edges
    return 1.0 / sqrt(1.0 + (r / scale) ** 2)


# -------------------------------------------------------------------------------
# Usage example
#
P = gc2cc(0.0, 0.0)  # lon, lat (in degrees)
Q = gc2cc(0.0, 0.001)  # lon, lat (in degrees)
arc_length = arc_length_cartesian(P, Q)

mean_characteristic_length = earth_radius * arc_length  # (used for sphere)
mean_dual_edge_length = mean_characteristic_length  # (approx, used for torus)


# -------------------------------------------------------------------------------
# RBF scales (mo_interpol_config.f90::configure_interpolation)
resol = mean_characteristic_length / 1000  # resolution in km

# Cell
if resol >= 2.5:
    rbf_vec_scale_c = 0.5
else:
    rbf_vec_scale_c = 0.5 / (1.0 + 1.8 * log(2.5 / resol) ** 3.75)
if resol <= 0.125:
    rbf_vec_scale_c = rbf_vec_scale_c * (resol / 0.125) ** 0.9

# Vertex
if resol >= 2.0:
    rbf_vec_scale_v = 0.5
else:
    rbf_vec_scale_v = 0.5 / (1.0 + 1.8 * log(2.0 / resol) ** 3)
if resol <= 0.125:
    rbf_vec_scale_v = rbf_vec_scale_v * (resol / 0.125) ** 0.96

# Edge
if resol >= 2.0:
    rbf_vec_scale_e = 0.5
else:
    rbf_vec_scale_e = 0.5 / (1.0 + 0.4 * log(2.0 / resol) ** 2)
if resol <= 0.125:
    rbf_vec_scale_e = rbf_vec_scale_e * (resol / 0.125) ** 0.325


# Torus (all the same)
t_rbf_vec_scale_c = mean_dual_edge_length
t_rbf_vec_scale_e = t_rbf_vec_scale_c
t_rbf_vec_scale_v = t_rbf_vec_scale_c

# -------------------------------------------------------------------------------
# RBF kernels

# sphere
dist = 0.5*arc_length_cartesian(P, Q)
kernel_c = rbf_gaussian_kernel(dist, rbf_vec_scale_c)
kernel_v = rbf_gaussian_kernel(dist, rbf_vec_scale_v)
kernel_e = rbf_inv_multiquadric_kernel(dist, rbf_vec_scale_e)

# torus (let's use two points at the same distance as the two on the sphere)
t_dist = earth_radius * arc_length_cartesian(P, Q) / earth_radius
t_kernel_c = rbf_gaussian_kernel(t_dist, t_rbf_vec_scale_c)
t_kernel_e = rbf_inv_multiquadric_kernel(t_dist, t_rbf_vec_scale_e)

t_dist_before_fix = earth_radius * arc_length_cartesian(P, Q)
t_kernel_c_before_fix = rbf_gaussian_kernel(t_dist_before_fix, t_rbf_vec_scale_c)
t_kernel_e_before_fix = rbf_inv_multiquadric_kernel(t_dist_before_fix, t_rbf_vec_scale_e)

# -------------------------------------------------------------------------------
# Print
#
print(f"Arc length between P and Q (radians): {arc_length:.6f}")
print(f"                           (meters):  {earth_radius * arc_length:.6f}")
print()
print(f"RBF vector scale for cell:   {rbf_vec_scale_c:.3f}")
print(f"                     vertex: {rbf_vec_scale_v:.3f}")
print(f"                     edge:   {rbf_vec_scale_e:.3f}")
print(f"               torus cell:  {t_rbf_vec_scale_c:.3f}")
print(f"               torus edge:  {t_rbf_vec_scale_e:.3f}")
print()
print(f"RBF kernels cell:   {kernel_c:.6f}")
print(f"            vertex: {kernel_v:.6f}")
print(f"            edge:   {kernel_e:.6f}")
print(f" torus      cell:   {t_kernel_c:.6f}")
print(f" torus bfix cell:   {t_kernel_c_before_fix:.6f}")
print(f" torus      edge:   {t_kernel_e:.6f}")
print(f" torus bfix edge:   {t_kernel_e_before_fix:.6f}")
