import os
import pickle

import gt4py.next as gtx
import matplotlib.pyplot as plt
import numpy as np

from icon4py.model.common.io import plots

QSCALE = 10

hill_x = 500.0
hill_y = 500.0
hill_height = 100.0
hill_width = 100.0
compute_distance_from_hill = lambda x, y: ((x - hill_x) ** 2 + (y - hill_y) ** 2) ** 0.5
compute_hill_elevation = lambda x, y: hill_height * np.exp(
    -((compute_distance_from_hill(x, y) / hill_width) ** 2)
)
x = np.linspace(0, 2*hill_x, 500)
y = hill_y
hill = compute_hill_elevation(x, y)

main_dir = os.getcwd() + "/../icon4py/"
grid_file_path = "testdata/grids/gauss3d_torus/Torus_Triangles_1000m_x_1000m_res10m.nc"

# -------------------------------------------------------------------------------
# Some serialized data path
#
savepoint_path = "testdata/ser_icondata/mpitask1/gauss3d_torus/ser_data/"
plot = plots.Plot(
    savepoint_path=main_dir + savepoint_path,
    grid_file_path=main_dir + grid_file_path,
    backend=gtx.gtfn_cpu,
)

# -------------------------------------------------------------------------------
# Load data

with open(main_dir + "imgs/00004_diffusion_before.pkl", "rb") as f:
    state_b = pickle.load(f)
with open(main_dir + "imgs/00005_diffusion_after.pkl", "rb") as f:
    state_a = pickle.load(f)

data_b = state_b.vn
data_a = state_a.vn

data = data_a - data_b

# -------------------------------------------------------------------------------
# Plot
#

# levels
axs = plot.plot_levels(data, num_levels=4, label="vvec_edge", qscale=QSCALE)
