import os
import pickle
import xarray as xr

import gt4py.next as gtx
import matplotlib.pyplot as plt
import numpy as np


from icon4py.model.common.io import plots

QSCALE = 50
PEVERY = 1

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

## fortran
#ds = xr.open_dataset(f"/scratch/l_jcanton/plot_data/torus_exclaim/torus_exclaim_insta_DOM01_ML_0005.nc")
#data = ds.w.values[0,:,:].T

# python
with open(main_dir + "imgs/00076_end_of_timestep.pkl", "rb") as f:
    state = pickle.load(f)
data = state.w

# -------------------------------------------------------------------------------
# Plot
#
axs, x_coords_i, y_coords_i, u_i, w_i, idxs = plot.plot_sections(
    data=data,
    sections_x=[],
    sections_y=[hill_y],
    label="w",
    plot_every=PEVERY,
    qscale=QSCALE,
)
plt.show(block=False)
axs[0].plot(x, hill, "--", color="black")
axs[0].set_aspect("equal")
# axs[0].set_xlim([150,850])
#axs[0].set_ylim([-1, 500])
axs[0].set_xlabel("x [m]")
axs[0].set_ylabel("z [m]")
plt.draw()
