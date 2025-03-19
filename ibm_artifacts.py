import os
import pickle
from icon4py.model.common.states import prognostic_state
import xarray as xr

import gt4py.next as gtx
import matplotlib.pyplot as plt
import numpy as np

from icon4py.model.common.io import plots

# mpl.use("tkagg")

Z_TOP = 200
QSCALE = 50
PEVERY = 2

hill_x = 500.0
hill_y = 500.0
hill_height = 100.0
hill_width = 100.0
compute_distance_from_hill = lambda x, y: ((x - hill_x) ** 2 + (y - hill_y) ** 2) ** 0.5
compute_hill_elevation = lambda x, y: hill_height * np.exp(
    -((compute_distance_from_hill(x, y) / hill_width) ** 2)
)
x = np.linspace(0, 1000, 500)
y = 500
hill = compute_hill_elevation(x, y)

main_dir = os.getcwd() + "/"
grid_file_path = "testdata/grids/gauss3d_torus/Torus_Triangles_1000m_x_1000m_res10m.nc"

# -------------------------------------------------------------------------------
# Some save path
#
savepoint_path = "/scratch/l_jcanton/ser_data/torus_exclaim.iccarus_hill/ser_data"
plot = plots.Plot(
    savepoint_path=savepoint_path,
    grid_file_path=main_dir + grid_file_path,
    backend=gtx.gtfn_cpu,
)

# # fortran
# ds = xr.open_dataset(f"/scratch/l_jcanton/plot_data/torus_exclaim.iccarus_hill/torus_exclaim_insta_DOM01_ML_0066.nc")
# data  = ds.w.values[0,:,:].T


pickle_prog = main_dir + "imgs/00000_after_predictor_prognostic.pkl"
pickle_diag = main_dir + "imgs/00001_after_predictor_diagnostic.pkl"

with open(pickle_prog, "rb") as f:
    prognostic_state = pickle.load(f)
with open(pickle_diag, "rb") as f:
    diagnostic_state = pickle.load(f)

data = prognostic_state.w

axs, x_coords_i, y_coords_i, u_i, w_i, idxs = plot.plot_sections(
    data=data,
    sections_x=[],
    sections_y=[500],
    label="w",
    plot_every=1,
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
#plt.savefig("w.png", bbox_inches="tight")
