import os, pickle

import gt4py.next as gtx
from icon4py.model.common.io import plots
# from icon4py.model.atmosphere.dycore import ibm


# -------------------------------------------------------------------------------
# Some serialized data
#

#ICON4PY_SAVEPOINT_PATH="ser_data/exclaim_gauss3d_250x250.uniform400_flat/ser_data"
#ICON4PY_GRID_FILE_PATH="testdata/grids/gauss3d_torus/Torus_Triangles_250m_x_250m_res2.5m.nc"

#ICON4PY_SAVEPOINT_PATH="ser_data/exclaim_gauss3d_250x250x250.uniform200_flat/ser_data"
#ICON4PY_GRID_FILE_PATH="testdata/grids/gauss3d_torus/Torus_Triangles_250m_x_250m_res1.25m.nc"

ICON4PY_SAVEPOINT_PATH="ser_data/exclaim_channel_950x350x100_5m_nlev20/ser_data"
ICON4PY_GRID_FILE_PATH="testdata/grids/gauss3d_torus/Channel_950m_x_350m_res5m.nc"

icon4py_dir = os.path.join(os.getcwd(), "../icon4py")
grid_file_path = os.path.join(icon4py_dir, ICON4PY_GRID_FILE_PATH)
savepoint_path = os.path.join(icon4py_dir, ICON4PY_SAVEPOINT_PATH)

plot = plots.Plot(
    savepoint_path=savepoint_path,
    grid_file_path=grid_file_path,
    backend=gtx.gtfn_cpu,
)
# _ibm = ibm.ImmersedBoundaryMethod(
#     grid=plot.grid,
#     savepoint_path=savepoint_path,
#     grid_file_path=grid_file_path,
#     backend=gtx.gtfn_cpu,
# )
del plot.tri._cpp_triangulation

#with open("data/plotting_250x250x1000_2.5.pkl", "wb") as f:
#with open("data/plotting_250x250x250_1.25.pkl", "wb") as f:
with open("data/plotting_channel_950x350x100_5m_nlev20.pkl", "wb") as f:
    pickle.dump({
        "tri": plot.tri,
        "full_level_heights": plot.full_level_heights,
        "half_level_heights": plot.half_level_heights,
        #"full_cell_mask": _ibm.full_cell_mask.asnumpy().astype(float),
        #"half_cell_mask": _ibm.half_cell_mask.asnumpy().astype(float),
        #"full_edge_mask": _ibm.full_edge_mask.asnumpy().astype(float),
        #"half_edge_mask": _ibm.half_edge_mask.asnumpy().astype(float),
    }, f)
