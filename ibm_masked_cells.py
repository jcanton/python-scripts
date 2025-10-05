import os
import importlib
import numpy as np
import matplotlib.pyplot as plt

import gt4py.next as gtx
from icon4py.model.common.io import plots
from icon4py.model.atmosphere.dycore import ibm

main_dir = os.getcwd() + "/../icon4py.ibm_02/"
#grid_file_path = main_dir + "testdata/grids/gauss3d_torus/Torus_Triangles_1000m_x_1000m_res10m.nc"
#savepoint_path = main_dir + "ser_data/exclaim_gauss3d.uniform800_flat/ser_data/"

#grid_file_path = main_dir + "testdata/grids/gauss3d_torus/Torus_Triangles_250m_x_250m_res2.5m.nc"
#savepoint_path = main_dir + "ser_data/exclaim_gauss3d_250x250.uniform400_flat/ser_data/"

#grid_file_path = main_dir + "testdata/grids/gauss3d_torus/Channel_950m_x_350m_res5m.nc"
#savepoint_path = main_dir + "ser_data/exclaim_channel_950x350x100_5m_nlev20/ser_data"
#grid_file_path = main_dir + "testdata/grids/gauss3d_torus/Channel_950m_x_350m_res2.5m.nc"
#savepoint_path = main_dir + "ser_data/exclaim_channel_950x350x100_2.5m_nlev40/ser_data"
grid_file_path = main_dir + "testdata/grids/gauss3d_torus/Channel_950m_x_350m_res1.5m.nc"
savepoint_path = main_dir + "ser_data/exclaim_channel_950x350x100_1.5m_nlev64/ser_data"
#grid_file_path = main_dir + "testdata/grids/gauss3d_torus/Channel_950m_x_350m_res1.25m.nc"
#savepoint_path = main_dir + "ser_data/exclaim_channel_950x350x100_1.25m_nlev80/ser_data"
#grid_file_path = main_dir + "testdata/grids/gauss3d_torus/Channel_950m_x_350m_res1m.nc"
#savepoint_path = main_dir + "ser_data/exclaim_channel_950x350x100_1m_nlev100/ser_data"

importlib.reload(plots)
plot = plots.Plot(
    savepoint_path=savepoint_path,
    grid_file_path=grid_file_path,
    backend=gtx.gtfn_cpu,
)

importlib.reload(ibm)
_ibm_masks = ibm.ImmersedBoundaryMethodMasks(
    grid=plot.grid,
    savepoint_path=savepoint_path,
    grid_file_path=grid_file_path,
    backend=gtx.gtfn_cpu,
)

data = _ibm_masks.full_cell_mask.asnumpy().astype(float)[:,-1]

plt.figure(1); plt.clf(); plt.show(block=False)
plt.tripcolor(plot.tri, data, edgecolor='none', shading='flat')
#plt.xticks(np.arange(0,1000,50))
#plt.yticks(np.arange(0,1000,50))
plt.xticks(np.arange(0,1000,50))
plt.yticks(np.arange(0, 400,50))
plt.axis('equal')
plt.draw()
plt.savefig("ibm_masked_cells.png", dpi=1200)

