import os
import importlib
import numpy as np
import matplotlib.pyplot as plt

import gt4py.next as gtx
from icon4py.model.common.io import plots
from icon4py.model.atmosphere.dycore import ibm

main_dir = os.getcwd() + "/../icon4py/"
#grid_file_path = main_dir + "testdata/grids/gauss3d_torus/Torus_Triangles_1000m_x_1000m_res10m.nc"
#savepoint_path = main_dir + "ser_data/exclaim_gauss3d.uniform800_flat/ser_data/"

#grid_file_path = main_dir + "testdata/grids/gauss3d_torus/Torus_Triangles_250m_x_250m_res2.5m.nc"
#savepoint_path = main_dir + "ser_data/exclaim_gauss3d_250x250.uniform400_flat/ser_data/"

grid_file_path = main_dir + "testdata/grids/gauss3d_torus/Channel_950m_x_350m_res5m.nc"
savepoint_path = main_dir + "ser_data/exclaim_channel_950x350x100_5m_nlev20/ser_data"

plot = plots.Plot(
    savepoint_path=savepoint_path,
    grid_file_path=grid_file_path,
    backend=gtx.gtfn_cpu,
)

importlib.reload(ibm)
_ibm = ibm.ImmersedBoundaryMethod(
    grid=plot.grid,
    savepoint_path=savepoint_path,
    grid_file_path=grid_file_path,
    backend=gtx.gtfn_cpu,
)

data = _ibm.full_cell_mask.asnumpy().astype(float)[:,-1]

plt.figure(1); plt.clf(); plt.show(block=False)
plt.tripcolor(plot.tri, data, edgecolor='black', shading='flat')
#plt.xticks(np.arange(0,1000,50))
#plt.yticks(np.arange(0,1000,50))
plt.xticks(np.arange(0,1000,50))
plt.yticks(np.arange(0, 400,50))
plt.draw()
plt.savefig("ibm_masked_cells.png", dpi=600)

