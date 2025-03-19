import os, pickle
import numpy as np
import matplotlib.pyplot as plt

import gt4py.next as gtx
from icon4py.model.common.io import plots
from icon4py.model.atmosphere.dycore import ibm
from icon4py.model.common import dimension as dims
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.atmosphere.dycore.stencils.mo_math_gradients_grad_green_gauss_cell_dsl import mo_math_gradients_grad_green_gauss_cell_dsl
program_mo_math_gradients_grad_green_gauss_cell_dsl = mo_math_gradients_grad_green_gauss_cell_dsl.with_backend(gtx.gtfn_cpu)

main_dir = os.getcwd() + "/"
state_fname = 'testdata/prognostic_state_initial.pkl'
savepoint_path = 'testdata/ser_icondata/mpitask1/gauss3d_torus/ser_data'
grid_file_path = "testdata/grids/gauss3d_torus/Torus_Triangles_2000m_x_2000m_res250m.nc"

with open(main_dir + state_fname, "rb") as ifile:
    prognostic_state = pickle.load(ifile)

rho = prognostic_state.rho
exner = prognostic_state.exner

_plot = plots.Plot(
    savepoint_path = main_dir + savepoint_path,
    grid_file_path = main_dir + grid_file_path,
    backend = gtx.gtfn_cpu,
    )
interpolation_savepoint = _plot.interpolation_savepoint
grid = _plot.grid

_ibm = ibm.ImmersedBoundaryMethod(grid)
_ibm._dirichlet_value_rho=0.0
_ibm.set_dirichlet_value_rho(rho)

# Extend the cell mask to neighboring cells
neighbor_table = grid.offset_providers['C2E2CO'].table
c2e2c = grid.connectivities[dims.C2E2CDim]
full_cell_mask_np = _ibm.full_cell_mask.ndarray
neigh_full_cell_mask_np = np.zeros((grid.num_cells, grid.num_levels), dtype=bool)
for k in range(grid.num_levels):
    neigh_full_cell_mask_np[c2e2c[np.where(full_cell_mask_np[:,k])], k] = True
axs = _plot.plot_data(full_cell_mask_np.astype('int'), 1,       label=f"rho", fig_num=1)
axs = _plot.plot_data(neigh_full_cell_mask_np.astype('int'), 1, label=f"rho", fig_num=2)
plt.show(block=False)



# Test Green-Gauss gradient approximation
z_grad_rth_1 = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, backend=gtx.gtfn_cpu)
z_grad_rth_2 = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, backend=gtx.gtfn_cpu)
z_grad_rth_3 = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, backend=gtx.gtfn_cpu)
z_grad_rth_4 = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, backend=gtx.gtfn_cpu)
#
geofac_grg_x = interpolation_savepoint.geofac_grg()[0]
geofac_grg_y = interpolation_savepoint.geofac_grg()[1]
#
geofac_grg_x.ndarray[17,1] = 0.0
#
program_mo_math_gradients_grad_green_gauss_cell_dsl(
    p_grad_1_u=z_grad_rth_1,
    p_grad_1_v=z_grad_rth_2,
    p_grad_2_u=z_grad_rth_3,
    p_grad_2_v=z_grad_rth_4,
    p_ccpr1=rho,
    p_ccpr2=exner,
    geofac_grg_x=geofac_grg_x,
    geofac_grg_y=geofac_grg_y,
    horizontal_start=0,
    horizontal_end=grid.num_cells,
    vertical_start=0,
    vertical_end=grid.num_levels,
    offset_provider=grid.offset_providers,
)
# axs = _plot.plot_data(prognostic_state.rho, 1, label=f"rho",     fig_num=1)
# axs = _plot.plot_data(z_grad_rth_1,         1, label=f"ddx_rho", fig_num=2)
# axs = _plot.plot_data(z_grad_rth_2,         1, label=f"ddx_rho", fig_num=3)
plt.show(block=False)