import os
import pickle

import gt4py.next as gtx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from icon4py.model.atmosphere.dycore import ibm
from icon4py.model.common.io import plots

# savepoint_path = 'testdata/ser_icondata/mpitask1/torus_hill.flat.ser_data/ser_data'
# grid_file_path = "testdata/grids/gauss3d_torus/Torus_Triangles_2000m_x_2000m_res100m.nc"

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
# W
#
# savepoint_path = '/scratch/l_jcanton/ser_data/torus_exclaim.iccarus_hill/ser_data'
savepoint_path = "/scratch/l_jcanton/ser_data/torus_exclaim.iccarus_flat/ser_data"
plot = plots.Plot(
    savepoint_path=savepoint_path,
    grid_file_path=main_dir + grid_file_path,
    backend=gtx.gtfn_cpu,
)
for i in range(0, 101):
    ## fortran
    # ds = xr.open_dataset(f"/scratch/l_jcanton/plot_data/torus_exclaim.iccarus_hill/torus_exclaim_insta_DOM01_ML_{i:04d}.nc")
    # w  = ds.w.values[0,:,:].T
    # python
    with open(main_dir + f"states.iccarus/{i:05d}_end_of_timestep.pkl", "rb") as f:
        state9 = pickle.load(f)
        w = state9.w.asnumpy()
    axs, x_coords_f, y_coords_f, w_f, _, idxs = plot.plot_sections(
        data=w,
        sections_x=[],
        sections_y=[500],
        label="w",
        plot_every=1,
        qscale=QSCALE,
    )
    for ax in axs:
        ax.plot(x, hill, "--", color="black")
        ax.set_aspect("equal")
        # ax.set_xlim([150,850])
        ax.set_ylim([-1, 500])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
    plt.draw()
    # plt.savefig(f"time_series/hill_fortran_{i:04d}.pdf", bbox_inches="tight")
    plt.savefig(f"time_series/hill_python_{i:04d}.pdf", bbox_inches="tight")

# -------------------------------------------------------------------------------
# ICON4Py
savepoint_path = "/scratch/l_jcanton/ser_data/torus_exclaim.iccarus_hill/ser_data"
plot = plots.Plot(
    savepoint_path=savepoint_path,
    grid_file_path=main_dir + grid_file_path,
    backend=gtx.gtfn_cpu,
)
_ibm = ibm.ImmersedBoundaryMethod(
    grid=plot.grid,
    savepoint_path=savepoint_path,
    grid_file_path=grid_file_path,
    backend=gtx.gtfn_cpu,
)
# average
with open(main_dir + "states.buildings3/00099_end_of_timestep.pkl", "rb") as f:
    state9 = pickle.load(f)
vn = state9.vn.asnumpy()
w = state9.w.asnumpy()
tsteps = 1
# for i in range(11,15):
#    with open(main_dir + f"imgs.iccarus/{i:05d}_end_of_timestep.pkl", "rb") as f:
#        state9 = pickle.load(f)
#        vn += state9.vn.asnumpy()
#        w  += state9.w.asnumpy()
#        tsteps+=1
axs, x_coords_i, y_coords_i, u_i, w_i, idxs = plot.plot_sections(
    data=vn / tsteps,
    data2=w / tsteps,
    sections_x=[],
    sections_y=[500],
    label="vvec_cell",
    plot_every=PEVERY,
    qscale=QSCALE,
)
# plt.show(block=False)
axs[0].plot(x, hill, "--", color="black")
axs[0].set_aspect("equal")
# axs[0].set_xlim([150,850])
axs[0].set_xlim([300, 700])
axs[0].set_ylim([-1, Z_TOP])
axs[0].set_xlabel("x [m]")
axs[0].set_ylabel("z [m]")
plt.draw()
# plot masked cells
s_mask = _ibm.half_cell_mask.ndarray[idxs, :]
x_coords = plot.tri.x[plot.tri.triangles[idxs]]
for i in range(s_mask.shape[0]):
    for j in range(s_mask.shape[1] - 1):
        if s_mask[i, j]:
            axs[0].fill_between(
                x_coords[i, 0:2],
                plot.half_level_heights[i, j] * np.ones(2),
                plot.half_level_heights[i, j + 1] * np.ones(2),
                facecolor="gainsboro",
                edgecolor=None,
                zorder=0,
            )
plt.draw()
plt.savefig("hill_python_quiver_buildings3.pdf", bbox_inches="tight")
# axs = plot.plot_levels(vn, 4, label=f"vvec_edge")


# ICON-fortran
savepoint_path = "/scratch/l_jcanton/ser_data/torus_exclaim.iccarus_hill/ser_data"
plot = plots.Plot(
    savepoint_path=savepoint_path,
    grid_file_path=main_dir + grid_file_path,
    backend=gtx.gtfn_cpu,
)
ds = xr.open_dataset(
    "/scratch/l_jcanton/plot_data/torus_exclaim.iccarus_hill/torus_exclaim_insta_DOM01_ML_0101.nc"
)
vn = ds.vn.values[0, :, :].T
w = ds.w.values[0, :, :].T
tsteps = 1
# for i in range(11,15):
#    ds = xr.open_dataset(f"/scratch/l_jcanton/plot_data/torus_exclaim.iccarus_hill/torus_exclaim_insta_DOM01_ML_{i:04d}.nc")
#    vn+=ds.vn.values[0,:,:].T
#    w +=ds.w.values[0,:,:].T
#    tsteps+=1
axs, x_coords_f, y_coords_f, u_f, w_f, idxs = plot.plot_sections(
    data=vn / tsteps,
    data2=w / tsteps,
    sections_x=[],
    sections_y=[500],
    label="vvec_cell",
    plot_every=PEVERY,
    qscale=QSCALE,
)
# plt.show(block=False)
axs[0].plot(x, hill, "--", color="black")
axs[0].set_aspect("equal")
# axs[0].set_xlim([150,850])
axs[0].set_xlim([300, 700])
axs[0].set_ylim([-1, Z_TOP])
axs[0].set_xlabel("x [m]")
axs[0].set_ylabel("z [m]")
for k in range(0, y_coords_f.shape[1], PEVERY):
    axs[0].plot(x_coords_f[:, k], y_coords_f[:, k], "-", color="gainsboro", zorder=0)
plt.draw()
plt.savefig("hill_fortran_quiver.pdf", bbox_inches="tight")

# -------------------------------------------------------------------------------
# vert profiles
#
markers = ["o", "s", "d"]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
x_profiles = [450, 500, 600]
fig = plt.figure(3, figsize=(8, 12))
plt.clf()
plt.show(block=False)
axs = fig.subplots(nrows=2, ncols=len(x_profiles), sharex=False, sharey=True)
# find x_coords closest to x_profiles
x_idxs = np.array([np.abs(x_coords_f[:, 0] - x).argmin() for x in x_profiles])
for i, x_idx in enumerate(x_idxs):
    axs[0, i].plot(
        u_f[x_idx, :],
        y_coords_f[x_idx, :],
        color=colors[i],
        markevery=10,
        label=f"x={x_coords_f[x_idx, 0]:.0f}m",
    )  # , marker=markers[i])
    axs[1, i].plot(
        w_f[x_idx, :],
        y_coords_f[x_idx, :],
        color=colors[i],
        markevery=10,
        label=f"x={x_coords_f[x_idx, 0]:.0f}m",
    )  # , marker=markers[i])
    #
    axs[0, i].plot(
        np.where(np.abs(u_i[x_idx, :]) > 5e-1, u_i[x_idx, :], np.nan),
        y_coords_i[x_idx, :],
        color=colors[i],
        markevery=13,
        linestyle="--",
    )  # , marker=markers[i])
    axs[1, i].plot(
        w_i[x_idx, :],
        y_coords_i[x_idx, :],
        color=colors[i],
        markevery=13,
        linestyle="--",
    )  # , marker=markers[i])
    #
    axs[0, i].set_xlabel("u [m/s]")
    axs[1, i].set_xlabel("w [m/s]")
axs[0, 0].set_ylabel("z [m]")
axs[1, 0].set_ylabel("z [m]")
# axs[0,0].set_ylim([-1,Z_TOP])
axs[0, 0].set_ylim([-1, 300])
# axs[0].legend(fontsize="small")
plt.draw()
plt.savefig("hill_compare.pdf", bbox_inches="tight")


# -------------------------------------------------------------------------------
# grid
tri = plot.tri
fig = plt.figure(1)
plt.clf()
plt.show(block=False)
ax = fig.subplots(nrows=1, ncols=1)
ax.triplot(tri, color="k", linewidth=0.25)
for i in range(len(tri.triangles)):
    ax.text(tri.cell_x[i], tri.cell_y[i], f"{i}")
ax.set_aspect("equal")
plt.draw()

# -------------------------------------------------------------------------------
# buildings
#
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

savepoint_path = "/scratch/l_jcanton/ser_data/torus_exclaim.iccarus_hill/ser_data"
plot = plots.Plot(
    savepoint_path=savepoint_path,
    grid_file_path=main_dir + grid_file_path,
    backend=gtx.gtfn_cpu,
)
with open(main_dir + "imgs/00009_end_of_timestep.pkl", "rb") as f:
    state9 = pickle.load(f)
axs, x_coords_i, y_coords_i, u_i, w_i, idxs = plot.plot_sections(
    data=state9.vn,
    data2=state9.w,
    sections_x=[],
    sections_y=[500],
    label="vvec_cell",
    plot_every=1,
    qscale=QSCALE,
)
axs[0].plot(x, hill, "--", color="black")
axs[0].set_aspect("equal")
axs[0].set_xlim([300, 550])
axs[0].set_ylim([0, 150])
axs[0].set_xlabel("x [m]")
axs[0].set_ylabel("z [m]")
plt.draw()
plt.show(block=False)
plt.savefig("hill_extra.pdf", bbox_inches="tight")


# -------------------------------------------------------------------------------
# Time series to figure out oscillations
#
savepoint_path = "/scratch/l_jcanton/ser_data/torus_exclaim.iccarus_hill/ser_data"
# savepoint_path = '/scratch/l_jcanton/ser_data/torus_exclaim.iccarus_flat/ser_data'
plot = plots.Plot(
    savepoint_path=savepoint_path,
    grid_file_path=main_dir + grid_file_path,
    backend=gtx.gtfn_cpu,
)
for i in range(1, 101):
    # fortran
    ds = xr.open_dataset(
        f"/scratch/l_jcanton/plot_data/torus_exclaim.iccarus_hill/torus_exclaim_insta_DOM01_ML_{i:04d}.nc"
    )
    vn = ds.vn.values[0, :, :].T
    w = ds.w.values[0, :, :].T
    ### python
    ##with open(main_dir + f"imgs.iccarus/{i-1:05d}_end_of_timestep.pkl", "rb") as f:
    ##    state9 = pickle.load(f)
    ##    vn = state9.vn.asnumpy()
    ##    w  = state9.w.asnumpy()
    axs, x_coords_f, y_coords_f, u_f, w_f, idxs = plot.plot_sections(
        data=vn,
        data2=w,
        sections_x=[],
        sections_y=[500],
        label="vvec_cell",
        plot_every=PEVERY,
        qscale=QSCALE,
    )
    axs[0].plot(x, hill, "--", color="black")
    axs[0].set_aspect("equal")
    # axs[0].set_xlim([150,850])
    # axs[0].set_ylim([-5,Z_TOP])
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("z [m]")
    plt.draw()
    plt.savefig(f"time_series_300/hill_fortran_{i:04d}.png", bbox_inches="tight")
    # plt.savefig(f"time_series/hill_python_{i:04d}.png", bbox_inches="tight")
