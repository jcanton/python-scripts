import importlib
import pickle

import matplotlib.pyplot as plt
import numpy as np

import coordinates_functions as cf

#-------------------------------------------------------------------------------
# Domain and topography
#
case_name = "teamx"

match case_name:
    case "hill_small":
        # Small ICCARUS hill
        nx = 1000
        x0=0; x1=1000;
        y0=0; y1=1000;
        # hill topography
        hh = 100; hw = 100;
        hx = (x0+x1)/2; hy = (y0+y1)/2;
        # vertical coordinates
        s1 = 150; s2 = 150;
        Zt = 600; flat_height = 500;
        lowest_layer_thickness = 3 # min layer thickness
        stretch_factor = 1
        num_levels = 100
    case "hill_smooth":
        # Small hill with good SLEVE parameters
        nx = 1000
        x0=0; x1=1000;
        y0=0; y1=1000;
        # hill topography
        hh = 100; hw = 100;
        hx = (x0+x1)/2; hy = (y0+y1)/2;
        # vertical coordinates
        s1 = 250; s2 = 250;
        Zt = 1000; flat_height = 900;
        lowest_layer_thickness = 0.1 # min layer thickness
        stretch_factor = -1
        num_levels = 100
    case "hill_paper":
        # Tall mountain
        nx = 1000
        x0=0; x1=50000;
        y0=0; y1=50000;
        # hill topography
        hh = 7000; hw = 2000;
        hx = (x0+x1)/2; hy = (y0+y1)/2;
        # vertical coordinates
        s1 = 7000; s2 = 7500;
        Zt = 35000; flat_height = 16000;
        lowest_layer_thickness = 10 # min layer thickness
        stretch_factor = 1
        num_levels = 50
    case "brigitta":
        # load icon data
        with open("data/coordinates_discrete_brigitta.pkl", "rb") as f:
            x_coords, topography = pickle.load(f)
        # vertical coordinates
        s1 = 4000; s2 = 2500;
        Zt = 22000; flat_height = 16000;
        lowest_layer_thickness = 20
        stretch_factor = 0.65
        num_levels = 80
    case "teamx":
        # TeamX cosine hill
        nx = 1000
        x0=0; x1=10240;
        y0=0; y1=20480;
        # hill topography
        hh = 1000
        wave_length = 512 * 20
        # vertical coordinates
        s1 = 3000; s2 = 350;
        Zt = 5000; flat_height = 5000;
        lowest_layer_thickness = 10 # min layer thickness
        maximal_layer_thickness = 10
        top_height_limit_for_maximal_layer_thickness = 1000
        stretch_factor = 1.0
        num_levels = 250

if "hill" in case_name:
    hill = lambda x,y: hh * np.exp( -( (x - hx)**2 + (y - hy)**2 ) / hw**2 )
    # discrete arrays
    x_coords = np.linspace(x0,x1,nx)
    y_coords = hy*np.ones(nx)
    topography = hill(x_coords, y_coords)
elif "teamx" in case_name:
    hill = lambda x,y: hh/2 * (1 - np.cos(2*np.pi*x/wave_length))
    # discrete arrays
    x_coords = np.linspace(x0,x1,nx)
    y_coords = np.ones(nx)
    topography = hill(x_coords, y_coords)

#-------------------------------------------------------------------------------
# Smooth topography: h1 and h2

smoothed_topography = cf.smooth_topography(x_coords=x_coords, topography=topography)
small_scale_topography = topography - smoothed_topography


#-------------------------------------------------------------------------------
# Vertical coordinates

importlib.reload(cf)
vct_a = cf.compute_vct_a(
    lowest_layer_thickness=lowest_layer_thickness,
    maximal_layer_thickness=maximal_layer_thickness,
    top_height_limit_for_maximal_layer_thickness=top_height_limit_for_maximal_layer_thickness,
    model_top=Zt,
    stretch_factor=stretch_factor,
    num_levels=num_levels
)

z_ifc = cf.compute_SLEVE_coordinate(
    x_coords=x_coords,
    vct_a=vct_a,
    topography=topography,
    flat_height=flat_height,
    model_top=Zt,
    decay_scale_1=s1,
    decay_scale_2=s2,
    num_levels=num_levels,
)

cc_z_ifc = cf.check_and_correct_layer_thickness(
    vct_a=vct_a,
    vertical_coordinate=z_ifc.copy(),
    lowest_layer_thickness=lowest_layer_thickness,
)

#-------------------------------------------------------------------------------
# full levels, Jacobian and others
#

z_mc    = cf.compute_mc(vertical_coordinate=z_ifc)
cc_z_mc = cf.compute_mc(vertical_coordinate=cc_z_ifc)

ddqz_z    = cf.compute_ddqz(vertical_coordinate=z_ifc)
cc_ddqz_z = cf.compute_ddqz(vertical_coordinate=cc_z_ifc)

#-------------------------------------------------------------------------------
# plot
#
pX = np.tile(x_coords, (num_levels,1)).T

fig=plt.figure(1); plt.clf(); plt.show(block=False)
plt.plot(x_coords, topography,             "-",  color="black",  label="Topography")
plt.plot(x_coords, smoothed_topography,    "--", color="orange", label="Smoothed topography")
plt.plot(x_coords, small_scale_topography, ":",  color="blue",   label="Small-scale topo")
plt.legend()
plt.draw()
#plt.savefig("imgs/topography.png", bbox_inches="tight")

fig=plt.figure(2); plt.clf(); plt.show(block=False)
#plt.plot(vct_a, '-+')
plt.plot(vct_a[:-1], vct_a[:-1]-vct_a[1:], '-+')
plt.xlabel(r"Height [m]")
plt.ylabel(r"Layer thickness $\Delta z$ [m]")
plt.draw()

fig=plt.figure(3, figsize=(14,6)); plt.clf(); plt.show(block=False)
(ax1, ax2, ax3) = fig.subplots(1, 3, sharex=True, sharey=True)
for k in range(num_levels+1):
    ax1.plot(x_coords,    z_ifc[:,k], "-",  color="black")
    ax2.plot(x_coords,    z_ifc[:,k], "-",  color="black")
    ax2.plot(x_coords, cc_z_ifc[:,k], "--", color="blue")
    ax3.plot(x_coords, cc_z_ifc[:,k], "-",  color="blue")
#ax1.set_aspect('equal')
#ax2.set_aspect('equal')
#ax1.set_ylim([0,3100])
plt.draw()
#plt.savefig("imgs/z_ifc2.pdf", bbox_inches="tight")

fig=plt.figure(4, figsize=(14,6)); plt.clf(); plt.show(block=False)
cmax = ddqz_z.max()
cmin = ddqz_z.min()
(ax1, ax2) = fig.subplots(1, 2, sharex=True, sharey=True)
im1 = ax1.pcolormesh(pX,    z_mc,    ddqz_z, vmin=cmin, vmax=cmax, cmap="YlOrRd")
im2 = ax2.pcolormesh(pX, cc_z_mc, cc_ddqz_z, vmin=cmin, vmax=cmax, cmap="YlOrRd")
cbar = plt.colorbar(im1)
cbar = plt.colorbar(im2)
#ax1.set_aspect('equal')
#ax2.set_aspect('equal')
plt.draw()
#plt.savefig("imgs/jacobian.png", bbox_inches="tight")

fig=plt.figure(5); plt.clf(); plt.show(block=False)
plt.plot(z_mc[nx//2,:], cc_ddqz_z[nx//2,:], '-+')
plt.title("Layer thickness in the middle of the domain")
plt.xlabel(r"Height [m]")
plt.ylabel(r"Layer thickness $\Delta z$ [m]")
plt.draw()
