import os
import pickle
import xarray as xr

# ------------------------------------------------------------------------------
# handling variables
#
variables = ["vn", "w", "rho", "exner", "theta_v"]


def load_state(fname, variables):
    state = {}
    if fname.endswith(".nc"):
        ds = xr.open_dataset(fname)
        for variable in variables:
            state[variable] = ds[variable].values[0,:,:].T
    elif fname.endswith(".pkl"):
        with open(fname, "rb") as ifile:
            data = pickle.load(ifile)
            for variable in variables:
                state[variable] = data[variable]
    return state


# ------------------------------------------------------------------------------
# reference solution
#
# fname = os.path.join("data", "ibm_channel_reference_solution.pkl")
fname = os.path.join(
    "../icon4py/testdata/ser_icondata/mpitask1/test_channel_ibm/",
    "end_of_timestep_000000000.pkl",
    #"../icon-exclaim/build_cpu2py/experiments/exclaim_gauss3d_sb/",
    #"torus_insta_DOM01_ML_0002.nc",
    #"../icon-exclaim/build_cpu2py/experiments/exclaim_gauss3d_sb/undefined_output_runxxx/",
    #"initial_condition_ibm_test.pkl"
)
ref_state = load_state(fname, variables)


# ------------------------------------------------------------------------------
# current solution
#
fname = os.path.join(
    #"../icon-exclaim/build_cpu2py/experiments/exclaim_gauss3d_sb/",
    #"torus_insta_DOM01_ML_0002.nc",
    "../icon-exclaim/build_cpu2py/experiments/exclaim_gauss3d_sb/undefined_output_runxxx/",
    "000001_debug_after_diffu.pkl",
    #"../icon-exclaim/build_cpu2py/experiments/exclaim_gauss3d_sb/undefined_output_runxxx/",
    #"initial_condition_ibm.pkl"
)
cur_state = load_state(fname, variables)

# ------------------------------------------------------------------------------
# compare
#
for variable in variables:
    ref_shape = ref_state[variable].shape # needed for nproma in icon-exclaim
    print(
        f"Variable: {variable} max abs diff: {abs(ref_state[variable] - cur_state[variable][:ref_shape[0],:ref_shape[1]]).max()}"
    )
