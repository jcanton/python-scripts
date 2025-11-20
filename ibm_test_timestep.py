import os
import glob
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
            state[variable] = ds[variable].values[0, :, :].T
    elif fname.endswith(".pkl"):
        with open(fname, "rb") as ifile:
            data = pickle.load(ifile)
            for variable in variables:
                state[variable] = data[variable]
    return state


fname = "000039_*.pkl"
ref_path = "../ref_gauss3d_test"
cur_path = "../icon-exclaim.main/build_gpu2py_verify/experiments/exclaim_gauss3d_sb/undefined_output_runxxx"
#cur_path = "../icon-exclaim/build_gpu2py/experiments/exclaim_gauss3d_sb/undefined_output_runxxx"

# ------------------------------------------------------------------------------
# reference solution
#
ref_fpath = glob.glob(os.path.join(ref_path, fname))[0]
ref_state = load_state(ref_fpath, variables)

# ------------------------------------------------------------------------------
# current solution
#
#fname = "000003_*.pkl"
cur_fpath = glob.glob(os.path.join(cur_path, fname))[0]
cur_state = load_state(cur_fpath, variables)

# ------------------------------------------------------------------------------
# compare
#
print(f"comparing\n{ref_fpath}\nto\n{cur_fpath}\n")

for variable in variables:
    ref_shape = ref_state[variable].shape  # needed for nproma in icon-exclaim
    delta_field = abs(
        ref_state[variable] - cur_state[variable][: ref_shape[0], : ref_shape[1]]
    )
    print(
        f"Variable: {variable:8s} max abs diff: {delta_field.max():.21e} at location ({delta_field.argmax() // delta_field.shape[1]}, {delta_field.argmax() % delta_field.shape[1]})"
    )
