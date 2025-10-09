import os
import pickle

# ------------------------------------------------------------------------------
# mesh and levels
#
plotting_data_file = "plotting_channel_950x350x100_5m_nlev20.pkl"

with open(os.path.join("data", plotting_data_file), "rb") as f:
    plotting = pickle.load(f)
    tri = plotting["tri"]
    full_level_heights = plotting["full_level_heights"]
    half_level_heights = plotting["half_level_heights"]
    full_cell_mask = plotting["full_cell_mask"]
    half_cell_mask = plotting["half_cell_mask"]
    full_edge_mask = plotting["full_edge_mask"]
    half_edge_mask = plotting["half_edge_mask"]
full_levels = full_level_heights[0, :]
half_levels = half_level_heights[0, :]

num_cells = len(tri.cell_x)
num_levels = len(full_levels)

# ------------------------------------------------------------------------------
# handling variables
#
variables = ["vn", "w", "rho", "exner", "theta_v"]


def load_state(fname, variables):
    state = {}
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
    "../runs/icon4py/test_channel_950m_x_350m_res5m_nlev20.reference",
    "end_of_timestep_000000009.pkl",
)
ref_state = load_state(fname, variables)


# ------------------------------------------------------------------------------
# current solution
#
fname = os.path.join(
    "../runs/icon4py/test_channel_950m_x_350m_res5m_nlev20.updated",
    "end_of_timestep_000000009.pkl",
)
cur_state = load_state(fname, variables)

# ------------------------------------------------------------------------------
# compare
#
for variable in variables:
    print(
        f"Variable: {variable} max abs diff: {abs(ref_state[variable] - cur_state[variable]).max()}"
    )
