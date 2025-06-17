import os, pickle
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

icon4py_dir = os.path.join(os.getcwd(), "../icon4py")

#-------------------------------------------------------------------------------
# Def function
#
def tridiag(
    vwind_impl_wgt,
    theta_v_ic,
    ddqz_z_half,
    z_alpha,
    z_beta,
    z_w_expl,
    z_exner_expl,
    z_q,
    dtime,
    cpd,
    ibm_cells,
):

    jc = ibm_cells

    ncells=theta_v_ic.shape[0]
    nlev=theta_v_ic.shape[1] - 1

    n_ibm = 5
    modify_matrix_loop = 1

    if modify_matrix_loop == 1:
        # apply IBM method modifying the matrix and solving the entire column
        theta_v_ic[jc, -5:] = 0
        z_w_expl[jc, -5:] = 0
        column_end = nlev
    elif modify_matrix_loop == 2:
        # reduce the depth of the column to the new surface
        column_end = nlev - n_ibm
    else:
        # do not edit the matrix, solve the entire column
        column_end = nlev


    w = np.zeros((ncells, nlev+1))
    for jk in range(1,column_end):

        gamma = dtime * cpd * vwind_impl_wgt * theta_v_ic[jc,jk] / ddqz_z_half[jc,jk]
        a = - gamma * z_beta[jc,jk-1] * z_alpha[jc,jk-1]
        c = - gamma * z_beta[jc,jk]   * z_alpha[jc,jk+1]
        b = 1 + gamma * z_alpha[jc,jk] * (z_beta[jc,jk-1] + z_beta[jc,jk])

        g = 1 / (b + a * z_q[jc,jk-1])
        z_q[jc,jk] = - c * g

        d = z_w_expl[jc,jk] - gamma * (z_exner_expl[jc,jk-1] - z_exner_expl[jc,jk])

        print(f"a: {a}, b: {b}, c: {c}, d: {d}")

        w[jc,jk] = (d - a * w[jc,jk-1]) * g

    for jk in range(column_end-1,0,-1):
        w[jc,jk] = w[jc,jk] + w[jc,jk+1]*z_q[jc,jk]

    return w

#-------------------------------------------------------------------------------
# Load data
#

fname = os.path.join(icon4py_dir, "runxx_w_tests", f"init_cond.pkl")
with open(fname, "rb") as ifile:
    state = pickle.load(ifile)
w_ini = state["w"]

fname = os.path.join(icon4py_dir, "runxx_w_tests", f"end_of_timestep_c_000000.pkl")
with open(fname, "rb") as ifile:
    state = pickle.load(ifile)
w_end_c = state["w"]

fname = os.path.join(icon4py_dir, "runxx_w_tests", f"w_matrix_c.pkl")
with open(fname, "rb") as ifile:
    state = pickle.load(ifile)

w_ref = state["w"]

vwind_impl_wgt = state["vwind_impl_wgt"][0]
theta_v_ic = state["theta_v_ic"]
ddqz_z_half = state["ddqz_z_half"]
z_alpha = state["z_alpha"]
z_beta = state["z_beta"]
z_w_expl = state["z_w_expl"]
z_exner_expl = state["z_exner_expl"]
z_q = state["z_q"]
dtime = state["dtime"]
cpd = state["cpd"]

#-------------------------------------------------------------------------------
# Solve system
#
ibm_cells = 13
w = tridiag( vwind_impl_wgt, theta_v_ic, ddqz_z_half, z_alpha, z_beta, z_w_expl, z_exner_expl, z_q, dtime, cpd, ibm_cells)

print((w[ibm_cells,:] - w_ref[ibm_cells,:]))
