import glob
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import gt4py.next as gtx
import numpy as np
import icon4py_plots


def process_files(args):
    output_files, out_path, savepoint_path, grid_file_path = args

    # Actually don't skip, to allow re-processing because I am computing them
    # while the sim is running (but some are now very time-consuming to
    # re-compute all the time)
    #if os.path.exists(out_path):
    #    print(f"Skipping {out_path}, already exists.", flush=True)
    #    return  # Skip if already processed

    num_files = len(output_files)
    print(f"Processing {num_files} files to create {out_path}", flush=True)

    if num_files == 0:
        return

    # NOTE: plot must be created in each process, as it is not picklable.
    plot = icon4py_plots.Plot(
        savepoint_path=savepoint_path,
        grid_file_path=grid_file_path,
        backend=gtx.gtfn_cpu,
    )

    vn = w = rho = exner = theta_v = None

    for i, file_path in enumerate(output_files):
        with open(file_path, "rb") as ifile:
            state = pickle.load(ifile)
            if i == 0:
                vn = state["vn"].copy()
                w = state["w"].copy()
                rho = state["rho"].copy()
                exner = state["exner"].copy()
                theta_v = state["theta_v"].copy()
            else:
                vn += state["vn"]
                w += state["w"]
                rho += state["rho"]
                exner += state["exner"]
                theta_v += state["theta_v"]

    vn /= num_files
    w /= num_files
    rho /= num_files
    exner /= num_files
    theta_v /= num_files

    u_cf, v_cf = plot._vec_interpolate_to_cell_center(vn)
    w_cf = plot._scal_interpolate_to_full_levels(w)

    state_dict = {
        "vn": vn,
        "w": w,
        "rho": rho,
        "exner": exner,
        "theta_v": theta_v,
        "wind_cf": np.stack([u_cf, v_cf, w_cf], axis=-1),
    }
    with open(out_path, "wb") as f:
        pickle.dump(state_dict, f)


# ===============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 7:
        print(
            "Usage: python temporal_average.py <num_workers> <python_files_dir> <savepoint_path> <grid_file_path> <dtime> <plot_frequency>"
        )
        sys.exit(1)
    num_workers = int(sys.argv[1])
    output_files_dir = sys.argv[2]
    savepoint_path = sys.argv[3]
    grid_file_path = sys.argv[4]
    dtime = float(sys.argv[5])
    plot_frequency = int(sys.argv[6])

    avgs_dir = os.path.join(output_files_dir, "avgs")
    if not os.path.exists(avgs_dir):
        os.makedirs(avgs_dir)

    # files
    output_files = glob.glob(
        os.path.join(output_files_dir, "??????_end_of_timestep_*pkl")
    )
    if len(output_files) == 0:
        output_files = glob.glob(
            os.path.join(output_files_dir, "end_of_timestep_*.pkl")
        )
    output_files.sort()

    print("")
    print(f"Using {num_workers} workers")
    print(f"Output files directory: {output_files_dir}")
    print(f"Savepoint path: {savepoint_path}")
    print(f"Grid file path: {grid_file_path}")
    print(f"Found {len(output_files)} output files in {output_files_dir}")
    print("")

    # timestamps
    sim_hours = 10
    files_per_hour = int(3600 / dtime / plot_frequency)

    # Prepare arguments for each file
    tasks = []
    for hour in range(sim_hours):
        filename = f"avg_hour{hour:03d}"
        out_path = os.path.join(avgs_dir, f"{filename}.pkl")
        tasks.append(
            (
                output_files[
                    int(hour * files_per_hour) : int((hour + 1) * files_per_hour)
                ],
                out_path,
                savepoint_path,
                grid_file_path,
            )
        )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_files, tasks))
