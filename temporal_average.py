import glob
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import gt4py.next as gtx
import numpy as np
from icon4py.model.common.io import plots


def process_files(args):
    output_files, out_path, savepoint_path, grid_file_path = args

    # Actually don't skip, to allow re-processing because I am computing them
    # while the sim is running
    #if os.path.exists(out_path):
    #    print(f"Skipping {out_path}, already exists.", flush=True)
    #    return  # Skip if already processed

    # NOTE: plot must be created in each process, as it is not picklable.
    plot = plots.Plot(
        savepoint_path=savepoint_path,
        grid_file_path=grid_file_path,
        backend=gtx.gtfn_cpu,
    )

    print(f"Processing {len(output_files)} files to create {out_path}", flush=True)

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

    n = len(output_files)
    if n > 0:
        vn /= n
        w /= n
        rho /= n
        exner /= n
        theta_v /= n

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
    if len(sys.argv) < 5:
        print(
            "Usage: python temporal_average.py <num_workers> <python_files_dir> <savepoint_path> <grid_file_path>"
        )
        sys.exit(1)
    num_workers = int(sys.argv[1])
    output_files_dir = sys.argv[2]
    savepoint_path = sys.argv[3]
    grid_file_path = sys.argv[4]

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
    dt = 0.04
    out_int = 1500
    sim_hours = 10
    files_per_hour = int(3600 / dt / out_int)

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
