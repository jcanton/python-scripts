import glob
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

def process_files(args):

    output_files, out_path = args
    if os.path.exists(out_path):
        print(f"Skipping {out_path}, already exists.", flush=True)
        return  # Skip if already processed

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
    vn /= n
    w /= n
    rho /= n
    exner /= n
    theta_v /= n

    state_dict = {
        "vn": vn,
        "w": w,
        "rho": rho,
        "exner": exner,
        "theta_v": theta_v,
    }
    with open(out_path, "wb") as f:
        pickle.dump(state_dict, f)


# ===============================================================================
if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python whatever_this_is.py <num_workers> <python_files_dir>")
        sys.exit(1)
    num_workers = int(sys.argv[1])
    output_files_dir = sys.argv[2]


    avgs_dir = os.path.join(output_files_dir, 'avgs')
    if not os.path.exists(avgs_dir):
        os.makedirs(avgs_dir)

    # files
    output_files = glob.glob(os.path.join(output_files_dir, '??????_end_of_timestep_??????.pkl'))
    if len(output_files) == 0:
        output_files = glob.glob(os.path.join(output_files_dir, 'end_of_timestep_*.pkl'))
    output_files.sort()

    print(f"Using {num_workers} workers to process files in {output_files_dir}")
    print(f"Found {len(output_files)} output files in {output_files_dir}")

    # timestamps
    dt = 0.04
    out_int = 25
    sim_hours = 24
    files_per_hour = int(3600 / dt / out_int)

    # Prepare arguments for each file
    tasks = []
    for hour in range(sim_hours):
        filename = f"avg_hour{hour:03d}"
        out_path = os.path.join(avgs_dir, f"{filename}.pkl")
        tasks.append( (output_files[int(hour*files_per_hour):int((hour+1)*files_per_hour)], out_path) )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_files, tasks))

