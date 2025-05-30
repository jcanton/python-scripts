import os, glob
import pickle
import numpy as np

main_dir = os.getcwd() + "/../icon4py/"
#files_dir = main_dir + "run61_barray_2x2_nlev200_flatFaces/"
files_dir = main_dir + "run62_barray_4x4_nlev200_flatFaces/"
file_paths = glob.glob(os.path.join(files_dir, 'end_of_timestep_*.pkl'))
file_paths.sort()

# eliminate some
ini_tstamp = "036000" # hour 1
end_tstamp = "180000" # hour 5
#ini_tstamp = "180000" # hour 5
#end_tstamp = "324000" # hour 9
ini_id = [i for i, f in enumerate(file_paths) if ini_tstamp in f][0]
end_id = [i for i, f in enumerate(file_paths) if end_tstamp in f][0]
file_paths = file_paths[ini_id:end_id+1]

fileLabel = lambda file_path: file_path.split('/')[-1].split('.')[0]

avg_state = {}
counters = {}
for file_path in file_paths:

    with open(file_path, "rb") as f:
        print(f"processing {fileLabel(file_path)}")
        state = pickle.load(f)

        if not avg_state:
            for key, array in state.items():
                if isinstance(array, np.ndarray):
                    avg_state[key] = array
                    counters[key] = 1
        else:
            for key, array in state.items():
                if isinstance(array, np.ndarray):
                    avg_state[key] += array
                    counters[key] += 1

for key, array in avg_state.items():
    avg_state[key] = array / counters[key]

out_fname = f"avg_state_{fileLabel(file_paths[0])}-{fileLabel(file_paths[-1])}.pkl"
out_fpath = os.path.join(files_dir, out_fname)
with open(out_fpath, "wb") as f:
    pickle.dump(avg_state, f)

print(f"Saved average state to {out_fpath}")
