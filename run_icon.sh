#!/bin/bash

# =======================================
# USER-EDITABLE: Slurm job parameters
# =======================================
SLURM_ACCOUNT="cwd01"
SLURM_NODES=1

SLURM_UENV="icon/25.2:v3"
SLURM_UENV_VIEW="default"

SLURM_PARTITION="debug"
SLURM_TIME="00:30:00"

SLURM_JOBNAME="channel_950x350x100_5m_nlev20_leeMoser"

# =======================================
# USER-EDITABLE: Default run settings
# Positional args override:
#   ./job.sh [sim_type] [run_simulation] [run_postprocess]
#   sbatch job.sh [sim_type] [run_simulation] [run_postprocess]
# =======================================
sim_type="icon4py" # or "icon-exclaim"
run_simulation=true
run_postprocess=false

# Override defaults with positional args if provided
if [ -n "$1" ]; then sim_type="$1"; fi
if [ -n "$2" ]; then run_simulation="$2"; fi
if [ -n "$3" ]; then run_postprocess="$3"; fi

# ============================================================================
# Determine cluster name and options
#
case $CLUSTER_NAME in
balfrin)
  export SCRATCH=/scratch/mch/jcanton
  export PROJECTS_DIR=$SCRATCH
  export ICON4PY_BACKEND="gtfn_gpu"
  export SLURM_UENV_VIEW="$SLURM_UENV_VIEW,modules"
  module load gcc-runtime
  module load nvhpc
  ;;
santis)
  export SCRATCH=/capstor/scratch/cscs/jcanton
  export PROJECTS_DIR=$SCRATCH
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/nvhpc-25.1-tsfur7lqj6njogdqafhpmj5dqltish7t/Linux_aarch64/25.1/compilers/lib
  export ICON4PY_BACKEND="gtfn_gpu"
  ;;
squirrel)
  export SCRATCH=/scratch/l_jcanton
  export PROJECTS_DIR=/home/l_jcanton/projects
  export ICON4PY_BACKEND="gtfn_cpu"
  ;;
mac)
  export SCRATCH=/Users/jcanton/projects
  export PROJECTS_DIR=/Users/jcanton/projects
  export ICON4PY_BACKEND="gtfn_cpu"
  ;;
*)
  echo "cluster name not recognized: ${CLUSTER_NAME}"
  ;;
esac

# After SCRATCH is known, define the log directory
SLURM_LOGDIR="${SCRATCH}/logs"

# =======================================
# Wrapper: If not in Slurm, submit ourselves
# =======================================
if [ -z "$SLURM_JOB_ID" ]; then
  # Timestamp for unique log files
  timestamp=$(date +"%Y%m%d_%H%M%S")

  # Pick log suffix based on sim type + booleans
  if [ "$run_simulation" = true ] && [ "$run_postprocess" = true ]; then
    log_suffix="${sim_type}_both"
  elif [ "$run_simulation" = true ]; then
    log_suffix="${sim_type}_sim"
  elif [ "$run_postprocess" = true ]; then
    log_suffix="${sim_type}_post"
  else
    log_suffix="${sim_type}_idle"
  fi

  # override to debug queue if only postprocessing
  if [ "$run_postprocess" = true ] && [ "$run_simulation" = false ]; then
    SLURM_PARTITION="debug"
    SLURM_TIME="00:30:00"
  fi

  # Ensure log dir exists
  mkdir -p "$SLURM_LOGDIR"

  # Submit self to Slurm with parameters preserved
  sbatch \
    --account="$SLURM_ACCOUNT" \
    --nodes="$SLURM_NODES" \
    --uenv="$SLURM_UENV" \
    --view="$SLURM_UENV_VIEW" \
    --partition="$SLURM_PARTITION" \
    --time="$SLURM_TIME" \
    --job-name="$SLURM_JOBNAME" \
    --output="$SLURM_LOGDIR/%x_${log_suffix}_${timestamp}.log" \
    --error="$SLURM_LOGDIR/%x_${log_suffix}_${timestamp}.log" \
    "$0" "$sim_type" "$run_simulation" "$run_postprocess"
  exit
fi

# ==============================================================================
# Environment setup
#
export ICON4PY_SAVEPOINT_PATH="ser_data/exclaim_channel_950x350x100_5m_nlev20/ser_data"
export ICON4PY_GRID_FILE_PATH="testdata/grids/gauss3d_torus/Channel_950m_x_350m_res5m.nc"
export TOTAL_WORKERS=$((SLURM_NNODES * SLURM_TASKS_PER_NODE))

export ICON4PY_DIR=$PROJECTS_DIR/icon4py.ibm
export SCRIPTS_DIR=$PROJECTS_DIR/python-scripts
export ICON_EXCLAIM_DIR=$PROJECTS_DIR/icon-exclaim

# Unified output dir (per sim_type)
export OUTPUT_DIR=$SCRATCH/runs/$sim_type/$SLURM_JOB_NAME
mkdir -p "$OUTPUT_DIR"

# For icon4py, use environment variable (driver has no --output_dir)
export ICON4PY_OUTPUT_DIR="$OUTPUT_DIR"

echo ""
echo "Running on cluster   = $CLUSTER_NAME"
echo "SLURM_JOB_ID         = $SLURM_JOB_ID"
echo "sim_type             = $sim_type"
echo "run_simulation       = $run_simulation"
echo "run_postprocess      = $run_postprocess"
echo "OUTPUT_DIR           = $OUTPUT_DIR"
echo ""

# ==============================================================================
# Run simulation
#
if [ "$run_simulation" = true ]; then
  case $sim_type in
  icon4py)
    echo "[INFO] Running icon4py simulation..."

    cd "$ICON4PY_DIR" || exit
    source .venv/bin/activate

    export PYTHONOPTIMIZE=2
    export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE=1
    export GT4PY_BUILD_CACHE_LIFETIME=persistent
    export GT4PY_BUILD_CACHE_DIR=$SCRATCH/gt4py_cache

    python \
      model/driver/src/icon4py/model/driver/icon4py_driver.py \
      $ICON4PY_SAVEPOINT_PATH \
      --icon4py_driver_backend="$ICON4PY_BACKEND" \
      --experiment_type=gauss3d_torus \
      --grid_root=2 --grid_level=0 \
      --enable_output
    ;;

  icon-exclaim)
    echo "[INFO] Preparing and running icon-exclaim simulation..."

    experiment_name="exclaim_channel"
    build_folder="build_acc"

    cd "$ICON_EXCLAIM_DIR" || exit
    cp run/exp.${experiment_name} ${build_folder}/run/

    cd ${build_folder} || exit
    ./make_runscripts ${experiment_name}

    rm -rf experiments/${experiment_name}/*
    rm -f run/LOG.exp.${experiment_name}.run.*

    cd run || exit

    # run the experiment script directly under current allocation
    echo "[INFO] Launching icon-exclaim with srun..."
    srun ./exp.${experiment_name}.run

    # collect logs + outputs into unified output dir
    mkdir -p "$OUTPUT_DIR"
    cp LOG.exp.${experiment_name}.run.* "$OUTPUT_DIR"/
    cp -r ../experiments/${experiment_name}/* "$OUTPUT_DIR"/
    ;;
  esac
fi

# ==============================================================================
# Postprocess
#
if [ "$run_postprocess" = true ]; then
  echo "[INFO] Running postprocess..."

  if [ "$sim_type" = "icon4py" ]; then
    if [ -n "$VIRTUAL_ENV" ]; then deactivate; fi
    source "$SCRIPTS_DIR/.venv/bin/activate"

    # generate vtu files
    python "$SCRIPTS_DIR/plot_vtk.py" "$TOTAL_WORKERS" "$OUTPUT_DIR" "$ICON4PY_SAVEPOINT_PATH" "$ICON4PY_GRID_FILE_PATH"

    # compute temporal averages
    python "$SCRIPTS_DIR/temporal_average.py" "$TOTAL_WORKERS" "$OUTPUT_DIR"
  else
    echo "[WARN] No postprocessing pipeline defined for $sim_type"
  fi
fi

# ==============================================================================
echo "Finished running job: $SLURM_JOB_NAME"
