#!/bin/bash

# =======================================
# USER-EDITABLE: Slurm job parameters
# =======================================
SLURM_NODES=1

SLURM_UENV="icon/25.2:v3"
SLURM_UENV_VIEW="default"

SLURM_PARTITION="normal"
#SLURM_TIME="1-00:00:00"
SLURM_TIME="1:00:00"

#SLURM_JOBNAME="channel_950m_x_350m_res1m_nlev100_vdiff00001"
#SLURM_JOBNAME="channel_950m_x_350m_multibuilding_res1.5m_nlev64_vdiff00050"
#SLURM_JOBNAME="test_channel_950m_x_350m_res5m_nlev20.reference"
#SLURM_JOBNAME="test_channel_ibm"

SLURM_JOBNAME="test_channel_blueline"

# =======================================
# USER-EDITABLE: Default run settings
# Positional args override:
#   ./job.sh [sim_type] [run_simulation] [run_postprocess]
#   sbatch job.sh [sim_type] [run_simulation] [run_postprocess]
# =======================================
sim_type="iconf90" # or "iconf90"
run_simulation=true
run_postprocess=true

# Override defaults with positional args if provided
if [ -n "$1" ]; then sim_type="$1"; fi
if [ -n "$2" ]; then run_simulation="$2"; fi
if [ -n "$3" ]; then run_postprocess="$3"; fi
if [ -n "$4" ]; then SLURM_JOBNAME="$4"; fi

# ============================================================================
# Determine cluster name and options
#
case $CLUSTER_NAME in
balfrin)
    SLURM_ACCOUNT="s83"
    export SCRATCH=/scratch/mch/jcanton
    export PROJECTS_DIR=$SCRATCH
    export ICON4PY_BACKEND="gtfn_gpu"
    ;;
santis)
    SLURM_ACCOUNT="cwd01"
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
    #timestamp=""

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

    # Ensure log dir exists
    mkdir -p "$SLURM_LOGDIR"

    # If running on mac, skip sbatch and run directly
    if [ "$CLUSTER_NAME" = "mac" ]; then
        echo "[INFO] Running locally on macOS, skipping sbatch."
        # Continue script execution (do not exit)
    elif [ "$sim_type" != "iconf90" ]; then
        # Submit self to Slurm with parameters preserved
        sbatch \
            --account="$SLURM_ACCOUNT" \
            --nodes="$SLURM_NODES" \
            --uenv="$SLURM_UENV" \
            --view="$SLURM_UENV_VIEW" \
            --partition="$SLURM_PARTITION" \
            --time="$SLURM_TIME" \
            --job-name="${log_suffix}_$SLURM_JOBNAME" \
            --output="$SLURM_LOGDIR/%x_${timestamp}.log" \
            --error="$SLURM_LOGDIR/%x_${timestamp}.log" \
            "$0" "$sim_type" "$run_simulation" "$run_postprocess" "$SLURM_JOBNAME"
        exit
    fi
fi

# ==============================================================================
# Environment setup
#
export TOTAL_WORKERS=$((SLURM_NNODES * SLURM_TASKS_PER_NODE))

export ICON4PY_DIR=$PROJECTS_DIR/icon4py
export SCRIPTS_DIR=$PROJECTS_DIR/python-scripts
export ICONF90_DIR=$PROJECTS_DIR/icon-exclaim

# ------------------------------------------------------------------------------
# python
#
export ICON4PY_RESTART_FREQUENCY=10000
export ICON4PY_CHANNEL_PERTURBATION=0.001

if [[ "$SLURM_JOBNAME" == *vdiff* ]]; then
    # get the diffusion coefficient from the job name
    diff_digits="${SLURM_JOBNAME##*vdiff}"
    export ICON4PY_DIFFU_COEFF="0.${diff_digits}"
else
    export ICON4PY_DIFFU_COEFF="0.0"
fi

case $SLURM_JOBNAME in
*res5m*)
    export ICON4PY_SAVEPOINT_PATH="${ICON4PY_DIR}/ser_data/exclaim_channel_950x350x100_5m_nlev20/ser_data"
    export ICON4PY_GRID_FILE_PATH="${ICON4PY_DIR}/testdata/grids/gauss3d_torus/Channel_950m_x_350m_res5m.nc"
    export ICON4PY_PLOT_FREQUENCY=1500
    export ICON4PY_NUM_LEVELS=20
    export ICON4PY_DTIME=0.04
    ;;
*res2.5m*)
    export ICON4PY_SAVEPOINT_PATH="${ICON4PY_DIR}/ser_data/exclaim_channel_950x350x100_2.5m_nlev40/ser_data"
    export ICON4PY_GRID_FILE_PATH="${ICON4PY_DIR}/testdata/grids/gauss3d_torus/Channel_950m_x_350m_res2.5m.nc"
    export ICON4PY_PLOT_FREQUENCY=3000
    export ICON4PY_NUM_LEVELS=40
    export ICON4PY_DTIME=0.02
    ;;
*res1.5m*)
    export ICON4PY_SAVEPOINT_PATH="${ICON4PY_DIR}/ser_data/exclaim_channel_950x350x100_1.5m_nlev64/ser_data"
    export ICON4PY_GRID_FILE_PATH="${ICON4PY_DIR}/testdata/grids/gauss3d_torus/Channel_950m_x_350m_res1.5m.nc"
    export ICON4PY_PLOT_FREQUENCY=6000
    export ICON4PY_NUM_LEVELS=64
    export ICON4PY_DTIME=0.01
    ;;
*res1.25m*)
    export ICON4PY_SAVEPOINT_PATH="${ICON4PY_DIR}/ser_data/exclaim_channel_950x350x100_1.25m_nlev80/ser_data"
    export ICON4PY_GRID_FILE_PATH="${ICON4PY_DIR}/testdata/grids/gauss3d_torus/Channel_950m_x_350m_res1.25m.nc"
    export ICON4PY_PLOT_FREQUENCY=6000
    export ICON4PY_NUM_LEVELS=80
    export ICON4PY_DTIME=0.01
    ;;
*res1m*)
    export ICON4PY_SAVEPOINT_PATH="${ICON4PY_DIR}/ser_data/exclaim_channel_950x350x100_1m_nlev100/ser_data"
    export ICON4PY_GRID_FILE_PATH="${ICON4PY_DIR}/testdata/grids/gauss3d_torus/Channel_950m_x_350m_res1m.nc"
    export ICON4PY_PLOT_FREQUENCY=6000
    export ICON4PY_NUM_LEVELS=100
    export ICON4PY_DTIME=0.01
    ;;
*test_channel_ibm*)
    export ICON4PY_SAVEPOINT_PATH="${ICON4PY_DIR}/testdata/ser_icondata/mpitask1/gauss3d_torus/ser_data"
    export ICON4PY_GRID_FILE_PATH="${ICON4PY_DIR}/testdata/grids/torus_50000x5000_res500/Torus_Triangles_50000m_x_5000m_res500m.nc"
    export ICON4PY_NUM_LEVELS="35"
    export ICON4PY_END_DATE="0001-01-01T00:00:08"
    export ICON4PY_DTIME="4.0"
    export ICON4PY_PLOT_FREQUENCY="1"
    export ICON4PY_CHANNEL_SPONGE_LENGTH="5000.0"
    export ICON4PY_CHANNEL_PERTURBATION="0.0"
    export ICON4PY_DIFFU_COEFF="0.001"
    ;;
*)
    if [ "$run_simulation" = "icon4py" ]; then
        echo "invalid jobname"
        exit 1
    fi
    ;;
esac

case $SLURM_JOBNAME in
*multibuilding*)
    export ICON4PY_PLOT_FREQUENCY=100
    ;;
esac

# ------------------------------------------------------------------------------
# fortran
export ICONF90_EXPERIMENT_NAME="exclaim_channel"
export ICONF90_BUILD_FOLDER="build_gpu2py"

# ------------------------------------------------------------------------------
# Unified output dir (per sim_type)
export OUTPUT_DIR=$SCRATCH/runs/$sim_type/$SLURM_JOBNAME
mkdir -p "$OUTPUT_DIR"

# ------------------------------------------------------------------------------
echo ""
echo "[INFO] Running on cluster   = $CLUSTER_NAME"
echo "[INFO] SLURM_JOB_ID         = $SLURM_JOB_ID"
echo "[INFO] sim_type             = $sim_type"
echo "[INFO] run_simulation       = $run_simulation"
echo "[INFO] run_postprocess      = $run_postprocess"
echo "[INFO] OUTPUT_DIR           = $OUTPUT_DIR"
echo ""

# ==============================================================================
# Run simulation
#
if [ "$run_simulation" = true ]; then
    case $sim_type in
    icon4py)
        echo "[INFO] Running icon4py simulation..."
        echo "[INFO] ICON4PY_DIR = ${ICON4PY_DIR}"
        echo "[INFO] ICON4PY_SAVEPOINT_PATH = ${ICON4PY_SAVEPOINT_PATH}"
        echo "[INFO] ICON4PY_GRID_FILE_PATH = ${ICON4PY_GRID_FILE_PATH}"
        echo "[INFO] ICON4PY_PLOT_FREQUENCY = ${ICON4PY_PLOT_FREQUENCY}"
        echo "[INFO] ICON4PY_NUM_LEVELS     = ${ICON4PY_NUM_LEVELS}"
        echo "[INFO] ICON4PY_DTIME          = ${ICON4PY_DTIME}"
        echo "[INFO] ICON4PY_DIFFU_COEFF    = ${ICON4PY_DIFFU_COEFF}"
        echo ""

        cd "$ICON4PY_DIR" || exit
        source .venv/bin/activate

        export PYTHONOPTIMIZE=2
        export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE=1
        export GT4PY_BUILD_CACHE_LIFETIME=persistent
        export GT4PY_BUILD_CACHE_DIR=$SCRATCH/gt4py_cache

        export ICON4PY_OUTPUT_DIR="$OUTPUT_DIR"

        python \
            model/driver/src/icon4py/model/driver/icon4py_driver.py \
            "$ICON4PY_SAVEPOINT_PATH" \
            --icon4py_driver_backend="$ICON4PY_BACKEND" \
            --experiment_type=gauss3d_torus \
            --grid_file="$ICON4PY_GRID_FILE_PATH" #\
        #--enable_output
        ;;

    iconf90)
        echo "[INFO] Preparing and running iconf90 simulation..."
        echo "[INFO] ICONF90_DIR             = ${ICONF90_DIR}"
        echo "[INFO] ICONF90_EXPERIMENT_NAME = ${ICONF90_EXPERIMENT_NAME}"
        echo "[INFO] ICONF90_BUILD_FOLDER    = ${ICONF90_BUILD_FOLDER}"
        echo ""

        cd "$ICONF90_DIR" || exit
        cp run/exp.${ICONF90_EXPERIMENT_NAME} ${ICONF90_BUILD_FOLDER}/run/

        cd ${ICONF90_BUILD_FOLDER} || exit
        ./make_runscripts ${ICONF90_EXPERIMENT_NAME}

        cd run || exit

        # add/fix slurm stuff
        sed -i '/#SBATCH --job-name=/i #SBATCH --uenv='"$SLURM_UENV" exp.${ICONF90_EXPERIMENT_NAME}.run
        sed -i '/#SBATCH --job-name=/i #SBATCH --view='"$SLURM_UENV_VIEW" exp.${ICONF90_EXPERIMENT_NAME}.run
        sed -i '/#SBATCH --job-name=/i #SBATCH --account='"$SLURM_ACCOUNT" exp.${ICONF90_EXPERIMENT_NAME}.run
        sed -i '/#SBATCH --job-name=/i #SBATCH --time='"$SLURM_TIME" exp.${ICONF90_EXPERIMENT_NAME}.run
        sed -i '/#SBATCH --partition=/c\#SBATCH --partition='"$SLURM_PARTITION" exp.${ICONF90_EXPERIMENT_NAME}.run
        sed -i '/#SBATCH --nodes=/c\#SBATCH --nodes='"$SLURM_NODES" exp.${ICONF90_EXPERIMENT_NAME}.run
        #sed -i '/#SBATCH --ntasks-per-node=/c\#SBATCH --ntasks-per-node=4' exp.${ICONF90_EXPERIMENT_NAME}.run
        #sed -i '/export mpi_procs_pernode/c\export mpi_procs_pernode=4' exp.${ICONF90_EXPERIMENT_NAME}.run

        # submit the experiment
        echo "[INFO] Queuing iconf90 with sbatch..."
        output=$(sbatch exp.${ICONF90_EXPERIMENT_NAME}.run)
        job_id=$(echo "$output" | awk '{print $4}')
        logfile="LOG.exp.${ICONF90_EXPERIMENT_NAME}.run.${job_id}.o"

        # create postpro job
        postpro_script="move_outputs_${ICONF90_EXPERIMENT_NAME}.sh"
        cat <<EOF >"$postpro_script"
#!/bin/bash
mv $ICONF90_DIR/${ICONF90_BUILD_FOLDER}/experiments/${ICONF90_EXPERIMENT_NAME}/* $OUTPUT_DIR/
mv $ICONF90_DIR/${ICONF90_BUILD_FOLDER}/run/${logfile} $OUTPUT_DIR/logfile.log
EOF
        chmod +x "$postpro_script"

        # submit postpro job
        sbatch \
            --account="$SLURM_ACCOUNT" \
            --partition=debug \
            --time=00:10:00 \
            --dependency=afterany:"${job_id}" \
            --job-name="move_outputs_${ICONF90_EXPERIMENT_NAME}" \
            --output="postpro.log" \
            --error="postpro.log" \
            "$postpro_script"
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

        # compute temporal averages
        #python "$SCRIPTS_DIR/temporal_average.py" "$TOTAL_WORKERS"
        python "$SCRIPTS_DIR/temporal_average.py" " 24 " "$OUTPUT_DIR" "$ICON4PY_SAVEPOINT_PATH" "$ICON4PY_GRID_FILE_PATH" "$ICON4PY_DTIME" "$ICON4PY_PLOT_FREQUENCY"

        # generate vtu files
        python "$SCRIPTS_DIR/plot_vtk.py" " 24 " "$OUTPUT_DIR" "$ICON4PY_SAVEPOINT_PATH" "$ICON4PY_GRID_FILE_PATH"
    else
        echo "[WARN] No postprocessing pipeline defined for $sim_type"
    fi
fi
