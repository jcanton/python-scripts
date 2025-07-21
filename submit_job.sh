#!/bin/bash

#=============================================================================

#SBATCH --account=cwd01

#SBATCH --nodes=1
#SBATCH --uenv=icon/25.2:v3
#SBATCH --view=default

#SBATCH --partition=debug
#SBATCH --time=00:30:00

#SBATCH --job-name=export_figures

#SBATCH --output=log.%x.log
#SBATCH --error=log.%x.log

case $CLUSTER_NAME in
balfrin)
  export SCRATCH=/scratch/mch/jcanton
  export PROJECTS_DIR=$SCRATCH
  export ICON4PY_BACKEND="gtfn_gpu"
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
  export SCRATCH=/scratch/l_jcanton/
  export PROJECTS_DIR=/home/l_jcanton/projects/
  export ICON4PY_BACKEND="gtfn_cpu"
  ;;
*)
  echo "cluster name not recognized: ${CLUSTER_NAME}"
  ;;
esac
echo "Running on cluster: ${CLUSTER_NAME}"

export PYTHONOPTIMIZE=2
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE=1
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=$SCRATCH/gt4py_cache

source "$PROJECTS_DIR/icon4py/.venv/bin/activate"

export ICON4PY_OUTPUT_DIR=$SLURM_JOB_NAME
export ICON4PY_SAVEPOINT_PATH="ser_data/exclaim_gauss3d_250x250x250.uniform200_flat/ser_data"
export ICON4PY_GRID_FILE_PATH="testdata/grids/gauss3d_torus/Torus_Triangles_250m_x_250m_res1.25m.nc"
export TOTAL_WORKERS=$((SLURM_NNODES * SLURM_TASKS_PER_NODE))

# run python
python ibm_test_advection_time_series.py "$TOTAL_WORKERS"

echo "Finished running job: $SLURM_JOB_NAME, one way or another"
