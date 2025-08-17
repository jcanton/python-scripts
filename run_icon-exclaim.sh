#!/bin/bash

#===============================================================================
#
icon_folder="/capstor/scratch/cscs/jcanton/icon-exclaim"
#experiment_name="exclaim_gauss3d_sb"
experiment_name="exclaim_channel"
build_folder="build_acc" # build_serialize / build_acc

#===============================================================================
#
PWD=$(pwd)

cd ${icon_folder} || exit
cp run/exp.${experiment_name} ${build_folder}/run/

cd ${build_folder} || exit

./make_runscripts ${experiment_name}

rm -rf experiments/${experiment_name}/*
rm run/LOG.exp.${experiment_name}.run.*

#===============================================================================
#
cd run || exit

# add uenv and account
sed -i '/#SBATCH --job-name=/i #SBATCH --uenv=icon/25.2:v3' exp.${experiment_name}.run
sed -i '/#SBATCH --job-name=/i #SBATCH --view=default' exp.${experiment_name}.run
sed -i '/#SBATCH --job-name=/i #SBATCH --account=cwd01' exp.${experiment_name}.run

# serialization settings
sed -i '/#SBATCH --job-name=/i #SBATCH --time=0:30:00' exp.${experiment_name}.run
sed -i '/#SBATCH --partition=/c\#SBATCH --partition=debug' exp.${experiment_name}.run
sed -i '/#SBATCH --nodes=/c\#SBATCH --nodes=1' exp.${experiment_name}.run
sed -i '/#SBATCH --ntasks-per-node=/c\#SBATCH --ntasks-per-node=1' exp.${experiment_name}.run
sed -i '/export mpi_procs_pernode/c\export mpi_procs_pernode=1' exp.${experiment_name}.run

## performance settings
#sed -i '/#SBATCH --job-name=/i #SBATCH --time=1:30:00' exp.${experiment_name}.run
#sed -i '/#SBATCH --partition=/c\#SBATCH --partition=normal' exp.${experiment_name}.run
#sed -i '/#SBATCH --nodes=/c\#SBATCH --nodes=1' exp.${experiment_name}.run
#sed -i '/#SBATCH --ntasks-per-node=/c\#SBATCH --ntasks-per-node=1' exp.${experiment_name}.run
#sed -i '/export mpi_procs_pernode/c\export mpi_procs_pernode=1' exp.${experiment_name}.run

output=$(sbatch exp.${experiment_name}.run)
job_id="${output:20:8}"

logfile="LOG.exp.${experiment_name}.run.${job_id}.o"
until [ -e "${logfile}" ]; do
  echo "waiting for ${logfile}"
  sleep 0.1
done
tail -f "${logfile}"

cd ${build_folder} || exit
cp run/${logfile} experiments/${experiment_name}/

#===============================================================================
#
cd ${PWD} || exit
