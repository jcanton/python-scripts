#!/usr/bin/env python3
"""Run serialized MPI jobs, collect ser_data, and archive outputs."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ======================================
# USER CONFIGURATION
# ======================================
MPI_RANKS: List[int] = [1, 2, 4]

@dataclass(frozen=True)
class Experiment:
	name: str
	output_dir_name: str
	tar_name: str


# There is no consistency in the following names, so be careful when editing..
# The issue is that icon4py has hardcoded experiment names/folders in various
# places, and funny things such as '_torus' identifying the grid type..
EXPERIMENTS: List[Experiment] = [
	Experiment(
		name="exclaim_ch_r04b09_dsl_sb",
		output_dir_name="mch_ch_r04b09_dsl",
		tar_name="mch_ch_r04b09_dsl",
	),
	Experiment(
		name="exclaim_nh53_tri_jws_sb",
		output_dir_name="jabw_R02B04",
		tar_name="jabw_R02B04",
	),
	Experiment(
		name="exclaim_ape_R02B04_sb",
		output_dir_name="exclaim_ape_R02B04",
		tar_name="exclaim_ape_R02B04",
	),
	Experiment(
		name="exclaim_gauss3d_sb",
		output_dir_name="gauss3d_torus",
		tar_name="gauss_3d",
	),
	Experiment(
		name="exclaim_nh_weisman_klemp_sb",
		output_dir_name="weisman_klemp_torus",
		tar_name="weisman_klemp_torus",
	),
]

# Base directories (adjust if needed)
PROJECTS_DIR = Path(os.environ.get("SCRATCH", str(Path.home() / "projects")))
ICONF90_DIR = PROJECTS_DIR / "icon-exclaim.serialize"
ICONF90_BUILD_FOLDER = "build_serialize"

# Derived paths
BUILD_DIR = ICONF90_DIR / ICONF90_BUILD_FOLDER
RUNSCRIPTS_DIR = BUILD_DIR / "run"
EXPERIMENTS_DIR = BUILD_DIR / "experiments"

# Slurm settings
SBATCH_PARTITION = "debug"
SBATCH_TIME = "00:15:00"
SBATCH_ACCOUNT = "cwd01"
SBATCH_UENV = "icon/25.2:v3"
SBATCH_UENV_VIEW = "default"
POLL_SECONDS = 10

# Output location for copied ser_data and tarballs
OUTPUT_ROOT = EXPERIMENTS_DIR / "serialized_runs"

# ======================================
# END USER CONFIGURATION
# ======================================


def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
	return subprocess.run(cmd, check=check, text=True, capture_output=True)


def update_slurm_variables(script_path: Path) -> None:
	"""Update SBATCH directives in the Slurm script (partition, account, time, uenv, view)."""
	original = script_path.read_text()
	updated = original
	
	# Find the position after #SBATCH --job-name= line
	job_name_match = re.search(r"^#SBATCH\s+--job-name=.*$", updated, flags=re.MULTILINE)
	if not job_name_match:
		raise RuntimeError("Could not find #SBATCH --job-name= line in script")
	
	# Prepare the new SBATCH lines to insert
	new_lines = (
		f"#SBATCH --partition={SBATCH_PARTITION}\n"
		f"#SBATCH --account={SBATCH_ACCOUNT}\n"
		f"#SBATCH --time={SBATCH_TIME}\n"
		f"#SBATCH --uenv='{SBATCH_UENV}'\n"
		f"#SBATCH --view='{SBATCH_UENV_VIEW}'"
	)
	
	# Remove existing partition, account, time, uenv, and view lines if they exist
	updated = re.sub(r"^#SBATCH\s+--partition=.*$\n?", "", updated, flags=re.MULTILINE)
	updated = re.sub(r"^#SBATCH\s+--account=.*$\n?", "", updated, flags=re.MULTILINE)
	updated = re.sub(r"^#SBATCH\s+--time=.*$\n?", "", updated, flags=re.MULTILINE)
	updated = re.sub(r"^#SBATCH\s+--uenv=.*$\n?", "", updated, flags=re.MULTILINE)
	updated = re.sub(r"^#SBATCH\s+--view=.*$\n?", "", updated, flags=re.MULTILINE)
	
	# Re-find job-name position in the cleaned text
	job_name_match = re.search(r"^(#SBATCH\s+--job-name=.*$)", updated, flags=re.MULTILINE)
	if not job_name_match:
		raise RuntimeError("Could not find #SBATCH --job-name= line in script")
	
	# Insert new lines after the job-name line
	insertion_point = job_name_match.end()
	updated = updated[:insertion_point] + "\n" + new_lines + updated[insertion_point:]
	
	script_path.write_text(updated)


def update_slurm_ranks(script_path: Path, ranks: int) -> None:
	"""Update ranks in the Slurm script (ntasks-per-node and mpi_procs_pernode)."""
	original = script_path.read_text()

	updated = original
	
	# Update #SBATCH --ntasks-per-node=X
	updated = re.sub(
		r"^#SBATCH\s+--ntasks-per-node\s*=\s*\d+\s*$",
		f"#SBATCH --ntasks-per-node={ranks}",
		updated,
		flags=re.MULTILINE,
	)
	
	# Update : ${no_of_nodes:=1} ${mpi_procs_pernode:=X}
	updated = re.sub(
		r"^:\s+\$\{no_of_nodes:=\d+\}\s+\$\{mpi_procs_pernode:=\d+\}\s*$",
		f": ${{no_of_nodes:=1}} ${{mpi_procs_pernode:={ranks}}}",
		updated,
		flags=re.MULTILINE,
	)

	script_path.write_text(updated)


def submit_job(script_path: Path) -> str:
	cmd = ["sbatch", str(script_path)]
	result = run_command(cmd)
	match = re.search(r"Submitted batch job\s+(\d+)", result.stdout)
	if not match:
		raise RuntimeError(f"Unable to parse job id from sbatch output: {result.stdout}")
	return match.group(1)


def normalize_state(raw_state: str) -> str:
	cleaned = raw_state.strip().upper()
	cleaned = cleaned.split("+")[0]
	cleaned = cleaned.split(":")[0]
	return cleaned


def get_job_state(job_id: str) -> Optional[str]:
	# First try sacct for completed jobs
	try:
		result = run_command(["sacct", "-j", job_id, "--format=State", "--noheader"], check=False)
		if result.stdout.strip():
			return normalize_state(result.stdout.strip().splitlines()[0])
	except FileNotFoundError:
		pass

	# Fallback to squeue for running jobs
	try:
		result = run_command(["squeue", "-j", job_id, "-h", "-o", "%T"], check=False)
		if result.stdout.strip():
			return normalize_state(result.stdout.strip().splitlines()[0])
	except FileNotFoundError:
		pass

	return None


def wait_for_success(job_id: str) -> None:
	terminal_states = {
		"COMPLETED": True,
		"FAILED": False,
		"CANCELLED": False,
		"TIMEOUT": False,
		"OUT_OF_MEMORY": False,
		"NODE_FAIL": False,
	}

	while True:
		state = get_job_state(job_id)
		if state is None:
			time.sleep(POLL_SECONDS)
			continue

		if state in terminal_states:
			if terminal_states[state]:
				return
			raise RuntimeError(f"Job {job_id} finished unsuccessfully with state: {state}")

		time.sleep(POLL_SECONDS)


def copy_ser_data(exp: Experiment, ranks: int) -> Path:
	exp_dir = EXPERIMENTS_DIR / exp.name
	src = exp_dir / "ser_data"
	if not src.exists():
		raise FileNotFoundError(f"Missing ser_data folder: {src}")

	dest_dir = OUTPUT_ROOT / f"mpirank{ranks}" / exp.output_dir_name
	dest_dir.parent.mkdir(parents=True, exist_ok=True)

	if dest_dir.exists():
		shutil.rmtree(dest_dir)

	shutil.copytree(src, dest_dir)
	
	# Copy NAMELIST files
	namelist_files = [
		f"NAMELIST_{exp.name}",
		"NAMELIST_ICON_output_atm",
	]
	for namelist_file in namelist_files:
		src_file = exp_dir / namelist_file
		if not src_file.exists():
			raise FileNotFoundError(f"Missing namelist file: {src_file}")
		shutil.copy2(src_file, dest_dir / namelist_file)
	
	return dest_dir


def tar_folder(folder: Path, tar_name: str) -> Path:
	date_str = datetime.now().strftime("%Y%m%d")
	tar_filename = f"{tar_name}.{date_str}.tar.gz"
	tar_path = folder.parent / tar_filename
	if tar_path.exists():
		tar_path.unlink()

	with tarfile.open(tar_path, "w:gz") as tar:
		tar.add(folder, arcname=folder.name)

	return tar_path


def run_experiment_series() -> None:
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	os.chdir(RUNSCRIPTS_DIR)

	for ranks in MPI_RANKS:
		for exp in EXPERIMENTS:
			script_path = RUNSCRIPTS_DIR / f"exp.{exp.name}.run"
			if not script_path.exists():
				raise FileNotFoundError(f"Missing slurm script: {script_path}")

			update_slurm_variables(script_path)
			update_slurm_ranks(script_path, ranks)
			job_id = submit_job(script_path)
			wait_for_success(job_id)

			dest_dir = copy_ser_data(exp, ranks)
			tar_folder(dest_dir, exp.tar_name)


if __name__ == "__main__":
	run_experiment_series()
