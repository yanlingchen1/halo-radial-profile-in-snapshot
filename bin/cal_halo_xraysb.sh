#!/bin/bash
#SBATCH -J cal_halo_r
#SBATCH -o ./logs/job.cal_halo_xraysb.dump
#SBATCH -e ./logs/job.cal_halo_xraysb.err
#SBATCH -p cosma8-shm
#SBATCH --exclude=mad05
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 72:00:00
#SBATCH --ntasks 64

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

python3 cal_halo_lum_props_by_halo_cylinder.py



