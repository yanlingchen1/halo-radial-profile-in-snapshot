#!/bin/bash
#SBATCH -J cal_halo_r
#SBATCH -o ./logs/job.cal_halo_xraysb_prof.dump
#SBATCH -e ./logs/job.cal_halo_xraysb_prof.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 12:00:00
#SBATCH --ntasks 32

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

python3 cal_halo_xrayprofile_cylinder_coor_xraylum.py




