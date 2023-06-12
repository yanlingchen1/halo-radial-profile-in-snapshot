#!/bin/bash
#SBATCH -J prof_sph230612
#SBATCH -o ./logs/job.prof_sph230612.dump
#SBATCH -e ./logs/job.prof_sph230612.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 12:00:00
#SBATCH --ntasks 64

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

python3 cal_halo_xrayprofile_sph.py




