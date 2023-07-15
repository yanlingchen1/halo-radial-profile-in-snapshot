#!/bin/bash
#SBATCH -J cal_r200clum
#SBATCH -o ./logs/job.cal_r200clum.dump
#SBATCH -e ./logs/job.cal_r200clum.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 24:00:00
#SBATCH --ntasks 1

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

python3 cal_halo_r200c_xraylum.py




