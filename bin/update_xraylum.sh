#!/bin/bash
#SBATCH -J update_xraylum
#SBATCH -o ./logs/job.update_xraylum.dump
#SBATCH -e ./logs/job.update_xraylum.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 48:00:00
#SBATCH --ntasks 128

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

python3 update_xraylum.py




