#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=300:0:0
#SBATCH --output=slurm-logs/slurm-%j.out
#SBATCH --error=slurm-logs/slurm-%j.err

python run_optimization.py $1 $2 $3 $4
