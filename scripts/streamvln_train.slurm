#!/bin/bash
#SBATCH --job-name=streamvln-train-fixed
#SBATCH --nodes=4
#SBATCH --gres=gpu:8 
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=8G
#SBATCH --reservation=laser
#SBATCH --output=log/streamvln-train-4n-%j.out
#SBATCH --error=log/streamvln-train-4n-%j.err

srun bash scripts/streamvln_train.sh