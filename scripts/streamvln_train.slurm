#!/bin/bash
#SBATCH --job-name=streamvln-train
#SBATCH --nodes=2 
#SBATCH --gres=gpu:8 
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16G
#SBATCH --reservation=laser
#SBATCH --output=log/streamvln-%j.out
#SBATCH --error=log/streamvln-%j.err

srun bash scripts/streamvln_train.sh