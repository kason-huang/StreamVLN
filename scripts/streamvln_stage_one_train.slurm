#!/bin/bash
#SBATCH --job-name=streamvln-train-16frame
#SBATCH --nodes=2
#SBATCH --gres=gpu:8 
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=8G
#SBATCH --reservation=laser
#SBATCH --output=log/streamvln-train-16frame-%j.out
#SBATCH --error=log/streamvln-train-16frame-%j.err

srun bash scripts/streamvln_stage_one_train.sh