#!/bin/bash
#SBATCH --job-name=streamvln-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:8 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=8G
#SBATCH --reservation=laser
#SBATCH --time=96:00:00
#SBATCH --output=log/streamvln-eval-%j.out
#SBATCH --error=log/streamvln-eval-%j.err

srun bash scripts/streamvln_eval.sh