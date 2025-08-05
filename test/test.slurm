#!/bin/bash
#SBATCH --job-name=ddp_test
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=20G
#SBATCH --reservation=laser 
#SBATCH --time=01:00:00
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

srun bash scripts/test.sh