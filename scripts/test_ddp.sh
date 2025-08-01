#!/bin/bash


#SBATCH -N 2
#SBATCH --cpus-per-task=128
#SBATCH --gres gpu:1
#SBATCH --mem 512GB
#SBATCH --reservation=laser

#SBATCH --output=log/slurm-%j.out
#SBATCH --error=log/slurm-%j.err

source /home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln

NNODES=2
GPUS_PER_NODE=1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=53007

torchrun \
    --nnodes=${NNODES} \
    --node_rank=${SLURM_PROCID} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    test/test_ddp.py
