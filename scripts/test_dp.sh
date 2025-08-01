#!/bin/bash
#SBATCH --job-name=ddp_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:1
#SBATCH --reservation=laser 
#SBATCH --time=01:00:00
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

source /home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln

export NCCL_IB_HCA=ibp12s0

HOSTNAME=$(hostname)
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

echo "Running on: $SLURM_NODELIST"
echo "Master addr: $MASTER_ADDR"
echo "This node: $HOSTNAME"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "NNodes: $SLURM_NNODES"

echo "HOST_NODE_ADDR: $MASTER_ADDR"

# torchrun \
#   --nnodes=$SLURM_NNODES \
#   --nproc_per_node=1 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=dgx072:$MASTER_PORT \
#   test/test_ddp.py


echo "Starting DDP test script..."
echo "NNODES: $NNODES"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"


torchrun \
    --nnodes=${NNODES} \
    --node_rank=${SLURM_PROCID} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    test/test_ddp.py