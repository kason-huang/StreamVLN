#!/bin/bash

hostname

source /home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln

GPUS_PER_NODE=1

HOSTNAME=$(hostname)
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

echo "-----------------------------"
echo "Running on: $SLURM_NODELIST"
echo "Master addr: ${MASTER_ADDR}:${MASTER_PORT}"
echo "This node: $HOSTNAME"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOST_NODE_ADDR: $MASTER_ADDR"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "-----------------------------"

torchrun \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_PROCID} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    test/test_ddp.py

# start from the streamvln
# torchrun \
#     --nnodes=${SLURM_NNODES} \
#     --nproc_per_node=${GPUS_PER_NODE} \
#     --rdzv_id=$SLURM_JOB_ID \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#     test/test_ddp.py