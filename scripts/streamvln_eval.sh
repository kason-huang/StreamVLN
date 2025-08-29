hostname

source /home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln

export HF_HOME=/shared_space/jiangjiajun/hf_cache

GPUS_PER_NODE=8

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

CHECKPOINT="checkpoints/StreamVLN_Video_qwen_1_5_1epoch_196token_8history_32frame_128batchsize_refined"
echo "CHECKPOINT: ${CHECKPOINT}"

torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=$MASTER_PORT \
    streamvln/streamvln_eval_debug.py \
    --model_path $CHECKPOINT \
    --habitat_config_path config/vln_r2r_v1_3.yaml \
    --output_path './results/val_unseen/StreamVLN_Video_qwen_1_5_1epoch_196token_8history_32frame_128batchsize_refined-eval-r2r_v1_3' \
