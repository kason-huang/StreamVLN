export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))

# export CUDA_VISIBLE_DEVICES=1,2,4,5,6,7

export HF_HOME=/shared_space/jiangjiajun/hf_cache

# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

# CHECKPOINT="checkpoints/StreamVLN_Video_qwen_1_5_1epoch_196token_8history_32frame_r2r_rxr_envdrop"
CHECKPOINT="checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"
echo "CHECKPOINT: ${CHECKPOINT}"

torchrun \
    --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    streamvln/streamvln_eval_debug.py \
    --model_path $CHECKPOINT \
    --habitat_config_path config/vln_r2r_v1_3.yaml \
    --output_path './results/val_unseen/StreamVLN_debug' \
