export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export HF_HUB_OFFLINE=1
export VISION_MODEL_VERSION="checkpoints/google/siglip-so400m-patch14-384"
MASTER_PORT=$((RANDOM % 101 + 20000))

CHECKPOINT="./checkpoints/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"
echo "CHECKPOINT: ${CHECKPOINT}"

torchrun --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    streamvln/streamvln_eval_v100.py \
    --model_path $CHECKPOINT \
    --habitat_config_path "config/vln_r2r.yaml" \
    --eval_split "val_unseen" \
    --output_path "results/vals/_unseen/streamvln" \
    --num_future_steps 4 \
    --num_frames 16 \
    --num_history 8 \
    --model_max_length 2048 \
    # --save_video \
    --vision_tower_path $VISION_MODEL_VERSION \
    --quantization_bits 4
