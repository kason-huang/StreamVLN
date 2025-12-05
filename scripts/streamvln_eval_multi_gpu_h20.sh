export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export HF_HUB_OFFLINE=1
export VISION_MODEL_VERSION="checkpoints/google/siglip-so400m-patch14-384"
MASTER_PORT=$((RANDOM % 101 + 20000))

CHECKPOINT="./checkpoints/StreamVLN_Video_qwen_1_5_1epoch_196token_8history_32frame"
echo "CHECKPOINT: ${CHECKPOINT}"

# torchrun --nproc_per_node=2 \
#     --master_port=$MASTER_PORT \
#     streamvln/streamvln_eval_h20.py \
#     --model_path $CHECKPOINT \
#     --habitat_config_path "config/vln_r2r.yaml" \
#     --eval_split "val_unseen" \
#     --output_path "results/vals/_unseen/streamvln" \
#     --vision_tower_path $VISION_MODEL_VERSION

python streamvln/streamvln_eval_h20.py \
    --model_path $CHECKPOINT \
    --habitat_config_path "config/vln_r2r.yaml" \
    --eval_split "val_unseen" \
    --output_path "results/vals/_unseen/streamvln"
    # --vision_tower_path $VISION_MODEL_VERSION