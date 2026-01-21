export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export HF_HUB_OFFLINE=1
MASTER_PORT=$((RANDOM % 101 + 20000))

#CHECKPOINT="./checkpoints/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"
#CHECKPOINT="checkpoints/StreamVLN_Video_qwen_1_5_1epoch_196token_8history_32frame_epoch_1"
#CHECKPOINT="./checkpoints/cloudrobo/checkpoint-24000"
CHECKPOINT="./checkpoints/lora/StreamVLN_Video_qwen_1_5_1epoch_196token_8history_32frame_lora_hight_alpha"

echo "CHECKPOINT: ${CHECKPOINT}"

time torchrun --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    streamvln/streamvln_eval_v100_32g_with_lora.py \
    --base_model_path checkpoints/lmms-lab/LLaVA-Video-7B-Qwen2 \
    --lora_path $CHECKPOINT \
    --habitat_config_path "config/vln_r2r.yaml" \
    --eval_split "val_unseen" \
    --output_path "results/vals/unseen/vln/paper_data/final_lora_with_memory_cloudrobo_val_unseen_train"
