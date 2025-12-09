# export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))
export HF_HUB_OFFLINE=0

CHECKPOINT="checkpoints/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"
echo "CHECKPOINT: ${CHECKPOINT}"

CONFIG_PATH="config/objectnav_hm3d.yaml"

torchrun --nproc_per_node=2 \
        --standalone streamvln/objectnav_eval.py \
        --model_path $CHECKPOINT \
        --habitat_config_path $CONFIG_PATH \
        # --save_video
