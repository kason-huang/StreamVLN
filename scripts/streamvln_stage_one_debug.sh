#!/bin/bash

# 1. Activate Conda Environment
# This line ensures you are using the correct Python environment and packages.

# 2. Set Environment Variables
# Sets the cache directory for Hugging Face models.
export HF_HOME=/shared_space/jiangjiajun/hf_cache
# By setting CUDA_VISIBLE_DEVICES, you can explicitly choose which GPU to use.
# '0' means the first GPU. Change it if you want to use a different one (e.g., '1', '2').
# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

echo "-----------------------------"
echo "Starting single-GPU debug run..."
# echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
echo "-----------------------------"


# 3. Define Training Hyperparameters
# ====================== Hyperparameter Definitions ======================
# Task name for organizing outputs
TASK_NAME="debug"

# Training settings
NUM_EPOCHS=1
NUM_HISTORY=8
NUM_FUTURE_STEPS=4
NUM_FRAMES=32

# Batch size settings for debugging
PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUM_STEPS=1

# --- SIMPLIFIED: Total batch size calculation for a single GPU ---
# We are only using 1 GPU for this debug run.
TOTAL_GPUS=1
EFFECTIVE_BATCH_SIZE=$((${PER_DEVICE_BATCH_SIZE} * ${GRAD_ACCUM_STEPS} * ${TOTAL_GPUS}))
echo "Effective Batch Size: ${EFFECTIVE_BATCH_SIZE}"
# =======================================================================

# 4. Define Model, Data, and Output Paths
VIDEO_FOLDER="/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/R2R","/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/RxR_new","/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/EnvDrop"
LLM_VERSION="Qwen/Qwen2-7B-Instruct"
VISION_MODEL_VERSION="checkpoints/siglip-so400m-patch14-384"
PROMPT_VERSION="qwen_1_5"

MID_RUN_NAME="StreamVLN_Video_${PROMPT_VERSION}_${NUM_EPOCHS}epoch_196token_${NUM_HISTORY}history_${NUM_FRAMES}frame_${EFFECTIVE_BATCH_SIZE}batchsize_${NUM_FUTURE_STEPS}future"

OUT_DIR="checkpoints/${TASK_NAME}/${MID_RUN_NAME}"
PREV_STAGE_CHECKPOINT="checkpoints/LLaVA-Video-7B-Qwen2"

# 5. (Optional) Set up WandB (Weights & Biases) for logging
# If you don't want to log to WandB during debug, you can comment these lines out.
export WANDB_PROJECT=StreamVLN_Debug # Changed project name to avoid clutter
export WANDB_NAME=$MID_RUN_NAME
export WANDB_ENTITY=jjiang127-hkust
export WANDB_DIR=/shared_space/jiangjiajun/wandb_logs
# To disable WandB completely, set the following line:
# export WANDB_DISABLED=true

echo "Output directory: ${OUT_DIR}"
echo "-----------------------------"


# 6. Run the Training Script
# --- MODIFIED: Simplified torchrun for a single GPU ---
# We removed all multi-node arguments like --nnodes, --node_rank, --master_addr.
# --nproc_per_node=1 tells torchrun to launch only one process on this machine.
# If you want to debug with PyCharm or VSCode, you can replace this whole `torchrun`
# block with a simple `python` command. See the explanation below.
echo "Launching training script..."
torchrun \
    --nproc_per_node=1 \
    streamvln/streamvln_train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --video_folder ${VIDEO_FOLDER} \
    --group_by_task False \
    \
    --num_history ${NUM_HISTORY} \
    --num_future_steps ${NUM_FUTURE_STEPS} \
    --num_frames ${NUM_FRAMES} \
    --history_stride 2 \
    --current_stride 2 \
    \
    --data_augmentation True \
    \
    --mm_tunable_parts="mm_mlp_adapter" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir $OUT_DIR \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --mm_vision_tower_lr 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.075 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 1.85e-05}' \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --report_to none \
    --verbose_logging True \
    --debug_open False

echo "========================================================"
echo "Debug script finished."
echo "========================================================"