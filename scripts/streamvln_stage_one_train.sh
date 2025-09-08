hostname

source /home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln

export HF_HOME=/shared_space/jiangjiajun/hf_cache

GPUS_PER_NODE=8

HOSTNAME=$(hostname)
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# export NCCL_IB_HCA=ibp12s0

# If trainning using 4 nodes
export NCCL_DEBUG=INFO
# export NCCL_BUFFSIZE=2097152
# export NCCL_BUFFSIZE=3145728
# export NCCL_MAX_NCHANNELS=4

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


# 5. 定义训练任务的变量
# ====================== NEW: 超参数定义 (NEW: Hyperparameter Definitions) ======================
# Task name
TASK_NAME="future"

NUM_EPOCHS=1

NUM_HISTORY=8
NUM_FUTURE_STEPS=6
NUM_FRAMES=32

PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUM_STEPS=4


# 动态计算总批次大小 (Dynamically calculate total batch size)
# Total Batch Size = per_device_batch_size * grad_accum_steps * total_gpus
TOTAL_GPUS=$((${SLURM_NNODES} * ${GPUS_PER_NODE}))
EFFECTIVE_BATCH_SIZE=$((${PER_DEVICE_BATCH_SIZE} * ${GRAD_ACCUM_STEPS} * ${TOTAL_GPUS}))
# =========================================================================================

VIDEO_FOLDER="/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/R2R","/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/RxR_new","/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/EnvDrop"
LLM_VERSION="Qwen/Qwen2-7B-Instruct"
VISION_MODEL_VERSION="checkpoints/siglip-so400m-patch14-384"
PROMPT_VERSION="qwen_1_5"

# --- MODIFIED: 使用变量构建运行名称 (MODIFIED: Build run name using variables) ---
MID_RUN_NAME="StreamVLN_Video_${PROMPT_VERSION}_${NUM_EPOCHS}epoch_196token_${NUM_HISTORY}history_${NUM_FRAMES}frame_${EFFECTIVE_BATCH_SIZE}batchsize_${NUM_FUTURE_STEPS}future_test"

OUT_DIR="checkpoints/${TASK_NAME}/${MID_RUN_NAME}"
PREV_STAGE_CHECKPOINT="checkpoints/LLaVA-Video-7B-Qwen2"

# 6. 设置 WandB (Weights & Biases)
export WANDB_PROJECT=StreamVLN
export WANDB_NAME=$MID_RUN_NAME
export WANDB_ENTITY=jjiang127-hkust
export WANDB_DIR=/shared_space/jiangjiajun/wandb_logs

echo "Output directory: ${OUT_DIR}"
echo "W&B Project: ${WANDB_PROJECT}, Name: ${WANDB_NAME}"
echo "-----------------------------"

# 7. 运行训练脚本
echo "Launching torchrun on node $SLURM_PROCID..."
torchrun \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_PROCID} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
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
    \
    --data_augmentation True \
    \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
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
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --report_to wandb

echo "========================================================"
echo "Job finished on node $(hostname)."
echo "========================================================"