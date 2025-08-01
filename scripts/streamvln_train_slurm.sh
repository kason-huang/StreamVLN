#!/bin/bash
#SBATCH --job-name=blip3o    # Job name
#SBATCH --nodes=2                            # Number of nodes
#SBATCH --gres=gpu:8                         # Number of GPUs per node
#SBATCH --time=96:00:00                      # Time limit (hh:mm:ss)

#SBATCH --reservation=laser
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jjiang127@connect.hkust-gz.edu.cn
#SBATCH --output=log/slurm-%j.out
#SBATCH --error=log/slurm-%j.err


source /home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln

export HF_HOME=/shared_space/jiangjiajun/hf_cache

# 定义 master 节点的地址和端口
MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
# MASTER_PORT=$((RANDOM % 101 + 20001))
export MASTER_PORT=29500

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

scontrol show hostnames $SLURM_JOB_NODELIST


sleep 30

# 定义视频数据路径
VIDEO_FOLDER="/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/R2R","/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/RxR","/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/EnvDrop"

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="checkpoints/siglip2-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################
BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="StreamVLN_Video_${PROMPT_VERSION}_1epoch_196token_8history_32frame"
PREV_STAGE_CHECKPOINT="checkpoints/LLaVA-Video-7B-Qwen2"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT streamvln/streamvln_train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --video_folder ${VIDEO_FOLDER} \
    --group_by_task False \
    \
    --num_history 8 \
    --num_future_steps 4 \
    --num_frames 32 \
    --data_augmentation True \
    \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir checkpoints/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
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
    # --report_to wandb \
