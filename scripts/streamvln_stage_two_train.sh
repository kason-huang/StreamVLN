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
# export NCCL_DEBUG=INFO
# export NCCL_BUFFSIZE=3145728
# export NCCL_MAX_NCHANNELS=16

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


VIDEO_FOLDER=(
  "/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/R2R"
  "/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/RxR"
  "/shared_space/jiangjiajun/data/streamvln_datasets/dagger_data/EnvDrop"
  "/shared_space/jiangjiajun/data/streamvln_datasets/dagger_data/R2R"
)

# MMC4_VIDEO_FOLDER="data/co-training_data/MMC4-core/images"
# SCANQA_VIDEO_FOLDER="data/co-training_data/ScanNet"
QA_VIDEO_FOLDER="/shared_space/jiangjiajun/data/llava_video_178k"

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="checkpoints/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################
BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="StreamVLN_Video_${PROMPT_VERSION}_1epoch_196token_8history_32frame_stage_two_qa_dagger"
PREV_STAGE_CHECKPOINT="checkpoints/StreamVLN_Video_${PROMPT_VERSION}_1epoch_196token_8history_32frame_128batchsize_refined"

# wandb settings
export WANDB_PROJECT=StreamVLN
export WANDB_NAME=$MID_RUN_NAME
export WANDB_ENTITY=jjiang127-hkust
export WANDB_DIR=/shared_space/jiangjiajun/wandb_logs

echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"
echo "-----------------------------"

torchrun \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_PROCID} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    streamvln/streamvln_train.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --video_folder ${VIDEO_FOLDER} \
    --qa_video_folder ${QA_VIDEO_FOLDER} \
    --group_by_task True \
    --multi_task_training True \
    \
    --num_history 8 \
    --num_future_steps 4 \
    --num_frames 32 \
    --data_augmentation True \
    --data_path "config/co-training_data.yaml" \
    \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --frames_upbound 32 \
    --force_sample True \
    --add_time_instruction True \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir checkpoints/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --mm_vision_tower_lr 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --report_to wandb \