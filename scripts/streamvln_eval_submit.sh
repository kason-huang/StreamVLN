#!/bin/bash
#SBATCH --job-name=streamvln-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:8 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=8G
#SBATCH --reservation=laser
#SBATCH --time=96:00:00
#SBATCH --output=log/streamvln-eval-%j.out
#SBATCH --error=log/streamvln-eval-%j.err

echo "========================================================"
echo "Starting job script..."
echo "Job ID: $SLURM_JOB_ID"
echo "========================================================"

# 1. 设置 Conda 环境
echo "Activating Conda environment..."
source /home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln
echo "Conda environment activated."

# 2. 设置环境变量
export HF_HOME=/shared_space/jiangjiajun/hf_cache

# 3. 设置分布式评估所需的环境变量
GPUS_PER_NODE=8
HOSTNAME=$(hostname)
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29501 # 使用一个和训练不同的端口，避免冲突

# 4. 打印调试信息
echo "-----------------------------"
echo "Running on: $SLURM_NODELIST"
echo "Master addr: ${MASTER_ADDR}:${MASTER_PORT}"
echo "This node: $HOSTNAME"
echo "-----------------------------"

# 5. 定义评估任务的变量
# These parameters should match the training parameters of the model you are evaluating.
PROMPT_VERSION="qwen_1_5"
NUM_EPOCHS=1

NUM_HISTORY=12
NUM_FRAMES=32
EFFECTIVE_BATCH_SIZE=64

NUM_FUTURE_STEPS=4

# Task name
TASK_NAME="history"

# =================================================================================================

# --- 使用变量构建运行名称和路径 (Build run name and paths using variables) ---
EVAL_RUN_NAME="StreamVLN_Video_${PROMPT_VERSION}_${NUM_EPOCHS}epoch_196token_${NUM_HISTORY}history_${NUM_FRAMES}frame_${EFFECTIVE_BATCH_SIZE}batchsize"

CHECKPOINT="checkpoints/${TASK_NAME}/${EVAL_RUN_NAME}"
OUTPUT_PATH="results/r2r/${TASK_NAME}/${EVAL_RUN_NAME}"

echo "CHECKPOINT: ${CHECKPOINT}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "-----------------------------"

# 6. 运行评估脚本
echo "Launching torchrun for evaluation..."
torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    streamvln/streamvln_eval_debug.py \
    \
    --model_path ${CHECKPOINT} \
    --habitat_config_path config/vln_r2r_v1_3.yaml \
    --output_path ${OUTPUT_PATH} \
    --num_future_steps ${NUM_FUTURE_STEPS} \
    --num_history ${NUM_HISTORY} \
    --num_frames ${NUM_FRAMES}

echo "========================================================"
echo "Evaluation job finished."
echo "========================================================"