#!/bin/bash
export HF_HUB_OFFLINE=1
export HF_HOME=$PWD/checkpoints/hf_home/
# Distributed training parameters
NNODES=${NNODES:-1}                           # Number of nodes, default 1 (single node)
NPROC_PER_NODE=${NPROC_PER_NODE:-8}          # GPUs per node, default 8
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}        # Master node address, default localhost for single node
MASTER_PORT=${MASTER_PORT:-12000}  # Default port

# Auto-select a large free port if unset or occupied
if command -v python3 >/dev/null 2>&1; then
SEL_PORT=$(python3 - <<'PY'
import os, socket, random, sys

def is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False

env_port = os.environ.get("MASTER_PORT")
if env_port is not None:
    try:
        p = int(env_port)
        if 1024 <= p <= 65535 and is_free(p):
            print(p)
            sys.exit(0)
    except Exception:
        pass

start = int(os.environ.get("PORT_START", 29500))
end = int(os.environ.get("PORT_END", 65535))
for _ in range(2048):
    p = random.randint(start, end)
    if is_free(p):
        print(p)
        sys.exit(0)

print(0)
PY
)
    if [ -n "$SEL_PORT" ] && [ "$SEL_PORT" != "0" ]; then
        MASTER_PORT=$SEL_PORT
    fi
fi

echo "=== Distributed training config ==="
echo "Num nodes: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "Master endpoint: $MASTER_ADDR:$MASTER_PORT"
echo "HF cache dir: $HF_HOME"

VIDEO_FOLDER="data/trajectory_data/R2R","data/trajectory_data/RxR","data/trajectory_data/EnvDrop"

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
# POSIX-compatible replacement: replace '/' with '_' using tr
LLM_VERSION_CLEAN="$(printf '%s' "$LLM_VERSION" | tr '/' '_')"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="$(printf '%s' "$VISION_MODEL_VERSION" | tr '/' '_')"

############### Pretrain ################
BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="StreamVLN_Video_${PROMPT_VERSION}_1epoch_196token_8history_32frame"
PREV_STAGE_CHECKPOINT="checkpoints/lmms-lab/LLaVA-Video-7B-Qwen2"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

# Detect GPU count: prefer CUDA_VISIBLE_DEVICES, then nvidia-smi, then Python torch
GPU_COUNT=""
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _CVD_CLEAN=$(echo "$CUDA_VISIBLE_DEVICES" | tr -d '[:space:]' | sed 's/,,*/,/g;s/^,//;s/,$//')
    if [ -n "$_CVD_CLEAN" ]; then
        GPU_COUNT=$(echo "$_CVD_CLEAN" | awk -F, '{print NF}')
    else
        GPU_COUNT=0
    fi
elif command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')
fi

if [ -z "$GPU_COUNT" ] || ! echo "$GPU_COUNT" | grep -Eq '^[0-9]+$'; then
    if command -v python3 >/dev/null 2>&1; then
        GPU_COUNT=$(python3 - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(-1)
PY
)
        GPU_COUNT=$(echo "$GPU_COUNT" | tr -d '[:space:]')
    fi
fi

if ! echo "$GPU_COUNT" | grep -Eq '^[0-9]+$'; then
    echo "Error: Unable to determine GPU count. Set CUDA_VISIBLE_DEVICES or install NVIDIA drivers."
    exit 1
fi

if [ "$GPU_COUNT" -le 0 ]; then
    echo "Error: No GPUs visible to the process."
    exit 1
fi

echo "Detected GPUs: $GPU_COUNT"
if [ "$GPU_COUNT" -lt "$NPROC_PER_NODE" ]; then
    echo "Warning: NPROC_PER_NODE ($NPROC_PER_NODE) > available GPUs ($GPU_COUNT)"
    echo "Auto-adjust NPROC_PER_NODE to available GPUs"
    NPROC_PER_NODE=$GPU_COUNT
fi
echo "========================"

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=12345 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT streamvln/streamvln_train.py \
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
    --report_to tensorboard

# Usage:
# 1. Single node (default): bash scripts/streamvln_train.sh
# 2. Specify GPUs per node: NPROC_PER_NODE=4 bash scripts/streamvln_train.sh
# 3. Multi-node example:
#    Master: MASTER_ADDR=192.168.1.100 NNODES=2 bash scripts/streamvln_train.sh
#    Worker: MASTER_ADDR=192.168.1.100 NNODES=2 bash scripts/streamvln_train.sh
# 4. Custom HF cache: HF_HOME=/path/to/cache bash scripts/streamvln_train.sh

"""
完整 torchrun 命令参数分析
A. 分布式训练参数
torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=12345 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT
分布式协调: 确保多节点/多GPU训练时进程间正确通信
B. 训练脚本和配置
streamvln/streamvln_train.py \
--deepspeed scripts/zero2.json \
--model_name_or_path $PREV_STAGE_CHECKPOINT
streamvln/streamvln_train.py: 主训练脚本
--deepspeed scripts/zero2.json: 使用 DeepSpeed ZeRO-2 优化配置
--model_name_or_path: 预训练模型路径
C. 模型版本配置
--version $PROMPT_VERSION \
--video_folder ${VIDEO_FOLDER} \
--group_by_task False
--version: 提示版本（qwen_1_5）
--video_folder: 视频数据路径
--group_by_task False: 不按任务分组数据
D. 视频数据处理参数
--num_history 8 \
--num_future_steps 4 \
--num_frames 32 \
--data_augmentation True
--num_history 8: 历史对话轮数
--num_future_steps 4: 预测未来步数
--num_frames 32: 每个视频段帧数
--data_augmentation True: 启用数据增强
E. 多模态模型配置
--mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
--vision_tower ${VISION_MODEL_VERSION} \
--mm_projector_type mlp2x_gelu \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False
--mm_tunable_parts: 可训练的多模态组件
--vision_tower: 视觉编码器版本
--mm_projector_type: 多模态投影层类型
--mm_vision_select_layer -2: 选择视觉编码器倒数第二层
--mm_use_im_start_end False: 不使用图像开始/结束标记
F. 图像处理配置
--image_aspect_ratio anyres_max_9 \
--image_grid_pinpoints "(1x1),...,(6x6)"
--image_aspect_ratio anyres_max_9: 自适应宽高比，最大9倍
--image_grid_pinpoints: 图像网格分割点配置
G. 训练超参数
--bf16 True \
--num_train_epochs 1 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2 \
--learning_rate 2e-5 \
--mm_vision_tower_lr 5e-6 \
--weight_decay 0. \
--warmup_ratio 0.075
--bf16 True: 使用 bfloat16 混合精度训练
--num_train_epochs 1: 训练轮数
--per_device_train_batch_size 2: 每设备训练批次大小
--gradient_accumulation_steps 2: 梯度累积步数
--learning_rate 2e-5: 主学习率
--mm_vision_tower_lr 5e-6: 视觉塔学习率（更低）
--warmup_ratio 0.075: 预热比例
H. 学习率调度
--lr_scheduler_type "cosine_with_min_lr" \
--lr_scheduler_kwargs '{"min_lr": 1.85e-05}'
--lr_scheduler_type: 余弦退火调度器，带最小学习率
--lr_scheduler_kwargs: 最小学习率参数
I. 模型和优化配置
--model_max_length 32768 \
--gradient_checkpointing True \
--torch_compile True \
--torch_compile_backend "inductor" \
--tf32 True
--model_max_length 32768: 最大序列长度
--gradient_checkpointing True: 梯度检查点节省显存
--torch_compile True: PyTorch 2.0 编译优化
--torch_compile_backend "inductor": 使用 Inductor 后端
--tf32 True: 在 A100/H100 上使用 TF32
J. 数据加载和输出
--dataloader_num_workers 8 \
--lazy_preprocess True \
--dataloader_drop_last True \
--run_name $MID_RUN_NAME \
--output_dir checkpoints/$MID_RUN_NAME \
--save_strategy "epoch" \
--save_total_limit 1 \
--logging_steps 10 \
--report_to tensorboard
--dataloader_num_workers 8: 数据加载工作进程数
--lazy_preprocess True: 延迟预处理
--dataloader_drop_last True: 丢弃不完整批次
--save_strategy "epoch": 每轮保存检查点
--report_to tensorboard: 记录到 TensorBoard
参数设计特点
内存优化: 使用 DeepSpeed ZeRO-2 + 梯度检查点 + bf16
性能优化: PyTorch 编译 + 多进程数据加载
多模态配置: 专门针对视觉-语言导航任务优化
分布式支持: 完整的多节点多GPU训练支持
实验管理: 详细的日志记录和检查点保存
这个配置非常适合大规模 StreamVLN 模型的高效训练。
"""