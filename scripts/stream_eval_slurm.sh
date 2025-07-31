#!/bin/bash
#SBATCH --job-name=streamvln_eval       # 作业名称
#SBATCH --nodes=1                      # 申请 1 个计算节点
#SBATCH --ntasks-per-node=1            # 每个节点运行 1 个任务
#SBATCH --cpus-per-task=64             # 每个任务使用 64 个 CPU 核心
#SBATCH --mem-per-cpu=8GB              # 每个 CPU 分配 16GB 内存
#SBATCH --gres=gpu:8                   # 申请 8 张 GPU
#SBATCH --reservation=laser            # 指定分区为 laser
#SBATCH --time=12:00:00                # 最大运行时间 12 小时
#SBATCH --mail-type=ALL                 # 任务状态变更时发送邮件通知
#SBATCH --mail-user=jjiang127@connect.hkust-gz.edu.cn          # 请输入您的邮箱地址
#SBATCH --output=log/slurm-%j.out
#SBATCH --error=log/slurm-%j.err


source /home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate streamvln

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))

CHECKPOINT="checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"
echo "CHECKPOINT: ${CHECKPOINT}"

torchrun --nproc_per_node=8 --master_port=$MASTER_PORT streamvln/streamvln_eval.py --model_path $CHECKPOINT
