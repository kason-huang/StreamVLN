#!/bin/bash
umask 000
set -x

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))

GPUS_PER_NODE=8

# source /home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
# conda activate streamvln

# DAGGER_DATASET=R2R
# DAGGER_DATA_PATH=/shared_space/jiangjiajun/data/streamvln_datasets/datasets/r2r/train/train.json.gz
# DAGGER_GT_ANNOTATIONS_PATH=/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/R2R/annotations.json

DAGGER_DATASET=RxR
DAGGER_DATA_PATH=/shared_space/jiangjiajun/data/streamvln_datasets/datasets/rxr/train/train_guide_en.json.gz
DAGGER_GT_ANNOTATIONS_PATH=/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/RxR_new/annotations.json

# DAGGER_DATASET=EnvDrop
# DAGGER_DATA_PATH=/shared_space/jiangjiajun/data/streamvln_datasets/datasets/envdrop/envdrop/envdrop.json.gz
# DAGGER_GT_ANNOTATIONS_PATH=/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/EnvDrop/annotations.json


DAGGER_UPDATE_SIZE=160000
DAGGER_COMMIT_FREQ=50 # dump data every DAGGER_COMMIT_FREQ updates
DAGGER_P=0 # allow model inference
DAGGER_DATA_IT=3 # not used if DAGGER_P=0

# MID_RUN_NAME="StreamVLN_Video_qwen_1_5_1epoch_196token_8history_32frame_128batchsize_refined"
CHECKPOINT="checkpoints/StreamVLN_Video_qwen_1_5_1epoch_196token_8history_32frame_128batchsize_refined"
echo "CHECKPOINT: ${CHECKPOINT}"

DAGGER_OUTPUT_PATH=/shared_space/jiangjiajun/data/streamvln_datasets/dagger_data/RxR_new

mkdir -p ${DAGGER_OUTPUT_PATH}

torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=$MASTER_PORT streamvln/streamvln_dagger.py \
    --model_path $CHECKPOINT \
    --dagger_dataset ${DAGGER_DATASET} \
    --dagger_data_path ${DAGGER_DATA_PATH} \
    --dagger_update_size ${DAGGER_UPDATE_SIZE} \
    --dagger_commit_freq ${DAGGER_COMMIT_FREQ} \
    --dagger_p ${DAGGER_P} \
    --dagger_data_it ${DAGGER_DATA_IT} \
    --dagger_output_path ${DAGGER_OUTPUT_PATH} \
    --dagger_gt_annotations_path ${DAGGER_GT_ANNOTATIONS_PATH} \
    # --dagger_save_video