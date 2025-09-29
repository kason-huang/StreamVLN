# StreamVLN Training Pipeline Summary

Based on the paper [StreamVLN: Streaming Vision-and-Language Navigation (ICLR 2025)](https://arxiv.org/abs/2507.05240) and the codebase analysis, here's a comprehensive summary of the StreamVLN training pipeline.

## Overview

StreamVLN employs a **two-stage training pipeline** that extends LLaVA-Video for Vision-and-Language Navigation tasks. The training process combines trajectory-based pre-training with multi-task co-training to achieve robust streaming navigation capabilities.

## Training Configuration

### Base Model Architecture
- **Foundation**: LLaVA-Video-7B-Qwen2 (based on Qwen2-7B-Instruct)
- **Vision Encoder**: Google SigLIP-SO400M-Patch14-384
- **Vision-Language Connector**: MLP2x-GeLU projector
- **Context Length**: 32,768 tokens
- **Frame Processing**: 32 frames per video sequence

### Key Hyperparameters
```bash
# Training Configuration
--num_train_epochs 1
--per_device_train_batch_size 2
--gradient_accumulation_steps 2
--learning_rate 2e-5
--mm_vision_tower_lr 5e-6
--weight_decay 0.
--warmup_ratio 0.075 (Stage 1) / 0.03 (Stage 2)
--lr_scheduler_type "cosine_with_min_lr" (Stage 1) / "cosine" (Stage 2)
--gradient_checkpointing True
--bf16 True
--model_max_length 32768
```

## Training Stages

### Stage 1: Trajectory-based Pre-training

**Objective**: Learn navigation action patterns from pre-collected trajectories

**Data Sources**:
- R2R (Room-to-Room) trajectories
- RxR (Room-to-Room Extended) trajectories
- EnvDrop trajectories

**Training Parts**: `mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model"`

**Frozen Components**:
- Vision resampler (if not explicitly enabled)
- Selected parameters based on `mm_tunable_parts` configuration

**Key Features**:
- **History Context**: 8 history steps
- **Future Prediction**: 4 future steps
- **Data Augmentation**: Enabled (ColorJitter, RandomPosterize, etc.)
- **Multi-resolution**: `anyres_max_9` with grid pinpoints `(1x1)` to `(6x6)`

**Command**:
```bash
sbatch scripts/streamvln_train_slurm.sh
```

### Stage 2: Multi-task Co-training

**Objective**: Enhance general multimodal understanding through co-training with auxiliary tasks

**Data Sources**:
- **Navigation**: Stage 1 trajectories + Dagger-collected data
- **Visual QA**: LLaVA-Video-178K dataset
- **ScanQA**: ScanNet visual question answering
- **Multimodal**: MMC4-core dataset

**Training Configuration**:
- **Multi-task**: `--multi_task_training True`
- **Task Grouping**: `--group_by_task True`
- **Enhanced Video Processing**: `--force_sample True`, `--add_time_instruction True`

**Training Parts**: Same as Stage 1 - `mm_vision_tower,mm_mlp_adapter,mm_language_model`

**Command**:
```bash
sbatch scripts/streamvln_stage_two_train_slurm.sh
```

## Parameter Freezing Strategy

### Traditional Mode (Code Lines 1714-1740)
When `mm_tunable_parts` is None, the system uses traditional freezing:

```python
if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
    model.requires_grad_(False)  # Freeze entire model first

if model_args.tune_mm_mlp_adapter:
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True  # Unfreeze MLP adapter

if model_args.tune_mm_vision_resampler:
    for p in model.get_model().vision_resampler.parameters():
        p.requires_grad = True  # Unfreeze vision resampler
```

### Custom Mode (Code Lines 1742-1770)
When `mm_tunable_parts` is specified, selective unfreezing occurs:

```python
model.requires_grad_(False)  # Freeze everything first
vision_tower.requires_grad_(False)
model.get_model().mm_projector.requires_grad_(False)
model.get_model().vision_resampler.requires_grad_(False)

# Selectively unfreeze based on configuration
tunable_parts = model_args.mm_tunable_parts.split(",")

if "mm_vision_tower" in tunable_parts:
    for name, param in model.named_parameters():
        if "vision_tower" in name:
            param.requires_grad_(True)

if "mm_mlp_adapter" in tunable_parts:
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True

if "mm_language_model" in tunable_parts:
    for name, param in model.named_parameters():
        if "vision_tower" not in name and "mm_projector" not in name:
            param.requires_grad_(True)
```

## Multi-task Training Architecture

### Task Composition
```python
# From make_supervised_data_module()
nav_dataset = VLNActionDataset(task_id=0)  # Navigation task
QA_dataset = LazySupervisedDataset(task_id=1)  # Visual QA
SCANQA_dataset = LazySupervisedDataset(task_id=2)  # ScanQA
MMC4_dataset = LazyMMC4Dataset(task_id=3)  # Multimodal comprehension

dataset = CombineDataset([nav_dataset, QA_dataset, SCANQA_dataset, MMC4_dataset])
```

### Task-specific Data Paths
- **Navigation**: `data/trajectory_data/` + `data/dagger_data/`
- **Visual QA**: `data/co-training_data/LLaVA-Video-178K`
- **ScanQA**: `data/co-training_data/ScanNet`
- **MMC4**: `data/co-training_data/MMC4-core/images`

## Memory and Optimization Techniques

### Gradient Checkpointing
```python
if training_args.gradient_checkpointing:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
```

### Mixed Precision Training
- **bf16**: Enabled for memory efficiency
- **Compute dtype**: torch.bfloat16
- **Quantization**: 4-bit/8-bit support via BitsAndBytes

### Distributed Training
- **Backend**: DeepSpeed (zero2.json)
- **Multi-node**: 4 nodes × 8 GPUs = 32 GPUs total
- **Communication**: c10d backend

## Training Pipeline Flow

1. **Initialization**: Parse arguments, setup distributed training
2. **Model Loading**: Load LLaVA-Video checkpoint, apply config overwrites
3. **Gradient Setup**: Configure trainable parts based on `mm_tunable_parts`
4. **Vision Setup**: Initialize vision tower, configure image processor
5. **Data Preparation**: Create task-specific datasets, setup data collator
6. **Trainer Setup**: Initialize LLaVATrainer with DeepSpeed
7. **Training Loop**: Resume from checkpoint or start fresh training
8. **Model Saving**: Save trained weights and configuration

## Key Innovations from Paper

### 1. Streaming Context Management
- **Fast-streaming dialogue**: Sliding-window KV cache for efficient processing
- **Slow-updating memory**: Token pruning for long sequence context modeling

### 2. Multi-modal Action Modeling
- **Interleaved inputs**: Video frames + language instructions + action outputs
- **Real-time interaction**: Continuous video input with online response generation

### 3. Efficient Training
- **Parameter efficiency**: Selective unfreezing reduces trainable parameters
- **Memory efficiency**: Gradient checkpointing and mixed precision training
- **Data efficiency**: Multi-task co-training improves generalization

This training pipeline enables StreamVLN to achieve state-of-the-art performance on vision-and-language navigation tasks while maintaining computational efficiency.