# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StreamVLN is a streaming vision-and-language navigation system that generates action outputs from continuous video input in an online, multi-turn dialogue manner. Built on LLaVA-Video as the foundational Video-LLM, it extends it for interleaved vision, language, and action modeling with SlowFast context modeling for efficient real-time interaction.

## Development Environment

### Dependencies
- Python 3.9
- PyTorch 2.1.2
- CUDA 12.4
- Habitat-sim 0.2.4 and Habitat-lab v0.2.4

### Environment Setup
```bash
conda create -n streamvln python=3.9
conda install habitat-sim==0.2.4 cuda-cudart=12.4 cuda-libraries-dev=12.4 cuda-nvcc=12.4  withbullet headless -c conda-forge -c aihabitat # 不要用cuda-toolkit=12.4，会冲突
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
cd ..
pip install -r requirements.txt
```

## Key Commands

### Training Commands
- **Stage-one Training**: `sbatch scripts/streamvln_train_slurm.sh` (multi-node multi-GPU)
- **Dagger Collection**: `sh scripts/streamvln_dagger_collect.sh` (multi-GPU)
- **Stage-two Training**: `sbatch scripts/streamvln_stage_two_train_slurm.sh` (multi-node multi-GPU)

### Evaluation Commands
- **Multi-GPU Evaluation**: `sh scripts/streamvln_eval_multi_gpu.sh`
- **Single GPU Evaluation**: `python streamvln/streamvln_eval.py --model_path <checkpoint>`

### Data Processing
- **Trajectory Generation**: `sh scripts/streamvln_trajectory_generation.sh`

## Architecture Overview

### Core Components
1. **StreamVLN Model** (`streamvln/model/stream_video_vln.py`): Main model architecture extending LLaVA-Video for VLN tasks
2. **VLN Agent** (`streamvln/streamvln_agent.py`): Handles agent navigation and interaction
3. **VLN Evaluator** (`streamvln/streamvln_eval.py`): Evaluation pipeline for benchmarking
4. **Training Pipeline** (`streamvln/streamvln_train.py`): Multi-stage training with co-training support

### Key Features
- **Fast-streaming dialogue context**: Sliding-window KV cache for efficient processing
- **Slow-updating memory**: Token pruning for long sequence context modeling
- **Multi-modal input**: Video frames, language instructions, and action outputs
- **Real-world deployment**: Support for physical robot deployment (Unitree Go2)

### Data Pipeline
- **Trajectory Data**: Pre-collected observation-action pairs from R2R, RxR, EnvDrop, ScaleVLN
- **Co-training Data**: LLaVA-Video-178K, ScanQA, MMC4 for multi-modal understanding
- **Environment Data**: MP3D and HM3D scenes for simulation

## Configuration System

### Environment Configuration
- Main configs in `config/` directory (e.g., `vln_r2r.yaml`)
- Uses Hydra for configuration management
- Habitat-sim integration for 3D environment simulation

### Training Configuration
- DeepSpeed integration for distributed training
- Multi-GPU and multi-node support via SLURM
- Customizable hyperparameters through script arguments

## Development Notes

### Model Architecture
- Based on LLaVA-Video with VLN-specific modifications
- Supports multiple vision encoders (SigLIP, EVA-CLIP, etc.)
- Custom tokenizers for navigation actions and memory management

### Training Pipeline
- Two-stage training: pre-training on trajectory data, then co-training
- Dagger collection for iterative improvement
- Support for mixed precision training (bf16)

### Evaluation Framework
- Habitat-sim integration for realistic 3D navigation
- Multiple evaluation metrics (success rate, SPL, navigation error)
- Support for different splits (val_seen, val_unseen)

## Data Structure

The expected data folder structure:
```
data/
├── datasets/           # VLN-CE episodes
├── scene_datasets/     # 3D scenes (MP3D, HM3D)
├── trajectory_data/    # Pre-collected trajectories
├── dagger_data/        # Dagger collected data
└── co-training_data/   # Multi-modal training data
```

## Common Issues

- Ensure Habitat-sim and Habitat-lab versions match (v0.2.4)
- Check CUDA compatibility with PyTorch 2.1.2
- Verify data paths in configuration files
- Monitor GPU memory usage during training due to large video sequences

## Real-world Deployment

For deployment on physical robots, see `realworld/realworld.md` for Unitree Go2 integration instructions and safety considerations.
- streamvln的论文在 @"docs/Wei et al. - 2025 - StreamVLN Streaming Vision-and-Language Navigation via SlowFast Context Modeling.pdf" 你分析原理的时候可以使用这个文件