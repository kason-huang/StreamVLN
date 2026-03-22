# LeRobot Dataset Integration for StreamVLN

## Overview

This document describes the lightweight LeRobot dataset integration for StreamVLN, which allows training with LeRobot-format datasets without installing the full `lerobot` package (which has dependency conflicts with StreamVLN).

## Problem Statement

The user successfully converted R2R navigation data to LeRobot format but cannot install `lerobot=v0.4.3` due to dependency conflicts with StreamVLN's existing dependencies.

**Solution**: Extract the core LeRobot dataset loading functionality into StreamVLN as a lightweight implementation that only depends on compatible libraries.

## Architecture

### File Structure

```
streamvln/
├── dataset/
│   └── lerobot_action_dataset.py    # Core LeRobot dataset loader
├── args.py                           # Added LeRobotDataArguments
└── streamvln_train.py                # Added adapter & integration logic

docs/
└── lerobot_integration.md            # This document

tests/
└── test_lerobot_dataset.py          # Test script
```

### Component Design

#### 1. LeRobotActionDataset Class

Location: `streamvln/dataset/lerobot_action_dataset.py`

**Purpose**: Lightweight PyTorch Dataset that loads LeRobot-format data without the full lerobot package.

**Key Features**:
- Multi-chunk dataset support (multiple parquet/video files)
- PyAV (preferred) or OpenCV (fallback) video decoding
- Episode-level and frame-level indexing
- Lazy loading for memory efficiency

**Data Flow**:
```
info.json → metadata (total_episodes, total_frames, fps)
    ↓
episodes/*.parquet → episode boundaries (start_idx, end_idx, length)
    ↓
data/*.parquet → frame data (action, timestamp, task_index)
    ↓
videos/*.mp4 → decoded frames (RGB images)
    ↓
Sample dict {
    "observation.images.rgb": torch.Tensor [3, H, W],
    "action": torch.Tensor [1],
    "task": str (JSON),
    "episode_index": torch.Tensor,
    "frame_index": torch.Tensor,
    "timestamp": torch.Tensor,
    "task_index": torch.Tensor
}
```

**Video Decoding Strategy**:
```python
# Priority: PyAV → OpenCV
if video_backend == "auto":
    try:
        import av
        video_backend = "av"  # Efficient, supports AV1
    except ImportError:
        video_backend = "opencv"  # Fallback
```

#### 2. LeRobotDataArguments

Location: `streamvln/args.py`

**Purpose**: Command-line arguments for LeRobot dataset configuration.

```python
@dataclass
class LeRobotDataArguments:
    use_lerobot: bool = False
    lerobot_data_path: str = "./data/lerobot"
    lerobot_repo_id: str = "streamvln/r2r_navigation"
    video_backend: str = "auto"  # "auto", "av", "opencv"
```

#### 3. LeRobotActionDatasetAdapter

Location: `streamvln/streamvln_train.py`

**Purpose**: Adapts LeRobot dataset format to StreamVLN training format.

**Conversion**:
- LeRobot format → Training format
- `observation.images.rgb` → Preprocessed images with tokenization
- `action` (scalar) → Action text sequence (e.g., "↑←→STOP")
- `task` (JSON) → Conversation prompt with instruction

## Dependencies

### Required (Already in StreamVLN)

| Package | Version Used | Purpose |
|---------|--------------|---------|
| torch | ≥2.0 | Tensor operations |
| pandas | 2.3.0 | Parquet file reading |
| numpy | 1.26.1 | Array operations |
| pyarrow | **14.0.0** | Parquet backend (see note below) |
| pillow | 11.2.1 | Image processing |
| opencv-python | 4.11.0.86 | Video decoding (fallback) |

### Optional

| Package | Version | Purpose |
|---------|---------|---------|
| av (PyAV) | 15.1.0 | Efficient video decoding, AV1 support |

### Important: PyArrow Version

**Problem**: PyArrow 19.0.0 (original) is incompatible with parquet files created by lerobot.

**Error**: `OSError: Repetition level histogram size mismatch`

**Solution**: Downgrade to PyArrow 14.0.0

```bash
pip install 'pyarrow==14.0.0'
```

### Installation Commands

```bash
# Activate StreamVLN environment
conda activate streamvln

# Downgrade PyArrow for compatibility
pip install 'pyarrow==14.0.0' --no-deps

# Install PyAV for AV1 video support (recommended)
pip install av
```

## Usage

### Basic Training

```bash
python streamvln/streamvln_train.py \
    --use_lerobot True \
    --lerobot_data_path ./data \
    --lerobot_repo_id lerobot \
    --video_backend auto \
    --model_name_or_path <model_path> \
    --data_path <data_path> \
    [other training arguments...]
```

### Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--use_lerobot` | Enable LeRobot dataset | False | True |
| `--lerobot_data_path` | Path to data parent directory | "./data/lerobot" | "./data" |
| `--lerobot_repo_id` | LeRobot repository ID | "streamvln/r2r_navigation" | "lerobot" |
| `--video_backend` | Video decoder | "auto" | "av", "opencv" |

### Dataset Structure

The dataset should be organized as:

```
{lerobot_data_path}/{lerobot_repo_id}/
├── meta/
│   ├── info.json              # Dataset metadata
│   ├── stats.json             # Data statistics
│   ├── tasks.parquet           # Task descriptions
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet # Episode boundaries
├── data/
│   └── chunk-000/
│       └── file-000.parquet    # Frame data (action, timestamp)
└── videos/
    └── observation.images.rgb/
        └── chunk-000/
            └── file-000.mp4    # Encoded video frames
```

### Example: Converted R2R Dataset

```bash
# After converting R2R to LeRobot format
# Dataset location: ./data/lerobot/

python streamvln/streamvln_train.py \
    --use_lerobot True \
    --lerobot_data_path ./data \
    --lerobot_repo_id lerobot \
    --model_name_or_path path/to/model \
    --data_path path/to/json \
    --num_frames 32 \
    --num_future_steps 1 \
    --output_dir output/lerobot_finetune
```

## Testing

### Run Test Script

```bash
conda activate streamvln
python tests/test_lerobot_dataset.py
```

### Expected Output

```
============================================================
Test 1: Direct LeRobot Dataset Loading
============================================================
✓ Dataset loaded successfully!
  Total episodes: 60
  Total frames: 3738
  FPS: 3
  Video backend: av

✓ Sample loaded successfully!
  Sample keys: [...]
    observation.images.rgb: shape=torch.Size([3, 480, 640])
    action: shape=torch.Size([1])
    ...

============================================================
Test 2: LeRobot Dataset with Adapter
============================================================
...

============================================================
Test 3: Episode Loading
============================================================
✓ Episode loaded successfully!
  Instruction: ...
  Actions: 56 actions
  Images: 56 images
  Image shape: (480, 640, 3)
```

### Test Coverage

1. **Direct Loading**: Verify `LeRobotActionDataset` loads metadata and samples
2. **Adapter Test**: Verify `LeRobotActionDatasetAdapter` converts to training format
3. **Episode Test**: Verify full episode loading with actions and images

## Implementation Details

### Video Frame Decoding

#### PyAV Method (Preferred)

```python
with av.open(str(video_path)) as container:
    stream = container.streams.video[0]

    # Seek to closest keyframe
    timestamp = int(frame_idx / fps / stream.time_base)
    container.seek(timestamp, stream=stream)

    # Decode until target frame
    for packet in container.demux(stream):
        for frame in packet.decode():
            if current_frame == frame_idx:
                return frame.to_ndarray(format='rgb24')
```

**Advantages**:
- Supports AV1 codec
- Efficient seeking
- Frame-accurate decoding

#### OpenCV Method (Fallback)

```python
cap = cv2.VideoCapture(str(video_path))
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

**Limitations**:
- No AV1 support
- Less efficient seeking
- Used only when PyAV unavailable

### Episode Metadata Extraction

When parquet metadata files are incompatible:

1. **Primary**: Read from `meta/episodes/*.parquet`
2. **Fallback**: Extract from `data/*.parquet` (group by episode_index)
3. **Last Resort**: Create uniform episodes from `info.json` totals

```python
def _extract_episodes_from_data(self):
    total_episodes = self.info.get('total_episodes')
    total_frames = self.info.get('total_frames')
    avg_frames = total_frames // total_episodes

    for i in range(total_episodes):
        episodes_list.append({
            'episode_index': i,
            'length': avg_frames,
            'dataset_from_index': i * avg_frames,
            'dataset_to_index': (i + 1) * avg_frames,
        })
```

### Frame Indexing

**Problem**: Map global frame index to (chunk, file, position)

**Solution**: Cumulative sum with binary search

```python
# Build cumulative frame count
self.episodes_df['_cumsum'] = self.episodes_df['length'].cumsum()

# Find episode for frame idx
episode_idx = np.searchsorted(cumsum, idx + 1, side='right')
```

**Time Complexity**: O(log n) for episode lookup

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Dataset init time | ~1-2 seconds | Loads metadata only |
| Frame decode time | ~10-50ms | Depends on backend, frame size |
| Memory footprint | Low | Lazy loading, one frame at a time |
| Supported video codecs | AV1, H.264, etc. | Via PyAV; OpenCV limited |

## Troubleshooting

### PyArrow Version Error

**Symptom**: `Repetition level histogram size mismatch`

**Cause**: PyArrow 19.x incompatible with lerobot-created parquet files

**Fix**:
```bash
pip install 'pyarrow==14.0.0' --no-deps
```

### Video Decoding Errors

**Symptom**: `Failed to decode frame`, `Unsupported codec`

**Cause**: AV1 codec not supported by OpenCV

**Fix**:
```bash
pip install av  # Install PyAV
```

### Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'utils'`

**Cause**: Relative import issue in streamvln_train.py

**Fix**: Run from project root with proper PYTHONPATH

### Data Path Issues

**Symptom**: `Dataset root not found`

**Cause**: Incorrect `{root}/{repo_id}` path construction

**Fix**: Ensure:
- `--lerobot_data_path` points to parent directory
- `--lerobot_repo_id` matches immediate subdirectory
- Example: `./data/lerobot/` → `root="./data" repo_id="lerobot"`

## Future Enhancements

- [ ] Add LRU cache for decoded frames
- [ ] Support for multiple video keys (depth, other cameras)
- [ ] Data augmentation integration
- [ ] Distributed data loading optimization
- [ ] LMDB caching for faster random access
- [ ] Support for streaming datasets (no local copy)

## References

- Original R2R to LeRobot converter: `scripts/dataset_converters/r2r2lerobot.py`
- LeRobot dataset format: `docs/r2rlerobot_design.md`
- LeRobot documentation: https://github.com/huggingface/lerobot

## Changelog

### 2026-02-09

- Created `LeRobotActionDataset` lightweight loader
- Added `LeRobotDataArguments` to args.py
- Integrated into training pipeline with adapter
- Fixed PyArrow compatibility (v19 → v14)
- Added PyAV support for AV1 video decoding
- **Fixed task extraction**: Handle `tasks.parquet` where JSON is in index, and convert `float` task_index to `int`
- Verified with converted R2R dataset (60 episodes, 3738 frames)
