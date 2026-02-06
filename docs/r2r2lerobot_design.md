# R2R → LeRobot 数据集转换设计文档

**版本**: v1.0
**日期**: 2026-02-06
**LeRobot 版本**: v3.0 (codebase_version)
**状态**: 已实现

---

## 1. 概述

### 1.1 背景

本项目将 R2R 视觉语言导航数据集转换为 LeRobot v3.0 标准格式，以利用 LeRobot 生态系统进行机器人学习研究。

### 1.2 设计目标

| 目标 | 说明 |
|-----|------|
| **兼容性** | 使用 `lerobot.datasets.lerobot_dataset.LeRobotDataset`，确保与 LeRobot v3.0 生态完全兼容 |
| **数据完整性** | 保留原始图像、动作序列和指令信息 |
| **多指令支持** | 每个 episode 的多个描述指令拆分为独立 episode |
| **可扩展性** | 设计可支持 RxR、EnvDrop 等其他 VLN 数据集 |

### 1.3 核心决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| **Action 对齐** | 丢弃首个 -1 | actions 比 images 多 1 个（开头的 -1），丢弃后一一对应 |
| **Instruction 处理** | 拆分为独立 episode | 每个 instruction 生成一个 episode，共享图像和动作 |
| **FPS** | 3 fps | 模拟导航决策频率，平衡时间戳精度 |
| **图像预处理** | 保持原样 | 保留原始分辨率 (640x480)，由数据集 transform 处理 |

---

## 2. 系统架构

### 2.1 架构分层图

```
┌─────────────────────────────────────────────────────────────────┐
│                        应用层                                    │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  CLI 接口        │    │  数据验证工具     │                    │
│  │  r2r2lerobot.py │    │  verify_lerobot  │                    │
│  └─────────────────┘    └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      转换逻辑层                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  R2R → LeRobot Converter                                 │   │
│  │  - load_r2r_episode(): 加载 R2R episode                  │   │
│  │  - process_episode(): 处理单个 episode                   │   │
│  │  - instruction splitting: 按指令拆分                     │   │
│  │  - action alignment: 丢弃 -1，对齐图像                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       数据访问层                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  lerobot.datasets.lerobot_dataset.LeRobotDataset        │   │
│  │  - LeRobotDataset.create(): 创建数据集                   │   │
│  │  - add_frame(): 添加帧数据                               │   │
│  │  - save_episode(): 保存 episode                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       存储层 (LeRobot v3.0)                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │ meta/      │  │ data/      │  │ videos/    │                │
│  │ info.json  │  │ chunk-000/ │  │ *.mp4      │                │
│  │ stats.json │  │ file-000   │  │            │                │
│  │ tasks.parq │  │ .parquet   │  │            │                │
│  └────────────┘  └────────────┘  └────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心流程时序图

```
用户            CLI          Converter         LeRobotDataset      文件系统
 │               │               │                  │                │
 │─── args ─────>│               │                  │                │
 │               │               │                  │                │
 │               │─── load ─────>│                  │                │
 │               │  annotations │                  │                │
 │               │               │                  │                │
 │               │<── episodes ──│                  │                │
 │               │               │                  │                │
 │               │               │─── create() ────>│                │
 │               │               │                  │─── mkdir ─────>│
 │               │               │                  │                │
 │               │               │<── dataset ──────│                │
 │               │               │                  │                │
 │               │               │─── for each episode ────────────>│
 │               │               │                  │                │
 │               │               │  ┌─────────────────────────────┐│
 │               │               │  │ 1. load images              ││
 │               │               │  │ 2. load actions (skip -1)  ││
 │               │               │  │ 3. for each instruction:    ││
 │               │               │  │    a. add_frame() × N       ││
 │               │               │  │    b. save_episode()        ││
 │               │               │  └─────────────────────────────┘│
 │               │               │                  │                │
 │               │               │                  │─── parquet ────>│
 │               │               │                  │─── mp4 ────────>│
 │               │               │                  │─── json ───────>│
 │               │               │                  │                │
 │<── done ──────│<──────────────│                  │                │
```

---

## 3. 数据格式规范

### 3.1 输入格式 (R2R)

```
R2R/
├── annotations.json          # Episode 元数据
└── images/
    └── {scene_id}_{dataset}_r2r_{id}/
        └── rgb/
            ├── 000.jpg
            ├── 001.jpg
            └── ...
```

**annotations.json 结构**:
```json
[
  {
    "id": 577,
    "video": "images/17DRP5sb8fy_r2r_000577",
    "instructions": [
      "Walk past the dining table and take a left...",
      "Walk straight down the wall. At the entry way...",
      "Walk towards the kitchen area. Turn left..."
    ],
    "actions": [-1, 3, 3, 3, 2, 1, ...]
  }
]
```

**字段说明**:
| 字段 | 类型 | 说明 |
|-----|------|------|
| `id` | int | Episode 唯一标识 |
| `video` | str | 图像目录相对路径 |
| `instructions` | str[] | 同一指令的多种描述 (通常 3 个) |
| `actions` | int[] | 动作序列，首个为 -1 |

### 3.2 输出格式 (LeRobot v3.0)

```
{output_dir}/{repo_id}/
├── meta/
│   ├── info.json              # 数据集元信息
│   ├── stats.json             # 数据统计信息
│   ├── tasks.parquet           # 任务索引表
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet  # Episode 元数据表
├── data/
│   └── chunk-000/
│       └── file-000.parquet      # 帧数据表
└── videos/
    └── observation.images.rgb/
        └── chunk-000/
            └── file-000.mp4       # 编码后的视频
```

---

## 4. 元数据文件格式

### 4.1 meta/info.json

```json
{
  "codebase_version": "v3.0",
  "robot_type": null,
  "total_episodes": 6,
  "total_frames": 315,
  "total_tasks": 6,
  "chunks_size": 1000,
  "data_files_size_in_mb": 100,
  "video_files_size_in_mb": 200,
  "fps": 3,
  "splits": {
    "train": "0:6"
  },
  "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
  "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
  "features": { ... }
}
```

**关键字段**:
| 字段 | 说明 |
|-----|------|
| `codebase_version` | LeRobot 版本标识 (v3.0) |
| `total_episodes` | Episode 总数 (考虑 instruction 拆分) |
| `total_frames` | 总帧数 |
| `fps` | 帧率，用于 timestamp 计算 |
| `splits` | 数据划分 (本方案不做划分，全部为 train) |
| `features` | 特征定义，见 4.4 |

### 4.2 meta/stats.json

```json
{
  "observation.images.rgb": {
    "mean": [127.5, ...],
    "std": [65.2, ...],
    "min": [0.0, ...],
    "max": [255.0, ...]
  },
  "action": {
    "mean": 1.8,
    "std": 0.9,
    "min": 0,
    "max": 3
  }
}
```

### 4.3 meta/tasks.parquet

| 列名 | 类型 | 说明 |
|-----|------|------|
| `task` | str | JSON 序列化的指令 `{"instruction": "..."}` |
| `task_index` | int64 | 任务索引 |

### 4.4 meta/episodes/chunk-000/file-000.parquet

| 列名 | 类型 | 说明 |
|-----|------|------|
| `episode_index` | int64 | Episode 索引 |
| `tasks` | str | 关联的 task |
| `length` | int64 | Episode 帧数 |
| `dataset_from_index` | int64 | 在 data 表中的起始索引 |
| `dataset_to_index` | int64 | 在 data 表中的结束索引 |
| `videos/observation.images.rgb/...` | - | 视频文件路径信息 |
| `stats/...` | - | 各特征的统计信息 |

### 4.5 Features 定义

```python
R2R_FEATURES = {
    "observation.images.rgb": {
        "dtype": "video",
        "shape": [height, width, 3],  # 从第一张图推断
        "names": ["height", "width", "channel"]
    },
    "action": {
        "dtype": "int64",
        "shape": [1],
        "names": ["action_index"]
    }
}
```

**注意**: `task` 是每帧必需的特殊字段，但不在 `features` 中定义。

---

## 5. 数据转换规则

### 5.1 Action 对齐

| 原始 | actions | [-1, 3, 3, 3, 2, 1, ...] | 长度 N+1 |
|      | images  | [img0, img1, img2, ...]   | 长度 N   |

| 转换后 | actions | [3, 3, 3, 2, 1, ...] | 长度 N |
|        | images  | [img0, img1, ...]    | 长度 N |

**规则**: `actions = actions[1:]` （丢弃开头的 -1）

### 5.2 Instruction 拆分

| 原始 Episode | 1 个 R2R episode |
|-------------|------------------|
| instructions | [instr0, instr1, instr2] |
| images | [img0, ..., imgN] |
| actions | [act0, ..., actN] |

| 输出 Episodes | 3 个 LeRobot episodes |
|--------------|---------------------|
| episode 0 | images + actions + instr0 |
| episode 1 | images + actions + instr1 |
| episode 2 | images + actions + instr2 |

**规则**: 每个 instruction 生成独立 episode，共享图像和动作序列。

### 5.3 Action 值映射

| 值 | 含义 | 方向 |
|----|------|------|
| 0 | STOP | - |
| 1 | FORWARD | ↑ |
| 2 | LEFT | ← |
| 3 | RIGHT | → |

---

## 6. 数据存储规范

### 6.1 目录结构规范

```
{output_dir}/                    # 用户指定的输出根目录
└── {repo_id}/                   # LeRobot 数据集 ID
    ├── meta/                    # 元数据目录
    │   ├── info.json            # 必需：数据集信息
    │   ├── stats.json           # 必需：数据统计
    │   ├── tasks.parquet        # 必需：任务索引
    │   └── episodes/            # 必需：episode 元数据
    │       └── chunk-000/
    │           └── file-000.parquet
    ├── data/                    # 必需：帧数据
    │   └── chunk-000/
    │       └── file-000.parquet
    └── videos/                  # 必需：视频文件
        └── observation.images.rgb/
            └── chunk-000/
                └── file-000.mp4
```

### 6.2 文件命名规范

| 类型 | 格式 | 示例 |
|-----|------|------|
| 数据目录 | `chunk-{index:03d}` | `chunk-000`, `chunk-001` |
| 数据文件 | `file-{index:03d}.parquet` | `file-000.parquet` |
| 视频目录 | `{video_key}/chunk-{index:03d}` | `observation.images.rgb/chunk-000` |
| 视频文件 | `file-{index:03d}.mp4` | `file-000.mp4` |

### 6.3 Chunk 策略

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `chunks_size` | 1000 | 单个 chunk 的帧数限制 |
| `data_files_size_in_mb` | 100 | 单个数据文件的 MB 限制 |
| `video_files_size_in_mb` | 200 | 单个视频文件的 MB 限制 |

当达到任一限制时，创建新的 chunk。

---

## 7. 接口设计

### 7.1 CLI 接口

```bash
python scripts/dataset_converters/r2r2lerobot.py \
    --data_dir <input_dir> \
    --output_dir <output_dir> \
    --dataset_name <name> \
    --repo_id <repo_id> \
    --fps <fps> \
    --start_idx <idx> \
    --end_idx <idx> \
    --overwrite
```

### 7.2 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--data_dir` | str | `./data/trajectory_data` | 输入数据根目录 |
| `--output_dir` | str | `./data/lerobot` | 输出目录 |
| `--dataset_name` | str | `R2R` | 数据集名称 (R2R/RxR/EnvDrop) |
| `--repo_id` | str | `streamvln/r2r_navigation` | LeRobot repo ID |
| `--fps` | int | `3` | 帧率 |
| `--start_idx` | int | `0` | 起始 episode 索引 |
| `--end_idx` | int | `None` | 结束 episode 索引 |
| `--overwrite` | flag | `False` | 覆盖已存在的输出 |

### 7.3 Python API

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
dataset = LeRobotDataset(repo_id="streamvln/r2r_navigation", root="./data/lerobot")

# 获取样本
sample = dataset[0]
# {
#     "observation.images.rgb": torch.Tensor([3, 480, 640]),
#     "action": torch.Tensor([]),  # scalar
#     "task": '{"instruction": "..."}',
#     "timestamp": torch.Tensor([]),
#     "frame_index": torch.Tensor([]),
#     "episode_index": torch.Tensor([]),
#     "task_index": torch.Tensor([]),
# }
```

---

## 8. 验证与测试

### 8.1 验证工具

```bash
# 验证数据集完整性
python scripts/verify_lerobot_dataset.py \
    --repo_id streamvln/r2r_navigation \
    --root ./data/lerobot \
    --verbose
```

### 8.2 验证检查项

| 检查项 | 说明 |
|-------|------|
| Episode 计数 | info.json total_episodes 与实际一致 |
| Action-Image 对应 | 每个 episode 的 actions 数量 = frames 数量 |
| 图像完整性 | 所有帧可正常加载 |
| 元数据完整性 | 所有必需文件存在 |

### 8.3 数据导出

```bash
# 导出帧图片和元数据
python scripts/verify_lerobot_dataset.py \
    --repo_id streamvln/r2r_navigation \
    --root ./data/lerobot \
    --export_dir ./data/frames \
    --episode_index 0 \
    --max_frames 10
```

输出结构:
```
data/frames/
└── episode_000000/
    ├── metadata.json    # instruction, actions, num_frames
    ├── frame_000000.jpg
    ├── frame_000001.jpg
    └── ...
```

---

## 9. 性能估算

### 9.1 数据量估算

| 指标 | 值 |
|-----|-----|
| 原始 episodes | 3,603 |
| 平均 instructions/episode | ~3 |
| 转换后 episodes | ~10,800 |
| 平均帧数/episode | ~50 |
| 图像尺寸 | 640×480×3 |
| 估算总大小 | ~15-30 GB |

### 9.2 处理速度

| 操作 | 速度 |
|-----|------|
| 图像加载 | ~2000 帧/秒 |
| 视频编码 | ~10-15 秒/episode (50帧) |
| 总体速度 | ~13 秒/episode |

---

## 10. 扩展性设计

### 10.1 支持其他数据集

| 数据集 | 图像路径 | 动作定义 |
|-------|---------|---------|
| R2R | `{scene}_r2r_{id}/rgb/{idx:03d}.jpg` | [-1, 0-3] |
| RxR | `{scene}_rxr_{id}/rgb/{idx:03d}.jpg` | [-1, 0-3] |
| EnvDrop | `{scene}_envdrop_{id}/rgb/{idx:03d}.jpg` | [-1, 0-3] |

### 10.2 扩展点

| 扩展点 | 当前实现 | 扩展方向 |
|-------|---------|---------|
| Features | 固定定义 | 支持动态配置 |
| 图像预处理 | 保持原样 | 添加 resize、normalize |
| 并发处理 | 顺序 | Ray 分布式 |
| 数据划分 | 无 | 支持 train/val/test |

---

## 附录

### A. 文件结构

```
StreamVLN/
├── scripts/
│   └── dataset_converters/
│       └── r2r2lerobot.py          # 主转换脚本
├── streamvln/dataset/
│   └── lerobot_dataset.py          # (废弃) 自定义实现
└── docs/
    └── r2r2lerobot_design.md       # 本文档
```

### B. 参考实现

| 项目 | 链接 |
|-----|------|
| LeRobot | https://github.com/huggingface/lerobot |
| any4lerobot | https://github.com/Tavish9/any4lerobot |
| libero2lerobot | https://github.com/Tavish9/any4lerobot/blob/main/libero2lerobot/libero_h5.py |

### C. 版本历史

| 版本 | 日期 | 变更 |
|-----|------|------|
| v1.0 | 2026-02-06 | 初始版本，基于 LeRobot v3.0 |
