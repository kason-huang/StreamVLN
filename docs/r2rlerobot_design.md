# R2R → LeRobot 数据集转换设计文档

**版本**: v1.1
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

## 2. LeRobot v3.0 数据组织原则

### 2.1 核心设计理念

LeRobot v3.0 引入了全新的数据组织方式，以支持**百万级 episode** 的大规模机器人学习数据集。

| 设计原则 | 说明 | 好处 |
|---------|------|------|
| **File-based storage** | 多个 episodes 共用 Parquet/MP4 文件 | 减少文件数量，降低文件系统压力 |
| **Relational metadata** | Episode 边界通过元数据解析，而非文件名 | 灵活的查询和索引 |
| **Lower filesystem pressure** | 更少、更大的文件 | 更快的初始化速度，更好的扩展性 |
| **Unified organization** | 统一的目录布局和路径模板 | 一致的数据访问方式 |

### 2.2 目录结构职责

```
{output_dir}/{repo_id}/
├── meta/          # 元数据层：描述数据集和 episodes 的"索引"
├── data/          # 数据层：存储实际的特征数据
└── videos/        # 视频层：存储编码后的视觉数据
```

| 目录 | 职责 | 存储内容 |
|-----|------|---------|
| **meta/** | **Schema + Index** | 数据集定义、统计信息、episode 索引、任务索引 |
| **data/** | **Raw Features** | observation、action、timestamp 等特征值 |
| **videos/** | **Encoded Visuals** | 图像序列编码为 MP4 视频文件 |

### 2.3 文件组织详解

#### 2.3.1 meta/ - 元数据层

| 文件 | 格式 | 职责 | 关键内容 |
|-----|------|------|---------|
| **info.json** | JSON | **Dataset Schema** | codebase_version、features、fps、splits、路径模板 |
| **stats.json** | JSON | **Normalization Stats** | mean、std、min、max（用于数据归一化） |
| **tasks.parquet** | Parquet | **Task Index** | task 字符串、task_index（多任务学习支持） |
| **episodes/chunk-000/file-000.parquet** | Parquet | **Episode Directory** | episode_index、length、起止索引、视频路径、统计信息 |

**核心思想**：`meta/` 存储的是**"如何找到数据"** 的信息，而不是数据本身。

#### 2.3.2 data/ - 特征数据层

| 文件 | 格式 | 职责 | 存储内容 |
|-----|------|------|---------|
| **chunk-000/file-000.parquet** | Parquet | **Frame Data** | observation.images.rgb（引用）、action、timestamp、frame_index、episode_index、task_index、index |

**重要说明**：
- 图像数据不直接存储在 parquet 中，而是通过引用指向 `videos/` 中的 MP4 文件
- 多个 episodes 的帧**串联**存储在同一文件中
- Episode 边界通过 `episode_index` 和 `episodes/` 元数据确定

#### 2.3.3 videos/ - 视频数据层

| 文件 | 格式 | 职责 | 存储内容 |
|-----|------|------|---------|
| **observation.images.rgb/chunk-000/file-000.mp4** | MP4 | **Encoded Images** | 一个或多个 episodes 的图像序列（AV1 编码） |

**设计原因**：
- 压缩效率高（AV1 比原始 JPEG 节省 50-70% 空间）
- 顺序读取优化（视频编码设计用于流式读取）
- 按相机分片（每个相机独立的视频文件）

### 2.4 Chunk 策略

LeRobot 使用 **Chunk** 来平衡文件大小和访问效率：

| 参数 | 默认值 | 作用 |
|-----|--------|------|
| `chunks_size` | 1000 | 单个 chunk 的帧数软限制 |
| `data_files_size_in_mb` | 100 | 数据文件的 MB 软限制 |
| `video_files_size_in_mb` | 200 | 视频文件的 MB 软限制 |

**工作原理**：
```
Episode 0-99     → chunk-000/file-000.parquet + video-000.mp4
Episode 100-199  → chunk-000/file-001.parquet + video-001.mp4
Episode 200-299  → chunk-001/file-000.parquet + video-002.mp4
...
```

当达到任一限制时，创建新的 chunk。

### 2.5 Episode 分片存储示意图

```
┌─────────────────────────────────────────────────────────────────┐
│                    data/chunk-000/file-000.parquet               │
├─────────────────────────────────────────────────────────────────┤
│ Frame 0-56    │ Episode 0 │ observation.images.rgb → video[0:56]   │
│ Frame 57-113  │ Episode 1 │ observation.images.rgb → video[57:113]  │
│ Frame 114-170 │ Episode 2 │ observation.images.rgb → video[114:170] │
│ ...           │ ...       │ ...                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 引用
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│          videos/observation.images.rgb/chunk-000/file-000.mp4    │
├─────────────────────────────────────────────────────────────────┤
│ [Ep0 Frame 0] [Ep0 Frame 1] ... [Ep0 Frame 56]                   │
│ [Ep1 Frame 0] [Ep1 Frame 1] ... [Ep1 Frame 56]                   │
│ [Ep2 Frame 0] [Ep2 Frame 1] ... [Ep2 Frame 56]                   │
│ ...                                                              │
└─────────────────────────────────────────────────────────────────┘
```

**关键点**：
- Parquet 存储特征值和**视频索引**（帧偏移量）
- MP4 存储实际的图像数据
- 通过 `episodes/` 元数据建立 Parquet 行范围 ↔ Episode 的映射

### 2.6 数据查询流程

当用户请求 `dataset[idx]` 时：

```
1. 根据 frame_index 确定 episode_index（通过 episodes 元数据）
2. 根据 episode_index 确定 dataset_from_index 和 dataset_to_index
3. 从 data/chunk-XXX/file-XXX.parquet 读取帧数据
4. observation.images.rgb 是引用类型
5. 从 videos/observation.images.rgb/chunk-XXX/file-XXX.mp4 解码图像
6. 返回完整的 sample
```

---

## 3. 系统架构

### 3.1 架构分层图

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

### 3.2 核心流程时序图

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

## 4. 数据格式规范

### 4.1 输入格式 (R2R)

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

### 4.2 输出格式 (LeRobot v3.0)

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

## 5. 元数据文件格式

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

## 6. 数据转换规则

### 6.1 Action 对齐

| 原始 | actions | [-1, 3, 3, 3, 2, 1, ...] | 长度 N+1 |
|      | images  | [img0, img1, img2, ...]   | 长度 N   |

| 转换后 | actions | [3, 3, 3, 2, 1, ...] | 长度 N |
|        | images  | [img0, img1, ...]    | 长度 N |

**规则**: `actions = actions[1:]` （丢弃开头的 -1）

### 6.2 Instruction 拆分

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

### 6.3 Action 值映射

| 值 | 含义 | 方向 |
|----|------|------|
| 0 | STOP | - |
| 1 | FORWARD | ↑ |
| 2 | LEFT | ← |
| 3 | RIGHT | → |

---

## 7. 数据存储规范

### 7.1 目录结构规范

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

### 7.2 文件命名规范

| 类型 | 格式 | 示例 |
|-----|------|------|
| 数据目录 | `chunk-{index:03d}` | `chunk-000`, `chunk-001` |
| 数据文件 | `file-{index:03d}.parquet` | `file-000.parquet` |
| 视频目录 | `{video_key}/chunk-{index:03d}` | `observation.images.rgb/chunk-000` |
| 视频文件 | `file-{index:03d}.mp4` | `file-000.mp4` |

### 7.3 Chunk 策略

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `chunks_size` | 1000 | 单个 chunk 的帧数限制 |
| `data_files_size_in_mb` | 100 | 单个数据文件的 MB 限制 |
| `video_files_size_in_mb` | 200 | 单个视频文件的 MB 限制 |

当达到任一限制时，创建新的 chunk。

---

## 8. 接口设计

### 8.1 CLI 接口

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
例子：python scripts/dataset_converters/r2r2lerobot.py --data_dir "./data/trajectory_data" --output_dir "./data/lerobot" --dataset_name "R2R" --repo_id "streamvln/r2r_navigation" --fps 3 --start_idx 0 --end_idx 20 --overwrite

### 8.2 参数说明

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

### 8.3 Python API

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

## 9. 验证与测试

### 9.1 LeRobot 数据集完整性验证

```bash
# 验证数据集完整性
python scripts/verify_lerobot_dataset.py \
    --repo_id streamvln/r2r_navigation \
    --root ./data/lerobot \
    --verbose
```

### 9.2 验证检查项

| 检查项 | 说明 |
|-------|------|
| Episode 计数 | info.json total_episodes 与实际一致 |
| Action-Image 对应 | 每个 episode 的 actions 数量 = frames 数量 |
| 图像完整性 | 所有帧可正常加载 |
| 元数据完整性 | 所有必需文件存在 |

### 9.3 R2R 与 LeRobot 数据一致性验证

为确保转换后数据的正确性，提供了专门的一致性验证工具：

#### 9.3.1 验证内容

| 验证项 | 说明 |
|-------|------|
| **Instruction** | LeRobot 中的 instruction 与 R2R 原始数据完全匹配 |
| **Actions** | LeRobot 中的 actions 序列与 R2R 原始数据一致（丢弃首个 -1 后） |
| **图片质量** | 从 MP4 解码的图片与原始 R2R 图片进行视觉对比 |

#### 9.3.2 使用方法

```bash
python scripts/verify_r2r_lerobot_with_images.py \
    --r2r_data_dir ./data/trajectory_data \
    --dataset_name R2R \
    --root ./data/lerobot \
    --output_dir ./data/verify_output
```

#### 9.3.3 输出结构

```
data/verify_output/
├── episode_000/
│   ├── frame_0000_compare.jpg  # 左侧: 原始R2R | 右侧: LeRobot解码
│   ├── frame_0001_compare.jpg
│   ├── ...
│   └── README.txt              # 说明文件
├── episode_001/
└── ...
```

每张对比图片由两张并排组成：
- **左侧**：原始 R2R 图片
- **右侧**：LeRobot 从 MP4 视频解码后的图片

#### 9.3.4 验证结果示例

```
================================================================================
验证结果
================================================================================

验证的 episodes 数量: 6

✓ Instruction 匹配: 6/6
✓ Actions 匹配:     6/6

Episodes 详情:
--------------------------------------------------------------------------------
  ✓ Episode   0 | R2R ID:   577 | Instr: 0 | Frames:  57
  ✓ Episode   1 | R2R ID:   577 | Instr: 1 | Frames:  57
  ✓ Episode   2 | R2R ID:   577 | Instr: 2 | Frames:  57
  ✓ Episode   3 | R2R ID:   517 | Instr: 0 | Frames:  48
  ✓ Episode   4 | R2R ID:   517 | Instr: 1 | Frames:  48
  ✓ Episode   5 | R2R ID:   517 | Instr: 2 | Frames:  48

✓ 所有验证通过！
```

#### 9.3.5 关键验证逻辑

**Actions 对齐处理**：
- R2R 原始数据：actions 包含开头的 -1，长度为 N
- 转换后：丢弃首个 -1，长度变为 N-1
- 特殊情况：如果丢弃 -1 后 actions 少于图片数量，会重复最后一个 action 以匹配图片数量

**图片编码/解码验证**：
- 编码：R2R 原始 JPG → MP4 (AV1, 3fps, yuv420p)
- 解码：MP4 → 图片帧
- 验证：逐像素对比（可选）或视觉对比（推荐）

### 9.4 数据导出

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

## 10. 性能估算

### 10.1 数据量估算

| 指标 | 值 |
|-----|-----|
| 原始 episodes | 3,603 |
| 平均 instructions/episode | ~3 |
| 转换后 episodes | ~10,800 |
| 平均帧数/episode | ~50 |
| 图像尺寸 | 640×480×3 |
| 估算总大小 | ~15-30 GB |

### 10.2 处理速度

| 操作 | 速度 |
|-----|------|
| 图像加载 | ~2000 帧/秒 |
| 视频编码 | ~10-15 秒/episode (50帧) |
| 总体速度 | ~13 秒/episode |

---

## 11. 扩展性设计

### 11.1 支持其他数据集

| 数据集 | 图像路径 | 动作定义 |
|-------|---------|---------|
| R2R | `{scene}_r2r_{id}/rgb/{idx:03d}.jpg` | [-1, 0-3] |
| RxR | `{scene}_rxr_{id}/rgb/{idx:03d}.jpg` | [-1, 0-3] |
| EnvDrop | `{scene}_envdrop_{id}/rgb/{idx:03d}.jpg` | [-1, 0-3] |

### 11.2 扩展点

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
│   ├── dataset_converters/
│   │   └── r2r2lerobot.py          # 主转换脚本
│   ├── verify_lerobot_dataset.py   # LeRobot 数据集完整性验证
│   └── verify_r2r_lerobot_with_images.py  # R2R与LeRobot一致性验证
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
| v1.1 | 2026-02-06 | 添加 R2R 与 LeRobot 数据一致性验证章节（9.3） |
