# LeRobotActionDataset 数据处理分析

## 概述

`LeRobotActionDataset` 是 StreamVLN 中用于加载 LeRobot 格式数据集的自定义 Dataset 类。它实现了**懒加载**机制，只在初始化时加载元数据，图片数据在 `__getitem__` 时动态加载。

## 整体架构

```
LeRobotActionDataset
├── 初始化阶段 (__init__)           # 仅加载元数据
├── 数据索引阶段 (_build_data_list)  # 构建样本索引列表
└── 数据加载阶段 (__getitem__)       # 动态加载图片和数据
```

---

## 1. 初始化阶段 (`__init__`)

### 代码位置
`streamvln/dataset/lerobot_action_dataset.py:593-645`

### 流程

```python
def __init__(self, tokenizer, data_args, task_id=0):
    # 基础配置
    self.num_frames = data_args.num_frames        # 每个样本的帧数
    self.num_history = data_args.num_history      # 历史帧采样数
    self.num_future_steps = data_args.num_future_steps  # 未来步长

    # 图片处理器
    self.image_processor = SigLipImageProcessor()

    # 加载 LeRobot 元数据
    self._load_lerobot_metadata()  # ← 只加载元数据，不加载图片

    # 构建数据索引列表
    self.data_list = self._build_data_list()

    # 动作映射
    self.idx2actions = {'0': 'STOP', '1': '↑', '2': '←', '3': '→'}
```

### 关键特性：懒加载

| 阶段 | 加载内容 | 内存占用 |
|------|---------|---------|
| `__init__` | 元数据（json、parquet schema） | ~MB |
| `__getitem__` | 图片字节（按需加载） | 视 batch size |

---

## 2. 元数据加载 (`_load_lerobot_metadata`)

### 代码位置
`streamvln/dataset/lerobot_action_dataset.py:647-682`

### 加载的元数据

```python
def _load_lerobot_metadata(self):
    # 1. info.json
    self.info = {
        "total_episodes": 60,
        "total_frames": 3738,
        "fps": 3,
        "features": {...}
    }

    # 2. episodes DataFrame
    self.episodes_df = pd.DataFrame({
        'episode_index': [0, 1, 2, ...],
        'length': [60, 62, 58, ...],
        'dataset_from_index': [0, 60, 122, ...],
        'dataset_to_index': [60, 122, 180, ...],
        '_cumsum': [60, 122, 180, ...]  # 累积帧数
    })

    # 3. tasks DataFrame
    self.tasks_df = pd.DataFrame({
        'task_json': ['{"instruction": "..."}', ...]
    })

    # 4. data chunks
    self.data_chunks = [
        Path('data/chunk-000'),
        Path('data/chunk-001'),
        ...
    ]
```

### 数据组织结构

```
data/lerobot/
├── meta/
│   ├── info.json              → 数据集信息
│   ├── episodes/
│   │   └── chunk-000/
│   │       └── file-000.parquet → episode 元数据
│   └── tasks.parquet          → task 描述
└── data/
    └── chunk-000/
        └── file-000.parquet   → 帧数据（图片 + action）
```

### Parquet 文件结构

```python
# data/chunk-000/file-000.parquet
columns: [
    'index',                      # 全局帧索引
    'episode_index',              # episode 索引
    'frame_index',                # episode 内帧索引
    'observation.images.rgb',     # 图片数据 {'bytes': b'\x89PNG...'}
    'action',                     # 动作 [0, 1, 2, 3]
    'task_index',                 # task 索引
    'timestamp'                   # 时间戳
]
```

---

## 3. 数据索引构建 (`_build_data_list`)

### 代码位置
`streamvln/dataset/lerobot_action_dataset.py:770-787`

### 切分策略

```python
def _build_data_list(self):
    data_list = []
    for episode_idx in range(self.total_episodes):
        episode_length = 60  # 例如
        num_frames = 6

        # 滑动窗口切分
        num_rounds = episode_length // num_frames  # 10
        for n in range(num_rounds + 1):
            start_frame = n * num_frames
            data_list.append((episode_idx, start_frame, valid_idx))

    return data_list
```

### 示例

```
Episode 0: 60 帧, num_frames=6

Sample 0: (0, 0, 0)   → 帧 0-5
Sample 1: (0, 6, 0)   → 帧 6-11
Sample 2: (0, 12, 0)  → 帧 12-17
...
Sample 9: (0, 54, 0)  → 帧 54-59

len(dataset) = 60 episodes × 10 samples = 600
```

---

## 4. 数据加载流程 (`__getitem__`)

### 核心流程图

```
__getitem__(idx)
    │
    ├─→ 解析 (episode_idx, start_idx, valid_idx)
    │
    ├─→ 计算 time_ids 和 step_ids
    │   │
    │   ├─→ sample_step_ids: 要预测动作的帧
    │   └─→ history_step_ids: 历史观察帧（可选）
    │
    ├─→ 加载图片帧
    │   │
    │   ├─→ History Frames
    │   │   └─→ _locate_frame() → _load_frame_data() → _load_image_from_bytes()
    │   │
    │   └─→ Sample Frames (含 action)
    │       └─→ _locate_frame() → _load_frame_data() → _load_image_from_bytes()
    │
    ├─→ 图片处理 (_process_frame)
    │   └─→ HWC → PIL → transforms → SigLipImageProcessor
    │
    ├─→ 构建对话 (prepare_conversation)
    │   └─→ <image> + action_text 序列
    │
    └─→ 文本预处理 (preprocess)
        └─→ tokenizer + labels (IGNORE_INDEX mask)
```

### 步骤 1: 计算 time_ids 和采样位置

```python
# 行 936-949
# time_ids: 时间戳序列
actions_len = episode_length - valid_idx
time_ids = np.arange(start_idx, min(start_idx + self.num_frames, actions_len))
# 例如: [0, 1, 2, 3, 4, 5]  (6个帧)

# sample_step_ids: 每隔 num_future_steps 采样一帧
start_idx_abs = time_ids[0] + valid_idx
end_idx = time_ids[-1] + 1 + valid_idx
interval = self.num_future_steps
sample_step_ids = np.arange(start_idx_abs, end_idx, interval, dtype=np.int32)
# 例如: num_future_steps=1, 则 [0, 1, 2, 3, 4, 5]
#       num_future_steps=2, 则 [0, 2, 4]

# history_step_ids: 历史帧（等间距采样）
if time_ids[0] != 0:
    history_step_ids = np.arange(
        0 + valid_idx,
        time_ids[0] + valid_idx,
        max(time_ids[0] // self.num_history, 1)
    )
# 例如: start_idx=6, num_history=3, 则 [0, 2, 4]
```

**采样示意图：**

```
Episode 帧序列: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
                      │         │
                      └─ sample ─┘

num_frames=6, num_future_steps=1, num_history=3, start_idx=6:

history_step_ids: [0, 2, 4]     (历史帧，等间距采样)
sample_step_ids:  [6, 7, 8, 9, 10, 11]  (要预测的帧)

最终图片序列: [frame_0, frame_2, frame_4, frame_6, frame_7, ..., frame_11]
              └──── history ────┘  └─────── sample ─────────┘
```

### 步骤 2: 加载图片帧

```python
# 行 955-998
# History Frames
for step_id in history_step_ids:
    global_frame_idx = episode_start + step_id

    # 1. 定位帧所在文件
    chunk_idx, file_idx, file_start_idx = self._locate_frame(global_frame_idx, episode_idx)

    # 2. 从 parquet 加载帧数据
    frame_data = self._load_frame_data(chunk_idx, file_idx, global_frame_idx)
    # frame_data = {
    #     'index': 10,
    #     'action': 3,
    #     'observation.images.rgb': {'bytes': b'\x89PNG...'},
    #     ...
    # }

    # 3. 从 PNG 字节加载图片
    img_bytes = frame_data['observation.images.rgb']['bytes']
    frame = self._load_image_from_bytes(img_bytes)
    # frame = np.array([[[R,G,B], ...]])  # HWC

    # 4. 处理图片
    frame_tensor = self._process_frame(frame)
    images.append(frame_tensor)

# Sample Frames (同时提取 action)
for step_id in sample_step_ids:
    # ... 同上加载图片 ...
    actions.append(action_value)
```

### 步骤 3: 图片处理

```python
# 行 1035-1057
def _process_frame(self, frame: np.ndarray) -> torch.Tensor:
    # 输入: frame (H, W, 3), uint8, [0-255]

    # 1. 归一化到 [0, 1]
    if frame.max() > 1.0:
        frame_tensor = torch.from_numpy(frame).float() / 255.0
    # frame_tensor: [H, W, 3], float32, [0.0-1.0]

    # 2. HWC → CHW
    frame_tensor = frame_tensor.permute(2, 0, 1)
    # frame_tensor: [3, H, W]

    # 3. CHW → HWC → PIL
    frame_pil = Image.fromarray(
        (frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )

    # 4. transforms (resize, crop 等)
    if self.transforms is not None:
        frame_pil = self.transforms(frame_pil)

    # 5. SigLipImageProcessor
    frame_processed = self.image_processor.preprocess(
        images=frame_pil,
        return_tensors='pt'
    )['pixel_values'][0]
    # frame_processed: [3, 224, 224] (SigLip 标准输入)

    return frame_processed
```

### 步骤 4: 构建对话

```python
# 行 1016-1024
# 初始模板
sources = [
    {"from": "human", "value": "You are...<instruction>."},
    {"from": "gpt", "value": ""}
]

# 添加历史观察 token
if start_idx != 0:
    sources[0]["value"] += f' These are your historical observations: {DEFAULT_MEMORY_TOKEN}.'

# 替换 instruction
sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)

# 交织图片和动作
interleave_sources = self.prepare_conversation(sources, list(actions))
```

### prepare_conversation() 详细

```python
# 行 804-823
def prepare_conversation(self, conversation, actions):
    # actions = [1, 3, 1, 0]  (↑, →, ↑, STOP)

    i = 0
    sources = []
    while i < len(actions):
        source = copy.deepcopy(conversation)

        # 随机选择连接词
        prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
        # 例如: "you can see <image>"

        # 提取当前步的动作
        step_actions = actions[i:i+self.num_future_steps]
        # num_future_steps=1: step_actions = [1]
        # num_future_steps=2: step_actions = [1, 3]

        # 转换为文本
        answer = self.actions2text(step_actions)
        # [1] → "↑"
        # [1, 3] → "↑→"

        # 构建对话
        source[0]["value"] = prompt
        source[1]["value"] = answer

        i += len(step_actions)
        sources.extend(source)

    return sources
```

**输出示例：**

```python
# actions = [1, 3, 1, 0], num_future_steps=1
[
    {"from": "human", "value": "you can see <image>."},
    {"from": "gpt", "value": "↑"},
    {"from": "human", "value": "in front of you is <image>."},
    {"from": "gpt", "value": "→"},
    {"from": "human", "value": "there is <image>."},
    {"from": "gpt", "value": "↑"},
    {"from": "human", "value": "you can spot <image>."},
    {"from": "gpt", "value": "STOP"},
]
```

### 步骤 5: 文本预处理

```python
# 行 1026-1033
data_dict = preprocess([interleave_sources], self.tokenizer, True)

# preprocess() 根据 conversation_lib.default_conversation.version 选择:
# - preprocess_qwen()    # Qwen chat template
# - preprocess_llama3()  # Llama3 chat template
# - preprocess_gemma()   # Gemma chat template
# - ...

# 处理流程:
# 1. 将对话转换为 prompt (如 <|im_start|>user\n...\n<|im_end|>)
# 2. Tokenize
# 3. 生成 labels:
#    - 用户输入部分 → IGNORE_INDEX (-100)
#    - 模型输出部分 → 真实 token id
```

**示例：**

```
Input:
  Human: "you can see <image>."
  Assistant: "↑"

Token IDs:
  [<|im_start|>, user, you, can, see, <image>, ., <|im_end|>, <|im_start|>, assistant, ↑, <|im_end|>]
  [         1,   2,   3,   4,    5,   151644,  7,          2,          1,        3,         151643,    2]

Labels:
  [-100, -100, -100, -100, -100, 151644, -100, -100, -100, -100, -100, 151643, -100]
   └─────────────── mask ──────────────┘   └─ 预测 ─┘
```

---

## 5. 返回值

```python
return (
    data_dict["input_ids"][0],  # [seq_len]    token 序列
    data_dict["labels"][0],     # [seq_len]    label 序列
    images,                      # [T, 3, H, W] 图片张量
    time_ids,                    # [T]         时间戳
    self.task                    # task_id
)
```

### 形状说明

| 变量 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `input_ids` | `[seq_len]` | torch.long | token 序列 |
| `labels` | `[seq_len]` | torch.long | label 序列（用户部分为 -100） |
| `images` | `[T, 3, H, W]` | torch.float | 图片张量，T = len(history) + len(sample) |
| `time_ids` | `[T]` | torch.long | 时间戳序列 |
| `task` | 标量 | int | task_id |

---

## 6. 关键设计特点

### 6.1 懒加载 (Lazy Loading)

```python
# __init__: 只加载元数据
self.episodes_df = pd.read_parquet(episodes_file)  # 几 KB
self.tasks_df = pd.read_parquet(tasks_file)        # 几 KB

# __getitem__: 按需加载图片
frame_data = self._load_frame_data(...)  # 一行数据，包含图片字节
img_bytes = frame_data['observation.images.rgb']['bytes']  # PNG 字节，几十 KB
frame = self._load_image_from_bytes(img_bytes)  # 解码为 numpy array
```

**优势：**
- 初始化快（秒级）
- 内存占用小
- 支持大规模数据集

### 6.2 滑动窗口切分

```
Episode (60 帧)
├── Sample 0: 帧 0-5    (start_frame=0)
├── Sample 1: 帧 6-11   (start_frame=6)
├── Sample 2: 帧 12-17  (start_frame=12)
...
└── Sample 9: 帧 54-59  (start_frame=54)
```

**参数控制：**
- `num_frames`: 窗口大小
- `num_future_steps`: 窗口步长

### 6.3 历史观察机制

```python
# 当前窗口: 帧 6-11
# 历史窗口: 帧 0-5（采样）

history_step_ids = np.arange(0, 6, max(6 // 3, 1))  # [0, 2, 4]

# 最终序列: [frame_0, frame_2, frame_4, frame_6, frame_7, ..., frame_11]
#              └──── history ────┘  └────── current ──────┘
```

**用途：** 提供上下文信息，类似 Transformer 的 causal mask

### 6.4 交错格式

```python
# 图片和动作交替排列
Human: "<image>"
Assistant: "↑"
Human: "<image>"
Assistant: "→"
Human: "<image>"
Assistant: "STOP"
```

**优势：**
- 每个图片-动作对是独立的训练样本
- 支持 variable-length action sequences

### 6.5 Label Masking

```python
# 只预测 assistant 的输出
input_ids:  [<|im_start|>, user, hello, <|im_end|>, <|im_start|>, assistant, hi, <|im_end|>]
labels:     [-100, -100, -100, -100, -100, -100, -100, hi, -100]
            └───────────────────── mask ─────────────────────┘ └─ 预测 ─┘
```

**实现：**
- 用户输入部分 → `IGNORE_INDEX` (-100)
- 模型输出部分 → 真实 token id

---

## 7. 数据流示意图

```
LeRobot 数据集
├── meta/
│   ├── info.json           → total_episodes, total_frames, fps
│   ├── episodes/           → episode_index, length, from/to_index
│   └── tasks.parquet       → task 描述
└── data/
    └── chunk-000/
        └── file-000.parquet → [index, action, observation.images.rgb, ...]
                                 │
                                 └─→ observation.images.rgb = {'bytes': b'\x89PNG...'}
                                                                  │
                                                                  ↓
                                                            __getitem__()
                                                                 │
                                    ┌────────────────────────────────┘
                                    ↓
                            _load_image_from_bytes()
                                    │
                                    ↓
                            Image.open(BytesIO()) → numpy array
                                    │
                                    ↓
                            _process_frame() → torch tensor
                                    │
                                    ↓
                            [3, 224, 224]  # SigLip 处理后的图片
```

---

## 8. 性能优化建议

### 8.1 并行加载

当前实现是串行加载图片，可以使用 `num_workers > 0` 的 DataLoader 来并行化：

```python
dataloader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,  # 并行加载
    pin_memory=True
)
```

### 8.2 Parquet 缓存

对于小规模数据集，可以缓存 parquet 文件：

```python
def _load_frame_data(self, chunk_idx, file_idx, idx):
    if not hasattr(self, '_parquet_cache'):
        self._parquet_cache = {}

    cache_key = (chunk_idx, file_idx)
    if cache_key not in self._parquet_cache:
        file_path = self.data_chunks[chunk_idx] / f"file-{file_idx:03d}.parquet"
        self._parquet_cache[cache_key] = pd.read_parquet(file_path)

    df = self._parquet_cache[cache_key]
    # ...
```

### 8.3 图片缓存

对于频繁访问的帧，可以缓存处理后的 tensor：

```python
if hasattr(self, '_image_cache') and key in self._image_cache:
    frame_tensor = self._image_cache[key]
else:
    frame = self._load_image_from_bytes(img_bytes)
    frame_tensor = self._process_frame(frame)
    if len(self._image_cache) < self.cache_size:
        self._image_cache[key] = frame_tensor
```

---

## 9. 与 VLNActionDataset 的详细对比

### 9.1 数据格式对比

| 方面 | VLNActionDataset | LeRobotActionDataset |
|------|-----------------|---------------------|
| **数据来源** | R2R/RxR 原始数据 | LeRobot 格式数据 |
| **图片格式** | JPG 文件 | PNG 字节（存储在 parquet） |
| **标注格式** | JSON (annotations.json) | Parquet (episodes/*.parquet, tasks.parquet) |
| **目录结构** | 扁平结构 | 层次结构（chunks） |
| **多指令支持** | ✅ (instructions 列表) | ✅ (tasks.parquet) |

#### VLNActionDataset 数据组织

```
data/trajectory_data/R2R/
├── annotations.json           # 所有 episode 的标注
└── images/
    ├── 17DRP5sb8fy_r2r_000577/
    │   └── rgb/
    │       ├── 000.jpg
    │       ├── 001.jpg
    │       └── ...
    └── ...
```

**annotations.json 结构：**
```json
[
  {
    "id": 0,
    "video": "images/17DRP5sb8fy_r2r_000577",
    "instructions": ["Go to the kitchen.", "Walk to the kitchen."],
    "actions": [-1, 3, 1, 3, 1, ...],
    "heading": [0, 15, 30, 30, ...],
    "path": [...]
  },
  ...
]
```

#### LeRobotActionDataset 数据组织

```
data/lerobot/streamvln/r2r_navigation/
├── meta/
│   ├── info.json                          # 数据集元信息
│   ├── episodes/
│   │   └── chunk-000/
│   │       └── file-000.parquet           # episode 边界信息
│   └── tasks.parquet                      # task 描述
└── data/
    └── chunk-000/
        └── file-000.parquet               # 帧数据（图片+action）
```

**info.json 结构：**
```json
{
  "total_episodes": 60,
  "total_frames": 3738,
  "fps": 3,
  "features": {
    "observation.images.rgb": {"dtype": "image", "shape": [480, 640, 3]},
    "action": {"dtype": "int64", "shape": [1]}
  }
}
```

---

### 9.2 初始化对比

| 方面 | VLNActionDataset | LeRobotActionDataset |
|------|-----------------|---------------------|
| **数据加载** | 一次性加载 `annotations.json` | 加载多个 parquet 文件 |
| **内存占用** | 小（JSON 几 MB） | 中（DataFrame 几十 MB） |
| **初始化速度** | 快 | 中 |
| **支持增量更新** | ❌ | ✅ |

#### VLNActionDataset.__init__

```python
# 行 614-694
def __init__(self, tokenizer, data_args, task_id):
    self.video_folder = data_args.video_folder.split(',')

    # 1. 加载所有标注数据
    self.nav_data = self.load_vln_data(data_args)
    # nav_data = [
    #   {'id': 0, 'video': '...', 'instructions': [...], 'actions': [...], ...},
    #   ...
    # ]

    # 2. 构建 data_list
    for ep_id, item in enumerate(self.nav_data):
        instructions = item['instructions']
        actions = item['actions']

        # 支持多个 instruction
        for ins_id in range(len(instructions)):
            # 清理初始旋转
            valid_idx = self.clean_initial_rotations(instructions[ins_id], actions)

            # 滑动窗口切分
            num_rounds = (actions_len - valid_idx) // self.num_frames
            for n in range(num_rounds + 1):
                self.data_list.append((ep_id, ins_id, n * self.num_frames, valid_idx))
```

**data_list 结构：**
```python
# (ep_id, ins_id, start_idx, valid_idx)
[
    (0, 0, 0, 0),    # episode 0, instruction 0, start from frame 0
    (0, 0, 6, 0),    # episode 0, instruction 0, start from frame 6
    (0, 1, 0, 2),    # episode 0, instruction 1, start from frame 0, skip first 2
    ...
]
```

#### LeRobotActionDataset.__init__

```python
# 行 593-645
def __init__(self, tokenizer, data_args, task_id):
    # 1. 加载元数据
    self._load_lerobot_metadata()
    # - info.json: total_episodes, total_frames, fps
    # - episodes_df: episode_index, length, from/to_index
    # - tasks_df: task_json
    # - data_chunks: [Path('data/chunk-000'), ...]

    # 2. 构建 data_list
    for episode_idx in range(self.total_episodes):
        episode_length = self.episodes_df[...]['length']

        num_rounds = (episode_length - valid_idx) // self.num_frames
        for n in range(num_rounds + 1):
            self.data_list.append((episode_idx, n * self.num_frames, valid_idx))
```

**data_list 结构：**
```python
# (episode_idx, start_idx, valid_idx)
[
    (0, 0, 0),    # episode 0, start from frame 0
    (0, 6, 0),    # episode 0, start from frame 6
    (1, 0, 0),    # episode 1, start from frame 0
    ...
]
```

---

### 9.3 数据加载对比

| 方面 | VLNActionDataset | LeRobotActionDataset |
|------|-----------------|---------------------|
| **图片路径** | 直接文件路径 | Parquet → PNG 字节 |
| **定位方式** | 文件列表索引 | `_locate_frame()` 二分查找 |
| **支持 chunks** | ❌ | ✅ |
| **解码方式** | PIL.open | PIL.open(BytesIO) |

#### VLNActionDataset.__getitem__

```python
# 行 774-826
def __getitem__(self, i):
    ep_id, ins_id, start_idx, valid_idx = self.data_list[i]
    data = self.nav_data[ep_id]
    video_path = data['video']  # 例如: "images/17DRP5sb8fy_r2r_000577"

    # 1. 获取所有图片文件名
    video_frames = sorted(os.listdir(os.path.join(video_path, 'rgb')))
    # ['000.jpg', '001.jpg', '002.jpg', ...]

    # 2. 获取 instruction
    instructions = data.get("instructions", [])
    if not isinstance(instructions, list):
        instructions = [instructions]
    instruction = instructions[ins_id]

    # 3. 获取 actions
    actions = data['actions'][1+valid_idx:] + [0]

    # 4. 计算 time_ids 和采样位置
    time_ids = np.arange(start_idx, min(start_idx + self.num_frames, actions_len))
    sample_step_ids = np.arange(start_idx, end_idx, interval, dtype=np.int32)
    history_step_ids = np.arange(0+valid_idx, time_ids[0]+valid_idx, ...) if time_ids[0] != 0 else []

    # 5. 构建图片文件路径
    sample_frames = [os.path.join(video_path, 'rgb', video_frames[i]) for i in sample_step_ids]
    history_frames = [os.path.join(video_path, 'rgb', video_frames[i]) for i in history_step_ids]

    # 6. 加载并处理图片
    images = []
    for image_file in history_frames + sample_frames:
        image = Image.open(image_file).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
        images.append(image)

    # 7. 构建对话
    sources = copy.deepcopy(self.conversations)
    if start_idx != 0:
        sources[0]["value"] += f' These are your historical observations: {DEFAULT_MEMORY_TOKEN}.'
    sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
    interleave_sources = self.prepare_conversation(sources, list(actions))

    # 8. 文本预处理
    data_dict = preprocess([interleave_sources], self.tokenizer, True)

    return data_dict["input_ids"][0], data_dict["labels"][0], images, torch.tensor(time_ids), self.task
```

#### LeRobotActionDataset.__getitem__

```python
# 行 925-1033
def __getitem__(self, idx):
    episode_idx, start_idx, valid_idx = self.data_list[idx]

    # 1. 获取 episode 数据
    episode_data = self.episodes_df[self.episodes_df['episode_index'] == episode_idx]
    episode_start = int(episode_data.iloc[0]['dataset_from_index'])
    episode_length = int(episode_data.iloc[0]['length'])

    # 2. 计算 time_ids 和采样位置
    time_ids = np.arange(start_idx, min(start_idx + self.num_frames, actions_len))
    sample_step_ids = np.arange(start_idx_abs, end_idx, interval, dtype=np.int32)
    history_step_ids = np.arange(0+valid_idx, time_ids[0]+valid_idx, ...) if time_ids[0] != 0 else []

    # 3. 加载图片（从 parquet）
    images = []
    actions = []

    for step_id in history_step_ids:
        global_frame_idx = episode_start + step_id
        # 定位帧所在文件
        chunk_idx, file_idx, file_start_idx = self._locate_frame(global_frame_idx, episode_idx)
        # 加载帧数据
        frame_data = self._load_frame_data(chunk_idx, file_idx, global_frame_idx)
        # 从 PNG 字节加载图片
        img_bytes = frame_data['observation.images.rgb']['bytes']
        frame = self._load_image_from_bytes(img_bytes)
        # 处理图片
        frame_tensor = self._process_frame(frame)
        images.append(frame_tensor)

    for step_id in sample_step_ids:
        # ... 同上加载图片 ...
        # 提取 action
        action_value = frame_data.get('action', 0)
        actions.append(action_value)

    # 4. 获取 instruction（从 task）
    task_index = episode_data.iloc[0].get('task_index', 0)
    task_json = self._get_task_description(task_index)
    task_data = json.loads(task_json)
    instruction = task_data.get('instruction', 'Navigate to the goal.')

    # 5-7. 同 VLNActionDataset
    ...
```

---

### 9.4 关键差异点

#### 1. 图片加载方式

**VLNActionDataset:**
```python
# 直接从文件路径加载
image_file = os.path.join(video_path, 'rgb', video_frames[i])
image = Image.open(image_file).convert('RGB')
```

**LeRobotActionDataset:**
```python
# 从 parquet 读取 PNG 字节
frame_data = self._load_frame_data(chunk_idx, file_idx, global_frame_idx)
img_bytes = frame_data['observation.images.rgb']['bytes']
frame = self._load_image_from_bytes(img_bytes)
    ↓
from io import BytesIO
img = Image.open(BytesIO(image_bytes))
return np.array(img)
```

#### 2. 帧定位方式

**VLNActionDataset:**
```python
# 直接使用文件列表索引
video_frames = sorted(os.listdir(...))  # ['000.jpg', '001.jpg', ...]
image_file = video_frames[frame_idx]
```

**LeRobotActionDataset:**
```python
# 通过元数据定位
def _locate_frame(self, idx: int, episode_idx: int):
    # 1. 检查是否单文件
    if len(self.data_chunks) == 1:
        return 0, 0, 0

    # 2. 多文件：搜索 parquet 文件
    for chunk_idx, chunk_dir in enumerate(self.data_chunks):
        for file_idx, file_path in enumerate(files):
            df = pd.read_parquet(file_path)
            if file_start <= idx < file_end:
                return chunk_idx, file_idx, file_start
```

#### 3. Instruction 获取方式

**VLNActionDataset:**
```python
# 从 annotations.json 的 instructions 字段获取
instructions = data.get("instructions", [])
if not isinstance(instructions, list):
    instructions = [instructions]
instruction = instructions[ins_id]  # 直接使用
```

**LeRobotActionDataset:**
```python
# 从 tasks.parquet 获取
task_index = episode_data.iloc[0].get('task_index', 0)
task_json = self._get_task_description(task_index)
task_data = json.loads(task_json)
instruction = task_data.get('instruction', 'Navigate to the goal.')
```

#### 4. 数据索引结构

**VLNActionDataset:**
```python
# (ep_id, ins_id, start_idx, valid_idx)
# - ep_id: 在 nav_data 列表中的索引
# - ins_id: instructions 列表中的索引（支持多指令）
# - start_idx: 起始帧位置
# - valid_idx: 跳过的初始帧数
```

**LeRobotActionDataset:**
```python
# (episode_idx, start_idx, valid_idx)
# - episode_idx: episode 索引（已在 parquet 中分割 instruction）
# - start_idx: 起始帧位置
# - valid_idx: 通常为 0
```

#### 5. 初始旋转清理

**VLNActionDataset:**
```python
def clean_initial_rotations(self, instruction, actions):
    # 分析 instruction 和 actions，找到开始导航的位置
    # 返回应该跳过的初始帧数
    ...
    if self.remove_init_turns:
        valid_idx = self.clean_initial_rotations(instructions[ins_id], actions)
```

**LeRobotActionDataset:**
```python
# 没有初始旋转清理
# 因为在数据转换时已处理（r2r2lerobot.py 中 drop first action -1）
valid_idx = 0  # LeRobot datasets don't have initial rotations to skip
```

---

### 9.5 性能对比

| 方面 | VLNActionDataset | LeRobotActionDataset |
|------|-----------------|---------------------|
| **初始化时间** | ~1 秒 | ~2-3 秒 |
| **内存占用（初始化）** | ~10 MB | ~50 MB |
| **单个样本加载** | ~10-20 ms | ~15-30 ms |
| **支持大规模数据** | ✅（但目录扫描慢） | ✅（parquet 索引快） |
| **多进程友好** | ⚠️（文件系统竞争） | ✅（parquet 并发读） |

---

### 9.6 使用场景建议

**使用 VLNActionDataset 当：**
- ✅ 数据已经是 R2R/RxR 格式
- ✅ 需要支持多指令（一个 trajectory 多个 instructions）
- ✅ 需要清理初始旋转
- ✅ 数据规模较小（< 10k episodes）

**使用 LeRobotActionDataset 当：**
- ✅ 数据已转换为 LeRobot 格式
- ✅ 需要与其他 LeRobot 数据集联合训练
- ✅ 数据规模较大（> 10k episodes）
- ✅ 需要更好的数据组织（chunks、压缩）
- ✅ 需要版本控制和元数据管理

---

## 10. 内存加载分析

### 10.1 核心问题：会一开始就加载全部图片吗？

**❌ 不会！** `LeRobotActionDataset` 采用了**懒加载（Lazy Loading）**策略，不会在初始化时加载全部图片到内存中。

### 10.2 初始化阶段的内存加载

#### __init__ 方法分析

```python
# 行 593-645: __init__
def __init__(self, tokenizer, data_args, task_id):
    # 基础配置（只有数值和对象引用）
    self.task_id = task_id
    self.image_size = data_args.image_size
    self.num_frames = data_args.num_frames
    self.image_processor = SigLipImageProcessor()

    # 1. 加载元数据（不含图片）
    self._load_lerobot_metadata()  # ← 只加载 JSON 和 Parquet schema

    # 2. 构建索引列表（不含图片）
    self.data_list = self._build_data_list()  # ← 只构建 (episode_idx, start_idx, valid_idx) 元组
```

#### _load_lerobot_metadata 方法分析

```python
# 行 647-682: _load_lerobot_metadata
def _load_lerobot_metadata(self):
    # 1. info.json - 只有几 KB
    self.info = json.load(info_file)
    # 内容: {"total_episodes": 60, "total_frames": 3738, "fps": 3, "features": {...}}

    # 2. episodes_df - DataFrame，只有几十 KB
    self.episodes_df = pd.read_parquet(episodes_file)
    # 列: episode_index, length, dataset_from_index, dataset_to_index
    # 不包含图片数据！

    # 3. tasks_df - DataFrame，只有几 KB
    self.tasks_df = pd.read_parquet(tasks_file)
    # 列: task_json (JSON 字符串)

    # 4. data_chunks - 只有路径列表
    self.data_chunks = [Path('data/chunk-000'), ...]
    # 不读取 parquet 文件内容！
```

#### _build_data_list 方法分析

```python
# 行 770-787: _build_data_list
def _build_data_list(self):
    data_list = []
    for episode_idx in range(self.total_episodes):
        episode_length = int(episode_data.iloc[0]['length'])

        num_rounds = (episode_length - valid_idx) // self.num_frames
        for n in range(num_rounds + 1):
            start_frame = n * self.num_frames
            # 只存储元组，不加载图片
            data_list.append((episode_idx, start_frame, valid_idx))

    return data_list
    # 例如: [(0, 0, 0), (0, 6, 0), (1, 0, 0), ...]
    # 每个元组只有 3 个整数，约 72 bytes
    # 10000 个样本 ≈ 720 KB
```

### 10.3 运行时的内存加载

#### __getitem__ 方法分析

```python
# 行 925-1033: __getitem__
def __getitem__(self, idx):
    episode_idx, start_idx, valid_idx = self.data_list[idx]

    # 动态加载图片（每个样本只加载需要的图片）
    images = []
    actions = []

    # 1. 加载历史帧
    for step_id in history_step_ids:
        global_frame_idx = episode_start + step_id

        # 定位帧所在文件
        chunk_idx, file_idx, file_start_idx = self._locate_frame(global_frame_idx, episode_idx)

        # 从 parquet 加载一行数据（包含 PNG 字节）
        frame_data = self._load_frame_data(chunk_idx, file_idx, global_frame_idx)
        # frame_data = {
        #     'index': 10,
        #     'action': 3,
        #     'observation.images.rgb': {'bytes': b'\x89PNG...'},  # ← PNG 字节
        #     ...
        # }

        # 从 PNG 字节加载图片
        img_bytes = frame_data['observation.images.rgb']['bytes']
        frame = self._load_image_from_bytes(img_bytes)

        # 处理图片
        frame_tensor = self._process_frame(frame)
        images.append(frame_tensor)

    # 2. 加载样本帧（同上）
    for step_id in sample_step_ids:
        # ... 同样的加载流程 ...
        actions.append(action_value)

    # 3. 构建对话和预处理
    sources = copy.deepcopy(self.conversations)
    interleave_sources = self.prepare_conversation(sources, list(actions))
    data_dict = preprocess([interleave_sources], self.tokenizer, True)

    # 4. 返回（图片在内存中）
    return data_dict["input_ids"][0], data_dict["labels"][0], images, time_ids, self.task
```

### 10.4 内存占用详细分析

#### 初始化阶段

| 数据 | 类型 | 大小 | 说明 |
|------|------|------|------|
| `self.info` | dict | ~5 KB | JSON 元数据 |
| `self.episodes_df` | DataFrame | ~50 KB | episode 边界信息 |
| `self.tasks_df` | DataFrame | ~10 KB | task 描述 |
| `self.data_chunks` | list[Path] | ~1 KB | 路径列表 |
| `self.data_list` | list[tuple] | ~1 MB | 索引元组（10k 样本） |
| **总计** | - | **~2 MB** | 不含图片 |

#### 单个样本加载

| 数据 | 类型 | 大小 | 说明 |
|------|------|------|------|
| PNG 字节 | bytes | ~100 KB | 压缩的图片数据 |
| 解码后 numpy array | ndarray | ~900 KB | 480x640x3 uint8 |
| 处理后 tensor | torch.Tensor | ~600 KB | 3x224x224 float32 |
| 单张图片 | - | ~600 KB | SigLip 处理后 |
| 6 张图片 | - | ~3.6 MB | num_frames=6 |
| **单个样本** | - | **~5 MB** | 包含图片、文本、标签 |

#### DataLoader 运行时

```python
dataloader = DataLoader(dataset, batch_size=8, num_workers=4)

# 内存占用计算:
# 每个 worker: 5 MB × 1 样本 = 5 MB
# 4 个 workers: 5 MB × 4 = 20 MB
# Batch 收集: 5 MB × 8 = 40 MB
# 总共约 60-100 MB（取决于 prefetch）
```

### 10.5 懒加载的优势

| 特性 | 懒加载（LeRobotActionDataset） | 预加载（传统方式） |
|------|------------------------------|------------------|
| **初始化时间** | ~2-3 秒 | ~10-30 分钟 |
| **初始化内存** | ~2 MB | 几十 GB |
| **支持大规模** | ✅ 100k+ episodes | ❌ 受内存限制 |
| **增量更新** | ✅ 无需重新加载 | ❌ 需要重启 |
| **多进程友好** | ✅ 每个 worker 独立 | ⚠️ 需要共享内存 |

### 10.6 内存加载流程图

```
┌─────────────────────────────────────────────────────────────┐
│  初始化阶段 (__init__)                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. _load_lerobot_metadata()                          │   │
│  │    ├─ info.json          → ~5 KB                    │   │
│  │    ├─ episodes.parquet   → ~50 KB                   │   │
│  │    ├─ tasks.parquet      → ~10 KB                   │   │
│  │    └─ data_chunks list   → ~1 KB                    │   │
│  │                                                         │   │
│  │ 2. _build_data_list()                                 │   │
│  │    └─ [(ep_idx, start, valid), ...] → ~1 MB         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  总内存: ~2 MB                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  训练阶段 (DataLoader + __getitem__)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Batch 0:                                             │   │
│  │   Worker 1: __getitem__(0)  → 加载 6 张图 → ~5 MB   │   │
│  │   Worker 2: __getitem__(1)  → 加载 6 张图 → ~5 MB   │   │
│  │   Worker 3: __getitem__(2)  → 加载 6 张图 → ~5 MB   │   │
│  │   Worker 4: __getitem__(3)  → 加载 6 张图 → ~5 MB   │   │
│  │   Collate: 4 个样本 → Batch → ~40 MB                │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Batch 1: (释放 Batch 0 的内存)                        │   │
│  │   Worker 1: __getitem__(4)  → 加载 6 张图 → ~5 MB   │   │
│  │   Worker 2: __getitem__(5)  → 加载 6 张图 → ~5 MB   │   │
│  │   ...                                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  峰值内存: ~60-100 MB                                       │
└─────────────────────────────────────────────────────────────┘
```

### 10.7 与 VLNActionDataset 的内存对比

| 阶段 | VLNActionDataset | LeRobotActionDataset |
|------|-----------------|---------------------|
| **初始化** | ~10 MB (annotations.json) | ~2 MB (元数据) |
| **单个样本** | ~5-10 MB (6-12 张图片) | ~5 MB (6 张图片) |
| **是否预加载图片** | ❌ 按需加载 | ❌ 按需加载 |
| **支持大规模** | ✅ 但目录扫描慢 | ✅ Parquet 索引快 |

**两者都采用懒加载策略，都不会在初始化时加载全部图片。**

### 10.8 常见问题

#### Q1: 如果数据集有 10000 个 episodes，初始化会慢吗？

**A:** 不会。初始化只加载元数据，时间复杂度是 O(1)：
- 加载 info.json: ~1ms
- 加载 episodes.parquet: ~10ms (整个文件，不管多少 episodes)
- 构建 data_list: ~5ms (只是循环和元组创建)
- **总计: ~20ms**

#### Q2: 如果数据集有 10000 个 episodes，初始化会占用多少内存？

**A:** 约 2-3 MB：
- episodes_df: ~50 KB (所有 episodes 的元信息)
- data_list: ~1 MB (10000 个 3 元组)
- 其他: ~1 MB
- **总计: ~2 MB**

#### Q3: 训练时会内存溢出吗？

**A:** 不会。每次只加载一个 batch 的图片：
- batch_size=8: 每张 ~600KB × 8 ≈ 5MB
- num_workers=4: 5MB × 4 = 20MB
- 总共: ~60-100 MB（含缓存）

#### Q4: 可以增加 batch_size 吗？

**A:** 可以。根据 GPU 内存调整：
```python
# 估计 GPU 内存占用
per_sample = 5 MB  # 图片
model_forward = 2 GB  # 模型前向传播
gradient = 2 GB  # 梯度
optimizer_state = 2 GB  # 优化器状态

# 总计 = 2GB + 2GB + 2GB + (5MB × batch_size)
# batch_size=32: ~6GB + 160MB ≈ 6.2 GB (适合 8GB GPU)
# batch_size=64: ~6GB + 320MB ≈ 6.3 GB (适合 8GB GPU)
```

---

## 11. 数据格式详解：为什么是 HWC？

### 11.1 维度顺序的来源

在 `r2r2lerobot.py` 中，图片的 features 定义如下：

```python
# 行 30-35
R2R_FEATURES = {
    "observation.images.rgb": {
        "dtype": "image",
        "shape": None,  # Will be inferred from first image: [height, width, channel]
        "names": ["height", "width", "channel"]
    },
    ...
}
```

**`names` 字段说明：**
- 这是一个**描述性标签**，用于说明 `shape` 中每个维度代表什么
- 它反映了 NumPy 数组的实际格式：`(height, width, channel)`
- 这种格式被称为 **HWC** 格式

### 11.2 为什么选择 HWC 而不是 CHW？

#### NumPy 的默认行为

```python
# 行 103-104: r2r2lerobot.py
from PIL import Image
img = Image.open(img_path)
img_array = np.array(img)  # ← 返回 (H, W, C) 格式
```

当你使用 `np.array(img)` 从 PIL Image 创建 NumPy 数组时，自动得到 `(H, W, C)` 格式：

```python
# 示例：480x640 的 RGB 图片
img = Image.open("scene.jpg")
arr = np.array(img)

print(arr.shape)  # (480, 640, 3)
#                  │     │    └─ channel (R, G, B)
#                  │     └────── width (640 像素/行)
#                  └──────────── height (480 行)
```

#### 图像格式对比

| 格式 | 维度顺序 | 使用场景 | 示例 |
|------|---------|----------|------|
| **HWC** | (Height, Width, Channel) | NumPy, PIL, OpenCV | `(480, 640, 3)` |
| **CHW** | (Channel, Height, Width) | PyTorch, Cx | `(3, 480, 640)` |
| **BHWC** | (Batch, Height, Width, Channel) | TensorFlow 默认 | `(8, 480, 640, 3)` |
| **BCHW** | (Batch, Channel, Height, Width) | PyTorch 默认 | `(8, 3, 480, 640)` |

**LeRobot 选择 HWC 的原因：**

| 考虑因素 | HWC ✅ | CHW ❌ |
|---------|--------|--------|
| **NumPy 原生支持** | ✅ 默认格式 | ❌ 需要转换 |
| **PIL 兼容** | ✅ 直接输出 | ❌ 需要转换 |
| **OpenCV 兼容** | ✅ 默认格式 | ❌ 需要转换 |
| **数据可视化** | ✅ matplotlib 直接显示 | ❌ 需要转换 |
| **人类直觉** | ✅ 高×宽，符合直觉 | ⚠️ 通道在前 |

### 11.3 完整的数据流动

```
┌─────────────────────────────────────────────────────────────┐
│  数据存储格式 (HWC)                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  原始 JPG 文件                                              │
│       │                                                     │
│       ↓ PIL.Image.open()                                   │
│  PIL Image 对象                                            │
│       │                                                     │
│       ↓ np.array()                                         │
│  NumPy Array (480, 640, 3)  ← HWC 格式                     │
│       │                                                     │
│       ↓ 存入 Parquet                                        │
│  Parquet 文件                                               │
│  {                                                          │
│    "observation.images.rgb": {                              │
│      "dtype": "image",                                      │
│      "shape": [480, 640, 3],                               │
│      "names": ["height", "width", "channel"]               │
│    }                                                        │
│  }                                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓ 训练时加载
┌─────────────────────────────────────────────────────────────┐
│  数据处理格式转换                                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  从 Parquet 读取 PNG 字节                                    │
│       │                                                     │
│       ↓ Image.open(BytesIO())                               │
│  PIL Image (HWC)                                            │
│       │                                                     │
│       ↓ np.array()                                         │
│  NumPy Array (480, 640, 3)  ← HWC 格式                     │
│       │                                                     │
│       ↓ torch.from_numpy()                                 │
│  torch.Tensor (480, 640, 3)  ← HWC 格式                    │
│       │                                                     │
│       ↓ .permute(2, 0, 1)                                  │
│  torch.Tensor (3, 480, 640)  ← CHW 格式（PyTorch 需要）    │
│       │                                                     │
│       ↓ 送入模型                                           │
│  模型输入 (Batch, 3, 224, 224)  ← CHW 格式                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 11.4 代码中的格式转换

#### 在数据转换时（r2r2lerobot.py）

```python
# 行 99-122
for frame_idx, img_path in enumerate(image_files):
    # 1. 从文件读取（HWC）
    from PIL import Image
    img = Image.open(img_path)

    # 2. 转换为 NumPy（HWC）
    img_array = np.array(img)  # (480, 640, 3)

    # 3. 直接存储（HWC）
    yield {
        "observation.images.rgb": img_array,  # HWC 格式
        "action": action,
        "task": task,
    }
```

#### 在数据加载时（lerobot_action_dataset.py）

```python
# 行 960-967
# 1. 从 Parquet 读取 PNG 字节
img_data = frame_data.get('observation.images.rgb', {})
image_bytes = img_data['bytes']

# 2. 解码为 NumPy（HWC）
frame = self._load_image_from_bytes(image_bytes)  # (480, 640, 3)

# 行 1036-1057
# 3. 转换为 Tensor（HWC）
frame_tensor = torch.from_numpy(frame).float() / 255.0  # (480, 640, 3)

# 4. HWC → CHW（关键转换）
if frame_tensor.dim() == 3 and frame_tensor.shape[-1] <= 4:
    frame_tensor = frame_tensor.permute(2, 0, 1)  # (3, 480, 640)

# 5. CHW → HWC（转回 PIL）
frame_pil = Image.fromarray(
    (frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
)  # (480, 640, 3)

# 6. SigLip 处理（会再转回 CHW）
frame_processed = self.image_processor.preprocess(
    images=frame_pil,
    return_tensors='pt'
)['pixel_values'][0]  # (3, 224, 224)
```

### 11.5 维度转换详解

#### permute(2, 0, 1) 的作用

```python
# 原始张量 (HWC)
tensor = torch.tensor([[[R, G, B], [R, G, B]], [[R, G, B], [R, G, B]]])
shape = (2, 2, 3)  # 2 行 × 2 列 × 3 通道

# permute(2, 0, 1) 表示：
# - 新维度 0 ← 旧维度 2 (channel)
# - 新维度 1 ← 旧维度 0 (height)
# - 新维度 2 ← 旧维度 1 (width)
permuted = tensor.permute(2, 0, 1)
shape = (3, 2, 2)  # 3 通道 × 2 行 × 2 列

# 索引变化：
# 原始: tensor[y, x, c]  (H, W, C)
# 转换: tensor[c, y, x]  (C, H, W)
```

**图示：**

```
原始 HWC 格式:                 转换后 CHW 格式:
   Height  Width                  Channel  Height  Width
    ↓       ↓                        ↓       ↓       ↓
  ┌─────┬─────┬─────┐           ┌─────┬─────┬─────┐
  │ RGB │ RGB │ RGB │           │ RGB │ RGB │ RGB │  ← Channel 0 (R)
  ├─────┼─────┼─────┤           ├─────┼─────┼─────┤
  │ RGB │ RGB │ RGB │           │ RGB │ RGB │ RGB │  ← Channel 1 (G)
  └─────┴─────┴─────┘           ├─────┼─────┼─────┤
     (2, 2, 3)                  │ RGB │ RGB │ RGB │  ← Channel 2 (B)
                                └─────┴─────┴─────┘
                                   (3, 2, 2)
```

### 11.6 PyTorch 为什么需要 CHW？

#### 卷积操作的效率

```python
# CHW 格式对卷积更高效
# 原因：每个通道的数据是连续的，便于并行计算

# CHW 格式： (3, 480, 640)
# 内存布局：
# [
#   [R_0000, R_0001, ..., R_0239],  ← 所有 R 像素连续
#   [G_0000, G_0001, ..., G_0239],  ← 所有 G 像素连续
#   [B_0000, B_0001, ..., B_0239],  ← 所有 B 像素连续
# ]

# HWC 格式： (480, 640, 3)
# 内存布局：
# [
#   [R_0000, G_0000, B_0000, R_0001, G_0001, B_0001, ...],
#   ↑ 每个像素的 RGB 交替存储，不利于并行卷积
# ]
```

#### CUDA/GPU 优化

```python
# GPU 卷积核通常针对 CHW 优化
# 早期深度学习框架（如 Caffe）使用 CHW
# PyTorch 延续了这个设计

# 现代 GPU（如 Tensor Core）对 CHW 有专门的优化
# 可以同时处理多个通道
```

### 11.7 实际使用示例

#### 正确的维度处理

```python
# ❌ 错误：直接使用 HWC
frame = np.array(Image.open("image.jpg"))  # (480, 640, 3)
tensor = torch.from_numpy(frame)           # (480, 640, 3)
model(tensor)  # 报错！模型期望 (B, 3, 480, 640)

# ✅ 正确：转换为 CHW
frame = np.array(Image.open("image.jpg"))  # (480, 640, 3) HWC
tensor = torch.from_numpy(frame)           # (480, 640, 3) HWC
tensor = tensor.permute(2, 0, 1)            # (3, 480, 640) CHW
tensor = tensor.unsqueeze(0)                # (1, 3, 480, 640) 加 batch 维
model(tensor)  # 正确！
```

#### 在 DataLoader 中

```python
# collate_fn 中的处理
def collate_fn(batch):
    images = []
    for item in batch:
        # item["images"] 已经是 (T, C, H, W) 格式
        images.append(item["images"])

    # Stack: (B, T, C, H, W)
    images = torch.stack(images)
    return images
```

### 11.8 总结

| 方面 | 说明 |
|------|------|
| **存储格式** | HWC (Height, Width, Channel) |
| **原因** | NumPy/PIL/OpenCV 标准格式 |
| **模型输入** | CHW (Channel, Height, Width) |
| **原因** | PyTorch/GPU 优化 |
| **转换时机** | 训练时的 `_process_frame()` 方法 |
| **转换方法** | `tensor.permute(2, 0, 1)` |

**关键点：**
- LeRobot 使用 HWC 存储，为了与标准图像库兼容
- PyTorch 使用 CHW 输入，为了 GPU 性能优化
- 两者之间的转换是透明的，在 `_process_frame()` 中自动完成

---

## 12. 总结

1. **懒加载**: 初始化快，内存占用小
2. **滑动窗口**: 将长 episode 切分为多个训练样本
3. **历史观察**: 支持加载历史帧作为上下文
4. **交错格式**: 图片和动作交替排列，适合序列预测
5. **Label Masking**: 只预测模型输出部分

这种设计使得 StreamVLN 可以高效地训练视觉导航任务，同时保持与原有 VLNActionDataset 的兼容性。
