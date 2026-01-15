# LeRobot数据集格式兼容方案

## 概述

修改 `vln_action_dataset.py` 以支持 LeRobot 格式的数据集，同时保持对原始格式的兼容性。支持通过参数配置切换数据集格式，并支持同时加载多个 LeRobot 数据集。

## 目标

1. 支持原始格式（annotations.json + 视频文件夹）
2. 支持 LeRobot 格式（NavDataset + parquet + 图像序列）
3. 支持同时加载多个数据集（逗号分隔配置）
4. 向后兼容，不破坏现有功能

---

## 需要修改的文件

| 文件 | 修改类型 |
|------|---------|
| `streamvln/dataset/vln_action_dataset.py` | 修改 |
| `streamvln/args.py` (DataArguments) | 修改 |

---

## 详细修改内容

### 1. 修改 DataArguments

**文件**: `streamvln/args.py`

在 `DataArguments` 类中添加以下参数：

```python
@dataclass
class DataArguments:
    # ... 现有参数 ...

    # 数据集格式选择
    dataset_format: str = field(
        default="original",
        metadata={"help": "数据集格式: 'original' 或 'lerobot'"}
    )

    # LeRobot 数据集配置（支持多个，逗号分隔）
    lerobot_roots: str = field(
        default=None,
        metadata={"help": "LeRobot 数据集根目录列表，逗号分隔。例: /path/r2r,/path/rxr"}
    )
```

---

### 2. 修改 vln_action_dataset.py

**文件**: `streamvln/dataset/vln_action_dataset.py`

#### 2.1 在文件开头添加导入

```python
# 在现有导入后添加
from typing import List, Dict, Any
from pathlib import Path
import copy
```

#### 2.2 添加新的 LeRobotVLNDataLoader 类

在 `VLNActionDataset` 类定义之前添加：

```python
class LeRobotVLNDataLoader:
    """从 LeRobot 格式数据集加载 VLN 数据，支持多数据集合并。"""

    def __init__(self, roots: str):
        """
        Args:
            roots: 逗号分隔的 LeRobot 数据集根目录列表
        """
        from vlnce2lerobot_v2 import NavDataset

        # 解析多个根目录
        root_dirs = [r.strip() for r in roots.split(',') if r.strip()]

        self.datasets: List[NavDataset] = []
        self.dataset_configs: List[Dict] = []

        for root_dir in root_dirs:
            root_path = Path(root_dir)

            # 检查目录是否存在
            if not root_path.exists():
                raise ValueError(f"LeRobot dataset root not found: {root_dir}")

            # 尝试从 info.json 读取 repo_id
            info_path = root_path / "meta" / "info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    repo_id = info.get("repo_id", root_path.name)
            else:
                repo_id = root_path.name

            # 加载数据集
            dataset = NavDataset(repo_id=repo_id, root=root_path)
            self.datasets.append(dataset)
            self.dataset_configs.append({
                'repo_id': repo_id,
                'root': root_path,
                'dataset': dataset
            })

        # 建立全局 episode 索引映射
        self._build_episode_index_mapping()

        print(f"[LeRobotVLNDataLoader] Loaded {len(self.datasets)} dataset(s):")
        for config in self.dataset_configs:
            print(f"  - {config['repo_id']}: {len(config['dataset'].episodes)} episodes")
        print(f"[LeRobotVLNDataLoader] Total episodes: {self.total_episodes}")

    def _build_episode_index_mapping(self):
        """建立全局 episode 索引到具体 dataset 的映射"""
        self.episode_mapping = {}  # {global_ep_idx: (dataset_idx, local_ep_idx)}
        global_idx = 0

        for ds_idx, config in enumerate(self.dataset_configs):
            dataset = config['dataset']
            for local_ep_idx in range(len(dataset.episodes)):
                self.episode_mapping[global_idx] = (ds_idx, local_ep_idx)
                global_idx += 1

        self.total_episodes = global_idx

    def get_all_episodes(self) -> List[Dict[str, Any]]:
        """
        获取所有 episode 数据，统一转换为原始格式。

        Returns:
            List of episode dictionaries in original format:
            {
                'id': str,
                'video': str,
                'actions': List[int],
                'instructions': List[str],
                'source_dataset': str  # 额外字段：标记来源数据集
            }
        """
        episodes = []
        for global_ep_idx in range(self.total_episodes):
            ep_data = self.get_episode(global_ep_idx)
            episodes.append(ep_data)
        return episodes

    def get_episode(self, global_ep_idx: int) -> Dict[str, Any]:
        """
        获取指定 episode 的数据（统一转换为原始格式）。

        Args:
            global_ep_idx: 全局 episode 索引

        Returns:
            Episode dictionary in original format
        """
        if global_ep_idx not in self.episode_mapping:
            raise IndexError(f"Episode index {global_ep_idx} out of range")

        ds_idx, local_ep_idx = self.episode_mapping[global_ep_idx]
        config = self.dataset_configs[ds_idx]
        dataset = config['dataset']

        # 获取该 episode 的所有帧
        frames = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample["episode_index"].item() == local_ep_idx:
                frames.append(sample)

        if not frames:
            raise ValueError(f"No frames found for episode {global_ep_idx}")

        # 解析 instruction
        task_str = frames[0]["task"]
        if isinstance(task_str, str):
            task_dict = json.loads(task_str)
            instruction = task_dict.get("instruction", task_str)
        else:
            instruction = str(task_str)

        # 提取 actions
        actions = [int(f["action"].item()) for f in frames]

        # 获取 video_path（用于加载图像）
        video_path = str(
            config['root'] /
            dataset.meta.get_video_file_path(local_ep_idx, "observation.images.rgb")
        )

        return {
            'id': f"{config['repo_id']}_{local_ep_idx}",
            'video': video_path,
            'actions': actions,
            'instructions': [instruction],
            'source_dataset': config['repo_id']  # 额外字段
        }

    def get_video_frames_list(self, global_ep_idx: int) -> List[str]:
        """
        获取指定 episode 的所有图像文件路径列表（排序后）。

        Args:
            global_ep_idx: 全局 episode 索引

        Returns:
            List of image file paths
        """
        if global_ep_idx not in self.episode_mapping:
            raise IndexError(f"Episode index {global_ep_idx} out of range")

        ds_idx, local_ep_idx = self.episode_mapping[global_ep_idx]
        config = self.dataset_configs[ds_idx]
        dataset = config['dataset']

        video_dir = Path(
            config['root'] /
            dataset.meta.get_video_file_path(local_ep_idx, "observation.images.rgb")
        )

        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")

        # 获取所有 jpg 文件并按文件名（数字）排序
        img_files = sorted(
            video_dir.glob("*.jpg"),
            key=lambda x: int(x.stem) if x.stem.isdigit() else 0
        )

        return [str(f) for f in img_files]
```

#### 2.3 修改 VLNActionDataset.__init__

在 `VLNActionDataset` 类中修改 `__init__` 方法，主要修改数据加载部分：

```python
class VLNActionDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_args,
        task_id
    ):
        super(VLNActionDataset, self).__init__()

        self.task_id = task_id
        self.image_size = data_args.image_size
        self.tokenizer = tokenizer
        self.transforms = data_args.transform_train
        self.image_processor = SigLipImageProcessor()

        self.num_frames = data_args.num_frames
        self.num_history = data_args.num_history
        self.num_future_steps = data_args.num_future_steps
        self.remove_init_turns = data_args.remove_init_turns

        # ========== 新增：数据集格式检测 ==========
        self.dataset_format = getattr(data_args, 'dataset_format', 'original')

        # ========== 修改：根据格式加载数据 ==========
        if self.dataset_format == 'lerobot':
            # LeRobot 格式
            lerobot_roots = getattr(data_args, 'lerobot_roots', None)
            if not lerobot_roots:
                raise ValueError(
                    "lerobot_roots must be provided when dataset_format='lerobot'. "
                    "Example: --lerobot_roots /path/r2r,/path/rxr"
                )
            self.lerobot_loader = LeRobotVLNDataLoader(roots=lerobot_roots)
            self.nav_data = self.lerobot_loader.get_all_episodes()
            self.video_folder = []  # LeRobot 格式不需要 video_folder
        else:
            # 原始格式
            self.lerobot_loader = None
            self.video_folder = data_args.video_folder.split(',')
            self.nav_data = self.load_vln_data(data_args)

        # ========== 剩余初始化逻辑保持不变 ==========
        self.data_list = []
        for ep_id, item in enumerate(self.nav_data):
            instructions = item['instructions']
            actions = item['actions']
            actions_len = len(actions)
            if actions_len < 4:
                continue

            if not isinstance(instructions, list):
                instructions = [instructions]

            for ins_id in range(len(instructions)):
                valid_idx = 0
                if self.remove_init_turns:
                    valid_idx = self.clean_initial_rotations(instructions[ins_id], actions)
                    if valid_idx != 0:
                        invalid_len += 1

                if actions_len - valid_idx < 4:
                    continue

                num_rounds = (actions_len - valid_idx) // self.num_frames
                for n in range(num_rounds + 1):
                    if n * self.num_frames == actions_len - valid_idx:
                        continue
                    self.data_list.append((ep_id, ins_id, n * self.num_frames, valid_idx))

        # ... 其余代码保持不变 ...
        self.idx2actions = {
            '0': 'STOP',
            '1': "↑",
            '2': "←",
            '3': "→",
        }

        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is '
        ]
        self.act_conjunctions = [
            'and then ',
            'after that ',
            'next ',
            'the next action is ',
            'followed by ',
            'leading to ',
            'continuing ',
            'subsequently ',
            'proceeding to '
        ]

        prompt = f"You are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        self.conversations = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
```

#### 2.4 修改 VLNActionDataset.__getitem__

修改 `__getitem__` 方法中的图像加载逻辑：

```python
def __getitem__(self, i):
    ep_id, ins_id, start_idx, valid_idx = self.data_list[i]
    data = self.nav_data[ep_id]

    # ========== 修改：支持 LeRobot 格式的图像加载 ==========
    if self.dataset_format == 'lerobot' and self.lerobot_loader is not None:
        # LeRobot 格式：直接获取排序后的图像文件列表
        video_frames = self.lerobot_loader.get_video_frames_list(ep_id)
        video_path = data['video']  # 图像目录路径
    else:
        # 原始格式：从 video_path/rgb 目录读取
        video_path = data['video']
        video_frames = sorted(os.listdir(os.path.join(video_path, 'rgb')))

    instructions = data.get("instructions", None)
    if not isinstance(instructions, list):
        instructions = [instructions]

    actions = data['actions'][1+valid_idx:] + [0]
    actions_len = len(actions)
    time_ids = np.arange(start_idx, min(start_idx + self.num_frames, actions_len))
    assert len(time_ids) > 0
    actions = np.array(actions)[time_ids]

    start_idx, end_idx, interval = time_ids[0]+valid_idx, time_ids[-1]+1+valid_idx, self.num_future_steps
    sample_step_ids = np.arange(start_idx, end_idx, interval, dtype=np.int32)

    # ========== 修改：根据格式构造完整图像路径 ==========
    if self.dataset_format == 'lerobot' and self.lerobot_loader is not None:
        # LeRobot: video_frames 已经是完整路径
        sample_frames = [video_frames[i] for i in sample_step_ids]
    else:
        # 原始格式: 需要拼接 video_path/rgb/
        sample_frames = [os.path.join(video_path, 'rgb', video_frames[i]) for i in sample_step_ids]

    if time_ids[0] != 0:
        history_step_ids = np.arange(0+valid_idx, time_ids[0]+valid_idx, max(time_ids[0] // self.num_history, 1))
        if self.dataset_format == 'lerobot' and self.lerobot_loader is not None:
            history_frames = [video_frames[i] for i in history_step_ids]
        else:
            history_frames = [os.path.join(video_path, 'rgb', video_frames[i]) for i in history_step_ids]
    else:
        history_frames = []

    images = []
    for image_file in history_frames + sample_frames:
        image = Image.open(image_file).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)

        image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
        images.append(image)

    images = torch.stack(images)

    sources = copy.deepcopy(self.conversations)

    if start_idx != 0:
        sources[0]["value"] += f' These are your historical observations: {DEFAULT_MEMORY_TOKEN}.'

    sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instructions[ins_id])
    interleave_sources = self.prepare_conversation(sources, list(actions))

    data_dict = preprocess([interleave_sources], self.tokenizer, True)

    return data_dict["input_ids"][0], \
        data_dict["labels"][0], \
        images, \
        torch.tensor(time_ids), \
        self.task
```

---

## 配置使用方法

### 原始格式（默认）

```bash
python train.py \
    --video_folder /data/r2r,/data/rxr \
    --dataset_format original
```

### LeRobot 格式 - 单个数据集

```bash
python train.py \
    --dataset_format lerobot \
    --lerobot_roots /shared/smartbot_new/liuyu/vln_ce_lerobot/r2r
```

### LeRobot 格式 - 多个数据集

```bash
python train.py \
    --dataset_format lerobot \
    --lerobot_roots /shared/smartbot_new/liuyu/vln_ce_lerobot/r2r,/shared/smartbot_new/liuyu/vln_ce_lerobot/rxr,/shared/smartbot_new/liuyu/vln_ce_lerobot/envdrop
```

---

## 数据流程对比

### 原始格式

```
video_folder/
  ├── R2R/
  │   ├── annotations.json
  │   └── images/
  │       └── 17DRP5sb8fy/
  │           └── rgb/
  │               ├── 0.jpg, 1.jpg, ...
  └── RxR/
      ├── annotations.json
      └── images/
          └── ...
```

### LeRobot 格式

```
lerobot_root/
  ├── meta/
  │   ├── info.json
  │   └── episodes_stats.json
  ├── data/
  │   └── chunk-000/
  │       ├── episode_000000.parquet
  │       └── episode_000001.parquet
  └── videos/
      └── chunk-000/
          └── observation.images.rgb/
              ├── episode_000000/
              │   ├── 0.jpg, 1.jpg, ...
              └── episode_000001/
                  └── ...
```

---

## 修改要点总结

| 组件 | 修改内容 |
|-----|---------|
| **DataArguments** | 添加 `dataset_format` 和 `lerobot_roots` 参数 |
| **新增类** | `LeRobotVLNDataLoader` - 封装多数据集加载逻辑 |
| **VLNActionDataset.__init__** | 根据格式选择加载方式 |
| **VLNActionDataset.__getitem__** | 根据格式构造图像路径 |
| **向后兼容** | 默认使用原始格式，无需修改现有配置 |

---

## 依赖关系

需要确保 `vlnce2lerobot_v2.py` 的 `NavDataset` 类可以被导入：

```python
# 在 vln_action_dataset.py 中
from vlnce2lerobot_v2 import NavDataset
```

可能需要调整导入路径，取决于 `vlnce2lerobot_v2.py` 的实际位置。

---

## 测试建议

1. **单元测试**：分别测试原始格式和 LeRobot 格式的数据加载
2. **集成测试**：测试多数据集合并场景
3. **性能测试**：对比两种格式的加载速度和内存占用
