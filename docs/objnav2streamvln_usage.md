# ObjectNav 转 StreamVLN 格式工具

`objnav2streamvln.py` 用于将 ObjectNav 数据集转换为 StreamVLN 训练格式，支持从命令行覆盖配置文件中的参数。

## 功能说明

- 从 ObjectNav 数据集提取 RGB 帧并保存为 StreamVLN 格式
- 支持命令行参数覆盖配置文件中的任何 Habitat 配置项
- 兼容 `--annot-path` 自定义标注文件路径

## 基本用法

### 使用默认配置

```bash
python scripts/objnav_converters/objnav2streamvln.py
```

### 自定义标注文件路径

```bash
python scripts/objnav_converters/objnav2streamvln.py \
  --annot-path /path/to/annotations.json
```

### 覆盖数据集路径

```bash
python scripts/objnav_converters/objnav2streamvln.py \
  habitat.dataset.data_path="data/custom/dataset.json.gz"
```

### 同时覆盖多个参数

```bash
python scripts/objnav_converters/objnav2streamvln.py \
  --annot-path /custom/path/annotations.json \
  habitat.dataset.data_path="data/custom/dataset.json.gz" \
  habitat.simulator.forward_step_size=0.5
```

## 参数说明

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--annot-path` | str | `data/trajectory_data/objectnav/cloudrobo_v1_l3mvn/annotations.json` | 标注文件路径 |
| `opts` | list | `[]` | 配置覆盖参数（REMAINDER 模式） |

### 可覆盖的配置项

所有 Habitat 配置文件中的字段都可以通过 `opts` 覆盖，常用配置项：

| 配置路径 | 说明 | 示例 |
|---------|------|------|
| `habitat.dataset.data_path` | 数据集文件路径 | `"data/dataset.json.gz"` |
| `habitat.simulator.forward_step_size` | 前进步长 | `0.25` |
| `habitat.environment.max_episode_steps` | 最大步数 | `1000` |
| `habitat.simulator.scene` | 场景文件路径 | `"scene.glb"` |

## 配置文件

默认配置文件：`config/objnav_image.yaml`

```yaml
habitat:
  dataset:
    data_path: data/trajectory_data_hm3d_format/objectnav/cloudrobo_v1_l3mvn/train/content/...
  simulator:
    forward_step_size: 0.25
```

## 输出目录结构

```
data/trajectory_data/objectnav/cloudrobo_v1_l3mvn/
├── annotations.json          # 标注文件
└── <video_id>/               # 每个 episode 的视频 ID
    └── rgb/
        ├── 000.jpg
        ├── 001.jpg
        └── ...
```

## 使用示例

### 示例 1: 转换默认数据集

```bash
cd /root/workspace/StreamVLN
python scripts/objnav_converters/objnav2streamvln.py
```

### 示例 2: 使用自定义数据集

```bash
python scripts/objnav_converters/objnav2streamvln.py \
  habitat.dataset.data_path="data/trajectory_data_hm3d_format/objectnav/my_dataset/train/content/my_dataset.json.gz"
```

### 示例 3: 调整模拟器参数

```bash
python scripts/objnav_converters/objnav2streamvln.py \
  habitat.simulator.forward_step_size=0.5 \
  habitat.environment.max_episode_steps=500
```

### 示例 4: 完整自定义

```bash
python scripts/objnav_converters/objnav2streamvln.py \
  --annot-path data/my_annotations.json \
  habitat.dataset.data_path="data/my_dataset.json.gz" \
  habitat.simulator.forward_step_size=0.3 \
  habitat.simulator.sensor_height=0.88
```

## 配置覆盖机制

参考了 `ovon/run.py` 的实现，使用 `argparse.REMAINDER` 模式：

```python
# objnav2streamvln.py (第19-24行)
parser.add_argument(
    'opts',
    default=None,
    nargs=argparse.REMAINDER,
    help='Modify config options from command line'
)

# 配置加载 (第37行)
config = get_config(CONFIG_PATH, args.opts)
```

## 向后兼容性

- ✅ 不传递 `opts` 时使用默认配置
- ✅ 保留原有的 `--annot-path` 参数
- ✅ 可以覆盖任何 Habitat 配置字段

## 相关文件

- **脚本**: `scripts/objnav_converters/objnav2streamvln.py`
- **配置**: `config/objnav_image.yaml`
- **参考实现**: `/root/workspace/ovon/ovon/run.py`

## 故障排查

### 导入错误

```
ModuleNotFoundError: No module named 'streamvln'
```

**解决**: 确保在正确的环境中运行：
```bash
conda activate streamvln
cd /root/workspace/StreamVLN
export PYTHONPATH=/root/workspace/StreamVLN:$PYTHONPATH
```

### 配置文件找不到

```
FileNotFoundError: config/objnav_image.yaml
```

**解决**: 确保在项目根目录运行脚本
```bash
cd /root/workspace/StreamVLN
python scripts/objnav_converters/objnav2streamvln.py
```

## 相关文档

- [ObjectNav 数据结构说明](objnav_data_structure.md)
- [Habitat 传感器扩展指南](habitat-sensor-extension-guide.md)
