# LeRobot Dataset Quick Start

快速开始使用 LeRobot 格式数据集训练 StreamVLN 模型。

## 环境准备

```bash
# 激活 StreamVLN 环境
conda activate streamvln

# 降级 PyArrow 以兼容 LeRobot 数据格式
pip install 'pyarrow==14.0.0' --no-deps

# 安装 PyAV (推荐，支持 AV1 视频解码)
pip install av
```

## 训练命令

```bash
python streamvln/streamvln_train.py \
    --use_lerobot True \
    --lerobot_data_path ./data \
    --lerobot_repo_id lerobot \
    --video_backend auto \
    --model_name_or_path <model_path> \
    --output_dir ./output/lerobot_train
```

## 参数说明

| 参数 | 说明 | 示例 |
|-----|------|-----|
| `--use_lerobot` | 启用 LeRobot 数据集 | True |
| `--lerobot_data_path` | 数据集父目录 | ./data |
| `--lerobot_repo_id` | 数据集仓库名 | lerobot |
| `--video_backend` | 视频解码器 | auto, av, opencv |

## 数据集目录结构

```
./data/lerobot/
├── meta/
│   ├── info.json
│   ├── episodes/chunk-000/file-000.parquet
│   └── tasks.parquet
├── data/
│   └── chunk-000/file-000.parquet
└── videos/
    └── observation.images.rgb/
        └── chunk-000/file-000.mp4
```

## 测试

```bash
python tests/test_lerobot_dataset.py
```

## 常见问题

### PyArrow 版本错误
```
OSError: Repetition level histogram size mismatch
```
**解决**: `pip install 'pyarrow==14.0.0' --no-deps`

### 视频解码失败
```
Failed to decode frame / Unsupported codec
```
**解决**: `pip install av`

### 找不到数据集
```
Dataset root not found
```
**解决**: 检查 `--lerobot_data_path` 和 `--lerobot_repo_id` 拼接后的路径是否存在

## 相关文档

详细文档: [docs/lerobot_integration.md](lerobot_integration.md)
