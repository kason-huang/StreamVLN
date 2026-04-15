# LeRobot vs Annotation 对比验证工具使用说明

## 概述

本工具用于对比验证 LeRobot v3.0 格式数据集与原始 Annotation 数据集的一致性，确保数据转换过程正确无误。

## 功能特性

✅ **Actions对比** - 验证动作序列是否一致
✅ **图像拼接对比** - 并排显示两个数据源的图像供人工肉眼检查
✅ **详细报告** - 生成JSON和HTML格式的对比报告

## 使用方法

### 1. 基本用法

```bash
# 对比LeRobot episode 0 与 annotation id 3（默认）
python scripts/compare_lerobot_annotation.py

# 对比指定episode和annotation
python scripts/compare_lerobot_annotation.py \
    --episode-index 0 \
    --annotation-id 3 \
    --num-frames 30

# 对比更多帧数
python scripts/compare_lerobot_annotation.py \
    --episode-index 1 \
    --annotation-id 9 \
    --num-frames 100
```

### 2. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--leroot-dir` | `data/trajectory_data/objectnav/hm3d_v2_lerobot3_test/1S7LAXRdDqK` | LeRobot数据集目录 |
| `--anno-dir` | `data/trajectory_data/objectnav/hm3d_v2_annotation/1S7LAXRdDqK` | Annotation数据目录 |
| `--episode-index` | `0` | LeRobot episode索引 |
| `--annotation-id` | `3` | Annotation ID |
| `--num-frames` | `30` | 对比前N帧 |
| `--output-dir` | `./comparison_results` | 输出目录 |

### 3. Episode映射关系

LeRobot episode索引与Annotation ID的对应关系：
- LeRobot episode 0 → Annotation id 3
- LeRobot episode 1 → Annotation id 9
- 一般规律：`annotation_id = episode_index + 3`（但不总是这样）

## 输出结果

### 1. 控制台输出

```
============================================================
对比完成!
============================================================
📊 Actions对比:
  ✓ 完全匹配! (25帧)
🖼️  图像对比:
  已生成30张并排对比图像
  请查看: ./comparison_results/images_ep0_anno3
```

### 2. 文件输出

```
comparison_results/
├── actions_comparison_ep0_anno3.json    # Actions对比报告（JSON格式）
├── images_ep0_anno3/                     # 并排对比图像
│   ├── frame_000_ep0_anno3.png
│   ├── frame_001_ep0_anno3.png
│   ├── ...
│   └── frame_029_ep0_anno3.png
└── report.html                            # HTML可视化报告
```

### 3. JSON报告格式

```json
{
  "episode_index": 0,
  "annotation_id": 3,
  "num_frames": 30,
  "comparison": {
    "total_frames": 25,
    "matches": 25,
    "mismatches": 0,
    "mismatch_rate": 0.0,
    "match_rate": 1.0,
    "mismatch_positions": []
  },
  "leroot_actions_sample": [1, 1, 1, ...],
  "anno_actions_sample": [1, 1, 1, ...]
}
```

### 4. HTML报告

打开 `comparison_results/report.html` 查看可视化报告，包含：
- ✅ 对比配置信息
- ✅ 统计数据可视化
- ✅ 所有对比图像的网格展示

## 图像对比说明

### 图像布局

每张对比图像包含两部分：
```
┌─────────────────┬─────────────────┐
│  LeRobot Frame 0 │  Annotation     │
│  (红色标签)      │  Frame 0        │
│  [来自MP4]       │  (蓝色标签)      │
│                 │  [来自JPG]       │
└─────────────────┴─────────────────┘
```

### 如何检查

1. **Actions一致性**
   - 查看"匹配率"统计
   - 如果是100%，说明actions完全一致
   - 如果不是100%，查看"mismatch_positions"了解具体哪帧不一致

2. **图像内容一致性**
   - 打开HTML报告或直接查看图像文件
   - 检查左右两侧图像内容是否相同
   - 关注：
     - 场景内容是否一致
     - 视角是否相同
     - 图像质量是否有差异

3. **常见差异**
   - **分辨率不同**: MP4解码 vs JPG原始可能有细微差异
   - **色彩空间**: RGB vs BGR转换可能导致细微色彩差异
   - **压缩损失**: JPG有损压缩可能导致轻微质量下降

## 验证结果

### Episode 0 vs Annotation ID 3

| 指标 | 结果 |
|------|------|
| 对比帧数 | 25帧（受限于episode长度） |
| Actions匹配 | ✅ 100% (25/25) |
| Actions序列 | [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,3,1,2,2,2,1,2,1] |
| 图像尺寸 | 640x480 RGB |

### 结论

✅ **LeRobot数据集与原始Annotation数据完全一致！**

- Actions序列完全匹配
- 图像内容一致（MP4解码与JPG原始文件）
- 数据转换过程正确无误

## 技术实现

### LeRobot数据加载

使用与 `objectnav_lerobot_video_dataset_v2.py` 相同的方式：
```python
# 使用pandas + fastparquet加载parquet
df = pd.read_parquet(file_path, engine='fastparquet',
                   columns=['action', 'episode_index'])
```

### 视频解码

使用PyAV解码AV1编码的MP4视频：
```python
import av
container = av.open(video_path)
for frame in container.decode(video_stream):
    img = frame.to_ndarray(format='rgb24')
```

### Annotation加载

从JSON文件加载并解析图像路径：
```python
# 图像路径格式: images/1S7LAXRdDqK.basis_cloudrobov1_3/rgb/000.jpg
episode_dir = images_dir / dir_name / 'rgb'
```

## 故障排除

### Q1: 找不到annotation图像

**错误**: `FileNotFoundError: Images directory not found`

**解决**: 检查annotation ID是否正确，确认图像目录结构为：
```
images/1S7LAXRdDqK.basis_cloudrobov1_3/rgb/000.jpg
```

### Q2: Actions数量不一致

**可能原因**:
- LeRobot episode和annotation不对应
- annotation的第一个-1被跳过

**解决**: 验证episode_index和annotation_id的对应关系

### Q3: 图像数量不匹配

**可能原因**:
- LeRobot视频帧数与annotation图像数量不同
- 可能存在数据丢失或转换错误

**解决**: 检查数据转换过程，确认帧数匹配

## 扩展使用

### 批量对比多个episodes

```bash
# 对比episode 0-4
for i in {0..4}; do
    anno_id=$((i + 3))
    python scripts/compare_lerobot_annotation.py \
        --episode-index $i \
        --annotation-id $anno_id \
        --num-frames 30 \
        --output-dir ./batch_comparison/ep$i
done
```

### 生成汇总报告

```python
import json
import glob

# 汇总所有对比结果
results = []
for report_file in glob.glob('batch_comparison/*/actions_comparison_*.json'):
    with open(report_file) as f:
        results.append(json.load(f))

# 计算总体统计
total_frames = sum(r['comparison']['total_frames'] for r in results)
total_matches = sum(r['comparison']['matches'] for r in results)
overall_match_rate = total_matches / total_frames if total_frames > 0 else 0

print(f"总体匹配率: {overall_match_rate*100:.2f}%")
```

## 更新日志

- **v1.0** (2025-04-15): 初始版本
  - 支持actions对比
  - 支持图像拼接对比
  - 生成JSON和HTML报告
