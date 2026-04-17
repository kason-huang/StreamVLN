# HM3D Object Nav 转 LeRobot 3.0 流程

## 信息
## 数据挂载

```bash
mkdir -p /mnt/sfs_turbo
apt-get install -y nfs-common
mount -t nfs -o vers=3,timeo=600,noresvport,nolock,tcp 3cb12372-c2d4-4380-aef8-c4eb73baa8d1.sfsturbo.internal:/ /mnt/sfs_turbo
```

---

## 前置信息

### 项目目录
`/root/workspace/StreamVLN`

### 需要处理的场景
`cloudrobo_output/scenes_episodes_only.csv`

### 数据路径

| 类型 | 路径 |
|------|------|
| 原始数据 | `data/trajectory_data/objectnav/hm3d_v2/train/merged` |
| annotation转化 | `data/trajectory_data/objectnav/hm3d_v2_annotation` |
| annotation+image转化 | `data/trajectory_data/objectnav/hm3d_v2_annotation` |
| lerobot3的格式 | `data/trajectory_data/objectnav/hm3d_v2_lerobot3/ceJTwFNjqCt` |

---

## 目录挂载

```bash
# 场景路径
ln -s /mnt/sfs_turbo/data-platform/data/versioned_data/hm3d-2.0/hm3d \
  /root/workspace/StreamVLN/data/scene_datasets/hm3d_v0.2

# 任务集路径
ln -s /mnt/sfs_turbo/data-platform/data/traj_datasets/objectnav/hm3d_v2 \
  /root/workspace/StreamVLN/data/trajectory_data/objectnav/hm3d_v2

# annotation路径
ln -s /mnt/sfs_turbo/dataplatform/00525479/objnav_1000k_cloudrobo/annotation/ \
  /root/workspace/StreamVLN/data/trajectory_data/objectnav/hm3d_v2_annotation

# lerobot3的格式
ln -s /mnt/sfs_turbo/dataplatform/00525479/objnav_1000k_cloudrobo/lerobot3/ \
  /root/workspace/StreamVLN/data/trajectory_data/objectnav/hm3d_v2_lerobot3
```

## Conda 环境安装

主要有两个conda环境：`streamvln-0226` 和 `lerobot-transfer`

### streamvln-0226 环境

按 StreamVLN 的方式安装：https://github.com/kason-huang/StreamVLN
```bash
git checkout feat-lerobot-1000k-objnav
conda create -n streamvln-0226 python=3.9
conda activate streamvln-0226
conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat

git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
pip install -r requirements.txt
pip install protobuf==3.20.1
```

### 安装 habitat_lab

由于代码还包含了GS的东西，所以需要安装GS

GS安装方式：https://github.com/EmbodiedAILab/panoptic_gs/tree/b9f82cfa847f12f978e33a0718d1f2998a9f0132

**需要重新安装torch：**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

export CUDA_HOME=/usr/local/cuda-12.1/
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

### lerobot-transfer 环境

```bash
conda create -n lerobot-transfer python=3.10
conda activate lerobot-transfer
pip install lerobot==0.4.3
```

### 环境配置

```bash
printf '{\n  "file_format_version": "1.0.0",\n  "ICD": { "library_path": "libEGL_nvidia.so.0" }\n}\n' > /usr/share/glvnd/egl_vendor.d/50_nvidia.json
```

## 数据前置处理

**目标：** 把全部的数据按场景分类为不同的HM3D的轨迹数据集，格式是json.gz

**数据的目录：** `obs://data-platformshanghai/data/traj_datasets/objectnav/hm3d_v2/train/content/`

### 执行脚本

```bash
python scripts/objnav_converters/merge_scene_episodes.py --num-workers 48
```

### 执行示例

```bash
(streamvln-0226) root@ecs-v100-0004:~/workspace/StreamVLN# python scripts/objnav_converters/merge_scene_episodes.py --num-workers 48

输出目录: data/trajectory_data/objectnav/hm3d_v2/train/merged
正在收集场景文件...
找到 224 个episode目录
找到 145 个唯一场景

场景统计:
  唯一场景数: 145
  总输入文件数: 31979

示例场景:
  PPTLa8SkUfo: 220 个文件
  U3oQjwTuMX8: 220 个文件
  DBBESbk4Y3k: 220 个文件
  8B43pG641ff: 221 个文件
  YHmAkqgwe2p: 221 个文件

============================================================
使用 20 个worker处理 145 个场景...
============================================================

开始处理 145 个场景 (分 8 批)...

[批次处理日志...]

============================================================
✅ 合并完成！
============================================================
总场景数: 145
成功: 145 | 失败: 0
合并总episodes: 1,076,917
包含多个文件的场景: 145
平均每场景episodes: 7427.0
总耗时: 24271.7秒 (404.5分钟)
平均处理速度: 167.4秒/场景

输出目录: data/trajectory_data/objectnav/hm3d_v2/train/merged
已保存文件数: 145
```

---

## HM3D Object Navigation 转 StreamVLN Annotation 格式

**目标：** 把每个场景的HM3D Object Navigation轨迹数据集转为不带图片的StreamVLN annotation格式的轨迹数据集

### 执行命令

```bash
conda activate streamvln-0226

python scripts/objnav_converters/objnav2r2r.py \
  --input data/trajectory_data/objectnav/hm3d_v2/train/merged/g8Xrdbe9fir.json.gz \
  --output data/trajectory_data/objectnav/hm3d_v2_annotation/g8Xrdbe9fir/annotations.json
```

### 统计结果
```
Episodes converted: 11478
```

### 参数解释

| 参数 | 说明 |
|------|------|
| `--input` | 输入文件：HM3D ObjectNav 的 JSON.gz 压缩文件，包含轨迹数据（episode_id, scene_id, object_category, reference_replay等） |
| `--output` | 输出文件：R2R 标注格式的 JSON 文件，包含转换后的标注数据 |

### 批量处理

```bash
./scripts/objnav_converters/batch_objnav2r2r.sh
```

---

## StreamVLN Annotation 获取图片

**目标：** 把不带图片的StreamVLN annotation格式的轨迹数据集转为带图片的StreamVLN annotation格式的轨迹数据集（需要配合原本的HM3D的任务数据）

**并行策略：** 直接把annotation给一分为4，关键的改动是从原本的episode（habitat里面的全部episode）遍历改为annotation的任务遍历

### 执行命令

```bash
conda activate streamvln-0226
cd /root/workspace/StreamVLN

# 切分annotations为4个部分
python scripts/objnav_converters/split_annotations.py \
  data/trajectory_data/objectnav/hm3d_v2_annotation/g8Xrdbe9fir/annotations.json \
  --num-parts 4

# 并行转换
./scripts/objnav_converters/hm3d_run_parallel_conversion.sh 0 \
  data/trajectory_data/objectnav/hm3d_v2_annotation/g8Xrdbe9fir \
  data/trajectory_data/objectnav/hm3d_v2/train/merged
```

### 脚本参数说明

这个脚本用于并行转换 ObjectNav 数据格式到 StreamVLN 格式，接收 3 个参数：

#### 1. `<gpu_id>` (示例中是 0)
- **用途：** 指定使用哪块 GPU
- **说明：** 虽然启动了 4 个并行任务，但它们都共享同一块 GPU（通过 CUDA_VISIBLE_DEVICES 设置）

#### 2. `<annot_dir>` (示例中是 `data/trajectory_data/objectnav/.../shanghai-zhujiajiao-room2-1-2025-07-15_14-52-28`)
- **用途：** 标注文件所在的目录
- **要求：** 该目录下需要包含以下文件：
  - `annotations.json` - 原始标注文件
  - `annotation_0.json` ~ `annotation_3.json` - 切分后的 4 个标注文件（通过 split_annotations.py 生成）

#### 3. `<data_dir>` (示例中是 `data/trajectory_data_hm3d_format/objectnav/cloudrobo_v1_l3mvn/train/content`)
- **用途：** HM3D 格式数据文件所在的目录
- **说明：** 脚本会根据场景名（从 annot_dir 提取）在该目录下查找对应的 <场景名>.json.gz 数据文件

#### 工作流程
1. 检查 4 个切分的标注文件是否存在
2. 根据 annot_dir 的目录名（场景名）在 data_dir 中查找对应的数据文件
3. 并行启动 4 个任务，分别处理 annotation_0.json ~ annotation_3.json
4. 等待所有任务完成并报告结果

#### 注意事项
运行此脚本前，需要先执行切分标注文件：
```bash
python scripts/objnav_converters/split_annotations.py <标注文件路径> --num-parts 4
```

切分的批量处理：
```bash
./scripts/objnav_converters/batch_split_annotations.sh
```

---

## StreamVLN Annotation（带图片）转为 LeRobot 流程

### 执行命令

```bash
conda activate lerobot-transfer

python scripts/dataset_converters/r2r2lerobot.py \
  --data_dir "xxxx" \
  --dataset_name "xxxx" \
  --output_dir "xxx" \
  --repo_id "xxx" \
  --fps 3
```

### 示例

```bash
python scripts/dataset_converters/r2r2lerobot.py \
  --data_dir "./data/trajectory_data" \
  --dataset_name "objectnav/hm3d_v2_annotation/g8Xrdbe9fir" \
  --output_dir "./data/trajectory_data/objectnav/hm3d_v2_lerobot3/" \
  --repo_id "g8Xrdbe9fir" \
  --fps 3
```

### 参数解释

| 参数 | 说明 |
|------|------|
| `--data_dir` | 所有轨迹数据的根目录 |
| `--dataset_name` | 多级子路径组织方式：`{task}/{model}/{scene_id}` 指向具体的采集场景 |
| `--output_dir` | LeRobot数据集的根目录（区别于默认的lerobot） |
| `--repo_id` | 数据集标识作为输出子目录名，这里直接用场景ID命名 |
| `--fps` | 帧率，用于时间戳生成 |

## 验证数据是否可以收敛

依赖主要参考：docs/objectnav-lerobot-dependencies.md
实现查看：docs/objectnav-lerobot-implementation.md
```bash
scripts/train_objnav_lerobot.sh
```
