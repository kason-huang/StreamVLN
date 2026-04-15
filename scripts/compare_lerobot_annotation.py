#!/usr/bin/env python3
"""
对比LeRobot格式数据与原始annotation数据的前30帧

验证：
1. Actions是否一致
2. 图像路径对应关系是否正确
3. 图像内容对比（拼接显示）
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

sys.path.insert(0, '/root/workspace/StreamVLN_up')
sys.path.insert(0, '/root/workspace/lerobot-0.4.3/src')

# LeRobot加载方式
try:
    import pandas as pd
    import av
    HAS_LEROBOT_DEPS = True
except ImportError as e:
    print(f"Error: Missing dependencies: {e}")
    print("Please install: pip install pandas av")
    sys.exit(1)


def load_annotation_data(anno_dir: str, annotation_id: int) -> Dict:
    """
    加载annotation数据

    Args:
        anno_dir: annotation目录路径
        annotation_id: annotation的ID（如3, 9等）

    Returns:
        annotation字典，包含instructions, actions列表
    """
    anno_file = os.path.join(anno_dir, 'annotations.json')

    with open(anno_file, 'r') as f:
        all_annotations = json.load(f)

    # 查找匹配的annotation
    for anno in all_annotations:
        if anno['id'] == annotation_id:
            return anno

    raise ValueError(f"Annotation id={annotation_id} not found in {anno_file}")


def load_lerobot_episode_actions(leroot_dir: str, episode_index: int) -> np.ndarray:
    """
    使用与objectnav_lerobot_video_dataset_v2.py相同的方式加载episode的actions

    参考v2文件的_load_single_lerobot_dataset方法
    """
    data_dir = Path(leroot_dir) / 'data'

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # 加载actions，使用pandas + fastparquet
    actions_cache = {}

    for chunk_dir in sorted(data_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue

        for file_path in sorted(chunk_dir.glob('*.parquet')):
            try:
                df = pd.read_parquet(file_path, engine='fastparquet', columns=['action', 'episode_index'])

                # 按episode分组
                for ep_idx in df['episode_index'].unique():
                    ep_idx = int(ep_idx)
                    if ep_idx not in actions_cache:
                        actions_cache[ep_idx] = []

                    # 获取该episode的所有actions
                    ep_actions = df[df['episode_index'] == ep_idx]['action'].values
                    actions_cache[ep_idx].extend(ep_actions.tolist())

            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

    if episode_index not in actions_cache:
        raise ValueError(f"Episode {episode_index} not found in dataset")

    return np.array(actions_cache[episode_index])


def load_lerobot_frame_images(leroot_dir: str, episode_index: int, num_frames: int = 30) -> List[np.ndarray]:
    """
    从LeRobot数据集加载前N帧图像

    Returns:
        图像列表（RGB numpy数组）
    """
    # 获取视频路径
    info_path = Path(leroot_dir) / 'meta' / 'info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    video_key = 'observation.images.rgb'

    # 使用v2的方式获取视频文件信息
    episodes_meta_dir = Path(leroot_dir) / 'meta' / 'episodes'

    # 尝试从episodes parquet读取视频文件信息
    chunk_idx, file_idx = 0, 0  # 默认值

    for chunk_dir in sorted(episodes_meta_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue

        for file_path in sorted(chunk_dir.glob('*.parquet')):
            try:
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(str(file_path))

                # 检查是否有video相关列
                schema = parquet_file.schema_arrow
                chunk_col = schema.get_field_index(f'videos/{video_key}/chunk_index')
                file_col = schema.get_field_index(f'videos/{video_key}/file_index')

                if chunk_col >= 0 and file_col >= 0:
                    # 读取episode_index和video文件信息
                    table = parquet_file.read(columns=[
                        'episode_index',
                        f'videos/{video_key}/chunk_index',
                        f'videos/{video_key}/file_index'
                    ])

                    ep_indices = table['episode_index'].to_pylist()
                    chunk_indices = table[f'videos/{video_key}/chunk_index'].to_pylist()
                    file_indices = table[f'videos/{video_key}/file_index'].to_pylist()

                    # 查找匹配的episode
                    for i, ep in enumerate(ep_indices):
                        if int(ep) == episode_index:
                            chunk_idx = int(chunk_indices[i])
                            file_idx = int(file_indices[i])
                            break

                    break  # 找到就退出
            except Exception as e:
                continue

    # 构建视频路径
    video_path = Path(leroot_dir) / 'videos' / video_key / f'chunk-{chunk_idx:03d}' / f'file-{file_idx:03d}.mp4'

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 使用PyAV解码前N帧
    print(f"Loading {num_frames} frames from {video_path}")

    container = av.open(str(video_path))
    video_stream = container.streams.video[0]

    frames = []
    current_frame = 0

    for frame in container.decode(video_stream):
        if current_frame >= num_frames:
            break

        # 转换为RGB numpy数组
        img = frame.to_ndarray(format='rgb24')
        frames.append(img)
        current_frame += 1

    container.close()

    return frames


def load_annotation_images(anno_dir: str, annotation_id: int, num_frames: int = 30) -> List[np.ndarray]:
    """
    从annotation目录加载图像

    Annotation图像结构：
    images/1S7LAXRdDqK.basis_cloudrobov1_1003/rgb/000.jpg
    images/1S7LAXRdDqK.basis_cloudrobov1_1003/rgb/001.jpg
    ...

    Returns:
        图像列表（RGB numpy数组）
    """
    # 找到对应的annotation
    anno = load_annotation_data(anno_dir, annotation_id)

    # 从video路径获取图像目录名
    video_filename = anno['video']  # 如 "images/1S7LAXRdDqK.basis_cloudrobov1_3"

    # 提取目录名（如 "1S7LAXRdDqK.basis_cloudrobov1_3"）
    dir_name = video_filename.split('/')[-1]

    # 构建图像目录路径
    episode_images_dir = Path(anno_dir) / 'images' / dir_name / 'rgb'

    if not episode_images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {episode_images_dir}")

    # 查找所有图像文件（按编号排序）
    image_files = sorted(episode_images_dir.glob('*.jpg'))

    if not image_files:
        raise FileNotFoundError(f"No image files found in {episode_images_dir}")

    print(f"Found {len(image_files)} image files for annotation {annotation_id}")
    print(f"  Directory: {episode_images_dir}")
    print(f"  First few files: {[f.name for f in image_files[:3]]}")

    # 加载前N帧图像
    from PIL import Image

    frames = []
    for i in range(min(num_frames, len(image_files))):
        img_path = image_files[i]

        # 读取图像
        img = Image.open(img_path)

        # 转换为RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        frames.append(np.array(img))

    return frames


def compare_actions(leroot_actions: np.ndarray, anno_actions: List[int], num_frames: int = 30) -> Dict:
    """
    对比actions

    Returns:
        对比结果字典
    """
    # annotation的actions第一个是-1，需要跳过
    anno_actions_valid = anno_actions[1:]  # 跳过第一个-1

    # 截取前num_frames
    leroot_actions_subset = leroot_actions[:num_frames]
    anno_actions_subset = anno_actions_valid[:num_frames]

    # 统计
    total_frames = min(len(leroot_actions_subset), len(anno_actions_subset))

    mismatches = 0
    mismatch_positions = []

    for i in range(total_frames):
        if leroot_actions_subset[i] != anno_actions_subset[i]:
            mismatches += 1
            mismatch_positions.append(i)

    match_rate = (total_frames - mismatches) / total_frames if total_frames > 0 else 0

    return {
        'total_frames': total_frames,
        'matches': total_frames - mismatches,
        'mismatches': mismatches,
        'mismatch_rate': mismatches / total_frames if total_frames > 0 else 0,
        'match_rate': match_rate,
        'mismatch_positions': mismatch_positions
    }


def create_side_by_side_comparison(leroot_images: List[np.ndarray],
                                   anno_images: List[np.ndarray],
                                   output_dir: str,
                                   episode_index: int,
                                   annotation_id: int):
    """
    创建并排对比图像

    Args:
        leroot_images: LeRobot数据集的图像列表
        anno_images: Annotation数据集的图像列表
        output_dir: 输出目录
        episode_index: LeRobot episode索引
        annotation_id: Annotation ID
    """
    from PIL import Image
    import os

    os.makedirs(output_dir, exist_ok=True)

    num_frames = min(len(leroot_images), len(anno_images))

    print(f"\n创建对比图像: {num_frames}帧")
    print(f"输出目录: {output_dir}")

    for i in range(num_frames):
        img1 = Image.fromarray(leroot_images[i])
        img2 = Image.fromarray(anno_images[i])

        # 获取图像尺寸
        w1, h1 = img1.size
        w2, h2 = img2.size

        # 创建并排图像
        new_width = w1 + w2
        new_height = max(h1, h2)

        side_by_side = Image.new('RGB', (new_width, new_height))

        # 粘贴两张图像
        side_by_side.paste(img1, (0, 0))
        side_by_side.paste(img2, (w1, 0))

        # 添加文字标签
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(side_by_side)

        # 尝试使用默认字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        # 绘制标签
        draw.text((10, 10), f"LeRobot - Frame {i}", fill='red', font=font)
        draw.text((w1 + 10, 10), f"Annotation - Frame {i}", fill='blue', font=font)

        # 保存
        output_path = os.path.join(output_dir, f"frame_{i:03d}_ep{episode_index}_anno{annotation_id}.png")
        side_by_side.save(output_path)

        if i < 5:  # 只打印前5个
            print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='对比LeRobot和Annotation数据')
    parser.add_argument('--leroot-dir', type=str,
                       default='data/trajectory_data/objectnav/hm3d_v2_lerobot3_test/1S7LAXRdDqK',
                       help='LeRobot数据集目录')
    parser.add_argument('--anno-dir', type=str,
                       default='data/trajectory_data/objectnav/hm3d_v2_annotation/1S7LAXRdDqK',
                       help='Annotation数据目录')
    parser.add_argument('--episode-index', type=int, default=0,
                       help='LeRobot episode索引（默认0）')
    parser.add_argument('--annotation-id', type=int, default=3,
                       help='Annotation ID（默认3，对应episode 0）')
    parser.add_argument('--num-frames', type=int, default=30,
                       help='对比前N帧（默认30）')
    parser.add_argument('--output-dir', type=str,
                       default='./comparison_results',
                       help='对比结果输出目录')

    args = parser.parse_args()

    print("="*60)
    print("LeRobot vs Annotation 数据对比工具")
    print("="*60)
    print(f"\n配置:")
    print(f"  LeRobot目录: {args.leroot_dir}")
    print(f"  Annotation目录: {args.anno_dir}")
    print(f"  LeRobot Episode Index: {args.episode_index}")
    print(f"  Annotation ID: {args.annotation_id}")
    print(f"  对比帧数: {args.num_frames}")
    print(f"  输出目录: {args.output_dir}")

    # ============================================================
    # 1. 加载Actions
    # ============================================================
    print("\n" + "="*60)
    print("步骤 1: 加载Actions")
    print("="*60)

    try:
        leroot_actions = load_lerobot_episode_actions(args.leroot_dir, args.episode_index)
        print(f"✓ LeRobot actions加载成功: {len(leroot_actions)} 个actions")
        print(f"  前10个: {leroot_actions[:10].tolist()}")
    except Exception as e:
        print(f"✗ LeRobot actions加载失败: {e}")
        return 1

    try:
        anno_data = load_annotation_data(args.anno_dir, args.annotation_id)
        anno_actions = anno_data['actions']
        print(f"✓ Annotation actions加载成功: {len(anno_actions)} 个actions")
        print(f"  前11个: {anno_actions[:11]}")
    except Exception as e:
        print(f"✗ Annotation actions加载失败: {e}")
        return 1

    # ============================================================
    # 2. 对比Actions
    # ============================================================
    print("\n" + "="*60)
    print("步骤 2: 对比Actions")
    print("="*60)

    comparison = compare_actions(leroot_actions, anno_actions, args.num_frames)

    print(f"\n对比结果 (前{args.num_frames}帧):")
    print(f"  总帧数: {comparison['total_frames']}")
    print(f"  匹配: {comparison['matches']}")
    print(f"  不匹配: {comparison['mismatches']}")
    print(f"  匹配率: {comparison['match_rate']*100:.2f}%")
    print(f"  不匹配率: {comparison['mismatch_rate']*100:.2f}%")

    if comparison['mismatches'] > 0:
        print(f"\n不匹配位置: {comparison['mismatch_positions'][:10]}")
        if len(comparison['mismatch_positions']) > 10:
            print(f"  ... (还有{len(comparison['mismatch_positions'])-10}个)")

    # 保存对比报告
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, f"actions_comparison_ep{args.episode_index}_anno{args.annotation_id}.json")

    with open(report_path, 'w') as f:
        json.dump({
            'episode_index': args.episode_index,
            'annotation_id': args.annotation_id,
            'num_frames': args.num_frames,
            'comparison': comparison,
            'leroot_actions_sample': leroot_actions[:args.num_frames].tolist(),
            'anno_actions_sample': anno_actions[1:args.num_frames+1]
        }, f, indent=2)

    print(f"\n对比报告已保存: {report_path}")

    # ============================================================
    # 3. 加载图像
    # ============================================================
    print("\n" + "="*60)
    print("步骤 3: 加载图像")
    print("="*60)

    try:
        leroot_images = load_lerobot_frame_images(args.leroot_dir, args.episode_index, args.num_frames)
        print(f"✓ LeRobot图像加载成功: {len(leroot_images)} 帧")
        print(f"  图像尺寸: {leroot_images[0].shape}")
    except Exception as e:
        print(f"✗ LeRobot图像加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    try:
        anno_images = load_annotation_images(args.anno_dir, args.annotation_id, args.num_frames)
        print(f"✓ Annotation图像加载成功: {len(anno_images)} 帧")
        print(f"  图像尺寸: {anno_images[0].shape}")
    except Exception as e:
        print(f"✗ Annotation图像加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ============================================================
    # 4. 创建并排对比图像
    # ============================================================
    print("\n" + "="*60)
    print("步骤 4: 创建对比图像")
    print("="*60)

    image_output_dir = os.path.join(args.output_dir, f"images_ep{args.episode_index}_anno{args.annotation_id}")
    create_side_by_side_comparison(leroot_images, anno_images, image_output_dir,
                                  args.episode_index, args.annotation_id)

    print(f"\n✓ 对比图像已保存到: {image_output_dir}")

    # ============================================================
    # 5. 总结
    # ============================================================
    print("\n" + "="*60)
    print("对比完成!")
    print("="*60)
    print(f"\n📊 Actions对比:")
    if comparison['match_rate'] == 1.0:
        print(f"  ✓ 完全匹配! ({comparison['total_frames']}帧)")
    else:
        print(f"  ⚠ 存在{comparison['mismatches']}处不一致")
        print(f"  匹配率: {comparison['match_rate']*100:.1f}%")

    print(f"\n🖼️  图像对比:")
    print(f"  已生成{len(leroot_images)}张并排对比图像")
    print(f"  请查看: {image_output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
