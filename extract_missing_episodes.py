#!/usr/bin/env python3
"""
提取EnvDrop数据集中缺失的episode，保持原始annotations.json格式
"""
import json
import os
from pathlib import Path
from typing import Set, List, Dict

def get_existing_video_names(images_dir: Path) -> Set[str]:
    """
    获取实际存在的video目录名称（包含'images/'前缀）

    Args:
        images_dir: images目录路径

    Returns:
        实际存在的video目录名称集合
    """
    existing_videos = set()

    if images_dir.exists():
        for item in images_dir.iterdir():
            if item.is_dir():
                # 保持与annotations.json中相同的格式："images/sceneid_envdrop_episodeid"
                video_name = f"images/{item.name}"
                existing_videos.add(video_name)

    return existing_videos

def extract_missing_episodes(annotations_file: Path, existing_videos: Set[str], output_file: str):
    """
    提取缺失的episode，保持原始格式

    Args:
        annotations_file: 原始annotations.json文件路径
        existing_videos: 实际存在的video目录名称集合
        output_file: 输出文件路径
    """
    missing_episodes = []
    missing_count = 0
    found_count = 0

    with open(annotations_file, 'r') as f:
        data = json.load(f)

    for item in data:
        video_path = item.get('video', '')

        if video_path in existing_videos:
            found_count += 1
        else:
            missing_episodes.append(item)  # 保持原始格式不变
            missing_count += 1

    # 保存缺失的episodes
    with open(output_file, 'w') as f:
        json.dump(missing_episodes, f, indent=2, ensure_ascii=False)

    return found_count, missing_count, len(data)

def create_summary_report(annotations_file: Path, output_file: str,
                         total_count: int, found_count: int, missing_count: int):
    """
    创建摘要报告

    Args:
        annotations_file: 原始annotations.json文件路径
        output_file: 输出文件路径
        total_count: 总数量
        found_count: 找到的数量
        missing_count: 缺失的数量
    """
    summary = {
        "analysis_time": "2025-11-17",
        "data_path": str(annotations_file.parent),
        "source_annotations": "annotations.json",
        "output_missing_episodes": "missing_episodes.json",
        "statistics": {
            "total_episodes": total_count,
            "existing_episodes": found_count,
            "missing_episodes": missing_count,
            "availability_percentage": (found_count / total_count * 100) if total_count > 0 else 0,
            "missing_percentage": (missing_count / total_count * 100) if total_count > 0 else 0
        },
        "regeneration_info": {
            "episodes_to_regenerate": missing_count,
            "output_file_format": "Same as original annotations.json",
            "regeneration_priority": "High - Missing episodes cannot be used for training",
            "recommended_approach": [
                "1. Generate missing video directories using Habitat-sim",
                "2. Ensure proper frame extraction and annotation consistency",
                "3. Validate generated data matches original annotations",
                "4. Update the main annotations.json after successful generation"
            ]
        }
    }

    summary_file = str(Path(output_file).parent / (Path(output_file).stem + '_summary.json'))
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary_file

def analyze_existing_scenes(images_dir: Path) -> Dict[str, int]:
    """
    分析现有scene的episode分布

    Args:
        images_dir: images目录路径

    Returns:
        每个scene的episode数量
    """
    scene_counts = {}

    if images_dir.exists():
        for item in images_dir.iterdir():
            if item.is_dir():
                episode_name = item.name
                if '_envdrop_' in episode_name:
                    scene_id = episode_name.split('_envdrop_')[0]
                    scene_counts[scene_id] = scene_counts.get(scene_id, 0) + 1

    return scene_counts

def main():
    """主函数"""
    # 设置路径
    base_dir = Path("/root/workspace/lab/StreamVLN/data/trajectory_data/EnvDrop")
    images_dir = base_dir / "images"
    annotations_file = base_dir / "annotations.json"
    output_file = base_dir / "missing_episodes.json"

    print("=== 提取缺失的EnvDrop Episodes ===")
    print(f"Images目录: {images_dir}")
    print(f"原始Annotations文件: {annotations_file}")
    print(f"输出文件: {output_file}")
    print()

    # 检查文件是否存在
    if not images_dir.exists():
        print(f"错误: images目录不存在: {images_dir}")
        return

    if not annotations_file.exists():
        print(f"错误: annotations.json文件不存在: {annotations_file}")
        return

    # 获取实际存在的video目录
    print("正在获取实际存在的video目录...")
    existing_videos = get_existing_video_names(images_dir)
    print(f"实际存在的video目录数量: {len(existing_videos)}")

    # 分析现有scene分布
    print("正在分析现有scene分布...")
    scene_counts = analyze_existing_scenes(images_dir)
    print(f"现有scene数量: {len(scene_counts)}")
    for scene_id, count in sorted(scene_counts.items(), key=lambda x: x[0])[:10]:
        print(f"  {scene_id}: {count} episodes")
    if len(scene_counts) > 10:
        print(f"  ... 还有 {len(scene_counts) - 10} 个scenes")

    # 提取缺失的episodes
    print("正在提取缺失的episodes...")
    found_count, missing_count, total_count = extract_missing_episodes(
        annotations_file, existing_videos, str(output_file)
    )

    # 创建摘要报告
    print("正在创建摘要报告...")
    summary_file = create_summary_report(
        annotations_file, output_file, total_count, found_count, missing_count
    )

    # 打印结果摘要
    print("\n=== 结果摘要 ===")
    print(f"总episodes数量: {total_count:,}")
    print(f"已存在的episodes: {found_count:,}")
    print(f"缺失的episodes: {missing_count:,}")
    print(f"数据可用率: {(found_count/total_count)*100:.2f}%")
    print(f"数据缺失率: {(missing_count/total_count)*100:.2f}%")

    print(f"\n=== 输出文件 ===")
    print(f"缺失episodes列表: {output_file}")
    print(f"摘要报告: {summary_file}")

    # 磁盘空间估算
    estimated_size_gb = missing_count * 0.05  # 假设每个episode约50MB
    print(f"\n=== 重新生成估算 ===")
    print(f"需要重新生成的episodes: {missing_count:,}")
    print(f"预估磁盘空间需求: {estimated_size_gb:.1f} GB")
    print(f"建议批量大小: 100-200 episodes/batch")

    print("\n提取完成!")
    print(f"使用 {output_file} 文件进行后续的episode重新生成。")

if __name__ == "__main__":
    main()