#!/usr/bin/env python3
"""
Verify a single merged scene against its source files.

This script checks:
1. Episode count consistency
2. Sample episode content verification
"""

import gzip
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any


def load_json_gz(file_path: str) -> Dict[str, Any]:
    """Load a gzipped JSON file."""
    with gzip.open(file_path, 'rt') as f:
        return json.load(f)


def collect_source_files(base_dir: str, scene_name: str) -> List[str]:
    """
    Collect all source files for a given scene.

    Args:
        base_dir: Base directory containing episode_num-xxx subdirectories
        scene_name: Name of the scene to collect files for

    Returns:
        List of file paths sorted in order
    """
    base_path = Path(base_dir)
    source_files = []

    # Find all episode_num-xxx directories
    episode_dirs = sorted(base_path.glob('episode_num_*'))

    for episode_dir in episode_dirs:
        if not episode_dir.is_dir():
            continue

        # Look for the scene file
        scene_file = episode_dir / f"{scene_name}.json.gz"
        if scene_file.exists():
            source_files.append(str(scene_file))

    return source_files


def verify_scene(scene_name: str, source_dir: str, merged_dir: str,
                 num_samples: int = 5) -> Dict[str, Any]:
    """
    Verify a merged scene against its source files.

    Args:
        scene_name: Name of the scene to verify
        source_dir: Directory containing episode_num-xxx subdirectories
        merged_dir: Directory containing merged scene files
        num_samples: Number of episodes to sample for content verification

    Returns:
        Dictionary containing verification results
    """
    print("=" * 70)
    print(f"校验场景: {scene_name}")
    print("=" * 70)

    # Step 1: Collect source files
    print(f"\n[步骤 1] 收集源文件...")
    print(f"源目录: {source_dir}")
    source_files = collect_source_files(source_dir, scene_name)

    if not source_files:
        return {
            'scene_name': scene_name,
            'status': 'ERROR',
            'message': f'未找到源文件'
        }

    print(f"找到 {len(source_files)} 个源文件:")
    for i, f in enumerate(source_files, 1):
        file_size = os.path.getsize(f) / (1024 * 1024)  # MB
        print(f"  {i}. {f} ({file_size:.2f} MB)")

    # Step 2: Load source data and collect episodes
    print(f"\n[步骤 2] 加载源数据并统计episodes...")
    source_episodes = []
    source_episode_ids = set()

    for i, source_file in enumerate(source_files, 1):
        data = load_json_gz(source_file)
        episodes = data.get('episodes', [])
        source_episodes.extend(episodes)

        for ep in episodes:
            ep_id = ep.get('episode_id')
            if ep_id:
                source_episode_ids.add(ep_id)

        print(f"  文件 {i}/{len(source_files)}: {len(episodes)} episodes, "
              f"当前累计: {len(source_episodes)} episodes")

    print(f"\n源数据总 episodes: {len(source_episodes)}")
    print(f"唯一 episode_id 数量: {len(source_episode_ids)}")

    # Check for duplicates
    if len(source_episodes) != len(source_episode_ids):
        print(f"⚠️  警告: 源数据中发现 {len(source_episodes) - len(source_episode_ids)} 个重复的 episode_id")

    # Step 3: Load merged data
    print(f"\n[步骤 3] 加载合并后的数据...")
    merged_file = os.path.join(merged_dir, f"{scene_name}.json.gz")

    if not os.path.exists(merged_file):
        return {
            'scene_name': scene_name,
            'status': 'ERROR',
            'message': f'合并文件不存在: {merged_file}'
        }

    merged_data = load_json_gz(merged_file)
    merged_episodes = merged_data.get('episodes', [])
    merged_episode_ids = set(ep.get('episode_id') for ep in merged_episodes if ep.get('episode_id'))

    print(f"合并文件: {merged_file}")
    file_size = os.path.getsize(merged_file) / (1024 * 1024)
    print(f"文件大小: {file_size:.2f} MB")
    print(f"合并后 episodes: {len(merged_episodes)}")
    print(f"唯一 episode_id 数量: {len(merged_episode_ids)}")

    # Step 4: Compare counts
    print(f"\n[步骤 4] 数量对比...")
    count_match = len(source_episodes) == len(merged_episodes)
    id_match = len(source_episode_ids) == len(merged_episode_ids)

    print(f"{'✓' if count_match else '✗'} Episode总数: 源={len(source_episodes)}, "
          f"合并={len(merged_episodes)} | {'一致' if count_match else '不一致!'}")
    print(f"{'✓' if id_match else '✗'} 唯一ID数: 源={len(source_episode_ids)}, "
          f"合并={len(merged_episode_ids)} | {'一致' if id_match else '不一致!'}")

    if len(merged_episodes) != len(merged_episode_ids):
        print(f"⚠️  警告: 合并数据中发现 {len(merged_episodes) - len(merged_episode_ids)} 个重复的 episode_id")

    # Step 5: Sample verification with full content comparison
    print(f"\n[步骤 5] 抽样验证 ({num_samples} 个episodes, 完整内容比对)...")

    # Build source episode lookup
    source_episodes_by_id = {ep.get('episode_id'): ep for ep in source_episodes}

    # Generate evenly distributed sample indices
    if len(merged_episodes) <= num_samples:
        # If total episodes less than sample size, check all
        sample_indices = list(range(len(merged_episodes)))
    else:
        # Evenly distribute samples across the entire range
        step = len(merged_episodes) / num_samples
        sample_indices = [int(i * step) for i in range(num_samples)]

    print(f"抽样索引: {sample_indices}")

    all_samples_match = True
    mismatch_details = []

    for idx in sample_indices:
        merged_ep = merged_episodes[idx]
        ep_id = merged_ep.get('episode_id')

        if not ep_id:
            print(f"  ✗ [{idx}] episode_id 为空")
            all_samples_match = False
            mismatch_details.append(f"索引{idx}: episode_id为空")
            continue

        source_ep = source_episodes_by_id.get(ep_id)

        if not source_ep:
            print(f"  ✗ [{idx}] episode_id={ep_id} 在源数据中不存在")
            all_samples_match = False
            mismatch_details.append(f"索引{idx}: {ep_id}不在源数据中")
            continue

        # Full content comparison
        if merged_ep == source_ep:
            print(f"  ✓ [{idx}] episode_id={ep_id}: 内容完全一致")
        else:
            print(f"  ✗ [{idx}] episode_id={ep_id}: 内容不一致")
            all_samples_match = False

            # Find differences
            source_keys = set(source_ep.keys())
            merged_keys = set(merged_ep.keys())

            missing_in_merged = source_keys - merged_keys
            extra_in_merged = merged_keys - source_keys
            common_keys = source_keys & merged_keys

            if missing_in_merged:
                print(f"     缺失字段: {missing_in_merged}")

            if extra_in_merged:
                print(f"     多余字段: {extra_in_merged}")

            # Check differing values for common keys
            differing_fields = []
            for key in sorted(common_keys):
                if source_ep[key] != merged_ep[key]:
                    differing_fields.append(key)

            if differing_fields:
                print(f"     值不匹配的字段 ({len(differing_fields)}个):")
                # Show first 5 differences in detail
                for key in differing_fields[:5]:
                    src_val = str(source_ep[key])[:100]
                    mrg_val = str(merged_ep[key])[:100]
                    if len(src_val) >= 100 or len(mrg_val) >= 100:
                        src_val += "..."
                        mrg_val += "..."
                    print(f"       - {key}:")
                    print(f"           源: {src_val}")
                    print(f"           合并: {mrg_val}")

                if len(differing_fields) > 5:
                    print(f"       ... 还有 {len(differing_fields) - 5} 个字段不匹配")

            mismatch_details.append(f"索引{idx}: {ep_id} 内容不一致")

    # Step 6: Final result
    print(f"\n" + "=" * 70)
    is_valid = count_match and id_match and all_samples_match

    if is_valid:
        print("✅ 校验通过: Episode数量一致且抽样验证通过")
    else:
        print("❌ 校验失败: 发现不一致")
        if mismatch_details:
            print(f"\n不一致详情:")
            for detail in mismatch_details:
                print(f"  - {detail}")

    print("=" * 70)

    return {
        'scene_name': scene_name,
        'status': 'PASS' if is_valid else 'FAIL',
        'source_file_count': len(source_files),
        'source_episode_count': len(source_episodes),
        'merged_episode_count': len(merged_episodes),
        'count_match': count_match,
        'sample_match': all_samples_match,
        'source_episode_ids': len(source_episode_ids),
        'merged_episode_ids': len(merged_episode_ids),
    }


def main():
    parser = argparse.ArgumentParser(description='Verify a merged scene against source files')
    parser.add_argument('--scene', type=str, default='YHmAkqgwe2p',
                        help='Scene name to verify (default: YHmAkqgwe2p)')
    parser.add_argument('--source-dir', type=str,
                        default='data/trajectory_data/objectnav/hm3d_v2/train/content',
                        help='Source directory containing episode_num-xxx subdirectories')
    parser.add_argument('--merged-dir', type=str,
                        default='data/trajectory_data/objectnav/hm3d_v2/train/merged',
                        help='Directory containing merged scene files')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of episodes to sample for verification (default: 20)')

    args = parser.parse_args()

    result = verify_scene(
        scene_name=args.scene,
        source_dir=args.source_dir,
        merged_dir=args.merged_dir,
        num_samples=args.num_samples
    )

    # Exit with appropriate code
    exit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
