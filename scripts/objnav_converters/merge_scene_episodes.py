#!/usr/bin/env python3
"""
Merge ObjectNav episode files by scene name with parallel processing.

This script merges all {scene_name}.json.gz files from episode_num-xxx directories
into a single {scene_name}.json.gz file per scene.

Memory Management:
    Uses ProcessPoolExecutor with as_completed to ensure each worker process
    exits after completing a task, preventing memory accumulation during long runs.

Usage:
    python merge_scene_episodes.py [--num-workers N]
"""

import gzip
import json
import os
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time


def load_json_gz(file_path: str) -> Dict[str, Any]:
    """Load a gzipped JSON file."""
    with gzip.open(file_path, 'rt') as f:
        return json.load(f)


def save_json_gz(data: Dict[str, Any], file_path: str) -> None:
    """Save data to a gzipped JSON file with fast compression."""
    with gzip.open(file_path, 'wt', compresslevel=1) as f:
        json.dump(data, f)


def process_scene(args: Tuple[str, List[str], str]) -> Tuple[str, int, float]:
    """
    Process a single scene - merge its files and save.

    Args:
        args: Tuple of (scene_name, file_paths, output_dir)

    Returns:
        Tuple of (scene_name, num_episodes, processing_time)
    """
    scene_name, file_paths, output_dir = args
    start_time = time.time()

    # Load all files in parallel using ThreadPoolExecutor
    # This significantly speeds up I/O-bound file reading
    with ThreadPoolExecutor(max_workers=min(16, len(file_paths))) as executor:
        all_data = list(executor.map(load_json_gz, file_paths))

    # Merge episodes - use the first file as base
    merged_data = all_data[0]
    for data in all_data[1:]:
        merged_data['episodes'].extend(data['episodes'])

    # Save merged file
    output_file = os.path.join(output_dir, f"{scene_name}.json.gz")
    save_json_gz(merged_data, output_file)

    elapsed = time.time() - start_time
    return (scene_name, len(merged_data['episodes']), elapsed)


def collect_scene_files(base_dir: str) -> Dict[str, List[str]]:
    """
    Collect all scene files grouped by scene name.

    Args:
        base_dir: Base directory containing episode_num-xxx subdirectories

    Returns:
        Dictionary mapping scene_name to list of file paths
    """
    base_path = Path(base_dir)
    scene_files = defaultdict(list)

    # Find all episode_num-xxx directories
    episode_dirs = sorted(base_path.glob('episode_num_*'))
    print(f"找到 {len(episode_dirs)} 个episode目录")

    for episode_dir in episode_dirs:
        if not episode_dir.is_dir():
            continue

        # Find all .json.gz files in this directory
        for json_file in episode_dir.glob('*.json.gz'):
            # Extract scene name (remove .json.gz extension)
            scene_name = json_file.stem
            if scene_name.endswith('.json'):
                scene_name = scene_name[:-5]  # Remove '.json'
            scene_files[scene_name].append(str(json_file))

    # Sort files for each scene to ensure consistent ordering
    for scene_name in scene_files:
        scene_files[scene_name].sort()

    print(f"找到 {len(scene_files)} 个唯一场景")
    return scene_files


def main():
    parser = argparse.ArgumentParser(description='Merge ObjectNav episode files by scene')
    parser.add_argument('--num-workers', type=int, default=5,
                        help='Number of parallel workers (default: 5)')
    parser.add_argument('--max-scenes', type=int, default=None,
                        help='Maximum number of scenes to process (for testing)')
    args = parser.parse_args()

    # Configuration
    source_dir = 'data/trajectory_data/objectnav/hm3d_v2/train/content'
    output_dir = 'data/trajectory_data/objectnav/hm3d_v2/train/merged'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}\n")

    # Collect all scene files
    print("正在收集场景文件...\n")
    scene_files = collect_scene_files(source_dir)

    # Print statistics
    total_input_files = sum(len(files) for files in scene_files.values())
    print(f"\n场景统计:")
    print(f"  唯一场景数: {len(scene_files)}")
    print(f"  总输入文件数: {total_input_files}")

    # Show some examples
    sample_scenes = list(scene_files.keys())[:5]
    print(f"\n示例场景:")
    for scene in sample_scenes:
        print(f"  {scene}: {len(scene_files[scene])} 个文件")

    # Limit scenes if specified
    if args.max_scenes:
        scene_files = dict(list(scene_files.items())[:args.max_scenes])
        print(f"\n测试模式：限制处理 {args.max_scenes} 个场景")

    # Prepare arguments for parallel processing
    process_args = [
        (scene_name, files, output_dir)
        for scene_name, files in sorted(scene_files.items())
    ]

    # Determine number of workers
    num_workers = args.num_workers
    print(f"\n{'='*60}")
    print(f"使用 {num_workers} 个worker处理 {len(process_args)} 个场景...")
    print(f"{'='*60}\n")

    # Batch processing configuration
    BATCH_SIZE = 20

    # Process scenes in parallel with batch processing
    total_episodes = 0
    scenes_with_multiple_files = sum(1 for files in scene_files.values() if len(files) > 1)

    overall_start = time.time()
    total_scenes = len(process_args)
    total_batches = (total_scenes + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"开始处理 {total_scenes} 个场景 (分 {total_batches} 批)...\n")

    # Process in batches to limit memory usage
    failed_scenes = []
    completed = 0

    for batch_idx in range(0, len(process_args), BATCH_SIZE):
        batch = process_args[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        batch_start_idx = batch_idx + 1
        batch_end_idx = min(batch_idx + BATCH_SIZE, total_scenes)

        print(f"\n{'='*60}")
        print(f"批次 {batch_num}/{total_batches}: 处理场景 {batch_start_idx}-{batch_end_idx}")
        print(f"{'='*60}")

        # Show scene names for visibility
        scene_names = [arg[0] for arg in batch]
        print(f"场景列表: {', '.join(scene_names[:5])}" + (f"... 等{len(batch)}个" if len(batch) > 5 else ""))
        print()

        batch_start = time.time()
        completed_in_batch = 0

        # Process this batch with ProcessPoolExecutor
        # Workers exit after batch completes, freeing memory
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_scene, arg): arg for arg in batch}

            # Print started messages
            for future in futures:
                scene_name = futures[future][0]
                print(f"  🚀 开始处理: {scene_name}")

            for future in as_completed(futures):
                arg = futures[future]
                scene_name_key = arg[0]
                try:
                    scene_name, num_episodes, proc_time = future.result()
                    total_episodes += num_episodes
                    completed += 1
                    completed_in_batch += 1

                    # Batch progress
                    batch_elapsed = time.time() - batch_start
                    print(f"  [{completed_in_batch}/{len(batch)}] ✅ {scene_name} - "
                          f"{num_episodes} episodes | {proc_time:.1f}s")
                except Exception as exc:
                    print(f"  [{completed_in_batch+1}/{len(batch)}] ❌ {scene_name_key} 失败: {exc}")
                    failed_scenes.append(arg)
                    completed += 1
                    completed_in_batch += 1

        batch_elapsed = time.time() - batch_start
        overall_elapsed = time.time() - overall_start

        # Overall progress
        avg_time = overall_elapsed / completed
        remaining = total_scenes - completed
        eta = avg_time * remaining

        print(f"✅ 批次 {batch_num}/{total_batches} 完成 | "
              f"耗时: {batch_elapsed:.1f}s | 总进度: {completed}/{total_scenes} | "
              f"ETA: {eta:.0f}s ({eta/60:.1f}min)")

        print(f"   内存已释放 (worker进程退出)\n")

    # Retry failed scenes
    if failed_scenes:
        print(f"\n{'='*60}")
        print(f"重试 {len(failed_scenes)} 个失败场景...")
        print(f"{'='*60}\n")
        for arg in failed_scenes:
            scene_name = arg[0]
            try:
                _, num_episodes, proc_time = process_scene(arg)
                total_episodes += num_episodes
                print(f"  ✅ {scene_name} 重试成功 - {num_episodes} episodes | {proc_time:.1f}s")
            except Exception as exc:
                print(f"  ❌ {scene_name} 重试失败: {exc}")

    print()  # New line after progress

    # Print final statistics
    total_time = time.time() - overall_start
    successful_scenes = total_scenes - len(failed_scenes)

    print(f"\n{'='*60}")
    print("✅ 合并完成！")
    print(f"{'='*60}")
    print(f"总场景数: {total_scenes}")
    print(f"成功: {successful_scenes} | 失败: {len(failed_scenes)}")
    print(f"合并总episodes: {total_episodes:,}")
    print(f"包含多个文件的场景: {scenes_with_multiple_files}")
    if successful_scenes > 0:
        print(f"平均每场景episodes: {total_episodes / successful_scenes:.1f}")
    print(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"平均处理速度: {total_time/total_scenes:.1f}秒/场景")
    print(f"\n输出目录: {output_dir}")
    print(f"已保存文件数: {successful_scenes}")


if __name__ == '__main__':
    main()
