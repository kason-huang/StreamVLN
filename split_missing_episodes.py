#!/usr/bin/env python3
"""
将EnvDrop的missing_episodes.json文件切分为n等份
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import math

def split_json_file(input_file: str, output_dir: str, n_splits: int, prefix: str = None) -> List[str]:
    """
    将JSON文件切分为n等份

    Args:
        input_file (str): 输入JSON文件路径
        output_dir (str): 输出目录路径
        n_splits (int): 切分的份数
        prefix (str, optional): 输出文件前缀，默认使用输入文件名

    Returns:
        List[str]: 生成的输出文件路径列表
    """
    # 验证输入参数
    if n_splits <= 0:
        raise ValueError("n_splits必须大于0")

    input_path = Path(input_file)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 设置输出文件前缀
    if prefix is None:
        prefix = input_path.stem

    print(f"正在读取输入文件: {input_file}")
    print(f"文件大小: {input_path.stat().st_size / (1024*1024):.2f} MB")

    # 读取原始JSON数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_items = len(data)
    print(f"总条目数: {total_items:,}")

    if total_items == 0:
        raise ValueError("输入文件中没有数据")

    # 计算每个文件的条目数
    items_per_file = math.ceil(total_items / n_splits)
    print(f"计划切分为 {n_splits} 份，每份约 {items_per_file:,} 个条目")

    # 生成输出文件路径列表
    output_files = []
    generated_files = []

    try:
        # 切分并保存数据
        for i in range(n_splits):
            start_idx = i * items_per_file
            end_idx = min((i + 1) * items_per_file, total_items)

            # 生成输出文件名
            output_filename = f"{prefix}_part_{i+1:03d}_of_{n_splits:03d}.json"
            output_filepath = output_path / output_filename
            output_files.append(str(output_filepath))

            # 切分数据
            chunk_data = data[start_idx:end_idx]
            chunk_size = len(chunk_data)

            # 保存切分后的数据
            print(f"正在生成第 {i+1}/{n_splits} 份: {output_filename} ({chunk_size:,} 个条目)")

            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)

            generated_files.append(str(output_filepath))

            print(f"  保存完成: {output_filepath} ({output_filepath.stat().st_size / (1024*1024):.2f} MB)")

        # 创建索引文件
        index_file = output_path / f"{prefix}_index.json"
        index_data = {
            "source_file": str(input_path),
            "total_items": total_items,
            "n_splits": n_splits,
            "items_per_file": items_per_file,
            "generated_files": output_files,
            "generation_time": "2025-11-17"
        }

        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

        generated_files.append(str(index_file))

        print(f"\n=== 切分完成 ===")
        print(f"生成了 {len(generated_files)} 个文件:")
        for file_path in generated_files:
            print(f"  - {file_path}")

        return generated_files

    except Exception as e:
        # 清理已生成的文件
        print(f"发生错误，正在清理已生成的文件...")
        for file_path in generated_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"  已删除: {file_path}")
            except Exception as cleanup_error:
                print(f"  删除文件时出错 {file_path}: {cleanup_error}")
        raise e

def split_missing_episodes(n_splits: int = 10, output_dir: str = None) -> List[str]:
    """
    专门用于切分missing_episodes.json的便捷函数

    Args:
        n_splits (int): 切分的份数，默认为10
        output_dir (str, optional): 输出目录，默认为data/trajectory_data/EnvDrop/tmp

    Returns:
        List[str]: 生成的输出文件路径列表
    """
    # 设置默认路径
    base_dir = Path("/root/workspace/lab/StreamVLN/data/trajectory_data/EnvDrop")
    input_file = base_dir / "missing_episodes.json"

    if output_dir is None:
        output_dir = base_dir / "tmp"

    print("=== 切分EnvDrop Missing Episodes ===")
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"切分份数: {n_splits}")
    print()

    return split_json_file(
        input_file=str(input_file),
        output_dir=str(output_dir),
        n_splits=n_splits,
        prefix="missing_episodes"
    )

def create_batch_script(output_dir: str, files: List[str], batch_size: int = 1):
    """
    创建批处理脚本来处理切分后的文件

    Args:
        output_dir (str): 输出目录
        files (List[str]): 切分后的文件列表
        batch_size (int): 每批处理的文件数量
    """
    # 过滤出数据文件（排除索引文件）
    data_files = [f for f in files if f.endswith('.json') and 'index' not in f]
    data_files.sort()  # 按文件名排序

    # 创建批处理脚本
    script_path = Path(output_dir) / "process_split_files.sh"

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write("#!/bin/bash\n")
        f.write("# 批处理切分后的missing_episodes文件\n\n")
        f.write("set -e  # 遇到错误立即退出\n\n")

        for i, data_file in enumerate(data_files):
            if i % batch_size == 0:
                f.write(f"\n# 批次 {(i // batch_size) + 1}\n")

            # 提取相对路径用于脚本
            rel_path = Path(data_file).name
            f.write(f"echo '处理文件: {rel_path}'\n")
            f.write(f"python3 trajectory_save_rgb.py --annot_path '{rel_path}'\n")
            f.write("echo '完成'\n\n")

    # 设置执行权限
    os.chmod(script_path, 0o755)

    print(f"已创建批处理脚本: {script_path}")
    return str(script_path)

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='切分missing_episodes.json文件')
    parser.add_argument('--n_splits', type=int, default=10,
                       help='切分的份数 (默认: 10)')
    parser.add_argument('--output_dir', type=str,
                       help='输出目录 (默认: data/trajectory_data/EnvDrop/tmp)')
    parser.add_argument('--create_batch', action='store_true',
                       help='创建批处理脚本')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='每批处理的文件数量 (默认: 1)')

    args = parser.parse_args()

    try:
        # 切分文件
        output_files = split_missing_episodes(
            n_splits=args.n_splits,
            output_dir=args.output_dir
        )

        # 如果需要，创建批处理脚本
        if args.create_batch:
            output_dir = args.output_dir or "/root/workspace/lab/StreamVLN/data/trajectory_data/EnvDrop/tmp"
            script_path = create_batch_script(output_dir, output_files, args.batch_size)
            print(f"\n使用方法:")
            print(f"  cd {output_dir}")
            print(f"  ./process_split_files.sh")

    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())