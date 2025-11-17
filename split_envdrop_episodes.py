#!/usr/bin/env python3
"""
将EnvDrop的annotations.json文件切分为n等份，并同时处理envdrop.json.gz文件
"""
import json
import os
import gzip
from pathlib import Path
from typing import List, Dict, Any
import math

def read_envdrop_gz(envdrop_gz_path: str) -> Dict[str, Any]:
    """
    读取envdrop.json.gz文件

    Args:
        envdrop_gz_path (str): envdrop.json.gz文件路径

    Returns:
        Dict[str, Any]: envdrop数据，包含episodes列表
    """
    print(f"正在读取envdrop数据文件: {envdrop_gz_path}")

    with gzip.open(envdrop_gz_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    total_episodes = len(data.get('episodes', []))
    print(f"envdrop总episodes数: {total_episodes:,}")

    return data

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

def split_both_files(input_annotations: str, input_envdrop_gz: str, output_dir: str, n_splits: int) -> Dict[str, List[str]]:
    """
    同时切分annotations.json和envdrop.json.gz文件

    Args:
        input_annotations (str): annotations.json文件路径
        input_envdrop_gz (str): envdrop.json.gz文件路径
        output_dir (str): 输出目录
        n_splits (int): 切分的份数

    Returns:
        Dict[str, List[str]]: 包含两种文件生成路径的字典
    """
    print("=== 切分EnvDrop Episodes 和环境数据 ===")
    print(f"annotations输入文件: {input_annotations}")
    print(f"envdrop输入文件: {input_envdrop_gz}")
    print(f"输出目录: {output_dir}")
    print(f"切分份数: {n_splits}")
    print()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 读取并切分annotations.json
    print("1. 处理annotations.json文件...")
    annotation_files = split_json_file(
        input_file=input_annotations,
        output_dir=str(output_path / "annotations"),
        n_splits=n_splits,
        prefix="annotations"
    )

    # 2. 读取envdrop.json.gz文件
    print("\n2. 读取envdrop.json.gz文件...")
    envdrop_data = read_envdrop_gz(input_envdrop_gz)
    envdrop_episodes = envdrop_data.get('episodes', [])

    # 3. 读取annotations数据以获取id到episode_id的映射关系
    print("\n3. 建立ID映射关系...")
    with open(input_annotations, 'r', encoding='utf-8') as f:
        annotations_data = json.load(f)

    # 创建episode_id到annotations的映射
    # 注意：annotations.json的id字段对应envdrop.json.gz的episode_id字段
    print("创建episode_id映射...")
    episode_id_to_annotation = {}
    for annotation in annotations_data:
        episode_id = annotation.get('id')
        if episode_id is not None:
            episode_id_to_annotation[episode_id] = annotation

    print(f"annotations中找到的episode_id数量: {len(episode_id_to_annotation)}")

    # 4. 按照annotations的切分方式切分envdrop数据
    print("\n4. 切分envdrop数据...")
    envdrop_output_files = []

    # 计算每个文件的episodes数量（基于annotations的切分）
    items_per_file = math.ceil(len(annotations_data) / n_splits)

    for i in range(n_splits):
        start_idx = i * items_per_file
        end_idx = min((i + 1) * items_per_file, len(annotations_data))

        # 获取这一部分annotations的episode_ids
        chunk_annotations = annotations_data[start_idx:end_idx]
        chunk_episode_ids = set(ann.get('id') for ann in chunk_annotations if ann.get('id') is not None)

        # 从envdrop数据中筛选对应的episodes
        chunk_envdrop_episodes = [
            episode for episode in envdrop_episodes
            if episode.get('episode_id') in chunk_episode_ids
        ]

        # 生成输出文件名
        envdrop_output_filename = f"envdrop_part_{i+1:03d}_of_{n_splits:03d}.json.gz"
        envdrop_output_filepath = output_path / "envdrop" / envdrop_output_filename

        # 创建envdrop输出目录
        (output_path / "envdrop").mkdir(parents=True, exist_ok=True)

        # 保存切分后的envdrop数据
        chunk_envdrop_data = {
            "episodes": chunk_envdrop_episodes
        }

        print(f"正在生成envdrop第 {i+1}/{n_splits} 份: {envdrop_output_filename} ({len(chunk_envdrop_episodes)} 个episodes)")

        with gzip.open(envdrop_output_filepath, 'wt', encoding='utf-8') as f:
            json.dump(chunk_envdrop_data, f, ensure_ascii=False)

        envdrop_output_files.append(str(envdrop_output_filepath))

        print(f"  保存完成: {envdrop_output_filepath}")

    return {
        "annotations": annotation_files,
        "envdrop": envdrop_output_files
    }

def split_envdrop_episodes(n_splits: int = 10, output_dir: str = None) -> List[str]:
    """
    专门用于切分envdrop annotations.json的便捷函数

    Args:
        n_splits (int): 切分的份数，默认为10
        output_dir (str, optional): 输出目录，默认为data/trajectory_data/EnvDrop/tmp

    Returns:
        List[str]: 生成的输出文件路径列表
    """
    # 设置默认路径
    base_dir = Path("/root/workspace/lab/StreamVLN/data/trajectory_data/EnvDrop")
    input_file = base_dir / "annotations.json"

    if output_dir is None:
        output_dir = base_dir / "tmp"

    print("=== 切分EnvDrop Episodes ===")
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"切分份数: {n_splits}")
    print()

    return split_json_file(
        input_file=str(input_file),
        output_dir=str(output_dir),
        n_splits=n_splits,
        prefix="episodes"
    )

def create_batch_script(output_dir: str, files_dict: Dict[str, List[str]], batch_size: int = 1):
    """
    创建批处理脚本来处理切分后的文件

    Args:
        output_dir (str): 输出目录
        files_dict (Dict[str, List[str]]): 包含annotations和envdrop文件路径的字典
        batch_size (int): 每批处理的文件数量
    """
    annotation_files = files_dict.get("annotations", [])
    envdrop_files = files_dict.get("envdrop", [])

    # 过滤出数据文件（排除索引文件）
    annotation_data_files = [f for f in annotation_files if f.endswith('.json') and 'index' not in f]
    annotation_data_files.sort()  # 按文件名排序

    envdrop_data_files = [f for f in envdrop_files if f.endswith('.json.gz')]
    envdrop_data_files.sort()  # 按文件名排序

    # 创建批处理脚本
    script_path = Path(output_dir) / "process_split_files.sh"

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write("#!/bin/bash\n")
        f.write("# 批处理切分后的episodes文件\n\n")
        f.write("set -e  # 遇到错误立即退出\n\n")

        # 处理annotations文件
        f.write("# 处理annotations文件\n")
        for i, data_file in enumerate(annotation_data_files):
            if i % batch_size == 0:
                f.write(f"\n# 批次 {(i // batch_size) + 1} - Annotations\n")

            # 提取相对路径用于脚本
            rel_path = Path(data_file).relative_to(Path(output_dir))
            f.write(f"echo '处理annotations文件: {rel_path}'\n")
            f.write(f"python3 trajectory_save_rgb.py --annot_path '{rel_path}'\n")
            f.write("echo 'annotations文件处理完成'\n\n")

        # 处理envdrop文件（如果需要）
        f.write("\n# EnvDrop文件已生成，可根据需要进行处理\n")
        for i, envdrop_file in enumerate(envdrop_data_files):
            rel_path = Path(envdrop_file).relative_to(Path(output_dir))
            f.write(f"echo 'envdrop文件已生成: {rel_path}'\n")

    # 设置执行权限
    os.chmod(script_path, 0o755)

    print(f"已创建批处理脚本: {script_path}")
    return str(script_path)

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='切分annotations.json和envdrop.json.gz文件')
    parser.add_argument('--n_splits', type=int, default=10,
                       help='切分的份数 (默认: 10)')
    parser.add_argument('--output_dir', type=str,
                       help='输出目录 (默认: data/trajectory_data/EnvDrop/tmp)')
    parser.add_argument('--create_batch', action='store_true',
                       help='创建批处理脚本')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='每批处理的文件数量 (默认: 1)')
    parser.add_argument('--annotations_only', action='store_true',
                       help='只处理annotations.json文件（兼容原有功能）')
    parser.add_argument('--annotations_path', type=str,
                       help='annotations.json文件路径 (默认: data/trajectory_data/EnvDrop/annotations.json)')
    parser.add_argument('--envdrop_path', type=str,
                       help='envdrop.json.gz文件路径 (默认: data/datasets/envdrop/envdrop.json.gz)')

    args = parser.parse_args()

    try:
        # 设置默认路径
        base_dir = Path("/root/workspace/lab/StreamVLN")
        default_annotations = base_dir / "data/trajectory_data/EnvDrop/annotations.json"
        default_envdrop = base_dir / "data/datasets/envdrop/envdrop.json.gz"

        annotations_path = args.annotations_path or str(default_annotations)
        envdrop_path = args.envdrop_path or str(default_envdrop)
        output_dir = args.output_dir or str(base_dir / "data/trajectory_data/EnvDrop/tmp")

        if args.annotations_only:
            # 兼容原有功能：只处理annotations.json
            print("=== 仅处理annotations.json文件 ===")
            output_files = split_envdrop_episodes(
                n_splits=args.n_splits,
                output_dir=output_dir
            )

            # 如果需要，创建批处理脚本
            if args.create_batch:
                script_path = create_batch_script(output_dir, {"annotations": output_files, "envdrop": []}, args.batch_size)
                print(f"\n使用方法:")
                print(f"  cd {output_dir}")
                print(f"  ./process_split_files.sh")

        else:
            # 新功能：同时处理两种文件
            print("=== 同时处理annotations.json和envdrop.json.gz文件 ===")
            output_files = split_both_files(
                input_annotations=annotations_path,
                input_envdrop_gz=envdrop_path,
                output_dir=output_dir,
                n_splits=args.n_splits
            )

            print(f"\n=== 切分完成 ===")
            print(f"生成的annotations文件:")
            for file_path in output_files["annotations"]:
                print(f"  - {file_path}")

            print(f"\n生成的envdrop文件:")
            for file_path in output_files["envdrop"]:
                print(f"  - {file_path}")

            # 如果需要，创建批处理脚本
            if args.create_batch:
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