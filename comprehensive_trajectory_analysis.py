#!/usr/bin/env python3
"""
StreamVLN轨迹数据综合分析报告
包含所有数据集的详细统计和对比
"""

import json
import os
import pandas as pd
from datetime import datetime

def load_dataset_stats(dataset_name, file_path):
    """加载单个数据集的统计信息"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None

    if not isinstance(data, list):
        data = [data]

    total_episodes = len(data)
    action_lengths = []
    instruction_counts = []

    for episode in data:
        if 'actions' in episode:
            action_lengths.append(len(episode['actions']))
        if 'instructions' in episode:
            instructions = episode['instructions']
            if isinstance(instructions, list):
                instruction_counts.append(len(instructions))
            else:
                instruction_counts.append(1)

    return {
        'dataset': dataset_name,
        'total_episodes': total_episodes,
        'avg_action_length': sum(action_lengths) / len(action_lengths) if action_lengths else 0,
        'min_action_length': min(action_lengths) if action_lengths else 0,
        'max_action_length': max(action_lengths) if action_lengths else 0,
        'total_instructions': sum(instruction_counts),
        'avg_instructions': sum(instruction_counts) / len(instruction_counts) if instruction_counts else 0
    }

def create_comprehensive_report():
    """创建综合分析报告"""

    trajectory_data_path = "/root/workspace/lab/StreamVLN/data/trajectory_data"

    # 所有数据集路径
    datasets = {
        'R2R': os.path.join(trajectory_data_path, 'R2R', 'annotations.json'),
        'RxR': os.path.join(trajectory_data_path, 'RxR', 'annotations.json'),
        'EnvDrop': os.path.join(trajectory_data_path, 'EnvDrop', 'annotations.json'),
        'ScaleVLN': os.path.join(trajectory_data_path, 'ScaleVLN', 'annotations.json')
    }

    print("StreamVLN 轨迹数据综合分析报告")
    print("=" * 60)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 加载所有数据集统计
    all_stats = []
    for dataset_name, file_path in datasets.items():
        if os.path.exists(file_path):
            stats = load_dataset_stats(dataset_name, file_path)
            if stats:
                all_stats.append(stats)
                print(f"✅ {dataset_name}: {stats['total_episodes']:,} 条轨迹")
        else:
            print(f"❌ {dataset_name}: 文件不存在")

    print()

    # 创建DataFrame便于分析
    df = pd.DataFrame(all_stats)

    # 总体统计
    total_episodes = df['total_episodes'].sum()
    total_instructions = df['total_instructions'].sum()

    print("📊 总体统计")
    print("-" * 30)
    print(f"总轨迹数: {total_episodes:,}")
    print(f"总指令数: {total_instructions:,}")
    print(f"平均每条轨迹指令数: {total_instructions/total_episodes:.2f}")
    print()

    # 排除ScaleVLN的统计
    non_scalevln_df = df[df['dataset'] != 'ScaleVLN']
    non_scalevln_episodes = non_scalevln_df['total_episodes'].sum()
    non_scalevln_instructions = non_scalevln_df['total_instructions'].sum()

    print("📊 排除ScaleVLN的统计")
    print("-" * 30)
    print(f"轨迹数: {non_scalevln_episodes:,}")
    print(f"指令数: {non_scalevln_instructions:,}")
    print(f"平均每条轨迹指令数: {non_scalevln_instructions/non_scalevln_episodes:.2f}")
    print()

    # 详细数据集对比
    print("📋 详细数据集对比")
    print("-" * 30)
    print(df[['dataset', 'total_episodes', 'avg_action_length', 'avg_instructions']].to_string(index=False))
    print()

    # 数据分布分析
    print("📈 数据分布分析")
    print("-" * 30)

    for _, row in df.iterrows():
        dataset_name = row['dataset']
        percentage = (row['total_episodes'] / total_episodes) * 100
        print(f"{dataset_name:10s}: {row['total_episodes']:8,} ({percentage:5.2f}%) - 平均动作长度: {row['avg_action_length']:6.2f}")

    print()

    # 对Stage 1训练的启示
    print("🎯 Stage 1 训练数据分析")
    print("-" * 30)
    print("基于分析结果，Stage 1训练可以考虑以下策略:")
    print()

    # 主要数据集 (排除ScaleVLN)
    main_datasets = ['R2R', 'RxR', 'EnvDrop']
    main_total = non_scalevln_df[non_scalevln_df['dataset'].isin(main_datasets)]['total_episodes'].sum()

    print(f"1. 主要训练数据集 (排除ScaleVLN):")
    print(f"   - 总轨迹数: {main_total:,}")
    print(f"   - 平均动作长度: {non_scalevln_df[non_scalevln_df['dataset'].isin(main_datasets)]['avg_action_length'].mean():.2f}")
    print(f"   - 这些数据质量高，适合作为预训练的主要数据源")
    print()

    print("2. ScaleVLN数据的特点:")
    scalevln_stats = df[df['dataset'] == 'ScaleVLN'].iloc[0]
    print(f"   - 轨迹数: {scalevln_stats['total_episodes']:,}")
    print(f"   - 平均动作长度较短: {scalevln_stats['avg_action_length']:.2f}")
    print(f"   - 建议在Dagger阶段或后期训练中引入，作为大规模数据增强")
    print()

    # 保存详细报告
    output_file = "/root/workspace/lab/StreamVLN/comprehensive_trajectory_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("StreamVLN 轨迹数据综合分析报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("📊 总体统计\n")
        f.write("-" * 30 + "\n")
        f.write(f"总轨迹数: {total_episodes:,}\n")
        f.write(f"总指令数: {total_instructions:,}\n")
        f.write(f"平均每条轨迹指令数: {total_instructions/total_episodes:.2f}\n\n")

        f.write("📊 排除ScaleVLN的统计 (Stage 1主要数据)\n")
        f.write("-" * 30 + "\n")
        f.write(f"轨迹数: {non_scalevln_episodes:,}\n")
        f.write(f"指令数: {non_scalevln_instructions:,}\n")
        f.write(f"平均每条轨迹指令数: {non_scalevln_instructions/non_scalevln_episodes:.2f}\n\n")

        f.write("📋 详细数据集对比\n")
        f.write("-" * 30 + "\n")
        f.write(df[['dataset', 'total_episodes', 'avg_action_length', 'avg_instructions']].to_string(index=False))
        f.write("\n\n")

        f.write("📈 数据分布分析\n")
        f.write("-" * 30 + "\n")
        for _, row in df.iterrows():
            dataset_name = row['dataset']
            percentage = (row['total_episodes'] / total_episodes) * 100
            f.write(f"{dataset_name:10s}: {row['total_episodes']:8,} ({percentage:5.2f}%) - 平均动作长度: {row['avg_action_length']:6.2f}\n")

        f.write("\n🎯 Stage 1 训练建议\n")
        f.write("-" * 30 + "\n")
        f.write("1. 使用R2R + RxR + EnvDrop作为主要训练数据 (排除ScaleVLN)\n")
        f.write("2. 总计约15.6万条高质量轨迹数据\n")
        f.write("3. ScaleVLN在Dagger阶段或后期引入\n")
        f.write("4. 注意数据不平衡，可能需要采样策略\n")

    print(f"💾 详细报告已保存到: {output_file}")

    return {
        'total_all': total_episodes,
        'total_excluding_scalevln': non_scalevln_episodes,
        'dataset_stats': all_stats
    }

if __name__ == "__main__":
    results = create_comprehensive_report()