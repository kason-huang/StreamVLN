#!/usr/bin/env python3
"""
分析StreamVLN trajectory_data目录中的轨迹数据统计
排除ScaleVLN，统计R2R、RxR、EnvDrop的轨迹数据量
"""

import json
import os
from collections import defaultdict
import pandas as pd

def load_annotations(file_path):
    """加载annotations.json文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def analyze_dataset(dataset_name, file_path):
    """分析单个数据集的统计信息"""
    print(f"\n=== 分析 {dataset_name} 数据集 ===")

    data = load_annotations(file_path)
    if data is None:
        return None

    if not isinstance(data, list):
        data = [data]

    # 基础统计
    total_episodes = len(data)

    # 动作序列统计
    action_lengths = []
    instruction_counts = []
    trajectory_analysis = defaultdict(int)

    for i, episode in enumerate(data):
        # 统计动作序列长度
        if 'actions' in episode:
            actions = episode['actions']
            action_lengths.append(len(actions))

        # 统计指令数量
        if 'instructions' in episode:
            instructions = episode['instructions']
            if isinstance(instructions, list):
                instruction_counts.append(len(instructions))
            else:
                instruction_counts.append(1)

        # 检查数据完整性
        has_video = 'video' in episode
        has_actions = 'actions' in episode
        has_instructions = 'instructions' in episode

        trajectory_analysis['has_video'] += int(has_video)
        trajectory_analysis['has_actions'] += int(has_actions)
        trajectory_analysis['has_instructions'] += int(has_instructions)
        trajectory_analysis['complete'] += int(has_video and has_actions and has_instructions)

    # 计算统计指标
    stats = {
        'dataset': dataset_name,
        'total_episodes': total_episodes,
        'avg_action_length': sum(action_lengths) / len(action_lengths) if action_lengths else 0,
        'min_action_length': min(action_lengths) if action_lengths else 0,
        'max_action_length': max(action_lengths) if action_lengths else 0,
        'avg_instructions': sum(instruction_counts) / len(instruction_counts) if instruction_counts else 0,
        'total_instructions': sum(instruction_counts),
        'complete_episodes': trajectory_analysis['complete'],
        'data_completeness_rate': trajectory_analysis['complete'] / total_episodes * 100 if total_episodes > 0 else 0
    }

    print(f"总轨迹数: {stats['total_episodes']:,}")
    print(f"完整轨迹数: {stats['complete_episodes']:,}")
    print(f"数据完整率: {stats['data_completeness_rate']:.2f}%")
    print(f"平均动作序列长度: {stats['avg_action_length']:.2f}")
    print(f"动作序列长度范围: {stats['min_action_length']} - {stats['max_action_length']}")
    print(f"总指令数: {stats['total_instructions']:,}")
    print(f"平均每条轨迹指令数: {stats['avg_instructions']:.2f}")

    return stats

def analyze_trajectory_data():
    """主函数：分析trajectory_data目录"""

    trajectory_data_path = "/root/workspace/lab/StreamVLN/data/trajectory_data"

    if not os.path.exists(trajectory_data_path):
        print(f"错误：找不到目录 {trajectory_data_path}")
        return

    # 要分析的数据集（排除ScaleVLN）
    datasets = {
        'R2R': os.path.join(trajectory_data_path, 'R2R', 'annotations.json'),
        'RxR': os.path.join(trajectory_data_path, 'RxR', 'annotations.json'),
        'EnvDrop': os.path.join(trajectory_data_path, 'EnvDrop', 'annotations.json'),
        # 注释掉ScaleVLN，根据要求排除
        # 'ScaleVLN': os.path.join(trajectory_data_path, 'ScaleVLN', 'annotations.json')
    }

    print("开始分析trajectory_data目录中的轨迹数据...")
    print("(排除ScaleVLN数据集)")

    all_stats = []
    total_episodes = 0
    total_instructions = 0
    total_complete_episodes = 0

    for dataset_name, file_path in datasets.items():
        if os.path.exists(file_path):
            stats = analyze_dataset(dataset_name, file_path)
            if stats:
                all_stats.append(stats)
                total_episodes += stats['total_episodes']
                total_instructions += stats['total_instructions']
                total_complete_episodes += stats['complete_episodes']
        else:
            print(f"警告：找不到文件 {file_path}")

    # 生成总结报告
    print(f"\n{'='*60}")
    print("总体统计报告 (排除ScaleVLN)")
    print(f"{'='*60}")

    print(f"\n📊 数据集汇总:")
    print(f"  参与统计的数据集数量: {len(all_stats)}")
    for stats in all_stats:
        print(f"  - {stats['dataset']}: {stats['total_episodes']:,} 条轨迹")

    print(f"\n🎯 关键指标:")
    print(f"  总轨迹数: {total_episodes:,}")
    print(f"  完整轨迹数: {total_complete_episodes:,}")
    print(f"  总体完整率: {total_complete_episodes/total_episodes*100:.2f}%")
    print(f"  总指令数: {total_instructions:,}")
    print(f"  平均每条轨迹指令数: {total_instructions/total_episodes:.2f}")

    # 生成数据框便于查看
    if all_stats:
        df = pd.DataFrame(all_stats)

        print(f"\n📋 详细统计表:")
        print(df[['dataset', 'total_episodes', 'complete_episodes', 'data_completeness_rate',
                  'avg_action_length', 'avg_instructions']].to_string(index=False))

        # 保存结果到文件
        output_file = "/root/workspace/lab/StreamVLN/trajectory_analysis_report.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("StreamVLN Trajectory Data Analysis Report\n")
            f.write("(排除ScaleVLN数据集)\n")
            f.write("="*50 + "\n\n")

            for stats in all_stats:
                f.write(f"{stats['dataset']} 数据集:\n")
                f.write(f"  总轨迹数: {stats['total_episodes']:,}\n")
                f.write(f"  完整轨迹数: {stats['complete_episodes']:,}\n")
                f.write(f"  数据完整率: {stats['data_completeness_rate']:.2f}%\n")
                f.write(f"  平均动作序列长度: {stats['avg_action_length']:.2f}\n")
                f.write(f"  总指令数: {stats['total_instructions']:,}\n")
                f.write(f"  平均每条轨迹指令数: {stats['avg_instructions']:.2f}\n\n")

            f.write("总体统计:\n")
            f.write(f"  总轨迹数: {total_episodes:,}\n")
            f.write(f"  完整轨迹数: {total_complete_episodes:,}\n")
            f.write(f"  总体完整率: {total_complete_episodes/total_episodes*100:.2f}%\n")
            f.write(f"  总指令数: {total_instructions:,}\n")
            f.write(f"  平均每条轨迹指令数: {total_instructions/total_episodes:.2f}\n")

        print(f"\n💾 详细报告已保存到: {output_file}")

    return {
        'total_episodes': total_episodes,
        'total_instructions': total_instructions,
        'total_complete_episodes': total_complete_episodes,
        'dataset_stats': all_stats
    }

if __name__ == "__main__":
    results = analyze_trajectory_data()