#!/usr/bin/env python3
"""
分析ScaleVLN数据集的统计信息
"""

import json
import os

def analyze_scalevln():
    """分析ScaleVLN数据集"""
    scalevln_path = "/root/workspace/lab/StreamVLN/data/trajectory_data/ScaleVLN/annotations.json"

    if not os.path.exists(scalevln_path):
        print(f"找不到ScaleVLN数据文件: {scalevln_path}")
        return None

    try:
        with open(scalevln_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取ScaleVLN数据失败: {e}")
        return None

    if not isinstance(data, list):
        data = [data]

    total_episodes = len(data)

    # 分析数据特征
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

    print("=== ScaleVLN 数据集分析 ===")
    print(f"总轨迹数: {total_episodes:,}")
    print(f"平均动作序列长度: {sum(action_lengths)/len(action_lengths):.2f}")
    print(f"动作序列长度范围: {min(action_lengths)} - {max(action_lengths)}")
    print(f"总指令数: {sum(instruction_counts):,}")
    print(f"平均每条轨迹指令数: {sum(instruction_counts)/len(instruction_counts):.2f}")

    return {
        'total_episodes': total_episodes,
        'avg_action_length': sum(action_lengths)/len(action_lengths),
        'total_instructions': sum(instruction_counts),
        'avg_instructions': sum(instruction_counts)/len(instruction_counts)
    }

if __name__ == "__main__":
    analyze_scalevln()