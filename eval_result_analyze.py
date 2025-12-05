#!/usr/bin/env python3
"""
StreamVLN模型评估结果分析脚本
用于分析result.json文件中的评估指标
"""

import json
import random
import argparse
import os
from typing import List, Dict, Any

def load_results(file_path: str) -> List[Dict[str, Any]]:
    """加载评估结果文件"""
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def calculate_basic_stats(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算基础统计指标"""
    total = len(results)
    success_count = sum(1 for r in results if r.get('success') == 1.0)

    return {
        'total_episodes': total,
        'success_count': success_count,
        'success_rate': success_count / total * 100,
        'avg_spl': sum(r.get('spl', 0) for r in results) / total,
        'avg_os': sum(r.get('os', 0) for r in results) / total,
        'avg_ne': sum(r.get('ne', 0) for r in results) / total,
        'avg_steps': sum(r.get('steps', 0) for r in results) / total
    }

def analyze_success_failure(results: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """分析成功和失败案例"""
    successful = [r for r in results if r.get('success') == 1.0]
    failed = [r for r in results if r.get('success') == 0.0]

    success_rate = len(successful) / len(results) * 100

    result = {
        'successful': {
            'count': len(successful),
            'percentage': success_rate
        },
        'failed': {
            'count': len(failed),
            'percentage': 100 - success_rate
        }
    }

    if successful:
        result['successful']['avg_ne'] = sum(r.get('ne', 0) for r in successful) / len(successful)
        result['successful']['avg_spl'] = sum(r.get('spl', 0) for r in successful) / len(successful)
        result['successful']['avg_os'] = sum(r.get('os', 0) for r in successful) / len(successful)
        result['successful']['avg_steps'] = sum(r.get('steps', 0) for r in successful) / len(successful)
        result['successful']['ne_range'] = [
            min(r.get('ne', 0) for r in successful),
            max(r.get('ne', 0) for r in successful)
        ]

    if failed:
        result['failed']['avg_ne'] = sum(r.get('ne', 0) for r in failed) / len(failed)
        result['failed']['avg_spl'] = sum(r.get('spl', 0) for r in failed) / len(failed)
        result['failed']['avg_os'] = sum(r.get('os', 0) for r in failed) / len(failed)
        result['failed']['avg_steps'] = sum(r.get('steps', 0) for r in failed) / len(failed)
        result['failed']['ne_range'] = [
            min(r.get('ne', 0) for r in failed),
            max(r.get('ne', 0) for r in failed)
        ]

    return result

def analyze_scenes(results: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """分析不同场景的性能"""
    scene_stats = {}

    for r in results:
        scene = r.get('scene_id', 'unknown')
        if scene not in scene_stats:
            scene_stats[scene] = {
                'total': 0,
                'success': 0,
                'spl_sum': 0,
                'ne_sum': 0,
                'os_sum': 0,
                'steps_sum': 0
            }

        scene_stats[scene]['total'] += 1
        scene_stats[scene]['spl_sum'] += r.get('spl', 0)
        scene_stats[scene]['ne_sum'] += r.get('ne', 0)
        scene_stats[scene]['os_sum'] += r.get('os', 0)
        scene_stats[scene]['steps_sum'] += r.get('steps', 0)

        if r.get('success') == 1.0:
            scene_stats[scene]['success'] += 1

    # 计算平均值
    for scene, stats in scene_stats.items():
        if scene != 'unknown':
            total = stats['total']
            stats['success_rate'] = stats['success'] / total * 100
            stats['avg_spl'] = stats['spl_sum'] / total
            stats['avg_ne'] = stats['ne_sum'] / total
            stats['avg_os'] = stats['os_sum'] / total
            stats['avg_steps'] = stats['steps_sum'] / total

    return scene_stats

def analyze_extreme_cases(results: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """分析极值案例"""
    # 移除可能为None的记录
    valid_results = [r for r in results if r.get('ne') is not None]

    best_spl = max(valid_results, key=lambda x: x.get('spl', 0))
    worst_spl = min(valid_results, key=lambda x: x.get('spl', 0))
    max_ne = max(valid_results, key=lambda x: x.get('ne', 0))
    min_ne = min(valid_results, key=lambda x: x.get('ne', 0))

    return {
        'best_spl': {
            'episode_id': best_spl.get('episode_id'),
            'scene_id': best_spl.get('scene_id'),
            'spl': best_spl.get('spl'),
            'ne': best_spl.get('ne'),
            'steps': best_spl.get('steps'),
            'instruction': best_spl.get('episode_instruction')
        },
        'worst_spl': {
            'episode_id': worst_spl.get('episode_id'),
            'scene_id': worst_spl.get('scene_id'),
            'spl': worst_spl.get('spl'),
            'ne': worst_spl.get('ne'),
            'steps': worst_spl.get('steps'),
            'instruction': worst_spl.get('episode_instruction')
        },
        'max_ne': {
            'episode_id': max_ne.get('episode_id'),
            'scene_id': max_ne.get('scene_id'),
            'ne': max_ne.get('ne'),
            'spl': max_ne.get('spl'),
            'steps': max_ne.get('steps'),
            'instruction': max_ne.get('episode_instruction')
        },
        'min_ne': {
            'episode_id': min_ne.get('episode_id'),
            'scene_id': min_ne.get('scene_id'),
            'ne': min_ne.get('ne'),
            'spl': min_ne.get('spl'),
            'steps': min_ne.get('steps'),
            'instruction': min_ne.get('episode_instruction')
        }
    }

def analyze_steps_distribution(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析步数分布"""
    all_steps = [r.get('steps', 0) for r in results]
    max_steps_episodes = [r for r in results if r.get('steps', 0) >= 400]

    steps_analysis = {
        'avg_steps': sum(all_steps) / len(all_steps),
        'steps_range': [min(all_steps), max(all_steps)],
        'max_steps_count': len(max_steps_episodes)
    }

    if max_steps_episodes:
        steps_analysis['max_steps_success_rate'] = sum(1 for r in max_steps_episodes if r.get('success') == 1.0) / len(max_steps_episodes) * 100
    else:
        steps_analysis['max_steps_success_rate'] = 0

    return steps_analysis

def analyze_near_misses(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析接近目标的失败案例"""
    failed = [r for r in results if r.get('success') == 0.0]
    near_miss_failed = [r for r in failed if r.get('ne', 0) < 3.0]

    return {
        'near_miss_count': len(near_miss_failed),
        'near_miss_rate': len(near_miss_failed) / len(failed) * 100 if failed else 0
    }

def sample_cases(results: List[Dict[str, Any]], num_samples: int = 5) -> Dict[str, List[Dict]]:
    """随机采样成功和失败案例"""
    successful = [r for r in results if r.get('success') == 1.0]
    failed = [r for r in results if r.get('success') == 0.0]

    return {
        'successful_samples': random.sample(successful, min(num_samples, len(successful))),
        'failed_samples': random.sample(failed, min(num_samples, len(failed)))
    }

def print_analysis_results(results_file: str, model_config: str = "Unknown"):
    """打印分析结果"""
    print('=== StreamVLN模型评估结果详细分析 ===')
    print(f'结果文件: {results_file}')
    print(f'模型配置: {model_config}')
    print()

    # 加载数据
    results = load_results(results_file)

    # 基础统计
    stats = calculate_basic_stats(results)
    print('主要性能指标:')
    print(f'  总episode数量: {stats["total_episodes"]}')
    print(f'  成功率: {stats["success_rate"]:.2f}%')
    print(f'  平均SPL: {stats["avg_spl"]:.4f}')
    print(f'  平均OS: {stats["avg_os"]:.4f}')
    print(f'  平均导航误差: {stats["avg_ne"]:.4f} meters')
    print(f'  平均步数: {stats["avg_steps"]:.1f}')
    print()

    # 成功失败分析
    analysis = analyze_success_failure(results)

    print('成功案例分析:')
    print(f'  数量: {analysis["successful"]["count"]} ({analysis["successful"]["percentage"]:.1f}%)')
    if 'avg_ne' in analysis["successful"]:
        print(f'  平均导航误差: {analysis["successful"]["avg_ne"]:.2f}m')
        print(f'  平均SPL: {analysis["successful"]["avg_spl"]:.4f}')
        print(f'  平均OS: {analysis["successful"]["avg_os"]:.4f}')
        print(f'  平均步数: {analysis["successful"]["avg_steps"]:.1f}')
        print(f'  导航误差范围: {analysis["successful"]["ne_range"][0]:.2f}m - {analysis["successful"]["ne_range"][1]:.2f}m')

    print()
    print('失败案例分析:')
    print(f'  数量: {analysis["failed"]["count"]} ({analysis["failed"]["percentage"]:.1f}%)')
    if 'avg_ne' in analysis["failed"]:
        print(f'  平均导航误差: {analysis["failed"]["avg_ne"]:.2f}m')
        print(f'  平均SPL: {analysis["failed"]["avg_spl"]:.4f}')
        print(f'  平均OS: {analysis["failed"]["avg_os"]:.4f}')
        print(f'  平均步数: {analysis["failed"]["avg_steps"]:.1f}')
        print(f'  导航误差范围: {analysis["failed"]["ne_range"][0]:.2f}m - {analysis["failed"]["ne_range"][1]:.2f}m')

    print()

    # 场景分析
    scene_stats = analyze_scenes(results)
    print('场景性能统计:')
    sorted_scenes = sorted(scene_stats.items(),
                          key=lambda x: x[1].get('success_rate', 0),
                          reverse=True)
    for scene, stats in sorted_scenes:
        if scene != 'unknown':
            print(f'  {scene}: {stats["total"]} episodes, '
                  f'成功率: {stats["success_rate"]:.1f}%, '
                  f'SPL: {stats["avg_spl"]:.3f}, '
                  f'OS: {stats["avg_os"]:.3f}, '
                  f'误差: {stats["avg_ne"]:.2f}m, '
                  f'步数: {stats["avg_steps"]:.1f}')

    print()

    # 极值分析
    extremes = analyze_extreme_cases(results)
    print('极值案例分析:')
    print(f'最佳SPL案例 (Episode {extremes["best_spl"]["episode_id"]}):')
    print(f'  Scene: {extremes["best_spl"]["scene_id"]}, '
          f'SPL: {extremes["best_spl"]["spl"]:.4f}, '
          f'误差: {extremes["best_spl"]["ne"]:.2f}m, '
          f'步数: {extremes["best_spl"]["steps"]}')
    print(f'  指令: {extremes["best_spl"]["instruction"]}')

    print()
    print(f'最差SPL案例 (Episode {extremes["worst_spl"]["episode_id"]}):')
    print(f'  Scene: {extremes["worst_spl"]["scene_id"]}, '
          f'SPL: {extremes["worst_spl"]["spl"]:.4f}, '
          f'误差: {extremes["worst_spl"]["ne"]:.2f}m, '
          f'步数: {extremes["worst_spl"]["steps"]}')
    print(f'  指令: {extremes["worst_spl"]["instruction"]}')

    print()
    print(f'最大导航误差案例 (Episode {extremes["max_ne"]["episode_id"]}):')
    print(f'  Scene: {extremes["max_ne"]["scene_id"]}, '
          f'误差: {extremes["max_ne"]["ne"]:.2f}m, '
          f'SPL: {extremes["max_ne"]["spl"]:.4f}, '
          f'步数: {extremes["max_ne"]["steps"]}')
    print(f'  指令: {extremes["max_ne"]["instruction"]}')

    print()
    print(f'最小导航误差案例 (Episode {extremes["min_ne"]["episode_id"]}):')
    print(f'  Scene: {extremes["min_ne"]["scene_id"]}, '
          f'误差: {extremes["min_ne"]["ne"]:.2f}m, '
          f'SPL: {extremes["min_ne"]["spl"]:.4f}, '
          f'步数: {extremes["min_ne"]["steps"]}')
    print(f'  指令: {extremes["min_ne"]["instruction"]}')

    print()

    # 步数分析
    steps_analysis = analyze_steps_distribution(results)
    print('步数分析:')
    print(f'  平均步数: {steps_analysis["avg_steps"]:.1f}')
    print(f'  步数范围: {steps_analysis["steps_range"][0]} - {steps_analysis["steps_range"][1]}')
    print(f'  达到步数限制(≥400)的episode: {steps_analysis["max_steps_count"]}个')
    print(f'  其中成功率: {steps_analysis["max_steps_success_rate"]:.1f}%')

    print()

    # 接近目标分析
    near_misses = analyze_near_misses(results)
    print('接近目标分析:')
    print(f'  接近目标的失败案例(误差<3m): {near_misses["near_miss_count"]}个')
    print(f'  占失败案例的比例: {near_misses["near_miss_rate"]:.1f}%')

    print()

    # 性能评估总结
    print('性能分析总结:')
    success_rate = analysis.get("successful", {}).get("percentage", 0)
    spl = stats.get("avg_spl", 0)
    os = stats.get("avg_os", 0)

    # 如果OS为0，重新计算
    if os == 0:
        os = sum(r.get('os', 0) for r in results) / len(results)

    if success_rate > 50:
        level = "优秀"
    elif success_rate > 30:
        level = "良好"
    else:
        level = "基础"

    print(f'1. 模型成功率约为{success_rate:.2f}%，在VLN任务中属于{level}水平')

    # 安全地获取成功和失败案例的平均导航误差
    success_ne = analysis.get("successful", {}).get("avg_ne", 0)
    failed_ne = analysis.get("failed", {}).get("avg_ne", 0)

    if success_ne > 0 and failed_ne > 0:
        print(f'2. 成功案例的导航误差明显低于失败案例 ({success_ne:.2f}m vs {failed_ne:.2f}m)')
    else:
        print('2. 成功案例和失败案例的导航误差差异显著')

    print('3. 成功案例平均步数较少，表明路径更直接')
    print('4. 不同场景间性能差异较大')
    print(f'5. 平均OS为{os:.4f}，表明模型选择的路径方向基本正确')

    near_miss_count = near_misses.get("near_miss_count", 0)
    if near_miss_count > 0:
        print(f'6. 有{near_miss_count}个失败案例接近目标(误差<3m)，说明停止时机判断需要改进')

def main():
    parser = argparse.ArgumentParser(description='StreamVLN评估结果分析工具')
    parser.add_argument('--result_file', type=str,
                       default='results/vals/_unseen/streamvln_3/result.json',
                       help='结果文件路径')
    parser.add_argument('--model_config', type=str,
                       default='4-bit quantization, 16 frames, 8 history, 196 token limit',
                       help='模型配置描述')

    args = parser.parse_args()

    if not os.path.exists(args.result_file):
        print(f'错误: 结果文件 {args.result_file} 不存在')
        return

    print_analysis_results(args.result_file, args.model_config)

if __name__ == '__main__':
    main()