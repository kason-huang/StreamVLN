#!/usr/bin/env python3
"""
从训练日志中提取loss数据并绘制曲线
使用正则表达式处理单引号格式的数据
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置matplotlib使用非交互式后端（适用于服务器环境）
import matplotlib
matplotlib.use('Agg')

def extract_loss_data(log_file):
    """
    从日志文件中提取loss数据
    """
    iterations = []
    losses = []

    with open(log_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # 使用正则表达式提取loss值
            # 匹配格式: 'loss': 数字
            loss_match = re.search(r"'loss':\s*([0-9.]+)", line)
            if loss_match:
                loss_value = float(loss_match.group(1))
                iterations.append(i + 1)
                losses.append(loss_value)

    return iterations, losses

def plot_loss_curve(iterations, losses, output_file='streamvln_loss_curve.png'):
    """
    绘制loss变化曲线并保存为图片
    """
    plt.figure(figsize=(12, 8))

    # 绘制原始loss曲线
    plt.plot(iterations, losses, 'b-', linewidth=1, alpha=0.6, label='Raw Loss')

    # 绘制平滑曲线（使用移动平均）
    if len(losses) > 10:
        window_size = min(50, len(losses) // 10)
        if window_size > 1:
            smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            smoothed_iterations = iterations[:len(smoothed_losses)]
            plt.plot(smoothed_iterations, smoothed_losses, 'r-', linewidth=2.5,
                    label=f'Smoothed Loss (window={window_size})')

    plt.title('StreamVLN Training Loss vs Iterations', fontsize=16, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 设置y轴范围，突出显示loss的变化
    min_loss = min(losses)
    max_loss = max(losses)
    margin = (max_loss - min_loss) * 0.05
    plt.ylim(min_loss - margin, max_loss + margin)

    plt.tight_layout()

    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Loss曲线图已保存到: {output_file}")

    plt.close()

def print_loss_stats(iterations, losses):
    """
    打印loss统计信息
    """
    print("=" * 60)
    print("StreamVLN训练Loss统计信息")
    print("=" * 60)
    print(f"总迭代次数: {len(iterations)}")
    print(f"初始Loss: {losses[0]:.4f}")
    print(f"最终Loss: {losses[-1]:.4f}")
    print(f"最小Loss: {min(losses):.4f}")
    print(f"最大Loss: {max(losses):.4f}")
    print(f"平均Loss: {np.mean(losses):.4f}")
    print(f"Loss下降幅度: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")

    # 计算学习最稳定区间（loss变化最小的连续区间）
    min_variance = float('inf')
    best_window_start = 0
    best_window_size = 0

    for window_size in range(10, min(50, len(losses) // 4)):
        for i in range(len(losses) - window_size + 1):
            window_losses = losses[i:i + window_size]
            variance = np.var(window_losses)
            if variance < min_variance:
                min_variance = variance
                best_window_start = i
                best_window_size = window_size

    print(f"\n最稳定的训练区间: 迭代{best_window_start+1}-{best_window_start+best_window_size}")
    print(f"  区间平均Loss: {np.mean(losses[best_window_start:best_window_start+best_window_size]):.4f}")
    print(f"  区间Loss方差: {min_variance:.6f}")
    print("=" * 60)

def main():
    # 设置文件路径
    log_file = 'results/vals/_unseen/issue_27/streamvln-12670.out'

    # 检查文件是否存在
    if not os.path.exists(log_file):
        print(f"错误: 找不到文件 {log_file}")
        return

    print(f"正在分析日志文件: {log_file}")

    # 提取loss数据
    iterations, losses = extract_loss_data(log_file)

    if not losses:
        print("错误: 未找到loss数据")
        return

    print(f"成功提取 {len(losses)} 个loss数据点")

    # 打印统计信息
    print_loss_stats(iterations, losses)

    # 绘制并保存loss曲线
    output_file = 'streamvln_loss_curve.png'
    plot_loss_curve(iterations, losses, output_file)

    print(f"\n分析完成! 图片已保存为: {output_file}")

if __name__ == "__main__":
    main()