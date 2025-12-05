#!/usr/bin/env python3
"""
从训练日志中提取多个指标数据并绘制变化图
提取loss、grad_norm、learning_rate三个指标
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FuncFormatter

# 设置matplotlib使用非交互式后端（适用于服务器环境）
import matplotlib
matplotlib.use('Agg')

def extract_metrics_data(log_file):
    """
    从日志文件中提取loss、grad_norm、learning_rate数据
    """
    iterations = []
    losses = []
    grad_norms = []
    learning_rates = []

    with open(log_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # 使用正则表达式提取各个指标
            loss_match = re.search(r"'loss':\s*([0-9.]+)", line)
            grad_match = re.search(r"'grad_norm':\s*([0-9.]+)", line)
            lr_match = re.search(r"'learning_rate':\s*([0-9.e-]+)", line)

            if loss_match:
                iterations.append(i + 1)
                losses.append(float(loss_match.group(1)))

                # 如果grad_norm存在，提取它
                if grad_match:
                    grad_norms.append(float(grad_match.group(1)))
                else:
                    grad_norms.append(0.0)

                # 如果learning_rate存在，提取它
                if lr_match:
                    learning_rates.append(float(lr_match.group(1)))
                else:
                    learning_rates.append(0.0)

    return iterations, losses, grad_norms, learning_rates

def plot_training_metrics(iterations, losses, grad_norms, learning_rates, output_file='streamvln_training_metrics.png'):
    """
    绘制多个训练指标的变化曲线
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('StreamVLN Training Metrics Analysis', fontsize=18, fontweight='bold')

    # 1. Loss变化图
    axes[0, 0].plot(iterations, losses, 'b-', linewidth=1, alpha=0.7, label='Raw Loss')

    # 添加平滑曲线
    if len(losses) > 20:
        window_size = min(50, len(losses) // 10)
        smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        smoothed_iterations = iterations[:len(smoothed_losses)]
        axes[0, 0].plot(smoothed_iterations, smoothed_losses, 'r-', linewidth=2.5,
                        label=f'Smoothed Loss (window={window_size})')

    axes[0, 0].set_title('Loss vs Iterations', fontweight='bold', fontsize=14)
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. Learning Rate变化图
    axes[0, 1].plot(iterations, learning_rates, 'g-', linewidth=2, alpha=0.8)
    axes[0, 1].set_title('Learning Rate vs Iterations', fontweight='bold', fontsize=14)
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[0, 1].set_yscale('log')  # 使用对数坐标显示学习率
    axes[0, 1].grid(True, alpha=0.3)

    # 格式化y轴标签
    def format_scientific(x, p):
        return f'{x:.1e}'
    axes[0, 1].yaxis.set_major_formatter(FuncFormatter(format_scientific))

    # 3. Gradient Norm变化图
    axes[1, 0].plot(iterations, grad_norms, 'm-', linewidth=1.5, alpha=0.8, label='Gradient Norm')

    # 添加平滑曲线
    if len(grad_norms) > 20:
        window_size = min(30, len(grad_norms) // 10)
        smoothed_grads = np.convolve(grad_norms, np.ones(window_size)/window_size, mode='valid')
        smoothed_iterations = iterations[:len(smoothed_grads)]
        axes[1, 0].plot(smoothed_iterations, smoothed_grads, 'orange', linewidth=2.5,
                        label=f'Smoothed Grad Norm (window={window_size})')

    axes[1, 0].set_title('Gradient Norm vs Iterations', fontweight='bold', fontsize=14)
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Gradient Norm', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 4. Loss与Gradient Norm的关系图（使用双y轴）
    ax2 = axes[1, 1]
    ax1 = ax2.twinx()

    # 绘制loss
    line1 = ax1.plot(iterations, losses, 'b-', linewidth=1.5, alpha=0.7, label='Loss')
    ax1.set_ylabel('Loss', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')

    # 绘制gradient norm（缩放到合适的范围以便在同一图中显示）
    scaled_grad_norms = [gn / max(grad_norms) * max(losses) for gn in grad_norms]
    line2 = ax2.plot(iterations, scaled_grad_norms, 'm-', linewidth=1.5, alpha=0.7, label='Grad Norm (scaled)')
    ax2.set_ylabel('Gradient Norm (scaled)', color='m', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='m')

    ax2.set_title('Loss vs Gradient Norm (Correlation)', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')

    plt.tight_layout()

    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"训练指标图表已保存到: {output_file}")

    plt.close()

def print_comprehensive_stats(iterations, losses, grad_norms, learning_rates):
    """
    打印所有指标的统计信息
    """
    print("=" * 80)
    print("StreamVLN训练指标综合统计")
    print("=" * 80)
    print(f"总迭代次数: {len(iterations)}")
    print()

    print("Loss统计:")
    print(f"  初始Loss: {losses[0]:.4f}")
    print(f"  最终Loss: {losses[-1]:.4f}")
    print(f"  最小Loss: {min(losses):.4f}")
    print(f"  最大Loss: {max(losses):.4f}")
    print(f"  平均Loss: {np.mean(losses):.4f}")
    print(f"  Loss下降幅度: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")
    print()

    print("学习率统计:")
    print(f"  初始学习率: {learning_rates[0]:.2e}")
    print(f"  最终学习率: {learning_rates[-1]:.2e}")
    print(f"  最大学习率: {max(learning_rates):.2e}")
    print(f"  最小学习率: {min(learning_rates):.2e}")
    print(f"  学习率变化倍数: {max(learning_rates)/min(learning_rates):.2f}")
    print()

    print("梯度范数统计:")
    print(f"  平均梯度范数: {np.mean(grad_norms):.4f}")
    print(f"  最大梯度范数: {max(grad_norms):.4f}")
    print(f"  最小梯度范数: {min(grad_norms):.4f}")
    print(f"  梯度范数标准差: {np.std(grad_norms):.4f}")
    print()

    # 计算相关性
    loss_grad_corr = np.corrcoef(losses, grad_norms)[0, 1]
    loss_lr_corr = np.corrcoef(losses, [np.log(lr) for lr in learning_rates])[0, 1]
    print("相关性分析:")
    print(f"  Loss与梯度范数相关系数: {loss_grad_corr:.4f}")
    print(f"  Loss与对数学习率相关系数: {loss_lr_corr:.4f}")
    print("=" * 80)

def analyze_training_phases(iterations, losses, grad_norms, learning_rates):
    """
    分析训练阶段
    """
    print("\n训练阶段分析:")
    print("-" * 40)

    # 找到最稳定的训练区间
    min_variance = float('inf')
    best_window_start = 0
    best_window_size = 0

    for window_size in range(20, min(100, len(losses) // 4)):
        for i in range(len(losses) - window_size + 1):
            window_losses = losses[i:i + window_size]
            variance = np.var(window_losses)
            if variance < min_variance:
                min_variance = variance
                best_window_start = i
                best_window_size = window_size

    print(f"最稳定训练阶段: 迭代 {best_window_start+1}-{best_window_start+best_window_size}")
    print(f"  平均Loss: {np.mean(losses[best_window_start:best_window_start+best_window_size]):.4f}")
    print(f"  平均梯度范数: {np.mean(grad_norms[best_window_start:best_window_start+best_window_size]):.4f}")
    print(f"  平均学习率: {np.mean(learning_rates[best_window_start:best_window_start+best_window_size]):.2e}")

def main():
    # 设置文件路径
    log_file = 'results/vals/_unseen/issue_27/streamvln-12670.out'

    # 检查文件是否存在
    if not os.path.exists(log_file):
        print(f"错误: 找不到文件 {log_file}")
        return

    print(f"正在分析日志文件: {log_file}")

    # 提取数据
    iterations, losses, grad_norms, learning_rates = extract_metrics_data(log_file)

    if not losses:
        print("错误: 未找到数据")
        return

    print(f"成功提取 {len(losses)} 个数据点")

    # 打印统计信息
    print_comprehensive_stats(iterations, losses, grad_norms, learning_rates)

    # 分析训练阶段
    analyze_training_phases(iterations, losses, grad_norms, learning_rates)

    # 绘制并保存图表
    output_file = 'streamvln_training_metrics.png'
    plot_training_metrics(iterations, losses, grad_norms, learning_rates, output_file)

    print(f"\n分析完成! 图片已保存为: {output_file}")

if __name__ == "__main__":
    main()