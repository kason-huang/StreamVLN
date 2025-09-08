#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import time
import os
import sys
from datetime import datetime

# --- 配置 ---
# 您要执行的命令
COMMAND = "pestat -G -w dgx[003-006,008,010,019,024,026,030-031,033-034,040-042,044-045,048,050-051,060,063,067]"

# 刷新间隔（秒）
REFRESH_INTERVAL = 43200

# 日志文件名
LOG_FILE = "node_monitor_history.log"

def clear_screen():
    """根据操作系统清空终端屏幕"""
    os.system('cls' if os.name == 'nt' else 'clear')

def run_and_parse():
    """执行命令并解析输出，返回节点总数、resv节点数和原始输出行 (带有增强的调试功能)"""
    print("--- [DEBUG] 正在执行 run_and_parse 函数 ---")
    try:
        result = subprocess.run(
            COMMAND,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        
        # [新增] 打印完整的原始输出，这是排查问题的关键！
        print("--- [DEBUG] pestat 命令返回的完整原始输出: ---")
        print(output)
        print("-------------------------------------------------")

    except subprocess.CalledProcessError as e:
        print(f"[错误] 命令执行失败: {e}")
        print(f"命令返回的错误信息 (stderr):\n{e.stderr}")
        return None, None, None, None
    except FileNotFoundError:
        print(f"[错误] 找不到命令 'pestat'。请确保该命令在您的 PATH 中。")
        sys.exit(1)

    total_nodes = 0
    resv_nodes = 0
    node_lines = []

    # 从原始输出中找到表头
    header = next((line for line in output.splitlines() if 'Hostname' in line and 'State' in line), "")
    if header:
        print(f"[DEBUG] 成功找到表头: {header}")
    else:
        print("[DEBUG] 未在输出中找到表头行。")

    print("[DEBUG] 开始逐行解析输出...")
    for line in output.splitlines():
        # strip() 移除前后的空格和换行符
        clean_line = line.strip()
        
        # 检查是否是有效的节点信息行
        if clean_line.startswith('dgx'):
            print(f"\n[DEBUG] 找到一个以 'dgx' 开头的行: '{clean_line}'")
            node_lines.append(line) # 保存原始行用于显示
            
            columns = clean_line.split()
            print(f"  [DEBUG] 按空格分割后的结果 (columns): {columns}")
            
            if len(columns) >= 3:
                total_nodes += 1
                state = columns[2]
                print(f"  [DEBUG] 识别到状态 (state): '{state}'")
                
                if 'resv*' in state:
                    resv_nodes += 1
                    print("    [DEBUG] >> 状态包含 'resv'，resv_nodes 计数器增加。")
            else:
                print(f"  [DEBUG] !! 警告: 该行分割后不足3列，已跳过。")
    
    # [新增] 循环结束后的最终诊断
    print("\n--- [DEBUG] 解析完成 ---")
    if not node_lines:
        print("[DEBUG] !! 重要警告: 未解析到任何有效的节点信息。请检查上面的 '完整原始输出'，确认是否存在以 'dgx' 开头的行。")
    else:
        print(f"[DEBUG] 共找到 {len(node_lines)} 条节点信息。")
    print("-------------------------\n")
        
    return total_nodes, resv_nodes, header, node_lines
def write_log(timestamp, current_perc, avg_perc):
    """将监控数据写入日志文件"""
    # 检查日志文件是否存在，如果不存在则写入表头
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w') as f:
            f.write("Timestamp,Current_Resv_Percentage,Average_Resv_Percentage\n")
    
    # 追加写入新的日志条目
    with open(LOG_FILE, 'a') as f:
        f.write(f"{timestamp},{current_perc:.2f},{avg_perc:.2f}\n")

def main():
    """主循环，用于持续监控和记录日志"""
    # 用于计算平均值的累积变量
    total_checks = 0
    cumulative_percentage = 0.0

    try:
        while True:
            # 获取数据
            total_nodes, resv_nodes, header, node_lines = run_and_parse()
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

            clear_screen()
            print(f"--- DGX 节点状态监控 --- (日志保存在: {LOG_FILE})")
            print(f"最后更新: {timestamp} | 监控命令: {COMMAND}\n")
            
            # 打印节点详情
            print("--- 节点详细状态 ---")
            if header:
                print(header)
            if node_lines:
                for line in node_lines:
                    print(line)
            print("-" * 20 + "\n")
            
            # 计算和显示摘要
            print("--- 状态摘要 ---")
            if total_nodes is not None and total_nodes > 0:
                # 1. 计算当前比例
                current_percentage = (resv_nodes / total_nodes) * 100
                
                # 2. 更新并计算平均比例
                total_checks += 1
                cumulative_percentage += current_percentage
                average_percentage = cumulative_percentage / total_checks

                print(f"监控的节点总数: {total_nodes}")
                print(f"处于 'resv' 状态的节点数: {resv_nodes}")
                print(f"当前 resv 状态节点比例: {current_percentage:.2f}%")
                print(f"平均 resv 状态节点比例: {average_percentage:.2f}% (基于 {total_checks} 次检测)")

                # 3. 写入日志
                write_log(timestamp, current_percentage, average_percentage)
                
            else:
                print("本次检测未能获取到有效的节点数据。")
            print("-" * 16)

            # 等待刷新
            time.sleep(REFRESH_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n监控脚本已停止。日志已保存在 {LOG_FILE}。")
        sys.exit(0)
    except Exception as e:
        print(f"\n发生未知错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()