#!/usr/bin/env python3
"""Parse log file and convert to CSV"""

import re
import csv

def parse_log_to_csv(log_file, output_csv):
    """Parse the log file and extract scene, episodes, and time data"""

    # Pattern: ✅ SceneID - N episodes | T.Ts
    pattern = r'✅\s+([A-Za-z0-9]+)\s+-\s+(\d+)\s+episodes\s+\|\s+([\d.]+)s'

    scenes = []

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                scene_id = match.group(1)
                episodes = int(match.group(2))
                time_sec = float(match.group(3))

                # Calculate episodes per second
                eps = episodes / time_sec if time_sec > 0 else 0

                scenes.append({
                    'scene_id': scene_id,
                    'episodes': episodes,
                    'time_seconds': time_sec,
                    'episodes_per_second': round(eps, 2)
                })

    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['scene_id', 'episodes', 'time_seconds', 'episodes_per_second']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for scene in scenes:
            writer.writerow(scene)

    return scenes


if __name__ == '__main__':
    log_file = '/root/workspace/StreamVLN/cloudrobo_output/log.txt'
    output_csv = '/root/workspace/StreamVLN/cloudrobo_output/scenes_stats.csv'

    scenes = parse_log_to_csv(log_file, output_csv)

    print(f"已处理 {len(scenes)} 个场景")
    print(f"CSV文件已保存到: {output_csv}")

    # Print summary statistics
    total_episodes = sum(s['episodes'] for s in scenes)
    total_time = sum(s['time_seconds'] for s in scenes)
    avg_eps = sum(s['episodes_per_second'] for s in scenes) / len(scenes)

    print(f"\n统计摘要:")
    print(f"  总场景数: {len(scenes)}")
    print(f"  总episodes: {total_episodes:,}")
    print(f"  总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"  平均处理速度: {avg_eps:.2f} episodes/秒")
