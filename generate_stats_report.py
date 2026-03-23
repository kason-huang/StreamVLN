#!/usr/bin/env python3
"""Generate comprehensive statistics report from log file"""

import re
import csv
from collections import defaultdict

def parse_log_data(log_file):
    """Parse the log file and extract all data"""
    pattern = r'✅\s+([A-Za-z0-9]+)\s+-\s+(\d+)\s+episodes\s+\|\s+([\d.]+)s'

    scenes = []

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                scene_id = match.group(1)
                episodes = int(match.group(2))
                time_sec = float(match.group(3))
                eps = episodes / time_sec if time_sec > 0 else 0

                scenes.append({
                    'scene_id': scene_id,
                    'episodes': episodes,
                    'time_seconds': time_sec,
                    'episodes_per_second': eps
                })

    return scenes


def generate_statistics(scenes):
    """Calculate statistics"""
    episodes_list = [s['episodes'] for s in scenes]
    time_list = [s['time_seconds'] for s in scenes]
    eps_list = [s['episodes_per_second'] for s in scenes]

    total_episodes = sum(episodes_list)
    total_time = sum(time_list)

    stats = {
        'total_scenes': len(scenes),
        'total_episodes': total_episodes,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'total_time_hours': total_time / 3600,
        'avg_episodes': sum(episodes_list) / len(episodes_list),
        'avg_time': sum(time_list) / len(time_list),
        'avg_eps': sum(eps_list) / len(eps_list),
        'max_episodes': max(episodes_list),
        'min_episodes': min(episodes_list),
        'median_episodes': sorted(episodes_list)[len(episodes_list) // 2],
        'max_time': max(time_list),
        'min_time': min(time_list),
        'max_eps': max(eps_list),
        'min_eps': min(eps_list),
    }

    # Top and bottom scenes
    top_by_episodes = sorted(scenes, key=lambda x: x['episodes'], reverse=True)[:10]
    bottom_by_episodes = sorted(scenes, key=lambda x: x['episodes'])[:10]
    top_by_time = sorted(scenes, key=lambda x: x['time_seconds'], reverse=True)[:10]
    top_by_eps = sorted(scenes, key=lambda x: x['episodes_per_second'], reverse=True)[:10]

    # Episode distribution
    distribution = {
        '0': sum(1 for e in episodes_list if e == 0),
        '1-100': sum(1 for e in episodes_list if 1 <= e <= 100),
        '101-500': sum(1 for e in episodes_list if 101 <= e <= 500),
        '501-1000': sum(1 for e in episodes_list if 501 <= e <= 1000),
        '1001-5000': sum(1 for e in episodes_list if 1001 <= e <= 5000),
        '5001-10000': sum(1 for e in episodes_list if 5001 <= e <= 10000),
        '10001-15000': sum(1 for e in episodes_list if 10001 <= e <= 15000),
        '15001-20000': sum(1 for e in episodes_list if 15001 <= e <= 20000),
    }

    return stats, top_by_episodes, bottom_by_episodes, top_by_time, top_by_eps, distribution


def save_report_txt(output_path, stats, top_by_episodes, bottom_by_episodes,
                    top_by_time, top_by_eps, distribution):
    """Save statistics report as text file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("数据合并统计报告\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        f.write("【总体统计】\n")
        f.write("-" * 80 + "\n")
        f.write(f"  总场景数:                {stats['total_scenes']:,}\n")
        f.write(f"  总Episodes数:            {stats['total_episodes']:,}\n")
        f.write(f"  总耗时:                  {stats['total_time_seconds']:.1f}秒\n")
        f.write(f"                           {stats['total_time_minutes']:.1f}分钟\n")
        f.write(f"                           {stats['total_time_hours']:.2f}小时\n")
        f.write(f"  平均每场景Episodes:      {stats['avg_episodes']:.1f}\n")
        f.write(f"  平均处理耗时:            {stats['avg_time']:.1f}秒\n")
        f.write(f"  平均处理速度:            {stats['avg_eps']:.2f} episodes/秒\n")
        f.write(f"  最大Episodes数:          {stats['max_episodes']:,}\n")
        f.write(f"  最小Episodes数:          {stats['min_episodes']:,}\n")
        f.write(f"  中位数Episodes:          {stats['median_episodes']:,}\n")
        f.write(f"  最长处理时间:            {stats['max_time']:.1f}秒 ({stats['max_time']/60:.1f}分钟)\n")
        f.write(f"  最短处理时间:            {stats['min_time']:.1f}秒\n")
        f.write(f"  最高处理速度:            {stats['max_eps']:.2f} episodes/秒\n")
        f.write(f"  最低处理速度:            {stats['min_eps']:.2f} episodes/秒\n")
        f.write("\n")

        # Episode distribution
        f.write("【Episodes分布】\n")
        f.write("-" * 80 + "\n")
        total = stats['total_scenes']
        for range_name, count in distribution.items():
            percentage = (count / total) * 100 if total > 0 else 0
            bar = '█' * int(percentage / 2)
            f.write(f"  {range_name:>12}: {count:4d} ({percentage:5.1f}%) {bar}\n")
        f.write("\n")

        # Top scenes by episodes
        f.write("【Top 10 场景 (Episodes最多)】\n")
        f.write("-" * 80 + "\n")
        f.write(f"  {'排名':<6} {'场景ID':<20} {'Episodes':<15} {'耗时(秒)':<12} {'速度(eps/s)'}\n")
        f.write("  " + "-" * 80 + "\n")
        for i, s in enumerate(top_by_episodes, 1):
            percentage = (s['episodes'] / stats['total_episodes']) * 100
            f.write(f"  {i:<6} {s['scene_id']:<20} {s['episodes']:>15,} "
                   f"{s['time_seconds']:>12.1f} {s['episodes_per_second']:>10.2f} "
                   f"({percentage:5.2f}%)\n")
        f.write("\n")

        # Bottom scenes by episodes
        f.write("【Bottom 10 场景 (Episodes最少)】\n")
        f.write("-" * 80 + "\n")
        f.write(f"  {'排名':<6} {'场景ID':<20} {'Episodes':<15} {'耗时(秒)':<12} {'速度(eps/s)'}\n")
        f.write("  " + "-" * 80 + "\n")
        for i, s in enumerate(bottom_by_episodes, 1):
            f.write(f"  {i:<6} {s['scene_id']:<20} {s['episodes']:>15,} "
                   f"{s['time_seconds']:>12.1f} {s['episodes_per_second']:>10.2f}\n")
        f.write("\n")

        # Top scenes by time
        f.write("【Top 10 场景 (耗时最长)】\n")
        f.write("-" * 80 + "\n")
        f.write(f"  {'排名':<6} {'场景ID':<20} {'Episodes':<15} {'耗时(秒)':<15} {'耗时(分钟)'}\n")
        f.write("  " + "-" * 80 + "\n")
        for i, s in enumerate(top_by_time, 1):
            f.write(f"  {i:<6} {s['scene_id']:<20} {s['episodes']:>15,} "
                   f"{s['time_seconds']:>15.1f} {s['time_seconds']/60:>10.1f}\n")
        f.write("\n")

        # Top scenes by processing speed
        f.write("【Top 10 场景 (处理速度最快)】\n")
        f.write("-" * 80 + "\n")
        f.write(f"  {'排名':<6} {'场景ID':<20} {'Episodes':<15} {'耗时(秒)':<12} {'速度(eps/s)'}\n")
        f.write("  " + "-" * 80 + "\n")
        for i, s in enumerate(top_by_eps, 1):
            f.write(f"  {i:<6} {s['scene_id']:<20} {s['episodes']:>15,} "
                   f"{s['time_seconds']:>12.1f} {s['episodes_per_second']:>10.2f}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")


def save_csv(output_csv, scenes):
    """Save detailed data to CSV"""
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['scene_id', 'episodes', 'time_seconds', 'time_minutes',
                      'episodes_per_second']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for scene in scenes:
            writer.writerow({
                'scene_id': scene['scene_id'],
                'episodes': scene['episodes'],
                'time_seconds': round(scene['time_seconds'], 2),
                'time_minutes': round(scene['time_seconds'] / 60, 2),
                'episodes_per_second': round(scene['episodes_per_second'], 2)
            })


if __name__ == '__main__':
    log_file = '/root/workspace/StreamVLN/cloudrobo_output/log.txt'
    output_dir = '/root/workspace/StreamVLN/cloudrobo_output'

    # Parse log
    scenes = parse_log_data(log_file)

    # Generate statistics
    stats, top_by_episodes, bottom_by_episodes, top_by_time, top_by_eps, distribution = \
        generate_statistics(scenes)

    # Save report
    report_txt = f'{output_dir}/statistics_report.txt'
    save_report_txt(report_txt, stats, top_by_episodes, bottom_by_episodes,
                    top_by_time, top_by_eps, distribution)
    print(f"报告已保存到: {report_txt}")

    # Save CSV
    csv_file = f'{output_dir}/scenes_stats.csv'
    save_csv(csv_file, scenes)
    print(f"CSV数据已保存到: {csv_file}")

    # Print summary
    print(f"\n处理完成!")
    print(f"  总场景数: {stats['total_scenes']}")
    print(f"  总episodes: {stats['total_episodes']:,}")
    print(f"  总耗时: {stats['total_time_hours']:.2f}小时")
