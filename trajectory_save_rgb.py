import os
import json
import sys
import time
import threading
import habitat

from habitat_baselines.config.default import get_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from streamvln.habitat_extensions import measures


class ProgressMonitor:
    """进度监控器，每20秒打印一次处理速度"""
    def __init__(self, total_episodes, description="episodes", log_file="log.txt"):
        self.total_episodes = total_episodes
        self.description = description
        self.log_file = log_file
        self.processed_count = 0
        self.start_time = time.time()
        self.last_print_time = time.time()
        self.last_processed_count = 0
        self.running = True
        self.lock = threading.Lock()

        # 清空日志文件并写入开始信息
        with open(self.log_file, 'w') as f:
            f.write(f"开始处理 {total_episodes:,} {description}\n")
            f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 60 + "\n")

        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_progress, daemon=True)
        self.monitor_thread.start()

    def increment(self, count=1):
        """增加已处理的数量"""
        with self.lock:
            self.processed_count += count

    def _log_message(self, message):
        """将消息写入日志文件"""
        with open(self.log_file, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    def _monitor_progress(self):
        """监控进度的线程函数"""
        while self.running:
            time.sleep(20)  # 每20秒打印一次

            with self.lock:
                current_time = time.time()
                time_elapsed = current_time - self.last_print_time
                total_time_elapsed = current_time - self.start_time

                if time_elapsed > 0:
                    # 最近20秒的处理速度
                    recent_rate = (self.processed_count - self.last_processed_count) / time_elapsed
                    # 总体处理速度
                    overall_rate = self.processed_count / total_time_elapsed if total_time_elapsed > 0 else 0

                    progress_percent = (self.processed_count / self.total_episodes) * 100 if self.total_episodes > 0 else 0

                    # 写入日志文件
                    self._log_message(f"[速度统计] 已处理: {self.processed_count:,}/{self.total_episodes:,} {self.description} ({progress_percent:.1f}%)")
                    self._log_message(f"[速度统计] 最近20秒速度: {recent_rate:.2f} {self.description}/秒")
                    self._log_message(f"[速度统计] 总体平均速度: {overall_rate:.2f} {self.description}/秒")

                    if overall_rate > 0:
                        eta_seconds = (self.total_episodes - self.processed_count) / overall_rate
                        eta_minutes = eta_seconds / 60
                        eta_hours = eta_minutes / 60
                        if eta_hours > 1:
                            self._log_message(f"[速度统计] 预计剩余时间: {eta_hours:.1f} 小时")
                        elif eta_minutes > 1:
                            self._log_message(f"[速度统计] 预计剩余时间: {eta_minutes:.1f} 分钟")
                        else:
                            self._log_message(f"[速度统计] 预计剩余时间: {eta_seconds:.1f} 秒")

                    self._log_message("-" * 40)

                    self.last_print_time = current_time
                    self.last_processed_count = self.processed_count

    def stop(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)

        # 写入完成信息
        with open(self.log_file, 'a') as f:
            f.write(f"\n处理完成！总共处理了 {self.processed_count:,} {self.description}\n")
            f.write(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            total_time = time.time() - self.start_time
            hours = total_time / 3600
            minutes = (total_time % 3600) / 60
            f.write(f"总耗时: {hours:.1f}小时 {minutes:.1f}分钟\n")
            f.write(f"平均速度: {self.processed_count / total_time:.2f} {self.description}/秒\n")


def save_rgb(annot_path, config_path):
    CONFIG_PATH = config_path  # Path to the Habitat config file
    ANNOT_PATH = annot_path  # Path to the annotations file
    GOAL_RADIUS = 0.25  # Radius for the goal in meters. not used if get actions from annotations

    # 创建日志目录和文件名
    log_dir = "data1/trajectory_data/EnvDrop/tmp"
    os.makedirs(log_dir, exist_ok=True)
    config_filename = os.path.splitext(os.path.basename(config_path))[0]
    log_file = os.path.join(log_dir, f"{config_filename}.log")

    env = habitat.Env(config=get_config(CONFIG_PATH))
    annotations = json.load(open(ANNOT_PATH, "r"))

    # 创建进度监控器
    total_episodes = len(env.episodes)
    monitor = ProgressMonitor(total_episodes, "episodes", log_file)

    print(f"开始处理 {total_episodes:,} 个episodes，监控日志将保存到 {log_file}")

    try:
        for i, episode in enumerate(env.episodes):
            env.current_episode = episode
            agent = ShortestPathFollower(sim=env.sim, goal_radius=GOAL_RADIUS, return_one_hot=False)
            observation = env.reset()

            annotation = next(annot for annot in annotations if annot["id"] == int(episode.episode_id))  # Get annotation for current episode
            reference_actions = annotation["actions"][1:] + [0]  # Pop the dummy action at the beginning and add stop action at the end
            step_id = 0  # Initialize step ID

            while not env.episode_over:
                rgb = observation["rgb"]  # Get the current rgb observation

                # TODO: Save RGB frame (customize as needed)
                # --------------------------------------------------------
                import PIL.Image as Image
                video_id = annotation["video"]  # Get the video ID from the annotation
                rgb_dir = f"data1/trajectory_data/EnvDrop/{video_id}/rgb"
                os.makedirs(rgb_dir, exist_ok=True)
                Image.fromarray(rgb).convert("RGB").save(os.path.join(rgb_dir, f"{step_id:03d}.jpg"))
                # --------------------------------------------------------

                action = reference_actions.pop(0)  # Get next action from our annotation
                observation = env.step(action)  # Update observation
                step_id += 1

            # 更新进度 - 每完成一个episode
            monitor.increment()

    finally:
        # 确保监控器停止
        monitor.stop()

    env.close()

    print(f"处理完成！详细日志请查看 {log_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        annot_path = sys.argv[1]  # 获取第一个参数
        config_path = sys.argv[2]

    else:
       annot_path = "data/trajectory_data/EnvDrop/annotations.json" 
       config_path = "config/vln_r2r.yaml"
    save_rgb(annot_path, config_path)
