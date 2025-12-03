"""
Object Navigation Evaluation for StreamVLN
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple
from collections import defaultdict

class ObjectNavMetrics:
    """
    ObjectNav专用评估指标
    """

    def __init__(self, success_threshold=0.5, object_detection_threshold=0.7):
        self.success_threshold = success_threshold  # 距离目标物体的阈值
        self.object_detection_threshold = object_detection_threshold  # 物体检测置信度阈值
        self.reset()

    def reset(self):
        """重置所有指标"""
        self.episodes = []
        self.total_episodes = 0
        self.successful_episodes = 0
        self.total_path_length = 0.0
        self.oracle_path_length = 0.0
        self.total_steps = 0
        self.success_steps = 0
        self.total_navigation_error = 0.0
        self.object_detections = 0
        self.correct_object_detections = 0

    def add_episode(self, episode_data: Dict):
        """
        添加一个episode的数据
        episode_data: {
            'id': int,
            'object_category': str,
            'target_location': [x, y, z],
            'predicted_actions': List[int],
            'final_position': [x, y, z],
            'path_length': float,
            'oracle_path_length': float,
            'object_detections': List[{'category': str, 'confidence': float, 'position': [x, y, z]}]
        }
        """
        self.episodes.append(episode_data)
        self.total_episodes += 1

        # 计算各种指标
        success = self.compute_success(episode_data)
        navigation_error = self.compute_navigation_error(episode_data)
        object_success = self.compute_object_success(episode_data)

        if success:
            self.successful_episodes += 1
            self.success_steps += len(episode_data['predicted_actions'])

        self.total_path_length += episode_data['path_length']
        self.oracle_path_length += episode_data['oracle_path_length']
        self.total_steps += len(episode_data['predicted_actions'])
        self.total_navigation_error += navigation_error

        # 物体检测统计
        detections = episode_data.get('object_detections', [])
        if detections:
            self.object_detections += len(detections)
            for detection in detections:
                if (detection['category'] == episode_data['object_category'] and
                    detection['confidence'] >= self.object_detection_threshold):
                    self.correct_object_detections += 1

    def compute_success(self, episode_data: Dict) -> bool:
        """
        计算基础成功指标：是否到达目标物体附近
        """
        try:
            final_position = np.array(episode_data['final_position'])
            target_position = np.array(episode_data['target_location'])
            distance = np.linalg.norm(final_position - target_position)
            return distance < self.success_threshold
        except:
            return False

    def compute_navigation_error(self, episode_data: Dict) -> float:
        """
        计算导航误差：到目标物体的距离
        """
        try:
            final_position = np.array(episode_data['final_position'])
            target_position = np.array(episode_data['target_location'])
            return np.linalg.norm(final_position - target_position)
        except:
            return float('inf')

    def compute_object_success(self, episode_data: Dict) -> bool:
        """
        计算物体成功指标：空间成功 + 物体识别成功
        """
        spatial_success = self.compute_success(episode_data)

        # 检查是否正确识别了目标物体
        detections = episode_data.get('object_detections', [])
        object_detection_success = False

        for detection in detections:
            if (detection['category'] == episode_data['object_category'] and
                detection['confidence'] >= self.object_detection_threshold):
                object_detection_success = True
                break

        return spatial_success and object_detection_success

    def compute_spl(self) -> float:
        """
        计算SPL (Success weighted by Path Length)
        """
        if self.total_episodes == 0:
            return 0.0

        spl_sum = 0.0
        for episode in self.episodes:
            success = self.compute_success(episode)
            if success:
                path_length = episode['path_length']
                oracle_length = episode['oracle_path_length']
                if oracle_length > 0:
                    spl_sum += oracle_length / max(path_length, oracle_length)

        return spl_sum / self.total_episodes

    def compute_success_rate(self) -> float:
        """
        计算成功率
        """
        if self.total_episodes == 0:
            return 0.0
        return self.successful_episodes / self.total_episodes

    def compute_object_finding_rate(self) -> float:
        """
        计算物体发现率
        """
        successful_episodes = 0
        for episode in self.episodes:
            if self.compute_object_success(episode):
                successful_episodes += 1

        if self.total_episodes == 0:
            return 0.0
        return successful_episodes / self.total_episodes

    def compute_time_to_success(self) -> float:
        """
        计算平均成功时间（步数）
        """
        if self.successful_episodes == 0:
            return float('inf')
        return self.success_steps / self.successful_episodes

    def compute_navigation_error(self) -> float:
        """
        计算平均导航误差
        """
        if self.total_episodes == 0:
            return float('inf')
        return self.total_navigation_error / self.total_episodes

    def compute_object_detection_accuracy(self) -> float:
        """
        计算物体检测准确率
        """
        if self.object_detections == 0:
            return 0.0
        return self.correct_object_detections / self.object_detections

    def get_metrics(self) -> Dict:
        """
        返回所有评估指标
        """
        return {
            'total_episodes': self.total_episodes,
            'success_rate': self.compute_success_rate(),
            'spl': self.compute_spl(),
            'object_finding_rate': self.compute_object_finding_rate(),
            'time_to_success': self.compute_time_to_success(),
            'navigation_error': self.compute_navigation_error(),
            'object_detection_accuracy': self.compute_object_detection_accuracy(),
            'average_path_length': self.total_path_length / max(self.total_episodes, 1),
            'average_steps': self.total_steps / max(self.total_episodes, 1)
        }

    def print_metrics(self):
        """
        打印评估结果
        """
        metrics = self.get_metrics()

        print("\n" + "="*60)
        print("Object Navigation Evaluation Results")
        print("="*60)
        print(f"Total Episodes: {metrics['total_episodes']}")
        print(f"Success Rate: {metrics['success_rate']:.2%}")
        print(f"SPL: {metrics['spl']:.4f}")
        print(f"Object Finding Rate: {metrics['object_finding_rate']:.2%}")

        if metrics['time_to_success'] != float('inf'):
            print(f"Time to Success: {metrics['time_to_success']:.2f} steps")
        else:
            print("Time to Success: N/A (no successful episodes)")

        if metrics['navigation_error'] != float('inf'):
            print(f"Navigation Error: {metrics['navigation_error']:.3f} meters")
        else:
            print("Navigation Error: N/A")

        print(f"Object Detection Accuracy: {metrics['object_detection_accuracy']:.2%}")
        print(f"Average Path Length: {metrics['average_path_length']:.2f} meters")
        print(f"Average Steps: {metrics['average_steps']:.2f}")
        print("="*60)

    def save_metrics(self, filepath: str):
        """
        保存评估结果到文件
        """
        metrics = self.get_metrics()

        # 添加额外信息
        metrics.update({
            'success_threshold': self.success_threshold,
            'object_detection_threshold': self.object_detection_threshold,
            'evaluation_timestamp': str(np.datetime64('now'))
        })

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to: {filepath}")


class ObjectNavEvaluator:
    """
    ObjectNav评估器主类
    """

    def __init__(self, config: Dict):
        self.config = config
        self.metrics = ObjectNavMetrics(
            success_threshold=config.get('success_distance', 0.5),
            object_detection_threshold=config.get('object_detection_threshold', 0.7)
        )

    def evaluate_episode(self, episode_data: Dict) -> Dict:
        """
        评估单个episode
        """
        self.metrics.add_episode(episode_data)

        return {
            'success': self.metrics.compute_success(episode_data),
            'object_success': self.metrics.compute_object_success(episode_data),
            'navigation_error': self.metrics.compute_navigation_error(episode_data),
            'path_length': episode_data.get('path_length', 0),
            'steps': len(episode_data.get('predicted_actions', []))
        }

    def evaluate_dataset(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        评估整个数据集
        """
        # 合并预测和真实数据
        for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            episode_data = {
                'id': gt.get('id', i),
                'object_category': gt.get('object_category', ''),
                'target_location': gt.get('target_location', [0, 0, 0]),
                'predicted_actions': pred.get('actions', []),
                'final_position': pred.get('final_position', [0, 0, 0]),
                'path_length': pred.get('path_length', 0),
                'oracle_path_length': gt.get('oracle_path_length', 1),
                'object_detections': pred.get('object_detections', [])
            }

            self.evaluate_episode(episode_data)

        return self.metrics.get_metrics()

    def evaluate_from_file(self, predictions_file: str, ground_truth_file: str, output_file: str = None):
        """
        从文件加载并评估
        """
        # 加载数据
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)

        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)

        # 评估
        metrics = self.evaluate_dataset(predictions, ground_truth)

        # 打印结果
        self.metrics.print_metrics()

        # 保存结果
        if output_file:
            self.metrics.save_metrics(output_file)

        return metrics


def evaluate_objectnav(model, dataloader, device, config):
    """
    运行ObjectNav评估
    """
    model.eval()
    evaluator = ObjectNavEvaluator(config)

    predictions = []
    ground_truth = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 获取batch数据
            input_ids = batch['input_ids'].to(device)
            images = batch['images'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # 模型预测
            outputs = model(
                input_ids=input_ids,
                images=images,
                attention_mask=attention_mask
            )

            # 处理预测结果
            predicted_actions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            # 保存结果
            for i in range(len(batch['task_type'])):
                pred_data = {
                    'episode_id': batch_idx,
                    'actions': predicted_actions[i].tolist(),
                    'final_position': [0, 0, 0],  # 需要从轨迹计算
                    'path_length': 0,  # 需要从轨迹计算
                    'object_detections': []  # 需要实现物体检测
                }

                gt_data = {
                    'id': batch_idx,
                    'object_category': 'unknown',  # 需要从数据获取
                    'target_location': [0, 0, 0],
                    'oracle_path_length': 1.0
                }

                predictions.append(pred_data)
                ground_truth.append(gt_data)

    # 评估
    metrics = evaluator.evaluate_dataset(predictions, ground_truth)
    return metrics


if __name__ == "__main__":
    # 示例使用
    config = {
        'success_distance': 0.5,
        'object_detection_threshold': 0.7
    }

    # 创建示例数据
    predictions = [
        {
            'actions': [1, 3, 1, 0],
            'final_position': [2.1, 0.0, 1.8],
            'path_length': 1.5,
            'object_detections': [
                {'category': 'chair', 'confidence': 0.8, 'position': [2.0, 0.0, 1.7]}
            ]
        }
    ]

    ground_truth = [
        {
            'id': 1,
            'object_category': 'chair',
            'target_location': [2.0, 0.0, 1.7],
            'oracle_path_length': 1.0
        }
    ]

    # 运行评估
    evaluator = ObjectNavEvaluator(config)
    metrics = evaluator.evaluate_dataset(predictions, ground_truth)
    evaluator.metrics.print_metrics()