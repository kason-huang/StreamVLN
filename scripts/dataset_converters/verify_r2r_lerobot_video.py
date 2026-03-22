#!/usr/bin/env python3
"""
R2R → LeRobot 数据验证脚本 (video dtype版本)

功能：
1. 验证 instruction 和 action 的一致性
2. 从mp4视频中提取帧并与原始R2R图片进行对比
3. 保存对比图片和详细的验证报告

用法:
    python scripts/dataset_converters/verify_r2r_lerobot_video.py
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import tempfile

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import pandas as pd
from tqdm import tqdm
from loguru import logger
import cv2


def load_r2r_annotations(r2r_data_dir: Path) -> List[Dict[str, Any]]:
    """加载 R2R annotations.json"""
    annotations_file = r2r_data_dir / "annotations.json"
    logger.info(f"加载 R2R annotations: {annotations_file}")
    with open(annotations_file, "r") as f:
        return json.load(f)


def load_r2r_image(r2r_data_dir: Path, video_path: str, frame_idx: int) -> np.ndarray:
    """
    从原始 R2R 数据加载图片

    Args:
        r2r_data_dir: R2R 数据根目录
        video_path: 图片目录相对路径 (e.g., "images/17DRP5sb8fy_r2r_000577")
        frame_idx: 帧索引

    Returns:
        numpy array (HWC), uint8
    """
    image_path = r2r_data_dir / video_path / "rgb" / f"{frame_idx:03d}.jpg"
    img = Image.open(image_path)
    return np.array(img, dtype=np.uint8)


def get_r2r_episode_data(ann: Dict[str, Any], r2r_data_dir: Path, instruction_idx: int) -> Dict[str, Any]:
    """
    获取原始 R2R episode 数据

    Args:
        ann: R2R annotation 字典
        r2r_data_dir: R2R 数据根目录
        instruction_idx: 指令索引

    Returns:
        包含 instruction, actions, images 的字典
    """
    video_path = ann["video"]
    instructions = ann["instructions"]
    actions = np.array(ann.get("actions", []), dtype=np.int64)

    # 获取指定指令
    if instruction_idx < len(instructions):
        instruction = instructions[instruction_idx]
    else:
        instruction = instructions[0] if instructions else "Navigation task"

    # 丢弃第一个 action (-1) 以对齐图片数量
    if len(actions) > 0 and actions[0] == -1:
        actions = actions[1:]

    # 加载图片
    images = []
    src_image_dir = r2r_data_dir / video_path / "rgb"
    image_files = sorted(src_image_dir.glob("*.jpg"))

    for img_path in image_files:
        img = Image.open(img_path)
        images.append(np.array(img, dtype=np.uint8))

    # 如果 actions 少于 images，重复最后一个 action（与转换脚本行为一致）
    if len(actions) < len(images):
        last_action = actions[-1] if len(actions) > 0 else -1
        actions = list(actions) + [last_action] * (len(images) - len(actions))
    else:
        actions = list(actions)

    return {
        "instruction": instruction,
        "actions": actions,
        "images": images,
        "num_frames": len(images),
        "video_path": video_path,
    }


def extract_frame_from_video(video_path: Path, frame_idx: int) -> np.ndarray:
    """
    从mp4视频中提取指定帧

    Args:
        video_path: mp4视频文件路径
        frame_idx: 帧索引

    Returns:
        numpy array (HWC), uint8
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"无法从视频中读取帧 {frame_idx}: {video_path}")

    # OpenCV读取的是BGR格式，转换为RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb.astype(np.uint8)


def get_lerobot_episode_data(dataset: LeRobotDataset, episode_index: int) -> Dict[str, Any]:
    """
    从 LeRobot 数据集获取 episode 数据

    Args:
        dataset: LeRobotDataset 实例
        episode_index: episode 索引

    Returns:
        包含 instruction, actions, images 的字典
    """
    # 加载 episodes metadata
    episodes_file = dataset.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_df = pd.read_parquet(episodes_file)

    # 找到对应的 episode
    ep_row = episodes_df[episodes_df['episode_index'] == episode_index]
    if len(ep_row) == 0:
        raise ValueError(f"Episode {episode_index} not found")

    start_idx = int(ep_row.iloc[0]['dataset_from_index'])
    end_idx = int(ep_row.iloc[0]['dataset_to_index'])

    instruction = None
    actions = []
    images = []

    # 加载info.json获取视频信息
    info_path = dataset.root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    video_key = "observation.images.rgb"
    video_info = info["features"][video_key]

    # 遍历帧
    for idx in range(start_idx, end_idx):
        sample = dataset[idx]

        # 获取 instruction（从第一帧）
        if instruction is None:
            task = sample.get("task")
            if isinstance(task, str):
                instruction = json.loads(task).get("instruction", "")

        # 获取 action
        action = sample.get("action")
        if hasattr(action, 'item'):
            action = action.item()
        actions.append(action)

        # 获取图片 - 对于video类型，从视频中提取帧
        # sample['observation.images.rgb'] 应该返回一个tensor
        img_tensor = sample['observation.images.rgb']

        # 转换为numpy数组
        img_np = img_tensor.numpy()
        # CHW to HWC
        img_hwc = np.transpose(img_np, (1, 2, 0))
        # 转换为 uint8
        if img_hwc.max() <= 1.0:
            img_hwc = (img_hwc * 255).astype(np.uint8)
        else:
            img_hwc = img_hwc.astype(np.uint8)
        images.append(img_hwc)

    return {
        "instruction": instruction,
        "actions": actions,
        "images": images,
        "num_frames": len(images),
    }


def compare_images(img1: np.ndarray, img2: np.ndarray) -> Dict[str, Any]:
    """
    对比两张图片

    Args:
        img1: 第一张图片 (HWC)
        img2: 第二张图片 (HWC)

    Returns:
        对比结果字典
    """
    if img1.shape != img2.shape:
        return {
            "match": False,
            "shape_match": False,
            "shape1": img1.shape,
            "shape2": img2.shape,
            "mse": None,
            "max_diff": None,
        }

    # 计算MSE (Mean Squared Error)
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)

    # 计算最大差异
    max_diff = np.max(np.abs(img1.astype(int) - img2.astype(int)))

    # 判断是否匹配（允许小的编码差异）
    match = bool(mse < 100)  # 允许一些编码损失，转为Python bool

    return {
        "match": match,
        "shape_match": True,
        "shape1": img1.shape,
        "shape2": img2.shape,
        "mse": float(mse),
        "max_diff": int(max_diff),
    }


def save_comparison_images(
    r2r_images: List[np.ndarray],
    lerobot_images: List[np.ndarray],
    output_dir: Path,
    episode_index: int,
    num_samples: int = None,
):
    """
    保存对比图片（原始 R2R vs LeRobot 解码后）

    每个保存两张图片并排：原始 | LeRobot

    Args:
        r2r_images: 原始 R2R 图片列表
        lerobot_images: LeRobot 解码后的图片列表
        output_dir: 输出目录
        episode_index: episode 索引
        num_samples: 保存的帧数（None 表示全部）
    """
    ep_dir = output_dir / f"episode_{episode_index:03d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    num_frames = min(len(r2r_images), len(lerobot_images))
    if num_samples is not None:
        num_frames = min(num_frames, num_samples)

    comparison_results = []

    for frame_idx in range(num_frames):
        r2r_img = r2r_images[frame_idx]
        lerobot_img = lerobot_images[frame_idx]

        # 对比图片
        comp_result = compare_images(r2r_img, lerobot_img)
        comp_result["frame_index"] = frame_idx
        comparison_results.append(comp_result)

        # 并排拼接
        h, w = r2r_img.shape[:2]
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
        combined[:, :w] = r2r_img
        combined[:, w:] = lerobot_img

        # 添加差异标记
        status = "✓" if comp_result["match"] else "✗"
        color = (0, 255, 0) if comp_result["match"] else (255, 0, 0)
        cv2.putText(combined, f"R2R", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, f"LeRobot", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if not comp_result["match"]:
            diff_text = f"MSE: {comp_result['mse']:.1f}"
            cv2.putText(combined, diff_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 保存
        output_path = ep_dir / f"frame_{frame_idx:04d}_compare.jpg"
        img_pil = Image.fromarray(combined)
        img_pil.save(output_path, quality=95)

    # 保存对比报告
    report_path = ep_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(comparison_results, f, indent=2)

    # 保存说明
    readme = ep_dir / "README.txt"
    with open(readme, "w") as f:
        f.write(f"Episode {episode_index} 对比图片\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"每张图片由两张并排组成：\n")
        f.write(f"- 左侧：原始 R2R 图片\n")
        f.write(f"- 右侧：LeRobot 解码后的图片\n\n")
        f.write(f"总帧数: R2R={len(r2r_images)}, LeRobot={len(lerobot_images)}\n")
        f.write(f"保存帧数: {num_frames}\n\n")

        # 统计信息
        match_count = sum(1 for r in comparison_results if r["match"])
        f.write(f"匹配统计: {match_count}/{len(comparison_results)} 帧匹配\n")

        if comparison_results:
            avg_mse = np.mean([r["mse"] for r in comparison_results if r["mse"] is not None])
            max_diff_all = max([r["max_diff"] for r in comparison_results if r["max_diff"] is not None])
            f.write(f"平均MSE: {avg_mse:.2f}\n")
            f.write(f"最大差异: {max_diff_all}\n")

    return comparison_results


def verify_and_compare(
    r2r_data_dir: Path,
    lerobot_dataset_dir: Path,
    output_dir: Path,
    num_image_samples: int = None,
    max_episodes: int = None,
) -> Dict[str, Any]:
    """
    验证 R2R 原始数据与 LeRobot 数据集的一致性，并保存对比图片

    Args:
        r2r_data_dir: R2R数据目录路径
        lerobot_dataset_dir: LeRobot数据集目录路径
        output_dir: 输出目录
        num_image_samples: 每个episode保存的对比图片数量
        max_episodes: 最大验证的episode数量

    Returns:
        验证结果字典
    """
    print("=" * 80)
    print("R2R → LeRobot 数据验证 (video dtype)")
    print("=" * 80)
    print()

    # 加载 R2R annotations
    r2r_annotations = load_r2r_annotations(r2r_data_dir)
    print(f"总共 {len(r2r_annotations)} 个原始 R2R episodes")
    print()

    # 加载 LeRobot 数据集
    print(f"加载 LeRobot 数据集: {lerobot_dataset_dir}")
    dataset = LeRobotDataset(repo_id=None, root=lerobot_dataset_dir)

    # 读取info.json
    info_path = lerobot_dataset_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    print(f"总共 {len(dataset)} 帧, {info['total_episodes']} 个 episodes")
    print(f"视频编码: {info['features']['observation.images.rgb']['info'].get('video.codec', 'unknown')}")
    print()

    # 加载 LeRobot episodes metadata
    episodes_file = lerobot_dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_df = pd.read_parquet(episodes_file)

    # 验证结果
    results = {
        "total_episodes_checked": 0,
        "total_frames_compared": 0,
        "instruction_match": [],
        "actions_match": [],
        "image_match": [],
        "actions_details": [],
        "image_comparison_stats": [],
        "issues": [],
    }

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("开始验证...")
    print()

    # 确定要验证的episode数量
    num_episodes_to_check = min(len(episodes_df), max_episodes) if max_episodes else len(episodes_df)

    # 遍历每个 LeRobot episode
    for ep_idx in tqdm(range(num_episodes_to_check), desc="验证 episodes"):
        try:
            # 获取 LeRobot 数据
            lerobot_data = get_lerobot_episode_data(dataset, ep_idx)

            # 获取该 episode 对应的 task
            tasks_array = episodes_df[episodes_df['episode_index'] == ep_idx]['tasks'].values[0]
            # tasks 是一个 pandas array，需要取出第一个元素作为字符串
            task_str = tasks_array[0] if len(tasks_array) > 0 else ""

            # 解析 task 字符串获取 instruction
            # task 格式可能是: {"instruction": "..."} 或 [{"instruction": "..."}]
            try:
                parsed = json.loads(task_str)
                if isinstance(parsed, dict):
                    lerobot_instruction = parsed.get("instruction", "")
                elif isinstance(parsed, list) and len(parsed) > 0:
                    lerobot_instruction = parsed[0].get("instruction", "")
                else:
                    lerobot_instruction = ""
            except:
                lerobot_instruction = ""

            # 根据 instruction 匹配找到对应的 R2R episode 和 instruction index
            r2r_data = None
            r2r_ann_idx = None
            r2r_instr_idx = None

            for ann_idx, ann in enumerate(r2r_annotations):
                instructions = ann.get("instructions", [])
                for instr_idx, instruction in enumerate(instructions):
                    if instruction == lerobot_instruction:
                        r2r_ann_idx = ann_idx
                        r2r_instr_idx = instr_idx
                        r2r_data = get_r2r_episode_data(ann, r2r_data_dir, instr_idx)
                        break
                if r2r_data is not None:
                    break

            if r2r_data is None:
                results["issues"].append(
                    f"Episode {ep_idx}: 未找到对应的原始 R2R 数据 "
                    f"(instruction: '{lerobot_instruction[:50]}...')"
                )
                continue

            results["total_episodes_checked"] += 1

            # 验证 instruction
            instr_match = (lerobot_data["instruction"] == r2r_data["instruction"])
            results["instruction_match"].append(instr_match)

            # 验证 actions
            actions_match = (lerobot_data["actions"] == r2r_data["actions"])
            results["actions_match"].append(actions_match)

            results["actions_details"].append({
                "episode_index": ep_idx,
                "r2r_ann_id": r2r_annotations[r2r_ann_idx]["id"],
                "r2r_instr_idx": r2r_instr_idx,
                "num_frames": len(lerobot_data["actions"]),
                "actions_match": actions_match,
            })

            if not actions_match:
                results["issues"].append(
                    f"Episode {ep_idx}: actions 不匹配 "
                    f"(lerobot: {len(lerobot_data['actions'])}, r2r: {len(r2r_data['actions'])})"
                )

            # 对比图片
            comp_results = save_comparison_images(
                r2r_images=r2r_data["images"],
                lerobot_images=lerobot_data["images"],
                output_dir=output_dir,
                episode_index=ep_idx,
                num_samples=num_image_samples,
            )

            # 统计图片匹配情况
            match_count = sum(1 for r in comp_results if r["match"])
            results["image_match"].append(match_count == len(comp_results))
            results["total_frames_compared"] += len(comp_results)

            # 计算统计信息
            if comp_results:
                avg_mse = np.mean([r["mse"] for r in comp_results if r["mse"] is not None])
                max_diff = max([r["max_diff"] for r in comp_results if r["max_diff"] is not None])

                results["image_comparison_stats"].append({
                    "episode_index": ep_idx,
                    "total_frames": len(comp_results),
                    "matched_frames": match_count,
                    "avg_mse": float(avg_mse),
                    "max_diff": int(max_diff),
                })

                # 如果有大量不匹配的帧，记录问题
                if match_count < len(comp_results) * 0.9:  # 少于90%匹配
                    results["issues"].append(
                        f"Episode {ep_idx}: 只有 {match_count}/{len(comp_results)} 帧图片匹配 "
                        f"(平均MSE: {avg_mse:.2f})"
                    )

        except Exception as e:
            results["issues"].append(f"Episode {ep_idx}: 处理时出错 - {str(e)}")
            logger.error(f"Episode {ep_idx} 出错: {e}")
            import traceback
            traceback.print_exc()

    return results


def print_results(results: Dict[str, Any]):
    """打印验证结果"""
    print()
    print("=" * 80)
    print("验证结果")
    print("=" * 80)
    print()

    print(f"验证的 episodes 数量: {results['total_episodes_checked']}")
    print(f"对比的总帧数: {results['total_frames_compared']}")
    print()

    # Instruction 匹配情况
    instr_match_count = sum(results["instruction_match"])
    print(f"✓ Instruction 匹配: {instr_match_count}/{results['total_episodes_checked']}")

    # Actions 匹配情况
    actions_match_count = sum(results["actions_match"])
    print(f"✓ Actions 匹配:     {actions_match_count}/{results['total_episodes_checked']}")

    # 图片匹配情况
    image_match_count = sum(results["image_match"])
    print(f"✓ 图片匹配:         {image_match_count}/{results['total_episodes_checked']}")

    print()

    # 详细信息
    print("Episodes 详情:")
    print("-" * 80)
    for detail, img_stat in zip(results["actions_details"], results["image_comparison_stats"]):
        status = "✓" if detail["actions_match"] else "✗"
        img_status = "✓" if img_stat["matched_frames"] == img_stat["total_frames"] else "⚠"
        print(f"  {status} Episode {detail['episode_index']:3d} | "
              f"R2R ID: {detail['r2r_ann_id']:5d} | "
              f"Instr: {detail['r2r_instr_idx']} | "
              f"Frames: {detail['num_frames']:3d} | "
              f"{img_status} 图片匹配: {img_stat['matched_frames']:3d}/{img_stat['total_frames']:3d} | "
              f"MSE: {img_stat['avg_mse']:.2f}")

    print()

    # 总体结果
    if results["issues"]:
        print("发现的问题:")
        for issue in results["issues"]:
            print(f"  ✗ {issue}")
    else:
        print("✓ 所有验证通过！")

    print()
    print("=" * 80)
    print("对比图片已保存，请人工检查图片质量")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="验证 R2R 原始数据与 LeRobot 数据集的一致性 (video dtype)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--r2r_data_dir",
        type=str,
        default="/root/workspace/StreamVLN/data/trajectory_data/objectnav/cloudrobo_v1_l3mvn/shanghai-zhujiajiao-room2-1-2025-07-15_14-52-28",
        help="R2R 数据目录"
    )
    parser.add_argument(
        "--lerobot_dataset_dir",
        type=str,
        default="/root/workspace/StreamVLN/data/lerobot3.0/shanghai-zhujiajiao-room2-1-2025-07-15_14-52-28",
        help="LeRobot 数据集目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/verify_output_video",
        help="对比图片输出目录"
    )
    parser.add_argument(
        "--num_image_samples",
        type=int,
        default=10,
        help="每个 episode 保存的对比图片数量（None 表示全部）"
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="最大验证的episode数量（None 表示全部）"
    )

    args = parser.parse_args()

    results = verify_and_compare(
        r2r_data_dir=Path(args.r2r_data_dir),
        lerobot_dataset_dir=Path(args.lerobot_dataset_dir),
        output_dir=Path(args.output_dir),
        num_image_samples=args.num_image_samples,
        max_episodes=args.max_episodes,
    )

    print_results(results)

    # 返回退出码
    if results["issues"]:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()
