#!/usr/bin/env python
"""
R2R → LeRobot 数据验证脚本

功能：
1. 自动验证 instruction 和 action 的一致性
2. 保存对比图片（原始 R2R vs LeRobot 解码后）供人工检查

用法:
    python scripts/verify_r2r_lerobot_with_images.py
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import pandas as pd
from tqdm import tqdm
from loguru import logger


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

        # 获取图片
        img_tensor = sample['observation.images.rgb']
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

    for frame_idx in range(num_frames):
        r2r_img = r2r_images[frame_idx]
        lerobot_img = lerobot_images[frame_idx]

        # 并排拼接
        h, w = r2r_img.shape[:2]
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
        combined[:, :w] = r2r_img
        combined[:, w:] = lerobot_img

        # 保存
        output_path = ep_dir / f"frame_{frame_idx:04d}_compare.jpg"
        img_pil = Image.fromarray(combined)
        img_pil.save(output_path, quality=95)

    # 保存说明
    readme = ep_dir / "README.txt"
    with open(readme, "w") as f:
        f.write(f"Episode {episode_index} 对比图片\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"每张图片由两张并排组成：\n")
        f.write(f"- 左侧：原始 R2R 图片\n")
        f.write(f"- 右侧：LeRobot 解码后的图片\n\n")
        f.write(f"总帧数: {len(r2r_images)}\n")
        f.write(f"保存帧数: {num_frames}\n")


def verify_and_compare(
    r2r_data_dir: Path,
    lerobot_repo_id: str,
    lerobot_root: Path,
    output_dir: Path,
    num_image_samples: int = None,
) -> Dict[str, Any]:
    """
    验证 R2R 原始数据与 LeRobot 数据集的一致性，并保存对比图片

    Returns:
        验证结果字典
    """
    print("=" * 80)
    print("R2R → LeRobot 数据验证")
    print("=" * 80)
    print()

    # 加载 R2R annotations
    r2r_annotations = load_r2r_annotations(r2r_data_dir)
    print(f"总共 {len(r2r_annotations)} 个原始 R2R episodes")
    print()

    # 加载 LeRobot 数据集
    print(f"加载 LeRobot 数据集: {lerobot_repo_id} @ {lerobot_root}")
    dataset = LeRobotDataset(repo_id=lerobot_repo_id, root=lerobot_root)
    print(f"总共 {len(dataset)} 帧, {dataset.meta.info['total_episodes']} 个 episodes")
    print()

    # 加载 LeRobot episodes metadata
    episodes_file = dataset.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_df = pd.read_parquet(episodes_file)

    # 验证结果
    results = {
        "total_episodes_checked": 0,
        "instruction_match": [],
        "actions_match": [],
        "actions_details": [],
        "issues": [],
    }

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("开始验证...")
    print()

    # 遍历每个 LeRobot episode
    for ep_idx in tqdm(range(len(episodes_df)), desc="验证 episodes"):
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

        # 保存对比图片
        save_comparison_images(
            r2r_images=r2r_data["images"],
            lerobot_images=lerobot_data["images"],
            output_dir=output_dir,
            episode_index=ep_idx,
            num_samples=num_image_samples,
        )

    return results


def print_results(results: Dict[str, Any]):
    """打印验证结果"""
    print()
    print("=" * 80)
    print("验证结果")
    print("=" * 80)
    print()

    print(f"验证的 episodes 数量: {results['total_episodes_checked']}")
    print()

    # Instruction 匹配情况
    instr_match_count = sum(results["instruction_match"])
    print(f"✓ Instruction 匹配: {instr_match_count}/{results['total_episodes_checked']}")

    # Actions 匹配情况
    actions_match_count = sum(results["actions_match"])
    print(f"✓ Actions 匹配:     {actions_match_count}/{results['total_episodes_checked']}")

    print()

    # 详细信息
    print("Episodes 详情:")
    print("-" * 80)
    for detail in results["actions_details"]:
        status = "✓" if detail["actions_match"] else "✗"
        print(f"  {status} Episode {detail['episode_index']:3d} | "
              f"R2R ID: {detail['r2r_ann_id']:5d} | "
              f"Instr: {detail['r2r_instr_idx']} | "
              f"Frames: {detail['num_frames']:3d}")

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
        description="验证 R2R 原始数据与 LeRobot 数据集的一致性",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--r2r_data_dir",
        type=str,
        default="./data/trajectory_data",
        help="R2R 数据根目录（包含 R2R/ 子目录）"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="R2R",
        help="数据集名称 (R2R/RxR/EnvDrop)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="streamvln/r2r_navigation",
        help="LeRobot dataset repo ID"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./data/lerobot",
        help="LeRobot 数据集根目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/verify_output",
        help="对比图片输出目录"
    )
    parser.add_argument(
        "--num_image_samples",
        type=int,
        default=None,
        help="每个 episode 保存的对比图片数量（None 表示全部）"
    )

    args = parser.parse_args()

    r2r_data_dir = Path(args.r2r_data_dir) / args.dataset_name

    results = verify_and_compare(
        r2r_data_dir=r2r_data_dir,
        lerobot_repo_id=args.repo_id,
        lerobot_root=Path(args.root),
        output_dir=Path(args.output_dir),
        num_image_samples=args.num_image_samples,
    )

    print_results(results)

    # 返回退出码
    if results["issues"]:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()