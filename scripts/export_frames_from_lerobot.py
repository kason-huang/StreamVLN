#!/usr/bin/env python
"""
Export frames from LeRobot dataset to images.

This script shows how to extract and save individual frames from a LeRobot dataset.
"""

import argparse
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from tqdm import tqdm


def export_episode_frames(
    dataset: LeRobotDataset,
    output_dir: Path,
    episode_index: int = None,
    max_frames: int = None,
):
    """
    Export frames from dataset to image files.

    Args:
        dataset: LeRobotDataset instance
        output_dir: Output directory for images
        episode_index: Specific episode to export (None for all)
        max_frames: Maximum frames per episode (None for all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine frame range
    if episode_index is not None:
        # Export specific episode
        start_idx = dataset.episodes[episode_index]["from_index"]
        end_idx = dataset.episodes[episode_index]["to_index"]
        frame_indices = range(start_idx, min(end_idx, start_idx + (max_frames or end_idx)))
    else:
        # Export all frames
        frame_indices = range(min(len(dataset), max_frames or len(dataset)))

    print(f"Exporting {len(list(frame_indices))} frames to {output_dir}")

    current_ep_idx = None
    ep_output_dir = None
    frame_count = 0

    for idx in tqdm(frame_indices, desc="Exporting frames"):
        sample = dataset[idx]

        # Get episode index
        ep_idx = sample.get("episode_index")
        if hasattr(ep_idx, 'item'):
            ep_idx = ep_idx.item()

        # Create episode directory if new episode
        if ep_idx != current_ep_idx:
            current_ep_idx = ep_idx
            ep_output_dir = output_dir / f"episode_{ep_idx:06d}"
            ep_output_dir.mkdir(parents=True, exist_ok=True)
            frame_count = 0

        # Get image tensor (CHW format: [3, H, W])
        image_tensor = sample['observation.images.rgb']

        # Convert to numpy (CHW)
        image_np = image_tensor.numpy()

        # Transpose to HWC for PIL
        image_hwc = np.transpose(image_np, (1, 2, 0))

        # Convert to uint8
        if image_hwc.dtype != np.uint8:
            if image_hwc.max() <= 1.0:
                image_hwc = (image_hwc * 255).astype(np.uint8)
            else:
                image_hwc = image_hwc.astype(np.uint8)

        # Save image
        frame_idx = sample.get("frame_index")
        if hasattr(frame_idx, 'item'):
            frame_idx = frame_idx.item()

        output_path = ep_output_dir / f"frame_{frame_idx:06d}.jpg"
        img_pil = Image.fromarray(image_hwc)
        img_pil.save(output_path, quality=95)

    print(f"Export complete!")


def main(
    repo_id: str,
    root: str,
    output_dir: str,
    episode_index: int = None,
    max_frames: int = None,
):
    """
    Main export function.
    """
    # Load dataset
    dataset = LeRobotDataset(repo_id=repo_id, root=root)

    print(f"Dataset: {repo_id}")
    print(f"Total samples: {len(dataset)}")
    # Get episode count from info.json
    info_file = Path(root) / repo_id / "meta" / "info.json"
    if info_file.exists():
        import json
        with open(info_file) as f:
            info = json.load(f)
            print(f"Total episodes: {info.get('total_episodes', 'unknown')}")

    if episode_index is not None:
        print(f"Exporting episode {episode_index} only")
    if max_frames is not None:
        print(f"Limiting to {max_frames} frames")
    print()

    # Export frames
    export_episode_frames(
        dataset=dataset,
        output_dir=Path(output_dir),
        episode_index=episode_index,
        max_frames=max_frames,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export frames from LeRobot dataset to images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        help="Root directory containing the dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/frames",
        help="Output directory for exported frames"
    )
    parser.add_argument(
        "--episode_index",
        type=int,
        default=None,
        help="Export specific episode only (None for all)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum frames to export (None for all)"
    )

    args = parser.parse_args()

    main(
        repo_id=args.repo_id,
        root=args.root,
        output_dir=args.output_dir,
        episode_index=args.episode_index,
        max_frames=args.max_frames,
    )
