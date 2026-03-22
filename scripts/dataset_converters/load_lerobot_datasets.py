#!/usr/bin/env python
"""
LeRobot Dataset Verification Script

This script loads a LeRobot dataset from local storage and verifies:
- Actions and images count correspondence
- Dataset structure integrity
- Export frames with instruction, action, and image
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import pandas as pd
from PIL import Image
from tqdm import tqdm


def verify_dataset(dataset: LeRobotDataset) -> dict:
    """
    Verify the dataset integrity.

    Returns a dictionary with verification results.
    """
    results = {
        "total_samples": len(dataset),
        "fps": dataset.fps,
        "episodes": defaultdict(lambda: {"frames": 0, "actions": 0}),
        "issues": [],
    }

    print(f"Loading dataset: {dataset.repo_id}")
    print(f"Total samples (frames): {len(dataset)}")
    print(f"FPS: {dataset.fps}")
    print()

    # Load episodes metadata
    episodes_file = dataset.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if episodes_file.exists():
        episodes_df = pd.read_parquet(episodes_file)
        print(f"Total episodes in metadata: {len(episodes_df)}")
        print(f"Episode columns: {list(episodes_df.columns)}")
        print()

        # Check each episode - column name is 'length' not 'num_frames'
        for idx, row in episodes_df.iterrows():
            ep_index = row.get('episode_index', idx)
            num_frames = row.get('length', 0)
            results["episodes"][ep_index]["frames"] = num_frames

    # Iterate through dataset to count actions
    print("Iterating through samples to verify action-image correspondence...")
    current_episode = None
    episode_frame_count = 0

    for idx in range(min(len(dataset), 10000)):  # Limit to prevent too long iterations
        sample = dataset[idx]

        episode_index = sample.get("episode_index")
        if isinstance(episode_index, type(sample["action"])):  # torch.Tensor
            episode_index = episode_index.item()

        # Track frames per episode
        if current_episode != episode_index:
            if current_episode is not None:
                results["episodes"][current_episode]["actions"] = episode_frame_count
            current_episode = episode_index
            episode_frame_count = 0
        episode_frame_count += 1

        # Check action shape
        action = sample["action"]
        if hasattr(action, 'shape'):
            if len(action.shape) == 0:
                pass  # Scalar action
            elif action.shape[0] != 1:
                results["issues"].append(f"Sample {idx}: action shape {action.shape} != (1,)")

    # Don't forget last episode
    if current_episode is not None:
        results["episodes"][current_episode]["actions"] = episode_frame_count

    return results


def print_results(results: dict):
    """Print verification results."""
    print("=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    print(f"Total samples: {results['total_samples']}")
    print(f"FPS: {results['fps']}")
    print(f"Episodes checked: {len(results['episodes'])}")
    print()

    # Check correspondence
    print("Action-Image Correspondence:")
    print("-" * 60)
    all_match = True
    for ep_idx, counts in sorted(results["episodes"].items()):
        frames = counts["frames"]
        actions = counts["actions"]
        match = "✓" if frames == actions else "✗"
        if frames != actions:
            all_match = False
        print(f"Episode {ep_idx}: frames={frames}, actions={actions} {match}")

    print()
    if all_match:
        print("✓ All episodes have matching action and image counts!")
    else:
        print("✗ Some episodes have mismatched counts!")

    if results["issues"]:
        print()
        print("Issues found:")
        for issue in results["issues"]:
            print(f"  - {issue}")

    print("=" * 60)


def get_episode_data(dataset: LeRobotDataset, episode_index: int) -> dict:
    """
    Get all data (instruction, actions, images) for a specific episode.

    Returns a dictionary with:
        - instruction: str, the task instruction
        - actions: list of int, action values
        - images: list of numpy arrays (HWC format), image data
        - num_frames: int, number of frames
    """
    # Load episodes metadata to find frame range
    episodes_file = dataset.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not episodes_file.exists():
        raise FileNotFoundError(f"Episodes file not found: {episodes_file}")

    episodes_df = pd.read_parquet(episodes_file)

    # Find the episode
    ep_row = episodes_df[episodes_df['episode_index'] == episode_index]
    if len(ep_row) == 0:
        raise ValueError(f"Episode {episode_index} not found")

    start_idx = int(ep_row.iloc[0]['dataset_from_index'])
    end_idx = int(ep_row.iloc[0]['dataset_to_index'])

    instruction = None
    actions = []
    images = []

    # Iterate through frames
    for idx in range(start_idx, end_idx):
        sample = dataset[idx]

        # Get instruction from first frame
        if instruction is None:
            task = sample.get("task")
            if isinstance(task, str):
                instruction = json.loads(task).get("instruction", "")

        # Get action
        action = sample.get("action")
        if hasattr(action, 'item'):
            action = action.item()
        actions.append(action)

        # Get image
        img_tensor = sample['observation.images.rgb']
        img_np = img_tensor.numpy()
        # CHW to HWC
        img_hwc = np.transpose(img_np, (1, 2, 0))
        # Convert to uint8 if needed
        if img_hwc.max() <= 1.0:
            img_hwc = (img_hwc * 255).astype(np.uint8)
        else:
            img_hwc = img_hwc.astype(np.uint8)
        images.append(img_hwc)

    return {
        "instruction": instruction,
        "actions": actions,
        "images": images,
        "num_frames": len(actions),
    }


def export_episode_images(
    dataset: LeRobotDataset,
    output_dir: str,
    episode_index: int = None,
    max_frames: int = None,
):
    """
    Export episode frames with metadata to image files.

    Creates:
    - episode_{index:06d}/frame_{frame:06d}.jpg
    - episode_{index:06d}/metadata.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all episodes if not specified
    if episode_index is None:
        episodes_file = dataset.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        if episodes_file.exists():
            episodes_df = pd.read_parquet(episodes_file)
            episode_indices = episodes_df['episode_index'].tolist()
        else:
            # Infer from dataset
            episode_indices = []
            for i in range(min(len(dataset), 1000)):
                ep_idx = dataset[i].get("episode_index")
                if hasattr(ep_idx, 'item'):
                    ep_idx = ep_idx.item()
                if ep_idx not in episode_indices:
                    episode_indices.append(ep_idx)
    else:
        episode_indices = [episode_index]

    print(f"Exporting {len(episode_indices)} episodes to {output_dir}")

    for ep_idx in tqdm(episode_indices, desc="Exporting episodes"):
        try:
            data = get_episode_data(dataset, ep_idx)
        except Exception as e:
            print(f"Error loading episode {ep_idx}: {e}")
            continue

        ep_dir = output_dir / f"episode_{ep_idx:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "episode_index": ep_idx,
            "instruction": data["instruction"],
            "actions": data["actions"],
            "num_frames": data["num_frames"],
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save images
        num_frames = min(data["num_frames"], max_frames) if max_frames else data["num_frames"]
        for frame_idx in range(num_frames):
            img = data["images"][frame_idx]
            img_pil = Image.fromarray(img)
            img_pil.save(ep_dir / f"frame_{frame_idx:06d}.jpg", quality=95)

        print(f"  Episode {ep_idx}: {num_frames} frames exported")

    print(f"Export complete! Output: {output_dir}")


def get_all_episodes_data(dataset: LeRobotDataset) -> list[dict]:
    """
    Get data for all episodes in the dataset.

    Returns a list of dictionaries, each containing:
        - episode_index: int
        - instruction: str, the task instruction
        - actions: list of int, action values
        - images: list of numpy arrays (HWC format), image data
        - num_frames: int, number of frames
    """
    # Get all episode indices
    episodes_file = dataset.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if episodes_file.exists():
        episodes_df = pd.read_parquet(episodes_file)
        episode_indices = episodes_df['episode_index'].tolist()
    else:
        # Infer from dataset
        episode_indices = []
        for i in range(min(len(dataset), 10000)):
            ep_idx = dataset[i].get("episode_index")
            if hasattr(ep_idx, 'item'):
                ep_idx = ep_idx.item()
            if ep_idx not in episode_indices:
                episode_indices.append(ep_idx)

    all_data = []
    for ep_idx in episode_indices:
        data = get_episode_data(dataset, ep_idx)
        data["episode_index"] = ep_idx
        all_data.append(data)

    return all_data


def main():
    #dataset = LeRobotDataset(repo_id='streamvln/r2r_navigation', root='./data/lerobot')
    dataset = LeRobotDataset(repo_id='streamvln/r2r_navigation', root='./data/lerobot3.0/lerobot-shengwei-reconstruction')

    # 方式1: 获取单个 episode 的数据
    # data = get_episode_data(dataset, episode_index=0)
    # print(data["instruction"])  # 任务指令
    # print(data["actions"])      # 动作列表 [3, 3, 3, 1, 2, ...]
    # print(data["images"])       # 图片列表 (HWC numpy array)
    # print(data["num_frames"])   # 帧数

    # 方式2: 获取全部 episodes 的数据
    all_episodes = get_all_episodes_data(dataset)

    print(f"Total episodes: {len(all_episodes)}")

    # 遍历每个 episode
    for ep_data in all_episodes:
        ep_idx = ep_data["episode_index"]
        instruction = ep_data["instruction"]
        actions = ep_data["actions"]
        images = ep_data["images"]
        num_frames = ep_data["num_frames"]

        print(f"\nEpisode {ep_idx}:")
        print(f"  Instruction: {instruction[:60]}...")
        print(f"  Frames: {num_frames}")
        print(f"  Actions: {len(actions)}")
        print(f"  Images shape: {images[0].shape if images else 'N/A'}")


def main1(
    repo_id: str,
    root: str,
    verbose: bool = False,
    export_dir: str = None,
    episode_index: int = None,
    max_frames: int = None,
):
    """
    Main verification function.
    """
    # Load dataset
    dataset = LeRobotDataset(repo_id=repo_id, root=root)

    # Verify
    results = verify_dataset(dataset)

    # Print results
    print_results(results)

    # Export images if requested
    if export_dir:
        print()
        print("=" * 60)
        print("EXPORTING IMAGES")
        print("=" * 60)
        export_episode_images(
            dataset=dataset,
            output_dir=export_dir,
            episode_index=episode_index,
            max_frames=max_frames,
        )

    # Show sample data if verbose
    if verbose:
        print()
        print("=" * 60)
        print("SAMPLE DATA (first 3 samples)")
        print("=" * 60)
        for idx in range(min(3, len(dataset))):
            sample = dataset[idx]
            print(f"\nSample {idx}:")
            for key, value in sample.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}, dtype={type(value).__name__}")
                    if key == "action":
                        print(f"    value={value}")
                    elif key == "task":
                        print(f"    value={value[:80]}...")
                else:
                    print(f"  {key}: {value}")

if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Verify LeRobot dataset integrity and export frames",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser.add_argument(
#         "--repo_id",
#         type=str,
#         default="streamvln/r2r_navigation",
#         help="LeRobot dataset repo ID"
#     )
#     parser.add_argument(
#         "--root",
#         type=str,
#         default="./data/lerobot",
#         help="Root directory containing the dataset"
#     )
#     parser.add_argument(
#         "--verbose",
#         action="store_true",
#         help="Print detailed sample information"
#     )
#     parser.add_argument(
#         "--export_dir",
#         type=str,
#         default=None,
#         help="Export episode frames to this directory"
#     )
#     parser.add_argument(
#         "--episode_index",
#         type=int,
#         default=None,
#         help="Export specific episode only (requires --export_dir)"
#     )
#     parser.add_argument(
#         "--max_frames",
#         type=int,
#         default=None,
#         help="Maximum frames per episode to export (requires --export_dir)"
#     )

#     args = parser.parse_args()

#     main(
#         repo_id=args.repo_id,
#         root=args.root,
#         verbose=args.verbose,
#         export_dir=args.export_dir,
#         episode_index=args.episode_index,
#         max_frames=args.max_frames,
#     )