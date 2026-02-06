#!/usr/bin/env python

import glob
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Tuple

import cv2
import datasets
import numpy as np
import PIL.Image
import torch
import torchvision
import tqdm
from datasets import concatenate_datasets
from streamvln.dataset.lerobot.common.datasets.compute_stats import (
    aggregate_stats,
    auto_downsample_height_width,
    sample_indices,
)
from streamvln.dataset.lerobot_dataset import (
    NavDataset,
    NavDatasetMetadata,
    compute_episode_stats,
    get_streamvln_features,
)
from loguru import logger

LEROBOT_HOME = Path(os.environ.get("LEROBOT_HOME", "/shared/smartbot_new/liuyu/"))


# Note: compute_episode_stats, NavDatasetMetadata, NavDataset are imported from streamvln.dataset.lerobot_dataset



def load_streamvln_episode(
    ann: Dict[str, Any],
    dataset_name: str,
    data_dir: Path,
    # img_size: Tuple[int, int] = (224, 224)
) -> Iterator[Dict[str, Any]]:
    """
    load StreamVLN episode data, return an iterator in LeRobot format

    Args:
        ann: single annotation dictionary
        dataset_name: dataset name (EnvDrop/R2R/RxR)
        data_dir: data root directory
        img_size: output image size (height, width)

    Yields:
        a dictionary of LeRobot format data for each frame
    """
    try:
        ann_id = ann["id"]
        video_path = ann["video"]

        # parse scene ID and episode ID
        parts = video_path.split("/")[-1].split("_")
        scene_id = parts[0]
        ann_id = parts[-1]
        # fix path parsing logic
        # original format: "video": "images/17DRP5sb8fy_envdrop_111702"
        # actual path: images/17DRP5sb8fy/rgb

        # src_image_dir = data_dir / dataset_name / "images" / "rgb" /scene_id

        # build source image directory
        src_image_dir = data_dir / dataset_name / video_path / "rgb"

        # get all image files
        image_files = sorted(glob.glob(str(src_image_dir / "*.jpg")))
        if not image_files:
            logger.warning(f"No image files found in {src_image_dir}")
            return

        # get actions and instructions
        actions = np.array(ann.get("actions", []), dtype=np.int64)
        instructions = ann.get("instructions", [])
        instruction = json.dumps({"instruction": instructions[0]}) if instructions else "Navigation task"

        # build file path mapping
        files = {"observation.images.rgb": str(src_image_dir)}

        for frame_idx, img_path in enumerate(image_files):
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            action_value = -1
            if frame_idx < len(actions):
                action_value = actions[frame_idx]

            action = np.array([action_value], dtype=np.int64)

            yield {
                'observation': {
                    'images.rgb': img,
                },
                'action': action,
                'language_instruction': instruction,
                'files': files,
            }

    except Exception as e:
        logger.error(f"Failed to load episode {ann_id}: {str(e)}", exc_info=True)
        return


def process_episode(
    dataset: NavDataset,
    ann: Dict[str, Any],
    dataset_name: str,
    data_dir: Path,
    episode_idx: int,
) -> Tuple[int, bool, str]:
    """
    Add a single episode to the existing dataset.

    Args:
        dataset: Existing NavDataset instance
        ann: Episode annotation
        dataset_name: Dataset name
        data_dir: Data directory
        episode_idx: Episode index

    Returns:
        (episode_idx, success, message)
    """
    try:
        episode_id = f"{dataset_name}_{ann['id']}"

        episode_iterator = load_streamvln_episode(ann, dataset_name, data_dir)
        frame_count = 0
        files = {}

        for step_data in episode_iterator:
            if frame_count == 0:
                files = step_data.pop('files', {})
            else:
                step_data.pop('files', {})

            dataset.add_frame(
                frame={
                    "observation.images.rgb": step_data["observation"]["images.rgb"],
                    "action": step_data["action"],
                },
                task=step_data["language_instruction"],
            )
            frame_count += 1

        if frame_count > 0:
            dataset.save_episode(files=files, episode_index=episode_idx)
            message = f"Successfully processed: {episode_id}, {frame_count} frames"
            return (episode_idx, True, message)
        else:
            message = f"No frames were processed, skipping: {episode_id}"
            return (episode_idx, False, message)

    except Exception as e:
        message = f"Failed to process episode: {str(e)}"
        logger.error(message, exc_info=True)
        return (episode_idx, False, message)


def process_dataset(
    dataset_name: str,
    data_dir: Path,
    repo_name: str,
    num_threads: int = 10,
    start_idx: int = 0,
    end_idx: int | None = None,
    overwrite: bool = False,
) -> Tuple[int, int]:
    """
    Process the entire dataset into a single LeRobot-format dataset.

    Args:
        dataset_name: Dataset name
        data_dir: Data root directory
        repo_name: Output dataset name
        num_threads: Number of threads (currently unused, sequential processing)
        start_idx: Start index for episodes
        end_idx: End index for episodes

    Returns:
        (total_episodes, success_episodes)
    """
    # Load annotations
    ann_file = data_dir / dataset_name / "annotations.json"
    if not ann_file.exists():
        logger.error(f"Annotation file not found: {ann_file}")
        return 0, 0

    with open(ann_file, "r") as f:
        annotations = json.load(f)

    total = len(annotations)
    end_idx = end_idx if end_idx is not None else total
    selected_anns = annotations[start_idx:end_idx]
    selected_count = len(selected_anns)

    if selected_count == 0:
        logger.warning(f"No episodes found in the index range [{start_idx}, {end_idx})")
        return 0, 0

    # Create output path for the entire dataset (not per-episode)
    output_path = LEROBOT_HOME / repo_name / dataset_name.lower()

    logger.info(
        f"Start processing dataset: {dataset_name} "
        f"(Total episodes: {total}, processing range: [{start_idx}, {end_idx}), actual processing: {selected_count})"
    )
    logger.info(f"Output directory: {output_path}")

    # Create a single dataset for all episodes
    features = get_streamvln_features()

    # Check if dataset already exists
    if overwrite and output_path.exists():
        logger.info(f"Overwrite enabled, removing existing dataset: {output_path}")
        try:
            shutil.rmtree(output_path)
            # Double-check that directory is actually removed
            if output_path.exists():
                raise OSError(f"Failed to remove directory: {output_path}")
        except Exception as e:
            logger.error(f"Failed to remove existing dataset: {e}")
            raise

    if output_path.exists() and (output_path / "meta" / "info.json").exists():
        logger.info(f"Dataset already exists at {output_path}")
        # Load existing dataset to continue
        dataset = NavDataset(repo_id=f"{repo_name}_{dataset_name.lower()}", root=output_path)
        # Count existing episodes from the parquet files
        data_chunk_dir = output_path / "data" / "chunk-000"
        if data_chunk_dir.exists():
            existing_episodes = sorted([int(f.stem.split("_")[1]) for f in data_chunk_dir.glob("episode_*.parquet")])
            start_episode = len(existing_episodes)
        else:
            start_episode = 0
        logger.info(f"Found {start_episode} existing episodes, continuing from episode {start_episode}")
        # Adjust selected_anns to start from where we left off
        selected_anns = selected_anns[start_episode:]
        selected_count = len(selected_anns)
        if selected_count == 0:
            logger.info("All episodes already processed")
            return total, total
    else:
        # Create new dataset - first remove incomplete directory if exists
        if output_path.exists():
            logger.warning(f"Removing incomplete dataset directory: {output_path}")
            shutil.rmtree(output_path)
        # NavDataset.create() will create the directory internally
        dataset = NavDataset.create(
            repo_id=f"{repo_name}_{dataset_name.lower()}",
            root=output_path,
            robot_type="unknown",
            fps=30,
            use_videos=True,
            features=features,
        )
        start_episode = 0

    # Process episodes sequentially
    success_count = 0
    progress_bar = tqdm.tqdm(
        enumerate(selected_anns, start=start_episode),
        total=selected_count,
        desc=f"处理 {dataset_name} [{start_idx}:{end_idx}]"
    )

    for episode_idx, ann in progress_bar:
        _, success, message = process_episode(
            dataset=dataset,
            ann=ann,
            dataset_name=dataset_name,
            data_dir=data_dir,
            episode_idx=episode_idx,
        )
        if success:
            success_count += 1
        progress_bar.set_postfix_str(
            f"Success: {success_count}/{selected_count} " f"({success_count/selected_count:.1%})"
        )
        progress_bar.set_description(f"{message}")

    return selected_count, success_count


def main(
    data_dir: str,
    repo_name: str = "nav_S1",
    num_threads: int = 10,
    start_index: int = None,
    end_index: int = None,
    datasets: str = None,
    overwrite: bool = False,
):
    """
    Main function to convert VLN-CE dataset to LeRobot format.

    Args:
        data_dir: Data root directory
        repo_name: Output dataset name
        num_threads: Number of threads (currently unused)
        start_index: Start index for episodes
        end_index: End index for episodes
        datasets: Dataset name to process (e.g., "R2R")
        overwrite: If True, delete existing dataset and start from scratch
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    total_episodes = 0
    success_episodes = 0
    dataset_name = datasets

    total, success = process_dataset(
        dataset_name=dataset_name,
        data_dir=data_path,
        repo_name=repo_name,
        num_threads=num_threads,
        start_idx=start_index if start_index is not None else 0,
        end_idx=end_index,
        overwrite=overwrite,
    )

    total_episodes += total
    success_episodes += success

    logger.info("=" * 50)
    logger.info("Conversion completed!")
    logger.info(f"Total episodes: {total_episodes}")
    logger.info(f"Success: {success_episodes}")
    logger.info(f"Failed: {total_episodes - success_episodes}")
    logger.info("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert VLN-CE dataset to LeRobot format")
    parser.add_argument("--data_dir", type=str, default="./data/", help="VLN-CE data root directory")
    parser.add_argument("--repo_name", type=str, default="vln_ce_lerobot", help="Output dataset name")
    parser.add_argument("--datasets", type=str, required=True, help="Dataset name (e.g., R2R)")
    parser.add_argument("--num_threads", type=int, default=10, help="Number of threads (currently unused)")
    parser.add_argument("--start_index", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end_index", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset")

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        repo_name=args.repo_name,
        num_threads=args.num_threads,
        start_index=args.start_index,
        end_index=args.end_index,
        datasets=args.datasets,
        overwrite=args.overwrite,
    )
