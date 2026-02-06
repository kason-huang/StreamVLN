#!/usr/bin/env python
"""
R2R to LeRobot Dataset Converter

Converts R2R trajectory data to LeRobot format.
Reference: any4lerobot/libero2lerobot/libero_h5.py

Key design decisions:
- Use lerobot.datasets.lerobot_dataset.LeRobotDataset
- Drop first action (-1) to align with image count
- Split each episode by instruction (multiple episodes per original)
- No image preprocessing (keep original size)
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Iterator

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from loguru import logger
from tqdm import tqdm


# R2R Dataset Features Definition
# Note: "task" is a special required field that is added to each frame,
# but is NOT defined in features as it's handled separately by LeRobot
R2R_FEATURES = {
    "observation.images.rgb": {
        "dtype": "video",
        "shape": None,  # Will be inferred from first image: [height, width, channel]
        "names": ["height", "width", "channel"]
    },
    "action": {
        "dtype": "int64",
        "shape": (1,),
        "names": ["action_index"]
    }
}


def load_r2r_episode(
    ann: Dict[str, Any],
    dataset_name: str,
    data_dir: Path,
    instruction_idx: int = 0,
) -> Iterator[Dict[str, Any]]:
    """
    Load R2R episode data, return an iterator in LeRobot format.

    Args:
        ann: Single annotation dictionary from annotations.json
        dataset_name: Dataset name (R2R/RxR/EnvDrop)
        data_dir: Data root directory
        instruction_idx: Which instruction to use from the instructions list

    Yields:
        A dictionary of LeRobot format data for each frame
    """
    try:
        ann_id = ann["id"]
        video_path = ann["video"]

        # Build source image directory
        # Format: images/{episode_id}/rgb/{000.jpg, 001.jpg, ...}
        src_image_dir = data_dir / dataset_name / video_path / "rgb"

        if not src_image_dir.exists():
            logger.warning(f"Image directory not found: {src_image_dir}")
            return

        # Get all image files (sorted by name: 000.jpg, 001.jpg, ...)
        image_files = sorted(src_image_dir.glob("*.jpg"))
        if not image_files:
            logger.warning(f"No image files found in {src_image_dir}")
            return

        # Get actions and instructions
        actions = np.array(ann.get("actions", []), dtype=np.int64)
        instructions = ann.get("instructions", [])

        # Drop first action (-1) to align with image count
        if len(actions) > 0 and actions[0] == -1:
            actions = actions[1:]

        # Get the specified instruction
        if instruction_idx < len(instructions):
            instruction = instructions[instruction_idx]
        else:
            instruction = instructions[0] if instructions else "Navigation task"

        # Format task as JSON string
        task = json.dumps({"instruction": instruction})

        # Yield frames
        for frame_idx, img_path in enumerate(image_files):
            # Read image
            try:
                from PIL import Image
                img = Image.open(img_path)
                img_array = np.array(img)
            except Exception as e:
                logger.error(f"Failed to load image {img_path}: {e}")
                continue

            # Get corresponding action
            if frame_idx < len(actions):
                action_value = actions[frame_idx]
            else:
                # Use last action if frame count exceeds actions
                action_value = actions[-1] if len(actions) > 0 else -1

            action = np.array([action_value], dtype=np.int64)

            yield {
                "observation.images.rgb": img_array,
                "action": action,
                "task": task,
            }

    except Exception as e:
        logger.error(f"Failed to load episode {ann.get('id', 'unknown')}: {e}", exc_info=True)
        return


def process_episode(
    dataset: LeRobotDataset,
    ann: Dict[str, Any],
    dataset_name: str,
    data_dir: Path,
    episode_idx: int,
) -> tuple[int, bool, str]:
    """
    Process a single R2R episode, splitting by instructions.

    Args:
        dataset: LeRobotDataset instance
        ann: Episode annotation
        dataset_name: Dataset name
        data_dir: Data directory
        episode_idx: Episode index (for progress tracking)

    Returns:
        (episodes_created, success, message)
    """
    try:
        ann_id = ann["id"]
        instructions = ann.get("instructions", [])

        if not instructions:
            instructions = ["Navigation task"]

        episodes_created = 0
        frame_count = 0

        # Create one episode per instruction
        for instr_idx, instruction in enumerate(instructions):
            episode_iterator = load_r2r_episode(
                ann,
                dataset_name,
                data_dir,
                instruction_idx=instr_idx,
            )

            frame_count = 0
            for frame_data in episode_iterator:
                dataset.add_frame(frame_data)
                frame_count += 1

            if frame_count > 0:
                dataset.save_episode()
                episodes_created += 1

        if episodes_created > 0:
            message = f"Processed episode {ann_id}: {episodes_created} sub-episodes, {frame_count} frames each"
            return episodes_created, True, message
        else:
            message = f"No frames processed for episode {ann_id}"
            return 0, False, message

    except Exception as e:
        message = f"Failed to process episode: {str(e)}"
        logger.error(message, exc_info=True)
        return 0, False, message


def process_dataset(
    dataset_name: str,
    data_dir: Path,
    output_dir: Path,
    repo_id: str,
    fps: int = 3,
    start_idx: int = 0,
    end_idx: int | None = None,
    overwrite: bool = False,
) -> tuple[int, int]:
    """
    Process the entire R2R dataset into LeRobot format.

    Args:
        dataset_name: Dataset name (R2R/RxR/EnvDrop)
        data_dir: Data root directory
        output_dir: Output directory
        repo_id: LeRobot dataset repo ID
        fps: Frames per second for timestamps
        start_idx: Start episode index
        end_idx: End episode index (exclusive)
        overwrite: Overwrite existing dataset

    Returns:
        (total_input_episodes, total_output_episodes)
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
        logger.warning(f"No episodes found in range [{start_idx}, {end_idx})")
        return 0, 0

    # Create output path
    output_path = output_dir / repo_id

    logger.info(
        f"Processing {dataset_name}: "
        f"(Total episodes: {total}, range: [{start_idx}, {end_idx}), processing: {selected_count})"
    )
    logger.info(f"Output directory: {output_path}")

    # Check if dataset already exists
    if overwrite and output_path.exists():
        logger.info(f"Overwrite enabled, removing existing dataset: {output_path}")
        shutil.rmtree(output_path)

    if output_path.exists() and (output_path / "meta" / "info.json").exists():
        logger.info(f"Dataset already exists at {output_path}")
        # Load existing dataset to continue
        dataset = LeRobotDataset(repo_id=repo_id, root=output_dir)
        # Count existing episodes
        data_chunk_dir = output_path / "data" / "chunk-000"
        if data_chunk_dir.exists():
            existing_episodes = len(list(data_chunk_dir.glob("episode_*.parquet")))
        else:
            existing_episodes = 0
        logger.info(f"Found {existing_episodes} existing episodes, continuing from episode {existing_episodes}")
        # Note: Resume is approximate due to instruction splitting
        selected_anns = selected_anns[existing_episodes:]
        selected_count = len(selected_anns)
        if selected_count == 0:
            logger.info("All episodes already processed")
            return total, dataset.total_episodes
    else:
        # Create new dataset
        if output_path.exists():
            logger.warning(f"Removing incomplete dataset directory: {output_path}")
            shutil.rmtree(output_path)

        # Infer image shape from first episode
        first_ann = selected_anns[0]
        video_path = first_ann["video"]
        first_image_dir = data_dir / dataset_name / video_path / "rgb"
        first_images = sorted(first_image_dir.glob("*.jpg"))
        if first_images:
            from PIL import Image
            first_img = Image.open(first_images[0])
            img_array = np.array(first_img)
            height, width = img_array.shape[:2]
            # Shape format: [height, width, channel]
            R2R_FEATURES["observation.images.rgb"]["shape"] = [height, width, 3]
            logger.info(f"Inferred image shape: [{height}, {width}, 3]")

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=output_dir,
            fps=fps,
            features=R2R_FEATURES,
        )

    # Process episodes
    total_output_episodes = 0
    success_count = 0
    progress_bar = tqdm(selected_anns, desc=f"Processing {dataset_name}")

    for ann in progress_bar:
        episodes_created, success, message = process_episode(
            dataset=dataset,
            ann=ann,
            dataset_name=dataset_name,
            data_dir=data_dir,
            episode_idx=0,
        )
        if success:
            success_count += 1
            total_output_episodes += episodes_created
        progress_bar.set_postfix_str(
            f"Output: {total_output_episodes}, Success: {success_count}/{selected_count}"
        )
        progress_bar.set_description(message[:50])

    logger.info(f"Conversion complete: {selected_count} input episodes -> {total_output_episodes} output episodes")
    return selected_count, total_output_episodes


def main(
    data_dir: str,
    output_dir: str,
    dataset_name: str,
    repo_id: str,
    fps: int,
    start_idx: int,
    end_idx: int | None,
    overwrite: bool,
):
    """
    Main conversion function.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    output_path = Path(output_dir)
    # Don't create parent directory here - LeRobotDataset.create() will handle it

    input_episodes, output_episodes = process_dataset(
        dataset_name=dataset_name,
        data_dir=data_path,
        output_dir=output_path,
        repo_id=repo_id,
        fps=fps,
        start_idx=start_idx,
        end_idx=end_idx,
        overwrite=overwrite,
    )

    logger.info("=" * 50)
    logger.info("Conversion completed!")
    logger.info(f"Input episodes: {input_episodes}")
    logger.info(f"Output episodes: {output_episodes}")
    logger.info(f"Output directory: {output_path / repo_id}")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert R2R trajectory data to LeRobot format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/trajectory_data",
        help="Input data root directory (containing R2R/ subdirectory)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/lerobot",
        help="Output directory for LeRobot dataset"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="R2R",
        help="Dataset name (R2R/RxR/EnvDrop)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="streamvln/r2r_navigation",
        help="LeRobot dataset repo ID"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=3,
        help="Frames per second for timestamp generation"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start episode index (inclusive)"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End episode index (exclusive), None for all"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory"
    )

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        repo_id=args.repo_id,
        fps=args.fps,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        overwrite=args.overwrite,
    )
