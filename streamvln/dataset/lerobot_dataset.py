"""
LeRobot dataset classes for VLN navigation data.

This module contains NavDataset and NavDatasetMetadata classes that extend
LeRobot's base classes to support image sequences instead of video files.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import PIL.Image
import datasets
from loguru import logger

from streamvln.dataset.lerobot.common.datasets.compute_stats import get_feature_stats
from streamvln.dataset.lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from streamvln.dataset.lerobot.common.datasets.utils import (
    check_timestamps_sync,
    embed_images,
    get_episode_data_index,
    hf_transform_to_torch,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
)
from streamvln.dataset.lerobot.common.datasets.video_utils import get_safe_default_codec
from streamvln.dataset.lerobot.common.constants import HF_LEROBOT_HOME
from streamvln.dataset.lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION


def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
    """calculate episode statistics

    Note: Images are expected in channel-last format [H,W,C], but stats must be in
    channel-first format [C,1,1] to match LeRobot's expected format.
    """
    ep_stats = {}
    for key, data in episode_data.items():
        if key not in features:  # skip non-feature data
            continue

        if features[key]["dtype"] == "string":
            continue
        elif features[key]["dtype"] in ["image", "video"]:
            if isinstance(data, (str, list)) and all(
                isinstance(item, str) for item in (data if isinstance(data, list) else [data])
            ):
                # string path, skip stats calculation
                continue
            # ensure data is in the correct shape
            ep_ft_array = np.array(data)
            if len(ep_ft_array.shape) == 3:  # [H, W, C]
                ep_ft_array = ep_ft_array[np.newaxis, ...]  # add time dimension [1, H, W, C]
            # Convert from channel-last [T,H,W,C] to channel-first [T,C,H,W]
            ep_ft_array = np.moveaxis(ep_ft_array, -1, 1)  # [T, C, H, W]
            # Reduce over time, height, width to keep channel dimension first: [T, C, 1, 1]
            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            # for non-image/video data, ensure it's a 2D array [N, D]
            ep_ft_array = np.array(data)
            if ep_ft_array.ndim == 1:
                if key == "episode_index":
                    ep_ft_array = ep_ft_array.reshape(-1, 1)
                else:
                    feature_shape = features[key]["shape"]
                    if len(feature_shape) > 1:
                        ep_ft_array = ep_ft_array.reshape(-1, np.prod(feature_shape))
                    else:
                        ep_ft_array = ep_ft_array.reshape(-1, 1)

            axes_to_reduce = (0,)  # calculate stats on the first dimension
            keepdims = True

        try:
            ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

            if features[key]["dtype"] in ["image", "video"]:
                value_norm = 1.0 if "depth" in key else 255.0
                # Remove batch dimension and normalize: [T, C, 1, 1] -> [C, 1, 1]
                ep_stats[key] = {
                    k: v if k == "count" else np.squeeze(v / value_norm, axis=0)
                    for k, v in ep_stats[key].items()
                }
        except Exception as e:
            logger.warning(f"Failed to calculate stats for feature {key}: {e}")
            continue

    return ep_stats


class NavDatasetMetadata(LeRobotDatasetMetadata):
    """Metadata class for NavDataset, supporting image sequences."""

    def get_data_file_path(self, ep_index: int) -> Path:
        chunk = self.get_episode_chunk(ep_index)
        return Path("data") / f"chunk-{chunk:03d}" / f"episode_{ep_index:06d}.parquet"

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        chunk = self.get_episode_chunk(ep_index)
        # Include episode_index in the path to separate different episodes
        # Format: videos/chunk-000/observation.images.rgb/episode_000000/
        return Path("videos") / f"chunk-{chunk:03d}" / vid_key / f"episode_{ep_index:06d}"

    def update_video_info(self) -> None:
        """
        Skip video info update for image sequences.
        The base method tries to open video files with av.open(), but we use image sequences.
        """
        # For image sequences, we don't need to extract video info
        pass

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
    ) -> None:
        """extend the base class's save_episode method, add action_config support"""
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()

        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        from streamvln.dataset.lerobot.common.datasets.compute_stats import aggregate_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)


class NavDataset(LeRobotDataset):
    """
    Navigation dataset that supports image sequences instead of video files.

    This class extends LeRobotDataset to handle VLN (Vision-Language Navigation)
    data where observations are stored as sequences of JPEG images in directories
    rather than as video files.
    """

    def _query_videos(
        self, query_timestamps: dict[str, list[float]], ep_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Override to handle image sequences (directories) instead of video files.
        Each episode has its own subdirectory: videos/chunk-000/observation.images.rgb/episode_000000/
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            # Get the video directory path
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)

            if video_path.is_dir():
                # Image sequence: load images from directory
                # Sort image files to get correct order
                img_files = sorted(video_path.glob("*.jpg"), key=lambda x: int(x.stem))
                frames = []
                for ts in query_ts:
                    # Find the closest frame based on timestamp
                    frame_idx = int(round(ts * self.fps))
                    if 0 <= frame_idx < len(img_files):
                        # Load image
                        img_path = img_files[frame_idx]
                        img = PIL.Image.open(img_path)
                        # Convert to numpy and then to tensor [C,H,W]
                        img_array = np.array(img)
                        # Convert from [H,W,C] to [C,H,W]
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                        frames.append(img_tensor)
                    else:
                        # Use first frame as fallback
                        img_path = img_files[0]
                        img = PIL.Image.open(img_path)
                        img_array = np.array(img)
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                        frames.append(img_tensor)

                # Stack frames: [T, C, H, W]
                if frames:
                    item[vid_key] = torch.stack(frames)
                else:
                    # Fallback: return zeros
                    c, h, w = self.features[vid_key]["shape"]
                    item[vid_key] = torch.zeros((len(query_ts), c, h, w), dtype=torch.uint8)
            else:
                # Video file: use base class method
                from streamvln.dataset.lerobot.common.datasets.video_utils import decode_video_frames
                frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
                item[vid_key] = frames.squeeze(0)

        return item

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        """
        Override __init__ to handle image sequences (directories) instead of video files.
        """
        # Import constants
        from streamvln.dataset.lerobot.common.constants import HF_LEROBOT_HOME
        from streamvln.dataset.lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
        import packaging.version

        # Initialize base attributes without calling super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata using NavDatasetMetadata to support image sequences
        self.meta = NavDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            from streamvln.dataset.lerobot.common.datasets.compute_stats import aggregate_stats
            self.stats = aggregate_stats(episodes_stats)

        # Load actual data - skip the file check for image sequences
        self.hf_dataset = self.load_hf_dataset()
        # Apply torch transform to convert Columns to Tensors
        self.hf_dataset.set_transform(hf_transform_to_torch)

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps - need to actually retrieve data to apply transform
        # Apply transform by accessing the data through the dataset
        timestamp_list = [self.hf_dataset[i]["timestamp"] for i in range(len(self.hf_dataset))]
        timestamps = torch.stack(timestamp_list).numpy()

        episode_index_list = [self.hf_dataset[i]["episode_index"] for i in range(len(self.hf_dataset))]
        episode_indices = torch.stack(episode_index_list).numpy()

        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            from streamvln.dataset.lerobot.common.datasets.utils import check_delta_timestamps, get_delta_indices
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def _check_local_files_exist(self) -> bool:
        """
        Override to check if local files exist, allowing directories for image sequences.
        """
        try:
            # Check data files (must be files)
            data_paths = [self.meta.get_data_file_path(ep_idx) for ep_idx in range(self.meta.total_episodes)]
            for fpath in data_paths:
                if not (self.root / fpath).is_file():
                    return False

            # Check video/image directories (can be directories for image sequences)
            if len(self.meta.video_keys) > 0:
                for ep_idx in range(self.meta.total_episodes):
                    for vid_key in self.meta.video_keys:
                        fpath = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
                        # For image sequences, check if directory exists and has files
                        if fpath.is_dir():
                            # Check if directory has at least one file
                            if not any(fpath.iterdir()):
                                return False
                        elif fpath.is_file():
                            # For actual video files, check if file exists
                            if not fpath.exists():
                                return False
                        else:
                            return False
            return True
        except Exception:
            return False

    def get_episodes_file_paths(self) -> list[Path]:
        """
        Override to handle image sequences (directories) instead of video files.
        For image sequences, we check if the directory exists and contains files.
        """
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        # For image sequences, we include the directory paths
        if len(self.meta.video_keys) > 0:
            video_dirs = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_dirs
        return fpaths

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
    ) -> "NavDataset":
        obj = cls.__new__(cls)
        obj.meta = NavDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        obj.episode_buffer = obj.create_episode_buffer()
        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        return obj

    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:

        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        # Use self.features directly for validation (video features are not in hf_features)
        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        # Add frame features to episode_buffer
        for key, value in frame.items():
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            self.episode_buffer[key].append(value)

        self.episode_buffer["size"] += 1

    def save_episode(self, files: dict, episode_index: int) -> None:
        """
        Extend the base class's save_episode method, add video file copying and image directory copying support.

        Args:
            files: Dictionary mapping feature keys to file/directory paths
            episode_index: The episode index to use for this episode
        """
        if not self.episode_buffer:
            return

        episode_buffer = self.episode_buffer
        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index already processed, image and video are handled separately
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["video", "image"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()

        # handle video/image files - copy from source directory
        for key, source_dir in files.items():
            if key.startswith("observation.images."):
                video_dir = self.root / self.meta.get_video_file_path(episode_index, key)
                video_dir.mkdir(parents=True, exist_ok=True)

                source_path = Path(source_dir)
                if source_path.is_file():
                    # Single video file - copy to episode_filename.mp4
                    dest_file = video_dir / f"episode_{episode_index:06d}.mp4"
                    shutil.copyfile(source_path, dest_file)
                    episode_buffer[key] = str(dest_file)
                elif source_path.is_dir():
                    # Directory of image files - copy all to the video directory
                    for img_file in source_path.glob("*"):
                        if img_file.is_file():
                            shutil.copy2(img_file, video_dir / img_file.name)
                    # For image sequences, store the directory path in parquet
                    episode_buffer[key] = str(video_dir)

        ep_stats = compute_episode_stats(episode_buffer, self.features)
        self._save_episode_table(episode_buffer, episode_index)

        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        self.episode_buffer = self.create_episode_buffer()

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        """save episode data to parquet file"""
        from datasets import concatenate_datasets

        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)


def get_streamvln_features() -> dict:
    """
    Define the feature structure of StreamVLN dataset.

    Returns:
        feature definition dictionary
    """
    return {
        "observation.images.rgb": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channel", "height", "width"]
        },
        "action": {"dtype": "int64", "shape": (1,), "names": ["action_index"]},
    }
