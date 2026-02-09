#!/usr/bin/env python
"""
Lightweight LeRobot Dataset Loader for StreamVLN

This module provides a minimal implementation of LeRobot dataset loading
without requiring the full lerobot package. It supports:
- Multi-chunk datasets
- PyAV (preferred) or OpenCV video decoding
- Episode-level and frame-level indexing
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class LeRobotActionDataset(Dataset):
    """
    Lightweight LeRobot dataset loader for StreamVLN training.

    This is a minimal implementation that only depends on StreamVLN-compatible
    libraries: pandas, pyarrow, numpy, torch, pillow, and opencv-python.

    Supports:
    - Multi-chunk datasets (multiple parquet files and videos)
    - PyAV (preferred) or OpenCV video decoding with automatic fallback
    - Episode-level and frame-level indexing
    - Lazy loading for memory efficiency

    Args:
        repo_id: Repository ID (e.g., "streamvln/r2r_navigation")
        root: Root directory path (parent of repo_id)
        video_backend: Video decoding backend preference: "auto", "av", or "opencv"
    """

    def __init__(
        self,
        repo_id: str,
        root: str,
        video_backend: str = "auto",
    ):
        self.repo_id = repo_id
        self.root = Path(root) / repo_id

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        # Load info.json
        info_file = self.root / "meta" / "info.json"
        if not info_file.exists():
            raise FileNotFoundError(f"info.json not found: {info_file}")

        with open(info_file, "r") as f:
            self.info = json.load(f)

        # Extract basic info
        self.fps = self.info.get("fps", 30)
        self.total_episodes = self.info.get("total_episodes", 0)
        self.total_frames = self.info.get("total_frames", 0)
        self.video_key = "observation.images.rgb"

        # Load ALL episodes metadata from all chunks
        self.episodes_df = self._load_all_episodes()

        # Build cumulative frame count mapping for efficient indexing
        if not self.episodes_df.empty:
            self.episodes_df = self.episodes_df.sort_values('episode_index').reset_index(drop=True)
            self.episodes_df['_cumsum'] = self.episodes_df['length'].cumsum()
        else:
            self.episodes_df['_cumsum'] = pd.Series([], dtype=int)

        # Load tasks metadata
        self.tasks_df = self._load_tasks()

        # Initialize video backend with fallback mechanism
        self.video_backend = self._init_video_backend(video_backend)

        # Find all data chunks
        self.data_chunks = self._find_data_chunks()

        # Find all video chunks
        self.video_chunks = self._find_video_chunks()

    def _load_all_episodes(self) -> pd.DataFrame:
        """Load episode metadata from all chunks."""
        all_episodes = []
        episodes_dir = self.root / "meta" / "episodes"

        if not episodes_dir.exists():
            warnings.warn(f"Episodes directory not found: {episodes_dir}")
            # Try to extract episode info from data files
            return self._extract_episodes_from_data()

        # Find all chunk directories (sorted)
        chunk_dirs = sorted(episodes_dir.glob("chunk-*"), key=lambda x: int(x.name.split("-")[1]))

        for chunk_dir in chunk_dirs:
            chunk_files = sorted(chunk_dir.glob("file-*.parquet"),
                                 key=lambda x: int(x.stem.split("-")[1]))
            for chunk_file in chunk_files:
                try:
                    df = pd.read_parquet(chunk_file)
                    all_episodes.append(df)
                except Exception as e:
                    warnings.warn(f"Failed to load episodes from {chunk_file}: {e}")

        if not all_episodes:
            warnings.warn("Failed to load episodes from metadata, extracting from data files")
            return self._extract_episodes_from_data()

        # Concatenate all episode metadata
        episodes_df = pd.concat(all_episodes, ignore_index=True)
        return episodes_df

    def _extract_episodes_from_data(self) -> pd.DataFrame:
        """Extract episode information from info.json when parquet metadata is unavailable."""
        episodes_list = []

        # Use info from info.json to create episode structure
        # The info.json contains total_episodes and total_frames
        total_episodes = self.info.get('total_episodes', self.total_episodes)
        total_frames = self.info.get('total_frames', self.total_frames)

        # If we have info about the number of episodes, distribute frames evenly
        # This is a fallback approximation - in production you'd want proper metadata
        if total_episodes > 0 and total_frames > 0:
            # Calculate average frames per episode
            avg_frames_per_episode = total_frames // total_episodes

            # Create episode entries
            for i in range(total_episodes):
                start_idx = i * avg_frames_per_episode
                if i == total_episodes - 1:
                    # Last episode gets remaining frames
                    end_idx = total_frames
                else:
                    end_idx = (i + 1) * avg_frames_per_episode

                episodes_list.append({
                    'episode_index': i,
                    'length': end_idx - start_idx,
                    'dataset_from_index': start_idx,
                    'dataset_to_index': end_idx,
                })
        else:
            # Fallback: create a single episode from total_frames
            episodes_list.append({
                'episode_index': 0,
                'length': total_frames,
                'dataset_from_index': 0,
                'dataset_to_index': total_frames,
            })

        return pd.DataFrame(episodes_list)

    def _load_tasks(self) -> pd.DataFrame:
        """Load tasks metadata from parquet file."""
        tasks_file = self.root / "meta" / "tasks.parquet"

        if not tasks_file.exists():
            return pd.DataFrame()

        try:
            df = pd.read_parquet(tasks_file)
            # The tasks.parquet file has JSON strings as the index, not as a column
            # Reset index to make the JSON strings accessible as a column
            if df.index.name is not None or len(df.index) > 0:
                # Check if the index contains JSON strings
                if df.index.astype(str).str.startswith('{').any():
                    df = df.reset_index()
                    # Rename the index column to 'task_json' if it doesn't have a name
                    if 'index' in df.columns:
                        df = df.rename(columns={'index': 'task_json'})
            return df
        except Exception as e:
            warnings.warn(f"Failed to load tasks from {tasks_file}: {e}")
            return pd.DataFrame()

    def _init_video_backend(self, backend: str) -> str:
        """Initialize video decoding backend with fallback."""
        if backend == "auto":
            # Try PyAV first, fallback to OpenCV
            try:
                import av
                return "av"
            except ImportError:
                warnings.warn("PyAV not available, falling back to OpenCV")
                return "opencv"
        elif backend == "av":
            try:
                import av
                return "av"
            except ImportError:
                warnings.warn("PyAV not available, falling back to OpenCV")
                return "opencv"
        else:
            return "opencv"

    def _find_data_chunks(self) -> List[Path]:
        """Find all data chunk directories."""
        data_dir = self.root / "data"
        if not data_dir.exists():
            return []

        chunks = sorted(data_dir.glob("chunk-*"), key=lambda x: int(x.name.split("-")[1]))
        return chunks

    def _find_video_chunks(self) -> List[Path]:
        """Find all video chunk directories."""
        video_dir = self.root / "videos" / self.video_key
        if not video_dir.exists():
            return []

        chunks = sorted(video_dir.glob("chunk-*"), key=lambda x: int(x.name.split("-")[1]))
        return chunks

    def __len__(self) -> int:
        """Return total number of frames across all episodes."""
        return self.total_frames

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample by frame index.

        Args:
            idx: Global frame index across all episodes

        Returns:
            Dictionary with keys:
                - observation.images.rgb: torch.Tensor [3, H, W]
                - action: torch.Tensor (scalar or [1])
                - task: str (JSON string)
                - episode_index: torch.Tensor
                - frame_index: torch.Tensor
                - timestamp: torch.Tensor
                - task_index: torch.Tensor
        """
        if idx < 0 or idx >= self.total_frames:
            raise IndexError(f"Index {idx} out of range [0, {self.total_frames})")

        # Find episode from frame index using binary search on cumulative sum
        episode_idx = self._find_episode_from_frame_index(idx)
        episode_row = self.episodes_df[self.episodes_df['episode_index'] == episode_idx]

        if episode_row.empty:
            raise ValueError(f"Episode {episode_idx} not found in metadata")

        episode_start_idx = int(episode_row.iloc[0]['dataset_from_index'])
        episode_length = int(episode_row.iloc[0]['length'])
        frame_index_in_episode = idx - episode_start_idx

        # Find chunk and file containing this frame
        chunk_idx, file_idx, file_start_idx = self._locate_frame(idx, episode_idx)

        # Load frame data from parquet
        frame_data = self._load_frame_data(chunk_idx, file_idx, idx)

        # Decode video frame
        video_path = self._get_video_path(chunk_idx, file_idx)
        frame = self._decode_video_frame(video_path, frame_index_in_episode)

        # Convert to CHW format and normalize to [0, 1]
        if isinstance(frame, np.ndarray):
            if frame.max() <= 1.0:
                frame_tensor = torch.from_numpy(frame).float()
            else:
                frame_tensor = torch.from_numpy(frame).float() / 255.0
        else:
            frame_tensor = frame

        # HWC to CHW if needed
        if frame_tensor.dim() == 3:
            if frame_tensor.shape[-1] <= 4:  # HWC format
                frame_tensor = frame_tensor.permute(2, 0, 1)

        # Build sample dict
        action_value = frame_data.get('action', 0)
        if isinstance(action_value, (list, tuple, np.ndarray)):
            action_value = action_value[0] if len(action_value) > 0 else 0

        task_index = frame_data.get('task_index', 0)
        task = self._get_task_description(task_index)

        sample = {
            "observation.images.rgb": frame_tensor,
            "action": torch.tensor([action_value], dtype=torch.long),
            "task": task,
            "episode_index": torch.tensor(episode_idx, dtype=torch.long),
            "frame_index": torch.tensor(frame_index_in_episode, dtype=torch.long),
            "timestamp": torch.tensor(frame_index_in_episode / self.fps, dtype=torch.float32),
            "task_index": torch.tensor(task_index, dtype=torch.long),
        }

        return sample

    def _find_episode_from_frame_index(self, idx: int) -> int:
        """Find episode index from global frame index using binary search."""
        if self.episodes_df.empty:
            return 0

        cumsum = self.episodes_df['_cumsum'].values
        episode_idx = np.searchsorted(cumsum, idx + 1, side='right')
        return int(self.episodes_df.iloc[episode_idx]['episode_index'])

    def _locate_frame(self, idx: int, episode_idx: int) -> Tuple[int, int, int]:
        """
        Locate which chunk and file contains a given frame index.

        Returns:
            (chunk_idx, file_idx, file_start_idx)
        """
        # This is a simplified implementation that assumes linear distribution
        # For production, you should track this in metadata
        for chunk_idx, chunk_dir in enumerate(self.data_chunks):
            files = sorted(chunk_dir.glob("file-*.parquet"),
                          key=lambda x: int(x.stem.split("-")[1]))
            for file_idx, file_path in enumerate(files):
                try:
                    df = pd.read_parquet(file_path)
                    if 'dataset_from_index' in df.columns and 'dataset_to_index' in df.columns:
                        file_start = int(df.iloc[0]['dataset_from_index'])
                        file_end = int(df.iloc[-1]['dataset_to_index'])
                        if file_start <= idx < file_end:
                            return chunk_idx, file_idx, file_start
                except Exception:
                    continue

        # Fallback: assume chunk-000/file-000
        return 0, 0, 0

    def _load_frame_data(self, chunk_idx: int, file_idx: int, idx: int) -> Dict[str, Any]:
        """Load frame data from parquet file."""
        chunk_dir = self.data_chunks[chunk_idx]
        file_path = chunk_dir / f"file-{file_idx:03d}.parquet"

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        try:
            df = pd.read_parquet(file_path)
            # Find the row with matching index
            row = df[df['index'] == idx]
            if row.empty:
                # Fall back to positional index
                pos_in_file = idx - int(df.iloc[0]['dataset_from_index'])
                row = df.iloc[pos_in_file:pos_in_file+1]
            return row.iloc[0].to_dict()
        except Exception as e:
            raise RuntimeError(f"Failed to load frame data from {file_path}: {e}")

    def _get_video_path(self, chunk_idx: int, file_idx: int) -> Path:
        """Get video file path for a given chunk and file index."""
        if chunk_idx >= len(self.video_chunks):
            chunk_idx = 0

        video_chunk_dir = self.video_chunks[chunk_idx]
        video_path = video_chunk_dir / f"file-{file_idx:03d}.mp4"

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        return video_path

    def _decode_video_frame(self, video_path: Path, frame_idx: int) -> np.ndarray:
        """Decode frame using PyAV (preferred) or OpenCV (fallback)."""
        if self.video_backend == "av":
            return self._decode_video_frame_av(video_path, frame_idx)
        else:
            return self._decode_video_frame_opencv(video_path, frame_idx)

    def _decode_video_frame_av(self, video_path: Path, frame_idx: int) -> np.ndarray:
        """Decode frame using PyAV (preferred for efficiency)."""
        import av

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]

            # Seek to the closest keyframe (container.seek, not stream.seek)
            try:
                # container.seek takes timestamp in stream timebase
                # Calculate timestamp: frame_idx / fps
                timestamp = int(frame_idx / self.fps / stream.time_base)
                container.seek(timestamp, stream=stream)
            except Exception:
                # If seeking fails, start from beginning
                container.seek(0, stream=stream)

            current_frame = 0
            for packet in container.demux(stream):
                for frame in packet.decode():
                    if current_frame == frame_idx:
                        # Convert to RGB numpy array
                        img = frame.to_ndarray(format='rgb24')
                        return img
                    current_frame += 1

        raise RuntimeError(f"Frame {frame_idx} not found in {video_path}")

    def _decode_video_frame_opencv(self, video_path: Path, frame_idx: int) -> np.ndarray:
        """Decode frame using OpenCV (fallback)."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Failed to decode frame {frame_idx} from {video_path}")

        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def _get_task_description(self, task_index: int) -> str:
        """Get task description for a given task index."""
        if not self.tasks_df.empty:
            # Convert task_index to int (may be float or numpy type)
            try:
                task_index_int = int(task_index)
            except (ValueError, TypeError):
                task_index_int = None

            if task_index_int is not None and task_index_int < len(self.tasks_df):
                task_row = self.tasks_df.iloc[task_index_int]
                # Try different column names for task JSON
                for col in ['task_json', 'task', 'tasks', 'instruction', 'text']:
                    if col in task_row and pd.notna(task_row[col]):
                        task = task_row[col]
                        if isinstance(task, str):
                            return task
                        else:
                            return json.dumps(task)
        return '{"instruction": "Navigation task"}'

    def get_episode(self, episode_index: int) -> Dict[str, Any]:
        """
        Load all frames from a specific episode.

        Args:
            episode_index: Episode index

        Returns:
            Dictionary with:
                - instruction: str, the task instruction
                - actions: list of int, action values
                - images: list of numpy arrays (HWC format), image data
                - num_frames: int, number of frames
        """
        # Find episode in metadata
        episode_row = self.episodes_df[self.episodes_df['episode_index'] == episode_index]

        if episode_row.empty:
            raise ValueError(f"Episode {episode_index} not found")

        start_idx = int(episode_row.iloc[0]['dataset_from_index'])
        end_idx = int(episode_row.iloc[0]['dataset_to_index'])
        task_index = int(episode_row.iloc[0].get('task_index', 0))

        # Parse task to get instruction
        task_json = self._get_task_description(task_index)
        try:
            task_data = json.loads(task_json)
            instruction = task_data.get('instruction', '')
        except json.JSONDecodeError:
            instruction = ""

        actions = []
        images = []

        # Load all frames in episode
        for idx in range(start_idx, end_idx):
            try:
                sample = self[idx]

                # Get action
                action = sample['action']
                if hasattr(action, 'item'):
                    action = action.item()
                elif len(action.shape) > 0:
                    action = action[0].item()
                actions.append(action)

                # Get image (convert back to HWC numpy)
                img = sample['observation.images.rgb']
                if isinstance(img, torch.Tensor):
                    img = img.numpy()
                    if img.shape[0] <= 4:  # CHW format
                        img = np.transpose(img, (1, 2, 0))
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                images.append(img)

            except Exception as e:
                warnings.warn(f"Failed to load frame {idx}: {e}")
                continue

        return {
            "instruction": instruction,
            "actions": actions,
            "images": images,
            "num_frames": len(actions),
        }