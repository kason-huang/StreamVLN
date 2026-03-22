#!/usr/bin/env python
"""
Test script for LeRobot dataset integration

This script tests:
1. Basic LeRobot dataset loading
2. Adapter class conversion to training format
3. Sample data retrieval and format validation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import torch
import json

# Test 1: Direct LeRobot dataset loading
print("=" * 60)
print("Test 1: Direct LeRobot Dataset Loading")
print("=" * 60)

try:
    from lerobot_dataset import LeRobotActionDataset

    # The dataset is at ./data/lerobot/ directly, so use root="./data" and repo_id="lerobot"
    dataset = LeRobotActionDataset(
        repo_id="lerobot",
        root="./data",
        video_backend="auto",
    )

    print(f"✓ Dataset loaded successfully!")
    print(f"  Total episodes: {dataset.total_episodes}")
    print(f"  Total frames: {dataset.total_frames}")
    print(f"  FPS: {dataset.fps}")
    print(f"  Video backend: {dataset.video_backend}")

    # Test loading a single sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n✓ Sample loaded successfully!")
        print(f"  Sample keys: {list(sample.keys())}")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"    {key}: type={type(value).__name__}")

except Exception as e:
    print(f"✗ Failed to load dataset: {e}")
    import traceback
    traceback.print_exc()

# Test 2: LeRobot dataset with adapter
print("\n" + "=" * 60)
print("Test 2: LeRobot Dataset with Adapter")
print("=" * 60)

try:
    from transformers import AutoTokenizer
    from streamvln.dataset.lerobot_action_dataset import LeRobotActionDataset
    from streamvln.streamvln_train import LeRobotActionDatasetAdapter

    # Load a minimal tokenizer (for testing purposes)
    # In practice, you'd use the actual model tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    # Load LeRobot dataset
    print("Loading LeRobot dataset...")
    lerobot_dataset = LeRobotActionDataset(
        repo_id="lerobot",
        root="./data",
        video_backend="auto",
    )

    # Wrap with adapter
    print("Wrapping with adapter...")
    adapter_dataset = LeRobotActionDatasetAdapter(
        lerobot_dataset=lerobot_dataset,
        tokenizer=tokenizer,
        num_frames=8,  # Use fewer frames for testing
        num_future_steps=1,
        num_history=None,
    )

    print(f"✓ Adapter dataset created successfully!")
    print(f"  Total samples: {len(adapter_dataset)}")

    # Test loading a single sample from adapter
    if len(adapter_dataset) > 0:
        input_ids, labels, images, time_ids, task = adapter_dataset[0]
        print(f"\n✓ Adapter sample loaded successfully!")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  labels shape: {labels.shape}")
        print(f"  images shape: {images.shape}")
        print(f"  time_ids shape: {time_ids.shape}")
        print(f"  task: {task}")

except Exception as e:
    print(f"✗ Failed to create adapter: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Episode loading
print("\n" + "=" * 60)
print("Test 3: Episode Loading")
print("=" * 60)

try:
    from streamvln.dataset.lerobot_action_dataset import LeRobotActionDataset

    dataset = LeRobotActionDataset(
        repo_id="lerobot",
        root="./data",
        video_backend="auto",
    )

    if dataset.total_episodes > 0:
        episode_data = dataset.get_episode(0)
        print(f"✓ Episode loaded successfully!")
        print(f"  Instruction: {episode_data['instruction'][:60]}...")
        print(f"  Actions: {len(episode_data['actions'])} actions")
        print(f"  Images: {len(episode_data['images'])} images")
        print(f"  Image shape: {episode_data['images'][0].shape if episode_data['images'] else 'N/A'}")
        print(f"  First few actions: {episode_data['actions'][:5]}")

except Exception as e:
    print(f"✗ Failed to load episode: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
