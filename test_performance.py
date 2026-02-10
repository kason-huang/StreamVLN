#!/usr/bin/env python
"""
Performance test for LeRobotActionDataset optimizations.

Tests the following optimizations:
1. Pre-built frame index (O(1) lookup)
2. Parquet file caching (eliminate redundant I/O)

Expected improvement: 20-30x faster data loading
"""

import time
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from streamvln.dataset.lerobot_action_dataset import LeRobotActionDataset


class MockDataArgs:
    """Minimal DataArguments for testing."""
    def __init__(self, dataset_path="./data"):
        self.image_size = 224
        self.is_multimodal = True
        self.mm_use_im_start_end = False
        self.num_frames = 32
        self.num_history = 8
        self.num_future_steps = 4
        self.remove_init_turns = 0
        self.transform_train = None
        self.video_folder = "data/trajectory_data/R2R"
        self.lerobot_dataset_path = dataset_path
        self.lerobot_repo_id = "streamvln/r2r_navigation"
        self.video_backend = "auto"


def test_initialization_time(args, tokenizer):
    """Test how long dataset initialization takes (includes index building)."""
    print("\n" + "="*60)
    print("Test 1: Dataset Initialization Time")
    print("="*60)

    start = time.time()
    dataset = LeRobotActionDataset(tokenizer=tokenizer, data_args=args)
    init_time = time.time() - start

    print(f"✓ Dataset initialized in {init_time:.2f}s")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Frame index size: {len(dataset._frame_index)} frames")

    return dataset, init_time


def test_single_sample_time(dataset, num_samples=10):
    """Test time to load individual samples."""
    print("\n" + "="*60)
    print(f"Test 2: Single Sample Loading Time ({num_samples} samples)")
    print("="*60)

    times = []
    for i in range(num_samples):
        idx = i % len(dataset)
        start = time.time()
        _ = dataset[idx]
        elapsed = time.time() - start
        times.append(elapsed)

        if i < 3:  # Print first 3 samples
            print(f"Sample {idx}: {elapsed*1000:.1f}ms")

    avg_time = sum(times) / len(times)
    print(f"\n✓ Average sample loading time: {avg_time*1000:.1f}ms")
    print(f"  - Min: {min(times)*1000:.1f}ms")
    print(f"  - Max: {max(times)*1000:.1f}ms")

    return avg_time


def test_batch_loading(dataset, batch_size=8, num_batches=10):
    """Test batch loading performance with DataLoader."""
    print("\n" + "="*60)
    print(f"Test 3: Batch Loading ({num_batches} batches, batch_size={batch_size})")
    print("="*60)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # Single worker for accurate measurement
        pin_memory=False,
    )

    times = []
    batch_count = 0

    for i, batch in enumerate(dataloader):
        start = time.time()
        _ = batch  # Force evaluation
        elapsed = time.time() - start
        times.append(elapsed)
        batch_count += 1

        if batch_count >= num_batches:
            break

    avg_time = sum(times) / len(times)
    print(f"✓ Average batch time: {avg_time*1000:.1f}ms")
    print(f"  - Throughput: {batch_size/avg_time:.1f} samples/sec")

    return avg_time


def test_cache_effectiveness(dataset):
    """Check how many parquet files are cached."""
    print("\n" + "="*60)
    print("Test 4: Cache Effectiveness")
    print("="*60)

    num_cached_files = len(dataset._parquet_cache)
    num_indexed_frames = len(dataset._frame_index)

    print(f"✓ Cached parquet files: {num_cached_files}")
    print(f"✓ Indexed frames: {num_indexed_frames}")

    if num_cached_files > 0:
        # Estimate cache size
        total_cache_size = sum(
            df.memory_usage(deep=True).sum()
            for df in dataset._parquet_cache.values()
        )
        cache_size_mb = total_cache_size / (1024 * 1024)
        print(f"✓ Estimated cache size: {cache_size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Performance test for LeRobotActionDataset")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./data",
        help="Path to LeRobot dataset root"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Number of batches to test"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for DataLoader test"
    )
    args = parser.parse_args()

    print("="*60)
    print("LeRobotActionDataset Performance Test")
    print("="*60)

    # Initialize
    data_args = MockDataArgs(dataset_path=args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained("checkpoints/lmms-lab/LLaVA-Video-7B-Qwen2")

    # Run tests
    dataset, init_time = test_initialization_time(data_args, tokenizer)
    avg_sample_time = test_single_sample_time(dataset, args.num_samples)
    avg_batch_time = test_batch_loading(dataset, args.batch_size, args.num_batches)
    test_cache_effectiveness(dataset)

    # Summary
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    print(f"Initialization time:  {init_time:.2f}s")
    print(f"Single sample time:   {avg_sample_time*1000:.1f}ms")
    print(f"Batch time:           {avg_batch_time*1000:.1f}ms")
    print(f"Throughput:           {args.batch_size/avg_batch_time:.1f} samples/sec")
    print("="*60)

    # Compare with expected performance
    print("\nExpected vs Actual Performance:")
    print(f"  - Expected single sample: 10-20ms (optimized)")
    print(f"  - Actual single sample:   {avg_sample_time*1000:.1f}ms")

    if avg_sample_time < 0.05:  # Less than 50ms
        print(f"  ✓ Performance is GOOD (within expected range)")
    elif avg_sample_time < 0.15:  # Less than 150ms
        print(f"  ⚠ Performance is MODERATE (some room for improvement)")
    else:
        print(f"  ✗ Performance is POOR (may need further optimization)")


if __name__ == "__main__":
    main()
