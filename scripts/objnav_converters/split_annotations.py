#!/usr/bin/env python3
"""
Split annotations.json into multiple parts for parallel processing.
"""
import json
import argparse
import math
from pathlib import Path


def split_annotations(input_path, num_parts=3):
    """
    Split annotations.json into multiple parts.

    Args:
        input_path: Path to the input annotations.json
        num_parts: Number of parts to split into (default: 3)
    """
    input_path = Path(input_path)

    # Load annotations
    print(f"Loading annotations from {input_path}...")
    with open(input_path, 'r') as f:
        annotations = json.load(f)

    total_items = len(annotations)
    print(f"Total items: {total_items}")

    # Calculate chunk size
    chunk_size = math.ceil(total_items / num_parts)
    print(f"Splitting into {num_parts} parts (~{chunk_size} items each)")

    # Create output directory (same as input directory)
    output_dir = input_path.parent

    # Split and save
    output_files = []
    for i in range(num_parts):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_items)
        chunk = annotations[start_idx:end_idx]

        # Output filename
        output_path = output_dir / f"annotation_{i}.json"

        # Save chunk
        with open(output_path, 'w') as f:
            json.dump(chunk, f, indent=2)

        output_files.append(output_path)
        print(f"  Part {i}: {len(chunk)} items -> {output_path}")

    print(f"\nSuccessfully split into {len(output_files)} files!")
    return output_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split annotations.json into multiple parts')
    parser.add_argument('input_path', type=str,
                        help='Path to the input annotations.json file')
    parser.add_argument('--num-parts', type=int, default=3,
                        help='Number of parts to split into (default: 3)')
    args = parser.parse_args()

    split_annotations(args.input_path, args.num_parts)
