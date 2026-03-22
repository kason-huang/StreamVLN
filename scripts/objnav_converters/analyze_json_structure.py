#!/usr/bin/env python3
"""
Analyze the data structure of HM3D ObjectNav trajectory data files.

Usage:
    python analyze_json_structure.py
"""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter


def analyze_value(value: Any, path: str = "root", max_depth: int = 5, current_depth: int = 0) -> Dict:
    """Recursively analyze the structure of a value."""
    if current_depth >= max_depth:
        return {"type": type(value).__name__, "note": "max depth reached"}

    if isinstance(value, dict):
        result = {
            "type": "dict",
            "length": len(value),
            "keys": list(value.keys()),
            "key_types": {k: type(v).__name__ for k, v in value.items()},
        }

        # Sample first few values for each key
        samples = {}
        for k, v in list(value.items())[:5]:  # Only first 5 keys
            samples[k] = analyze_value(v, f"{path}.{k}", max_depth, current_depth + 1)
        result["sample_values"] = samples

        return result

    elif isinstance(value, list):
        result = {
            "type": "list",
            "length": len(value),
        }

        if len(value) > 0:
            result["element_types"] = Counter(type(v).__name__ for v in value)

            # Sample first few elements
            samples = []
            for i, v in enumerate(value[:3]):
                samples.append(analyze_value(v, f"{path}[{i}]", max_depth, current_depth + 1))
            result["sample_elements"] = samples

            # Sample last element if different
            if len(value) > 3:
                result["last_element"] = analyze_value(
                    value[-1], f"{path}[{len(value)-1}]", max_depth, current_depth + 1
                )

        return result

    elif isinstance(value, str):
        return {"type": "str", "length": len(value), "value": value[:100] + ("..." if len(value) > 100 else "")}

    elif isinstance(value, (int, float, bool, type(None))):
        return {"type": type(value).__name__, "value": value}

    else:
        return {"type": type(value).__name__, "str": str(value)[:100]}


def find_all_paths(data: Any, base_path: str = "", max_paths: int = 1000) -> List[str]:
    """Find all paths to leaf nodes in the data structure."""
    paths = []

    def _find_paths(obj: Any, path: str = ""):
        if len(paths) >= max_paths:
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                if not isinstance(value, (dict, list)):
                    paths.append(new_path)
                else:
                    _find_paths(value, new_path)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                new_path = f"{path}[{i}]"
                if not isinstance(value, (dict, list)):
                    paths.append(new_path)
                else:
                    _find_paths(value, new_path)

    _find_paths(data, base_path)
    return paths


def analyze_file(file_path: str) -> Dict:
    """Analyze a JSON.gz file."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {file_path}")
    print(f"{'='*70}\n")

    # Load the file
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    # Basic info
    print("📦 File Information:")
    print(f"  File size: {Path(file_path).stat().st_size / (1024*1024):.2f} MB")
    print(f"  Top-level type: {type(data).__name__}")

    # Structure analysis
    print("\n🔍 Structure Analysis:")
    analysis = analyze_value(data)
    print(json.dumps(analysis, indent=2, ensure_ascii=False))

    # If data is a dict, show top-level keys
    if isinstance(data, dict):
        print("\n📋 Top-level Keys:")
        for key in data.keys():
            value = data[key]
            if isinstance(value, (dict, list)):
                print(f"  - {key}: {type(value).__name__} (length: {len(value)})")
            elif isinstance(value, str):
                print(f"  - {key}: str (length: {len(value)})")
            else:
                print(f"  - {key}: {type(value).__name__}")

    # Find common paths
    print("\n📍 Common Paths:")
    paths = find_all_paths(data)
    for path in paths[:50]:
        print(f"  {path}")

    # Check for specific ObjectNav-related fields
    print("\n🎯 ObjectNav-relevant Fields:")
    objnav_fields = [
        "episode_id", "episode_id", "scene_id", "object_category", "object_goal",
        "geodesic_distance", "start_position", "start_rotation", "goals",
        "actions", "observations", "rgb", "depth", "position", "rotation",
        "instruction", "trajectory", "info"
    ]

    found_fields = set()
    for field in objnav_fields:
        if any(f".{field}" in p or p.startswith(field) for p in paths):
            found_fields.add(field)
            print(f"  ✓ {field}")

    if not found_fields:
        print("  (No standard ObjectNav fields found)")

    # Detailed analysis for specific keys of interest
    if isinstance(data, dict):
        print("\n🔬 Detailed Analysis of Key Fields:")

        # Look for episode/trajectory data
        for key in ["episodes", "trajectory", "data", "instructions", "actions"]:
            if key in data:
                print(f"\n  [{key}]:")
                field_analysis = analyze_value(data[key], max_depth=3)
                print(json.dumps(field_analysis, indent=4, ensure_ascii=False))

    return analysis


def main():
    """Main entry point."""
    # Default file path
    file_path = "data/trajectory_data_hm3d_format/objectnav/cloudrobo_v1_l3mvn/train/content/suzhou-room-shengwei-metacam-2025-07-09_01-13-22.json.gz"

    # Check if file exists
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        print("\nSearching for similar files...")
        # Search for any json.gz files in the data directory
        data_dir = Path("data")
        if data_dir.exists():
            json_files = list(data_dir.rglob("*.json.gz"))
            if json_files:
                print(f"\nFound {len(json_files)} JSON.gz files:")
                for f in json_files[:10]:
                    print(f"  - {f}")
                if len(json_files) > 0:
                    file_path = str(json_files[0])
                    print(f"\n🔄 Analyzing first file: {file_path}")
            else:
                print("No JSON.gz files found in data directory")
                return
        else:
            print("data directory not found")
            return

    # Analyze the file
    analyze_file(file_path)


if __name__ == "__main__":
    main()
