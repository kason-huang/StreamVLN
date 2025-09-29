#!/usr/bin/env python3
"""
RxR (Room-to-Room Extended) Dataset Analysis Script
Analyzes the structure and content of the multilingual RxR vision-and-language navigation dataset.

Usage:
    python utils/data_analyze/rxr.py
    python utils/data_analyze/rxr.py --data_dir data/datasets/rxr/train
    python utils/data_analyze/rxr.py --analyze_follower --episode_id 5
    python utils/data_analyze/rxr.py --analyze_guide --show_multilingual
"""

import argparse
import json
import gzip
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Union


def load_rxr_file(file_path: str) -> Union[Dict, List]:
    """Load RxR dataset file from gzipped JSON."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def analyze_follower_dataset(data_path: str, sample_size: int = 100) -> Dict[str, Any]:
    """Analyze the train_follower.json.gz dataset."""
    print(f"\n👥 ANALYZING FOLLOWER DATASET")
    print("-" * 50)

    data = load_rxr_file(data_path)
    episodes = data['episodes']

    print(f"Total episodes: {len(episodes):,}")

    # Language distribution
    languages = []
    instruction_lengths = []
    path_lengths = []
    scenes = []

    for ep in episodes[:sample_size]:
        if 'instruction' in ep:
            instr = ep['instruction']
            languages.append(instr.get('language', 'unknown'))
            if 'instruction_text' in instr:
                instruction_lengths.append(len(instr['instruction_text'].split()))

        if 'reference_path' in ep:
            path_lengths.append(len(ep['reference_path']))

        if 'scene_id' in ep:
            scenes.append(ep['scene_id'])

    language_counts = Counter(languages)
    unique_scenes = len(set(scenes))

    print(f"Languages: {dict(language_counts)}")
    print(f"Unique scenes: {unique_scenes}/{sample_size}")
    print(f"Instruction word count - Min: {min(instruction_lengths)}, Max: {max(instruction_lengths)}, Avg: {sum(instruction_lengths)/len(instruction_lengths):.1f}")
    print(f"Reference path length - Min: {min(path_lengths)}, Max: {max(path_lengths)}, Avg: {sum(path_lengths)/len(path_lengths):.1f}")

    return {
        'episodes': episodes,
        'stats': {
            'total_episodes': len(episodes),
            'language_distribution': dict(language_counts),
            'unique_scenes': unique_scenes,
            'avg_instruction_length': sum(instruction_lengths)/len(instruction_lengths),
            'avg_path_length': sum(path_lengths)/len(path_lengths)
        }
    }


def analyze_guide_gt_dataset(data_path: str, sample_size: int = 100) -> Dict[str, Any]:
    """Analyze the train_guide_gt.json.gz dataset."""
    print(f"\n🎯 ANALYZING GUIDE GT DATASET")
    print("-" * 50)

    data = load_rxr_file(data_path)

    print(f"Total entries: {len(data):,}")

    # Sample keys to understand structure
    sample_keys = list(data.keys())[:10]
    print(f"Sample keys: {sample_keys}")

    # Analyze a sample entry
    if sample_keys:
        first_key = sample_keys[0]
        first_entry = data[first_key]
        print(f"\nFirst entry (key: {first_key}):")
        print(f"  Keys: {list(first_entry.keys())}")

        if 'locations' in first_entry:
            print(f"  Locations: {len(first_entry['locations'])} points")
        if 'actions' in first_entry:
            print(f"  Actions: {len(first_entry['actions'])} actions")
        if 'forward_steps' in first_entry:
            print(f"  Forward steps: {first_entry['forward_steps']}")

    return {
        'data': data,
        'stats': {
            'total_entries': len(data),
            'sample_keys': sample_keys
        }
    }


def analyze_guide_dataset(data_path: str, sample_size: int = 100) -> Dict[str, Any]:
    """Analyze the train_guide.json.gz dataset."""
    print(f"\n🗺️  ANALYZING GUIDE DATASET")
    print("-" * 50)

    data = load_rxr_file(data_path)
    episodes = data['episodes']

    print(f"Total episodes: {len(episodes):,}")

    # Language distribution
    languages = []
    roles = []

    for ep in episodes[:sample_size]:
        if 'instruction' in ep and isinstance(ep['instruction'], dict):
            languages.append(ep['instruction'].get('language', 'unknown'))

        if 'info' in ep and isinstance(ep['info'], dict):
            roles.append(ep['info'].get('role', 'unknown'))

    language_counts = Counter(languages)
    role_counts = Counter(roles)

    print(f"Languages: {dict(language_counts)}")
    print(f"Roles: {dict(role_counts)}")

    # Examine first episode
    if len(episodes) > 0:
        first_ep = episodes[0]
        print(f"\nFirst episode keys: {list(first_ep.keys())}")

        if 'instruction' in first_ep and isinstance(first_ep['instruction'], dict):
            instr = first_ep['instruction']
            print(f"  Instruction language: {instr.get('language', 'unknown')}")
            print(f"  Instruction ID: {instr.get('instruction_id', 'unknown')}")

        if 'path' in first_ep:
            print(f"  Path length: {len(first_ep['path'])}")

    return {
        'episodes': episodes,
        'stats': {
            'total_episodes': len(episodes),
            'language_distribution': dict(language_counts),
            'role_distribution': dict(role_counts)
        }
    }


def print_follower_example(episodes: List[Dict], episode_id: int = 0, show_timing: bool = False) -> None:
    """Print a complete follower example."""
    if episode_id >= len(episodes):
        print(f"Error: Episode ID {episode_id} out of range (0-{len(episodes)-1})")
        return

    example = episodes[episode_id]

    print("=" * 80)
    print(f"RxR FOLLOWER EXAMPLE - Episode {episode_id}")
    print("=" * 80)

    print(f"\n📋 METADATA:")
    print(f"  Episode ID: {example['episode_id']}")
    print(f"  Scene ID: {example['scene_id']}")

    print(f"\n🎯 STARTING POSITION:")
    print(f"  Position: {example['start_position']}")
    print(f"  Rotation: {example['start_rotation']}")

    print(f"\n🏆 GOALS:")
    if 'goals' in example and len(example['goals']) > 0:
        goal = example['goals'][0]
        print(f"  Target: {goal['position']}")
        print(f"  Radius: {goal['radius']} meters")

    print(f"\n💬 INSTRUCTION:")
    instr = example['instruction']
    print(f"  Language: {instr.get('language', 'unknown')}")
    print(f"  Instruction ID: {instr.get('instruction_id', 'unknown')}")
    print(f"  Annotator ID: {instr.get('annotator_id', 'unknown')}")
    print(f"  Edit Distance: {instr.get('edit_distance', 'unknown')}")
    print(f"  Text: {instr.get('instruction_text', 'unknown')}")

    if show_timing and 'timed_instruction' in instr:
        timed = instr['timed_instruction']
        print(f"  Timed Instruction: {len(timed)} segments")
        if isinstance(timed, list) and len(timed) > 0:
            print(f"  First timing segment: {timed[0]}")

    print(f"\n🗺️  REFERENCE PATH:")
    if 'reference_path' in example:
        path = example['reference_path']
        print(f"  Waypoints: {len(path)}")
        for i, waypoint in enumerate(path[:3]):  # Show first 3
            print(f"    Waypoint {i+1}: {waypoint}")
        if len(path) > 3:
            print(f"    ... ({len(path) - 3} more waypoints)")

    print(f"\n📊 INFO:")
    if 'info' in example:
        info = example['info']
        for key, value in info.items():
            print(f"  {key}: {value}")


def print_guide_example(episodes: List[Dict], episode_id: int = 0) -> None:
    """Print a complete guide example."""
    if episode_id >= len(episodes):
        print(f"Error: Episode ID {episode_id} out of range (0-{len(episodes)-1})")
        return

    example = episodes[episode_id]

    print("=" * 80)
    print(f"RxR GUIDE EXAMPLE - Episode {episode_id}")
    print("=" * 80)

    print(f"\n📋 METADATA:")
    print(f"  Episode ID: {example['episode_id']}")
    print(f"  Trajectory ID: {example['trajectory_id']}")
    print(f"  Scene ID: {example['scene_id']}")

    print(f"\n🎯 STARTING POSITION:")
    print(f"  Position: {example['start_position']}")
    print(f"  Rotation: {example['start_rotation']}")

    print(f"\n💬 INSTRUCTION:")
    instr = example['instruction']
    print(f"  Language: {instr.get('language', 'unknown')}")
    print(f"  Instruction ID: {instr.get('instruction_id', 'unknown')}")
    print(f"  Text: {instr.get('instruction_text', 'unknown')}")

    print(f"\n🗺️  PATH:")
    if 'path' in example:
        path = example['path']
        print(f"  Waypoints: {len(path)}")
        for i, waypoint in enumerate(path[:3]):  # Show first 3
            print(f"    Waypoint {i+1}: {waypoint}")
        if len(path) > 3:
            print(f"    ... ({len(path) - 3} more waypoints)")

    print(f"\n📊 INFO:")
    if 'info' in example:
        info = example['info']
        for key, value in info.items():
            print(f"  {key}: {value}")


def analyze_multilingual_content(episodes: List[Dict], sample_size: int = 100) -> None:
    """Analyze multilingual content across episodes."""
    print(f"\n🌍 MULTILINGUAL ANALYSIS")
    print("-" * 50)

    language_samples = defaultdict(list)

    for ep in episodes[:sample_size]:
        if 'instruction' in ep and isinstance(ep['instruction'], dict):
            instr = ep['instruction']
            lang = instr.get('language', 'unknown')
            if 'instruction_text' in instr:
                language_samples[lang].append(instr['instruction_text'])

    print(f"Languages found: {list(language_samples.keys())}")

    for lang, texts in language_samples.items():
        if texts:
            avg_length = sum(len(text.split()) for text in texts) / len(texts)
            print(f"\n{lang}:")
            print(f"  Samples: {len(texts)}")
            print(f"  Avg length: {avg_length:.1f} words")
            print(f"  Example: {texts[0][:100]}...")


def compare_datasets(follower_stats: Dict, guide_stats: Dict) -> None:
    """Compare follower and guide datasets."""
    print(f"\n📊 DATASET COMPARISON")
    print("-" * 50)

    print(f"Follower Dataset:")
    print(f"  Episodes: {follower_stats.get('total_episodes', 'N/A'):,}")
    print(f"  Languages: {len(follower_stats.get('language_distribution', {}))}")
    print(f"  Avg instruction length: {follower_stats.get('avg_instruction_length', 'N/A'):.1f} words")

    print(f"\nGuide Dataset:")
    print(f"  Episodes: {guide_stats.get('total_episodes', 'N/A'):,}")
    print(f"  Languages: {len(guide_stats.get('language_distribution', {}))}")
    print(f"  Roles: {guide_stats.get('role_distribution', {})}")


def main():
    parser = argparse.ArgumentParser(description='Analyze RxR dataset structure')
    parser.add_argument('--data_dir', type=str, default='data/datasets/rxr/train',
                       help='Path to RxR dataset directory')
    parser.add_argument('--analyze_follower', action='store_true',
                       help='Analyze follower dataset')
    parser.add_argument('--analyze_guide', action='store_true',
                       help='Analyze guide dataset')
    parser.add_argument('--analyze_guide_gt', action='store_true',
                       help='Analyze guide ground truth dataset')
    parser.add_argument('--episode_id', type=int, default=0,
                       help='Specific episode ID to display')
    parser.add_argument('--show_timing', action='store_true',
                       help='Show timing information in instructions')
    parser.add_argument('--show_multilingual', action='store_true',
                       help='Show multilingual content analysis')
    parser.add_argument('--sample_size', type=int, default=100,
                       help='Number of episodes to sample for analysis')
    parser.add_argument('--compare', action='store_true',
                       help='Compare follower and guide datasets')

    args = parser.parse_args()

    try:
        # File paths
        follower_path = os.path.join(args.data_dir, 'train_follower.json.gz')
        guide_path = os.path.join(args.data_dir, 'train_guide.json.gz')
        guide_gt_path = os.path.join(args.data_dir, 'train_guide_gt.json.gz')

        print("=" * 80)
        print("RxR (Room-to-Room Extended) DATASET ANALYSIS")
        print("=" * 80)

        # Default behavior: analyze all datasets
        if not (args.analyze_follower or args.analyze_guide or args.analyze_guide_gt):
            args.analyze_follower = True
            args.analyze_guide = True
            args.analyze_guide_gt = True

        follower_data = None
        guide_data = None

        # Analyze datasets
        if args.analyze_follower:
            follower_data = analyze_follower_dataset(follower_path, args.sample_size)
            if not args.analyze_guide and not args.analyze_guide_gt:
                print_follower_example(follower_data['episodes'], args.episode_id, args.show_timing)

        if args.analyze_guide:
            guide_data = analyze_guide_dataset(guide_path, args.sample_size)
            if not args.analyze_follower and not args.analyze_guide_gt:
                print_guide_example(guide_data['episodes'], args.episode_id)

        if args.analyze_guide_gt:
            guide_gt_data = analyze_guide_gt_dataset(guide_gt_path, args.sample_size)

        # Additional analyses
        if args.show_multilingual:
            if follower_data:
                analyze_multilingual_content(follower_data['episodes'], args.sample_size)
            elif guide_data:
                analyze_multilingual_content(guide_data['episodes'], args.sample_size)

        if args.compare and follower_data and guide_data:
            compare_datasets(follower_data['stats'], guide_data['stats'])

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")

        # Print summary
        print(f"\n📋 SUMMARY:")
        print(f"RxR is a multilingual extension of R2R with:")
        print(f"  - Multi-lingual instructions (English, Hindi, Telugu)")
        print(f"  - Both follower and guide agent perspectives")
        print(f"  - Detailed timing information")
        print(f"  - Expert demonstrations with action sequences")
        print(f"  - MP3D scene environments")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()