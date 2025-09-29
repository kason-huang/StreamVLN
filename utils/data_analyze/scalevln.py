#!/usr/bin/env python3
"""
ScaleVLN Dataset Analysis Script
Analyzes the structure and content of ScaleVLN vision-and-language navigation dataset.

Usage:
    python utils/data_analyze/scalevln.py
    python utils/data_analyze/scalevln.py --data_path data/datasets/scalevln/scalevln_subset_150k.json.gz
    python utils/data_analyze/scalevln.py --episode_id 5 --detailed
"""

import argparse
import json
import gzip
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional


def load_scalevln_dataset(data_path: str) -> Dict[str, Any]:
    """Load ScaleVLN dataset from gzipped JSON file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def analyze_dataset_structure(data: Dict[str, Any]) -> None:
    """Print overall dataset structure and statistics."""
    print("=" * 80)
    print("SCALEVLN DATASET STRUCTURE ANALYSIS")
    print("=" * 80)

    episodes = data['episodes']
    vocab = data['instruction_vocab']

    print(f"\n📊 DATASET OVERVIEW:")
    print(f"  Total Episodes: {len(episodes):,}")
    print(f"  Vocabulary Size: {len(vocab['word_list'])} (EMPTY)")
    print(f"  Top-level Keys: {list(data.keys())}")

    print(f"\n📝 VOCABULARY INFO:")
    print(f"  Word list length: {len(vocab['word_list'])} (empty)")
    print(f"  Word2idx dict size: {len(vocab['word2idx_dict'])} (empty)")
    print(f"  STOI dict size: {len(vocab['stoi'])} (empty)")
    print(f"  ITOS list size: {len(vocab['itos'])} (empty)")
    print(f"  UNK Index: {vocab['UNK_INDEX']}")
    print(f"  PAD Index: {vocab['PAD_INDEX']}")


def analyze_episodes(episodes: List[Dict[str, Any]], sample_size: int = 100) -> None:
    """Analyze episode structure and statistics."""
    print(f"\n🎯 EPISODE ANALYSIS (first {sample_size} episodes):")

    # Check keys consistency
    all_keys = set()
    for episode in episodes[:sample_size]:
        all_keys.update(episode.keys())
    print(f"  Episode Keys: {sorted(all_keys)}")

    # Check for None values
    none_geodesic = 0
    none_tokens = 0
    valid_episodes = []

    for ep in episodes[:sample_size]:
        if ep['info']['geodesic_distance'] is None:
            none_geodesic += 1
        if ep['instruction']['instruction_tokens'] is None:
            none_tokens += 1
        if ep['info']['geodesic_distance'] is not None and ep['instruction']['instruction_tokens'] is not None:
            valid_episodes.append(ep)

    print(f"  Episodes with None geodesic_distance: {none_geodesic}/{sample_size}")
    print(f"  Episodes with None instruction_tokens: {none_tokens}/{sample_size}")
    print(f"  Valid episodes (both fields present): {len(valid_episodes)}/{sample_size}")

    # Scene analysis
    scene_ids = [ep['scene_id'] for ep in episodes[:sample_size]]
    unique_scenes = len(set(scene_ids))
    scene_types = [s.split('/')[0] for s in scene_ids]
    scene_type_counts = Counter(scene_types)

    print(f"  Scene Types: {dict(scene_type_counts)}")
    print(f"  Unique Scenes: {unique_scenes}/{sample_size}")

    # Goal radius analysis
    radii = [ep['goals'][0]['radius'] for ep in episodes[:sample_size]]
    unique_radii = list(set(radii))
    print(f"  Goal Radii: {unique_radii}")

    # Path lengths (available since reference_path is always present)
    path_lengths = [len(ep['reference_path']) for ep in episodes[:sample_size]]
    print(f"  Reference Path - Min: {min(path_lengths)} waypoints, Max: {max(path_lengths)} waypoints, Avg: {sum(path_lengths)/len(path_lengths):.1f} waypoints")


def analyze_trajectory_patterns(episodes: List[Dict[str, Any]], sample_size: int = 100) -> None:
    """Analyze trajectory ID patterns and structure."""
    print(f"\n🔗 TRAJECTORY PATTERN ANALYSIS (first {sample_size} episodes):")

    trajectory_ids = [ep['trajectory_id'] for ep in episodes[:sample_size]]

    # Analyze patterns
    has_scalevln = sum(1 for tid in trajectory_ids if 'scalevln_' in tid)
    has_fake = sum(1 for tid in trajectory_ids if '__fake_' in tid)

    print(f"  Trajectory IDs with 'scalevln_' prefix: {has_scalevln}/{sample_size}")
    print(f"  Trajectory IDs with '__fake_' pattern: {has_fake}/{sample_size}")

    # Sample trajectory IDs
    print(f"  Sample Trajectory IDs:")
    for i, tid in enumerate(trajectory_ids[:5]):
        print(f"    {i+1}: {tid}")

    # Parse trajectory ID structure
    parsed_patterns = defaultdict(int)
    for tid in trajectory_ids[:20]:
        if '__fake_' in tid:
            parts = tid.split('__fake_')
            if len(parts) == 2:
                parsed_patterns['with_fake_suffix'] += 1
        if 'scalevln_' in tid:
            parsed_patterns['with_scalevln_prefix'] += 1

    print(f"  Pattern Analysis: {dict(parsed_patterns)}")


def print_complete_example(data: Dict[str, Any], episode_id: int = 0, detailed: bool = False) -> None:
    """Print a complete example episode."""
    episodes = data['episodes']

    if episode_id >= len(episodes):
        print(f"Error: Episode ID {episode_id} out of range (0-{len(episodes)-1})")
        return

    example_episode = episodes[episode_id]

    print("=" * 80)
    print(f"COMPLETE SCALEVLN DATASET EXAMPLE - Episode {episode_id}")
    print("=" * 80)

    print(f"\n📋 EPISODE METADATA:")
    print(f"  Episode ID: {example_episode['episode_id']}")
    print(f"  Trajectory ID: {example_episode['trajectory_id']}")
    print(f"  Scene ID: {example_episode['scene_id']}")

    print(f"\n🎯 STARTING POSITION:")
    print(f"  Position: {example_episode['start_position']}")
    print(f"  Rotation: {example_episode['start_rotation']}")

    print(f"\n🏆 GOAL:")
    print(f"  Target: {example_episode['goals'][0]['position']}")
    print(f"  Success Radius: {example_episode['goals'][0]['radius']} meters")

    print(f"\n📊 METADATA:")
    geodesic_dist = example_episode['info']['geodesic_distance']
    print(f"  Geodesic Distance: {geodesic_dist} ({'None' if geodesic_dist is None else f'{geodesic_dist:.2f}m'})")

    print(f"\n💬 INSTRUCTION:")
    print(f"  Text: {example_episode['instruction']['instruction_text']}")
    tokens = example_episode['instruction']['instruction_tokens']
    print(f"  Tokens: {tokens} ({'None' if tokens is None else f'{len(tokens)} tokens'})")

    print(f"\n🗺️  REFERENCE PATH:")
    ref_path = example_episode['reference_path']
    print(f"  Number of waypoints: {len(ref_path)}")

    if detailed:
        for i, waypoint in enumerate(ref_path):
            print(f"    Waypoint {i+1}: {waypoint}")

        # Calculate elevation changes
        if len(ref_path) > 1:
            elevations = [wp[1] for wp in ref_path]  # Y-coordinate is elevation
            elevation_change = max(elevations) - min(elevations)
            print(f"  Elevation Change: {elevation_change:.2f}m")
            print(f"  Elevation Range: {min(elevations):.2f}m to {max(elevations):.2f}m")
    else:
        # Show first few and last waypoint
        for i, waypoint in enumerate(ref_path[:3]):
            print(f"    Waypoint {i+1}: {waypoint}")
        if len(ref_path) > 6:
            print(f"    ... ({len(ref_path) - 6} waypoints omitted)")
            for i, waypoint in enumerate(ref_path[-3:], len(ref_path) - 2):
                print(f"    Waypoint {i}: {waypoint}")


def analyze_instructions(episodes: List[Dict[str, Any]], sample_size: int = 1000) -> None:
    """Analyze instruction patterns despite missing tokenization."""
    print(f"\n💬 INSTRUCTION ANALYSIS (first {min(sample_size, len(episodes))} episodes):")

    # Since tokens are None, analyze text directly
    instructions = [ep['instruction']['instruction_text'] for ep in episodes[:sample_size]]
    word_counts = [len(instr.split()) for instr in instructions]

    print(f"  Word Count - Min: {min(word_counts)}, Max: {max(word_counts)}, Avg: {sum(word_counts)/len(word_counts):.1f}")

    # Most common words
    all_words = []
    for instr in instructions:
        all_words.extend(instr.lower().split())

    word_counts_dict = Counter(all_words)
    most_common = word_counts_dict.most_common(20)
    print(f"  Most Common Words: {[word for word, count in most_common]}")

    # Navigation keywords analysis
    nav_keywords = ['left', 'right', 'forward', 'straight', 'turn', 'go', 'stop', 'continue', 'walk', 'up', 'down', 'stairs']
    nav_counts = {word: word_counts_dict.get(word, 0) for word in nav_keywords}
    print(f"  Navigation Keywords: {nav_counts}")

    # Multi-level navigation indicators
    multi_level_words = ['stairs', 'up', 'down', 'floor', 'level', 'elevator']
    multi_level_counts = {word: word_counts_dict.get(word, 0) for word in multi_level_words}
    print(f"  Multi-level Navigation Words: {multi_level_counts}")

    # Check for sentence structure
    sentence_counts = [instr.count('.') for instr in instructions]
    avg_sentences = sum(sentence_counts) / len(sentence_counts)
    print(f"  Average Sentences per Instruction: {avg_sentences:.1f}")


def analyze_navigation_complexity(episodes: List[Dict[str, Any]], sample_size: int = 100) -> None:
    """Analyze navigation complexity based on reference paths."""
    print(f"\n🧭 NAVIGATION COMPLEXITY ANALYSIS (first {sample_size} episodes):")

    # Analyze elevation changes
    elevation_changes = []
    path_lengths = []

    for ep in episodes[:sample_size]:
        ref_path = ep['reference_path']
        path_lengths.append(len(ref_path))

        if len(ref_path) > 1:
            elevations = [wp[1] for wp in ref_path]  # Y-coordinate
            elevation_change = max(elevations) - min(elevations)
            elevation_changes.append(elevation_change)

    print(f"  Path Lengths - Min: {min(path_lengths)}, Max: {max(path_lengths)}, Avg: {sum(path_lengths)/len(path_lengths):.1f} waypoints")
    print(f"  Elevation Changes - Min: {min(elevation_changes):.2f}m, Max: {max(elevation_changes):.2f}m, Avg: {sum(elevation_changes)/len(elevation_changes):.2f}m")

    # Classify navigation types
    flat_navigation = sum(1 for change in elevation_changes if change < 0.5)
    moderate_elevation = sum(1 for change in elevation_changes if 0.5 <= change < 2.0)
    high_elevation = sum(1 for change in elevation_changes if change >= 2.0)

    print(f"  Navigation Types:")
    print(f"    Flat navigation (<0.5m elevation change): {flat_navigation}/{len(elevation_changes)}")
    print(f"    Moderate elevation (0.5-2.0m): {moderate_elevation}/{len(elevation_changes)}")
    print(f"    High elevation (>=2.0m): {high_elevation}/{len(elevation_changes)}")


def compare_with_other_datasets() -> None:
    """Compare ScaleVLN dataset characteristics with R2R and EnvDrop."""
    print(f"\n🔄 COMPARISON WITH OTHER DATASETS:")
    print(f"  ScaleVLN: ~155K episodes, HM3D scenes, no vocabulary, synthetic trajectories")
    print(f"  EnvDrop: ~146K episodes, MP3D scenes, 2,504 vocab, multi-level focus")
    print(f"  R2R: ~11K episodes, MP3D scenes, 2,711 vocab, standard VLN")
    print(f"  Key ScaleVLN features:")
    print(f"    - Largest dataset size")
    print(f"    - HM3D environments (more diverse than MP3D)")
    print(f"    - Synthetic/augmented trajectories ('fake' in IDs)")
    print(f"    - Precise goal radius (0.25m vs 3.0m)")
    print(f"    - No preprocessing (no tokenization, no distances)")


def main():
    parser = argparse.ArgumentParser(description='Analyze ScaleVLN dataset structure')
    parser.add_argument('--data_path', type=str, default='data/datasets/scalevln/scalevln_subset_150k.json.gz',
                       help='Path to ScaleVLN dataset file')
    parser.add_argument('--episode_id', type=int, default=0,
                       help='Specific episode ID to display (default: 0)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed waypoint information and elevation analysis')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only show analysis, no complete example')
    parser.add_argument('--sample_size', type=int, default=100,
                       help='Number of episodes to sample for analysis')
    parser.add_argument('--compare', action='store_true',
                       help='Show comparison with other datasets')

    args = parser.parse_args()

    try:
        # Load dataset
        print(f"Loading dataset from: {args.data_path}")
        data = load_scalevln_dataset(args.data_path)

        # Analysis
        analyze_dataset_structure(data)
        analyze_episodes(data['episodes'], args.sample_size)
        analyze_trajectory_patterns(data['episodes'], args.sample_size)
        analyze_instructions(data['episodes'], args.sample_size)
        analyze_navigation_complexity(data['episodes'], args.sample_size)

        if args.compare:
            compare_with_other_datasets()

        # Complete example
        if not args.analyze_only:
            print_complete_example(data, args.episode_id, args.detailed)

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()