#!/usr/bin/env python3
"""
EnvDrop Dataset Analysis Script
Analyzes the structure and content of EnvDrop vision-and-language navigation dataset.

Usage:
    python utils/data_analyze/envdrop.py
    python utils/data_analyze/envdrop.py --data_path data/datasets/envdrop/envdrop.json.gz
    python utils/data_analyze/envdrop.py --episode_id 5 --show_tokens
"""

import argparse
import json
import gzip
import os
import sys
from collections import Counter
from typing import Dict, List, Any


def load_envdrop_dataset(data_path: str) -> Dict[str, Any]:
    """Load EnvDrop dataset from gzipped JSON file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def analyze_dataset_structure(data: Dict[str, Any]) -> None:
    """Print overall dataset structure and statistics."""
    print("=" * 80)
    print("ENVDROP DATASET STRUCTURE ANALYSIS")
    print("=" * 80)

    episodes = data['episodes']
    vocab = data['instruction_vocab']

    print(f"\n📊 DATASET OVERVIEW:")
    print(f"  Total Episodes: {len(episodes):,}")
    print(f"  Vocabulary Size: {len(vocab['word_list']):,}")
    print(f"  Top-level Keys: {list(data.keys())}")

    print(f"\n📝 VOCABULARY INFO:")
    print(f"  Word list length: {len(vocab['word_list']):,}")
    print(f"  Word2idx dict size: {len(vocab['word2idx_dict']):,}")
    print(f"  STOI dict size: {len(vocab['stoi']):,}")
    print(f"  ITOS list size: {len(vocab['itos']):,}")
    print(f"  UNK Index: {vocab['UNK_INDEX']}")
    print(f"  PAD Index: {vocab['PAD_INDEX']}")
    print(f"  Sample vocabulary words: {vocab['word_list'][:20]}")


def analyze_episodes(episodes: List[Dict[str, Any]], sample_size: int = 100) -> None:
    """Analyze episode structure and statistics."""
    print(f"\n🎯 EPISODE ANALYSIS (first {sample_size} episodes):")

    # Check keys consistency
    all_keys = set()
    for episode in episodes[:sample_size]:
        all_keys.update(episode.keys())
    print(f"  Episode Keys: {sorted(all_keys)}")

    # Statistics
    scene_ids = [ep['scene_id'] for ep in episodes[:sample_size]]
    unique_scenes = len(set(scene_ids))
    distances = [ep['info']['geodesic_distance'] for ep in episodes[:sample_size]]

    print(f"  Unique Scenes: {unique_scenes}")
    print(f"  Distance - Min: {min(distances):.2f}m, Max: {max(distances):.2f}m, Avg: {sum(distances)/len(distances):.2f}m")

    # Path lengths
    path_lengths = [len(ep['reference_path']) for ep in episodes[:sample_size]]
    print(f"  Reference Path - Min: {min(path_lengths)} waypoints, Max: {max(path_lengths)} waypoints, Avg: {sum(path_lengths)/len(path_lengths):.1f} waypoints")


def print_complete_example(data: Dict[str, Any], episode_id: int = 0, show_tokens: bool = False) -> None:
    """Print a complete example episode."""
    episodes = data['episodes']

    if episode_id >= len(episodes):
        print(f"Error: Episode ID {episode_id} out of range (0-{len(episodes)-1})")
        return

    example_episode = episodes[episode_id]
    vocab = data['instruction_vocab']

    print("=" * 80)
    print(f"COMPLETE ENVDROP DATASET EXAMPLE - Episode {episode_id}")
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
    print(f"  Geodesic Distance: {example_episode['info']['geodesic_distance']:.2f} meters")

    print(f"\n💬 INSTRUCTION:")
    print(f"  Text: {example_episode['instruction']['instruction_text']}")

    if show_tokens:
        tokens = example_episode['instruction']['instruction_tokens']
        print(f"  Tokens (all): {tokens}")

        # Convert token indices back to words
        itos = vocab['itos']
        decoded_tokens = [itos[idx] if idx < len(itos) else '<unk>' for idx in tokens]
        # Remove padding for cleaner display
        decoded_tokens_no_pad = [token for token in decoded_tokens if token != '<pad>']
        print(f"  Decoded (with padding): {' '.join(decoded_tokens)}")
        print(f"  Decoded (no padding): {' '.join(decoded_tokens_no_pad)}")
    else:
        tokens = example_episode['instruction']['instruction_tokens']
        print(f"  Tokens (first 20): {tokens[:20]}")

        # Show some decoded tokens
        itos = vocab['itos']
        decoded_sample = [itos[idx] if idx < len(itos) else '<unk>' for idx in tokens[:20]]
        print(f"  Decoded sample: {' '.join(decoded_sample)}")

    print(f"\n🗺️  REFERENCE PATH:")
    print(f"  Number of waypoints: {len(example_episode['reference_path'])}")
    for i, waypoint in enumerate(example_episode['reference_path']):
        print(f"    Waypoint {i+1}: {waypoint}")


def analyze_instructions(episodes: List[Dict[str, Any]], sample_size: int = 1000) -> None:
    """Analyze instruction patterns and statistics."""
    print(f"\n💬 INSTRUCTION ANALYSIS (first {min(sample_size, len(episodes))} episodes):")

    # Instruction lengths (without padding)
    instructions = []
    token_counts = []

    for ep in episodes[:sample_size]:
        tokens = ep['instruction']['instruction_tokens']
        # Remove padding tokens (index 0)
        tokens_no_pad = [t for t in tokens if t != 0]
        token_counts.append(len(tokens_no_pad))
        instructions.append(ep['instruction']['instruction_text'])

    word_counts = [len(instr.split()) for instr in instructions]

    print(f"  Token Count (no padding) - Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {sum(token_counts)/len(token_counts):.1f}")
    print(f"  Word Count - Min: {min(word_counts)}, Max: {max(word_counts)}, Avg: {sum(word_counts)/len(word_counts):.1f}")

    # Most common words
    all_words = []
    for instr in instructions:
        all_words.extend(instr.lower().split())

    word_counts_dict = Counter(all_words)
    most_common = word_counts_dict.most_common(20)
    print(f"  Most Common Words: {[word for word, count in most_common]}")

    # Direction words analysis
    direction_words = ['left', 'right', 'forward', 'straight', 'turn', 'go', 'stop', 'continue', 'walk', 'up', 'down', 'stairs']
    direction_counts = {word: word_counts_dict.get(word, 0) for word in direction_words}
    print(f"  Direction Words: {direction_counts}")

    # Check for multi-level navigation indicators
    multi_level_words = ['stairs', 'up', 'down', 'floor', 'level', 'elevator']
    multi_level_counts = {word: word_counts_dict.get(word, 0) for word in multi_level_words}
    print(f"  Multi-level Navigation Words: {multi_level_counts}")


def compare_with_r2r() -> None:
    """Compare EnvDrop dataset characteristics with R2R."""
    print(f"\n🔄 COMPARISON WITH R2R:")
    print(f"  EnvDrop is significantly larger: ~146K episodes vs ~11K in R2R")
    print(f"  EnvDrop has smaller vocabulary: 2,504 vs 2,711 in R2R")
    print(f"  EnvDrop includes explicit padding and unknown token handling")
    print(f"  EnvDrop features more multi-level navigation scenarios")
    print(f"  EnvDrop instructions are often longer and multi-sentence")


def main():
    parser = argparse.ArgumentParser(description='Analyze EnvDrop dataset structure')
    parser.add_argument('--data_path', type=str, default='data/datasets/envdrop/envdrop.json.gz',
                       help='Path to EnvDrop dataset file')
    parser.add_argument('--episode_id', type=int, default=0,
                       help='Specific episode ID to display (default: 0)')
    parser.add_argument('--show_tokens', action='store_true',
                       help='Show all instruction tokens and decoded text')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only show analysis, no complete example')
    parser.add_argument('--sample_size', type=int, default=100,
                       help='Number of episodes to sample for analysis')
    parser.add_argument('--compare', action='store_true',
                       help='Show comparison with R2R dataset')

    args = parser.parse_args()

    try:
        # Load dataset
        print(f"Loading dataset from: {args.data_path}")
        data = load_envdrop_dataset(args.data_path)

        # Analysis
        analyze_dataset_structure(data)
        analyze_episodes(data['episodes'], args.sample_size)
        analyze_instructions(data['episodes'], args.sample_size)

        if args.compare:
            compare_with_r2r()

        # Complete example
        if not args.analyze_only:
            print_complete_example(data, args.episode_id, args.show_tokens)

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()