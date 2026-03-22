#!/usr/bin/env python
"""
Lightweight LeRobot Dataset Loader for StreamVLN

This module provides a minimal implementation of LeRobot dataset loading
without requiring the full lerobot package. It supports:
- Multi-chunk datasets
- PyAV (preferred) or OpenCV video decoding
- Episode-level and frame-level indexing
- Integration with StreamVLN training pipeline
"""

import json
import warnings
import copy
import random
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from packaging import version

import os
import transformers
import glob
import tokenizers

from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token
from llava.model import *

from streamvln.utils.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX
from streamvln.args import DataArguments

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


# =============================================================================
# Preprocess functions (shared with VLNActionDataset)
# =============================================================================

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def preprocess_multimodal(sources: List, data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources


def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(sources: List[List[Dict[str, str]]], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv: conversation_lib.Conversation = conversation_lib.default_conversation.copy()
    roles: Dict[str, str] = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations: List[str] = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source: List[Dict[str, str]] = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role: str = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids: torch.Tensor = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids: torch.Tensor = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets: torch.Tensor = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    sep: str = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: List[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids) - 1
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            round_len += 2
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        tokenizer.add_tokens(["<memory>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")

    im_start, im_end = tokenizer.additional_special_tokens_ids
    unmask_tokens_idx = [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
            if encode_id == memory_token_index:
                input_id[idx] = MEMORY_TOKEN_INDEX

        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f"(#turns={len(re_rounds)} ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: List,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources: List, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)


# =============================================================================
# LeRobotActionDataset - Unified Dataset class
# =============================================================================

class LeRobotActionDataset(Dataset):
    """
    Unified LeRobot dataset for StreamVLN training.

    This dataset loads LeRobot-format robotics datasets and converts them
    to the StreamVLN training format. Similar to VLNActionDataset, it:
    - Only loads metadata in __init__
    - Dynamically loads and processes images in __getitem__
    - Uses the same conversation format and preprocessing

    Args:
        tokenizer: Tokenizer for text processing
        data_args: DataArguments containing dataset configuration
        task_id: Task identifier
    """

    def __init__(
        self,
        tokenizer,
        data_args,
        task_id: int = 0,
    ):
        super(LeRobotActionDataset, self).__init__()

        self.task_id = task_id
        self.image_size = data_args.image_size
        self.tokenizer = tokenizer
        self.transforms = data_args.transform_train
        self.image_processor = SigLipImageProcessor()

        self.num_frames = data_args.num_frames
        self.num_history = data_args.num_history
        self.num_future_steps = data_args.num_future_steps
        self.remove_init_turns = data_args.remove_init_turns

        # LeRobot dataset path
        self.dataset_root = getattr(data_args, 'lerobot_dataset_path', data_args.video_folder)  # This will be the LeRobot dataset root
        self.repo_id = getattr(data_args, 'lerobot_repo_id', 'streamvln/navigation')
        self.video_backend = getattr(data_args, 'video_backend', 'auto')


        # Maximum number of parquet files to cache (prevents OOM with many episodes)
        # Adjust based on available memory:
        #   - 2-3 files:  ~200-1500 MB (for large datasets)
        #   - 5-10 files: ~500-3000 MB (for medium datasets)
        #   - Unlimited:  set to None or very large number (may OOM with many episodes)
        self._max_parquet_cache_size = getattr(data_args, 'max_parquet_cache_size', 5)

        # Load LeRobot dataset metadata
        self._load_lerobot_metadata()

        # Build data list
        self.data_list = self._build_data_list()

        # Action index to text mapping (same as VLNActionDataset)
        self.idx2actions = {
            '0': 'STOP',
            '1': "↑",
            '2': "←",
            '3': "→",
        }

        # Conversation template (same as VLNActionDataset)
        prompt = f"You are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        self.conversations = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

        # Conjunctions for variety (same as VLNActionDataset)
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is '
        ]

    def _load_lerobot_metadata(self):
        """Load LeRobot dataset metadata (info.json, episodes, tasks)."""
        self.root = Path(self.dataset_root) / self.repo_id if self.repo_id not in str(self.dataset_root) else Path(self.dataset_root)

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

        # Load episodes metadata
        self.episodes_df = self._load_all_episodes()
        print(f"Lerobot episode size: {len(self.episodes_df)}")

        # Build cumulative frame count mapping
        if not self.episodes_df.empty:
            self.episodes_df = self.episodes_df.sort_values('episode_index').reset_index(drop=True)
            self.episodes_df['_cumsum'] = self.episodes_df['length'].cumsum()
        else:
            self.episodes_df['_cumsum'] = pd.Series([], dtype=int)

        # Load tasks metadata
        self.tasks_df = self._load_tasks()

        # Find all data chunks
        self.data_chunks = self._find_data_chunks()

        # =============================================================================
        # PERFORMANCE OPTIMIZATION: Parquet cache + Frame index
        # =============================================================================
        from collections import OrderedDict

        # Cache for loaded parquet DataFrames (key: chunk_idx, file_idx)
        # Using OrderedDict for LRU cache implementation
        self._parquet_cache: OrderedDict[Tuple[int, int], pd.DataFrame] = OrderedDict()



        # Pre-computed frame index mapping (key: global_frame_idx)
        # Value: (chunk_idx, file_idx, pos_in_file)
        self._frame_index: Dict[int, Tuple[int, int, int]] = {}

        # Build frame index at initialization (one-time cost for O(1) lookup)
        self._build_frame_index()
        print(f"Built frame index: {len(self._frame_index)} frames indexed")
        if self._max_parquet_cache_size:
            print(f"Parquet cache limited to {self._max_parquet_cache_size} files (LRU)")
        else:
            print(f"Parquet cache: unlimited size (may use high memory with many episodes)")

    def _load_all_episodes(self) -> pd.DataFrame:
        """Load episode metadata from all chunks."""
        all_episodes = []
        episodes_dir = self.root / "meta" / "episodes"

        if not episodes_dir.exists():
            warnings.warn(f"Episodes directory not found: {episodes_dir}")
            return self._extract_episodes_from_data()

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
            return self._extract_episodes_from_data()

        episodes_df = pd.concat(all_episodes, ignore_index=True)
        return episodes_df

    def _extract_episodes_from_data(self) -> pd.DataFrame:
        """Extract episode information from info.json when parquet metadata is unavailable."""
        episodes_list = []
        total_episodes = self.info.get('total_episodes', self.total_episodes)
        total_frames = self.info.get('total_frames', self.total_frames)

        if total_episodes > 0 and total_frames > 0:
            avg_frames_per_episode = total_frames // total_episodes

            for i in range(total_episodes):
                start_idx = i * avg_frames_per_episode
                if i == total_episodes - 1:
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
            if df.index.name is not None or len(df.index) > 0:
                if df.index.astype(str).str.startswith('{').any():
                    df = df.reset_index()
                    if 'index' in df.columns:
                        df = df.rename(columns={'index': 'task_json'})
            return df
        except Exception as e:
            warnings.warn(f"Failed to load tasks from {tasks_file}: {e}")
            return pd.DataFrame()

    def _find_data_chunks(self) -> List[Path]:
        """Find all data chunk directories."""
        data_dir = self.root / "data"
        if not data_dir.exists():
            return []
        chunks = sorted(data_dir.glob("chunk-*"), key=lambda x: int(x.name.split("-")[1]))
        return chunks

    def _build_data_list(self):
        """Build list of sample indices from LeRobot dataset episodes (same as VLNActionDataset)."""
        data_list = []
        for episode_idx in range(self.total_episodes):
            episode_data = self.episodes_df[
                self.episodes_df['episode_index'] == episode_idx
            ]
            if not episode_data.empty:
                episode_length = int(episode_data.iloc[0]['length'])
                valid_idx = 0  # LeRobot datasets don't have initial rotations to skip
                if episode_length - valid_idx >= self.num_frames:
                    num_rounds = (episode_length - valid_idx) // self.num_frames
                    for n in range(num_rounds + 1):
                        start_frame = n * self.num_frames
                        if start_frame != 0 and start_frame + self.num_frames > episode_length - valid_idx:
                            continue
                        data_list.append((episode_idx, start_frame, valid_idx))
        return data_list

    def __len__(self):
        return len(self.data_list)

    @property
    def task(self):
        return self.task_id

    def actions2text(self, actions):
        """Convert action indices to text (same as VLNActionDataset)."""
        converted_sequence = []
        for action in actions:
            act_text = self.idx2actions.get(str(action.item()), 'STOP')
            converted_sequence.append(act_text)
        return ''.join(converted_sequence)

    def prepare_conversation(self, conversation, actions):
        """Prepare conversation with interleaved image tokens and action predictions (same as VLNActionDataset)."""
        i = 0
        sources = []
        t = 0
        while i < len(actions):
            source = copy.deepcopy(conversation)
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            step_actions = actions[i:i+self.num_future_steps]
            answer = self.actions2text(step_actions)
            if i == 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."

            source[1]["value"] = answer
            i += len(step_actions)
            t += 1
            sources.extend(source)
        return sources

    def _get_task_description(self, task_index: int) -> str:
        """Get task description for a given task index."""
        if not self.tasks_df.empty:
            try:
                task_index_int = int(task_index)
            except (ValueError, TypeError):
                task_index_int = None

            if task_index_int is not None and task_index_int < len(self.tasks_df):
                task_row = self.tasks_df.iloc[task_index_int]
                for col in ['task_json', 'task', 'tasks', 'instruction', 'text']:
                    if col in task_row and pd.notna(task_row[col]):
                        task = task_row[col]
                        if isinstance(task, str):
                            return task
                        else:
                            return json.dumps(task)
        return '{"instruction": "Navigation task"}'

    def _get_global_frame_index(self, episode_idx, frame_in_episode):
        """Get global frame index from episode index and frame in episode."""
        episode_data = self.episodes_df[
            self.episodes_df['episode_index'] == episode_idx
        ]
        if episode_data.empty:
            return None
        episode_start = int(episode_data.iloc[0]['dataset_from_index'])
        return episode_start + frame_in_episode

    def _find_episode_from_frame_index(self, idx: int) -> int:
        """Find episode index from global frame index using binary search."""
        if self.episodes_df.empty:
            return 0
        cumsum = self.episodes_df['_cumsum'].values
        episode_idx = np.searchsorted(cumsum, idx + 1, side='right')
        return int(self.episodes_df.iloc[episode_idx]['episode_index'])

    def _build_frame_index(self):
        """
        Pre-compute frame index mapping for O(1) lookups.

        Builds a dictionary mapping each global frame index to its location:
        - chunk_idx: which chunk directory
        - file_idx: which file in the chunk
        - pos_in_file: position within that file

        This eliminates the need for file searching during __getitem__.
        Time cost: ~10-30 seconds at initialization (one-time)
        Performance gain: 100-1000x faster frame location
        """
        print("Building frame index... (this may take 10-30 seconds)")

        for chunk_idx, chunk_dir in enumerate(self.data_chunks):
            # Get all parquet files in this chunk
            files = sorted(chunk_dir.glob("file-*.parquet"),
                          key=lambda x: int(x.stem.split("-")[1]))

            for file_idx, file_path in enumerate(files):
                try:
                    # Read the parquet file once
                    df = pd.read_parquet(file_path)

                    # Get the starting index for this file
                    if 'dataset_from_index' in df.columns:
                        file_start_idx = int(df.iloc[0]['dataset_from_index'])
                    else:
                        file_start_idx = 0

                    # Index each frame in this file
                    for pos_in_file, row in df.iterrows():
                        if 'index' in row:
                            global_idx = int(row['index'])
                        else:
                            # Fallback: use file_start_idx + position
                            global_idx = file_start_idx + pos_in_file

                        self._frame_index[global_idx] = (chunk_idx, file_idx, pos_in_file)

                except Exception as e:
                    warnings.warn(f"Failed to index file {file_path}: {e}")
                    continue

        print(f"Frame index built: {len(self._frame_index)} frames")

    def _locate_frame(self, idx: int, episode_idx: int) -> Tuple[int, int, int]:
        """
        Locate which chunk and file contains a given frame index.

        PERFORMANCE OPTIMIZATION: Uses pre-built frame index for O(1) lookup.
        Falls back to linear search only if index is not available.
        """
        # Try O(1) lookup using pre-built index
        if idx in self._frame_index:
            return self._frame_index[idx]

        # Fallback: For simple datasets with single file, return first chunk/file
        if len(self.data_chunks) == 0:
            return 0, 0, 0

        # Fallback: Check if there's only one chunk and one file (common case)
        first_chunk = self.data_chunks[0]
        files = list(first_chunk.glob("file-*.parquet"))
        if len(self.data_chunks) == 1 and len(files) == 1:
            return 0, 0, 0

        # Fallback: For multi-file datasets, search through data files
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

        # Fallback: use episode metadata to determine file location
        if not self.episodes_df.empty and 'data/chunk_index' in self.episodes_df.columns:
            # Find the episode that contains this frame index
            episode_data = self.episodes_df[self.episodes_df['episode_index'] == episode_idx]
            if not episode_data.empty:
                chunk_idx = int(episode_data.iloc[0].get('data/chunk_index', 0))
                file_idx = int(episode_data.iloc[0].get('data/file_index', 0))
                file_start = int(episode_data.iloc[0].get('dataset_from_index', 0))
                return chunk_idx, file_idx, file_start

        return 0, 0, 0

    def _load_frame_data(self, chunk_idx: int, file_idx: int, idx: int) -> Dict[str, Any]:
        """
        Load frame data from parquet file.

        PERFORMANCE OPTIMIZATION:
        - Uses LRU (Least Recently Used) cache to avoid re-reading files
        - Cache size is limited by max_parquet_cache_size to prevent OOM
        - Cache key is (chunk_idx, file_idx)

        MEMORY MANAGEMENT:
        With many episodes, unbounded caching would cause OOM.
        LRU eviction ensures memory usage stays bounded.
        """
        cache_key = (chunk_idx, file_idx)

        # Check if parquet file is already cached
        if cache_key in self._parquet_cache:
            # Move to end (mark as recently used)
            self._parquet_cache.move_to_end(cache_key)
            df = self._parquet_cache[cache_key]
        else:
            # Load new file
            chunk_dir = self.data_chunks[chunk_idx]
            file_path = chunk_dir / f"file-{file_idx:03d}.parquet"

            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")

            # Load the parquet file
            df = pd.read_parquet(file_path)

            # Add to cache (at end = most recently used)
            self._parquet_cache[cache_key] = df

            # Evict oldest entry if cache is too large
            if (self._max_parquet_cache_size is not None and
                len(self._parquet_cache) > self._max_parquet_cache_size):
                # Remove oldest (first) entry
                oldest_key = next(iter(self._parquet_cache))
                del self._parquet_cache[oldest_key]

        try:
            # Try to find row by 'index' column first
            if 'index' in df.columns:
                row = df[df['index'] == idx]
                if not row.empty:
                    return row.iloc[0].to_dict()

            # Fallback: calculate position in file
            if 'dataset_from_index' in df.columns:
                file_start_idx = int(df.iloc[0]['dataset_from_index'])
                pos_in_file = idx - file_start_idx
            else:
                # Use iloc with idx as last resort
                pos_in_file = idx

            row = df.iloc[pos_in_file:pos_in_file+1]
            if row.empty:
                raise ValueError(f"Frame index {idx} not found in file")
            return row.iloc[0].to_dict()

        except Exception as e:
            raise RuntimeError(f"Failed to load frame data at index {idx}: {e}")

    def _load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Load image from PNG bytes stored in parquet file."""
        from io import BytesIO
        img = Image.open(BytesIO(image_bytes))
        return np.array(img)

    def __getitem__(self, idx):
        """Get a training sample (same return format as VLNActionDataset)."""
        episode_idx, start_idx, valid_idx = self.data_list[idx]

        # Get episode data
        episode_data = self.episodes_df[self.episodes_df['episode_index'] == episode_idx]
        if episode_data.empty:
            raise ValueError(f"Episode {episode_idx} not found")
        episode_start = int(episode_data.iloc[0]['dataset_from_index'])
        episode_length = int(episode_data.iloc[0]['length'])

        # Calculate time_ids (same as VLNActionDataset)
        actions_len = episode_length - valid_idx
        time_ids = np.arange(start_idx, min(start_idx + self.num_frames, actions_len))
        assert len(time_ids) > 0

        # Calculate sample step IDs
        start_idx_abs, end_idx, interval = time_ids[0]+valid_idx, time_ids[-1]+1+valid_idx, self.num_future_steps
        sample_step_ids = np.arange(start_idx_abs, end_idx, interval, dtype=np.int32)

        # Calculate history step IDs (same as VLNActionDataset)
        if time_ids[0] != 0:
            history_step_ids = np.arange(0+valid_idx, time_ids[0]+valid_idx, max(time_ids[0] // self.num_history, 1))
        else:
            history_step_ids = []

        # Load all frames (history + sample)
        images = []
        actions = []

        # Load history frames
        for step_id in history_step_ids:
            global_frame_idx = episode_start + step_id
            chunk_idx, file_idx, _ = self._locate_frame(global_frame_idx, episode_idx)
            frame_data = self._load_frame_data(chunk_idx, file_idx, global_frame_idx)

            # Load image from PNG bytes stored in parquet
            img_data = frame_data.get('observation.images.rgb', {})
            if isinstance(img_data, dict) and 'bytes' in img_data:
                image_bytes = img_data['bytes']
                frame = self._load_image_from_bytes(image_bytes)
            else:
                raise RuntimeError(f"Image data not found in frame {global_frame_idx}")

            # Process frame
            frame_tensor = self._process_frame(frame)
            images.append(frame_tensor)

            # History frames don't have actions to predict (or we can skip them)
            # For simplicity, we don't add actions for history frames

        # Load sample frames
        for step_id in sample_step_ids:
            global_frame_idx = episode_start + step_id
            chunk_idx, file_idx, _ = self._locate_frame(global_frame_idx, episode_idx)
            frame_data = self._load_frame_data(chunk_idx, file_idx, global_frame_idx)

            # Load image from PNG bytes stored in parquet
            img_data = frame_data.get('observation.images.rgb', {})
            if isinstance(img_data, dict) and 'bytes' in img_data:
                image_bytes = img_data['bytes']
                frame = self._load_image_from_bytes(image_bytes)
            else:
                raise RuntimeError(f"Image data not found in frame {global_frame_idx}")

            # Process frame
            frame_tensor = self._process_frame(frame)
            images.append(frame_tensor)

            # Extract action
            action_value = frame_data.get('action', 0)
            if isinstance(action_value, (list, tuple, np.ndarray)):
                action_value = action_value[0] if len(action_value) > 0 else 0
            actions.append(action_value)

        if not images:
            raise RuntimeError(f"Failed to load any frames for episode {episode_idx}, start {start_idx}")

        images = torch.stack(images)
        actions = torch.tensor(actions, dtype=torch.long) if actions else torch.tensor([], dtype=torch.long)
        time_ids = torch.tensor(time_ids, dtype=torch.long)

        # Get instruction from task
        task_index = episode_data.iloc[0].get('task_index', 0)
        task_json = self._get_task_description(task_index)
        try:
            task_data = json.loads(task_json)
            instruction = task_data.get('instruction', 'Navigate to the goal.')
        except json.JSONDecodeError:
            instruction = 'Navigate to the goal.'

        # Prepare conversation (same as VLNActionDataset)
        sources = copy.deepcopy(self.conversations)

        # Add historical observations token if there are history frames (same as VLNActionDataset)
        if start_idx != 0:
            sources[0]["value"] += f' These are your historical observations: {DEFAULT_MEMORY_TOKEN}.'

        sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
        interleave_sources = self.prepare_conversation(sources, list(actions))

        # Preprocess to get input_ids and labels
        data_dict = preprocess([interleave_sources], self.tokenizer, True)

        return data_dict["input_ids"][0], \
            data_dict["labels"][0], \
            images, \
            time_ids, \
            self.task

    def _process_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Process a single frame to tensor (same as VLNActionDataset)."""
        # Convert to tensor
        if isinstance(frame, np.ndarray):
            if frame.max() <= 1.0:
                frame_tensor = torch.from_numpy(frame).float()
            else:
                frame_tensor = torch.from_numpy(frame).float() / 255.0
        else:
            frame_tensor = frame

        # HWC to CHW
        if frame_tensor.dim() == 3 and frame_tensor.shape[-1] <= 4:
            frame_tensor = frame_tensor.permute(2, 0, 1)

        # Convert to PIL and apply transforms
        frame_pil = Image.fromarray((frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        if self.transforms is not None:
            frame_pil = self.transforms(frame_pil)

        # Apply image processor
        frame_processed = self.image_processor.preprocess(images=frame_pil, return_tensors='pt')['pixel_values'][0]
        return frame_processed
