import sys
import argparse
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import gzip
import itertools
import json
import math
import random
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import transformers
import habitat
from PIL import Image
from habitat import logger, Env
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat_extensions import measures

from transformers.image_utils import to_numpy_array

from model.stream_video_vln import StreamVLNForCausalLM
from utils.dist import *
from utils.utils import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_MEMORY_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IMAGE_TOKEN_INDEX,
    MEMORY_TOKEN_INDEX,
    dict_to_cuda,
)

try:
    from depth_camera_filtering import filter_depth
except ImportError:
    def filter_depth(depth, blur_type=None):
        return depth


class ObjectNavEvaluator:
    """
    HM3D ObjectNav evaluator using StreamVLN.
    """

    def __init__(
        self,
        config_path: str,
        split: str,
        env_num: int,
        output_path: Optional[str],
        model: Any,
        tokenizer: Any,
        epoch: int,
        args: argparse.Namespace,
    ):
        self.args = args
        self.device = torch.device(args.device)
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        self.epoch = epoch

        self.config_path = config_path
        self.config = get_habitat_config(self.config_path, overrides=[f"habitat.dataset.split={self.split}"])
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors
        self.idx_to_action: Dict[int, Dict[str, str]] = {
            0: {"action": "stop"},
            1: {"action": "move_forward"},
            2: {"action": "turn_left"},
            3: {"action": "turn_right"},
        }

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = self.model.get_vision_tower().image_processor

        self.num_frames = args.num_frames
        self.num_future_steps = args.num_future_steps
        self.num_history = args.num_history
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.dataset_path = self._resolve_dataset_path()
        self.total_dataset_episodes = self._count_dataset_episodes(self.dataset_path)

        depth_sensor = getattr(self.sim_sensors_config, "depth_sensor", None)
        rgb_sensor = getattr(self.sim_sensors_config, "rgb_sensor", None)
        self._camera_height = rgb_sensor.position[1] if rgb_sensor is not None else 0.0
        if depth_sensor is not None:
            self._min_depth = getattr(depth_sensor, "min_depth", 0.1)
            self._max_depth = getattr(depth_sensor, "max_depth", 10.0)
            camera_fov_rad = np.deg2rad(getattr(depth_sensor, "hfov", 90.0))
            self._camera_fov = camera_fov_rad
            self._fx = depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))
            self._fy = self._fx
        else:
            self._min_depth = None
            self._max_depth = None
            self._camera_fov = None
            self._fx = None
            self._fy = None
        self.axis_align_matrix = self.get_axis_align_matrix()

        requested_total = args.max_episodes if args.max_episodes is not None else -1
        if requested_total is None or requested_total <= 0:
            self.target_total_episodes = self.total_dataset_episodes
        else:
            if self.total_dataset_episodes > 0:
                self.target_total_episodes = min(requested_total, self.total_dataset_episodes)
            else:
                self.target_total_episodes = requested_total

        if self.target_total_episodes is None or self.target_total_episodes <= 0:
            self.max_episodes_per_worker = -1
        else:
            self.max_episodes_per_worker = math.ceil(self.target_total_episodes / max(1, self.env_num))
        self.nav_action_names = ["move_forward", "turn_left", "turn_right", "stop"]

        self._configure_prompt()

    def _sanitize_metric(self, value: Any, metric_name: str, episode_id: str, default: float = 0.0) -> float:
        """Ensure metrics do not propagate NaN/Inf downstream."""
        if value is None:
            logger.warning(
                "[Metric sanitize] episode=%s metric=%s missing; defaulting to %.4f",
                episode_id,
                metric_name,
                default,
            )
            return default
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            logger.warning(
                "[Metric sanitize] episode=%s metric=%s not convertible (%s); defaulting to %.4f",
                episode_id,
                metric_name,
                value,
                default,
            )
            return default

        if not math.isfinite(numeric_value):
            logger.warning(
                "[Metric sanitize] episode=%s metric=%s non-finite (%s); defaulting to %.4f",
                episode_id,
                metric_name,
                numeric_value,
                default,
            )
            return default
        return numeric_value

    def _resolve_dataset_path(self) -> str:
        data_path = self.config.habitat.dataset.data_path
        if isinstance(data_path, str) and "{split}" in data_path:
            try:
                data_path = data_path.format(split=self.split)
            except KeyError:
                logger.warning("Failed to format dataset path %s with split %s", data_path, self.split)
        if not os.path.isabs(data_path):
            data_path = os.path.abspath(os.path.join(self.repo_root, data_path))
        return data_path

    def _count_dataset_episodes(self, dataset_path: str) -> int:
        if not dataset_path:
            return -1
        try:
            with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            episodes = data.get("episodes", [])
            return len(episodes)
        except FileNotFoundError:
            logger.warning("Dataset file %s not found; defaulting to unlimited episodes", dataset_path)
        except json.JSONDecodeError:
            logger.warning("Dataset file %s is not valid JSON; defaulting to unlimited episodes", dataset_path)
        except OSError as exc:
            logger.warning("Failed to open dataset file %s: %s", dataset_path, exc)
        return -1

    def _configure_prompt(self) -> None:
        prompt = (
            "<video>\n"
            "You are an autonomous agent executing object navigation in HM3D.\n"
            "Your task is to reach the goal object following the available actions: "
            "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        )
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]
        self.actions2idx = OrderedDict(
            {
                "STOP": [0],
                "↑": [1],
                "←": [2],
                "→": [3],
            }
        )
        self.conjunctions = [
            "you can see ",
            "in front of you is ",
            "there is ",
            "you can spot ",
            "you are toward the ",
            "ahead of you is ",
            "in your sight is ",
        ]

    def config_env(self) -> Env:
        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = self.split
            measurement_cfg = self.config.habitat.task.measurements
            measurement_cfg.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=False,
                        fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )
            self.config.habitat.simulator.scene_dataset = "hm3d"
            self.config.habitat.seed = self.args.seed          # global seed
            self.config.habitat.simulator.seed = self.args.seed  # per-simulator seed
            self.config.habitat.simulator.create_renderer = self.args.render
            self.config.habitat.simulator.debug_render = self.args.render
        env = Env(config=self.config)
        return env

    def preprocess_depth_image(self, depth_image, do_depth_scale: bool = True, depth_scale: int = 1000):
        target_height = self.image_processor.crop_size["height"]
        target_width = self.image_processor.crop_size["width"]
        resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)

        img = to_numpy_array(resized_depth_image)
        if do_depth_scale:
            img = img / depth_scale
        return torch.from_numpy(img).float(), (target_width, target_height)

    def get_intrinsic_matrix(self, sensor_cfg) -> np.ndarray:
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        intrinsic_matrix = np.array(
            [
                [fx, 0.0, cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return intrinsic_matrix

    def preprocess_intrinsic(self, intrinsic, ori_size, target_size):
        intrinsic = copy.deepcopy(intrinsic)
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[None, ...]
        intrinsic[:, 0] /= ori_size[0] / target_size[0]
        intrinsic[:, 1] /= ori_size[1] / target_size[1]
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2
        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)
        return intrinsic

    def get_axis_align_matrix(self) -> torch.Tensor:
        return torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],
                [np.sin(yaw), np.cos(yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )
        return transformation_matrix

    def parse_actions(self, output: str) -> List[int]:
        action_patterns = "|".join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        return list(itertools.chain.from_iterable(actions))

    def preprocess_qwen(
        self,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        max_len: int = 2048,
        system_message: str = "You are a helpful assistant.",
        add_system: bool = False,
    ):
        roles = {"human": "user", "gpt": "assistant"}
        tokenizer = copy.deepcopy(tokenizer)
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            tokenizer.add_tokens(["<memory>"], special_tokens=True)

        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        im_start, im_end = tokenizer.additional_special_tokens_ids
        nl_tokens = tokenizer("\n").input_ids

        chat_template = (
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
        tokenizer.chat_template = chat_template

        conversations = []
        input_ids = []
        for source in sources:
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."
            if roles.get(source[0]["from"], roles["human"]) != roles["human"]:
                source = source[1:]

            tokens = []
            if add_system:
                tokens += tokenizer.apply_chat_template([
                    {"role": "system", "content": system_message}
                ])

            for conv in source:
                role = conv.get("role", conv.get("from"))
                content = conv.get("content", conv.get("value", ""))
                role = roles.get(role, role)
                conversations.append(content)
                tokens += tokenizer.apply_chat_template([
                    {"role": role, "content": content}
                ])

            for idx, token in enumerate(tokens):
                if token == image_token_index:
                    tokens[idx] = IMAGE_TOKEN_INDEX
                if token == memory_token_index:
                    tokens[idx] = MEMORY_TOKEN_INDEX

            input_ids.append(tokens[:max_len])

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids, conversations

    def eval_action(self, idx: int):
        env = self.config_env()
        all_episodes = env.episodes
        if len(all_episodes) == 0:
            env.close()
            empty_tensor = torch.empty(0, device=self.device)
            return empty_tensor, empty_tensor, empty_tensor, empty_tensor, torch.tensor([0], device=self.device)

        assigned_episodes = all_episodes[idx::self.env_num]
        if len(assigned_episodes) == 0:
            env.close()
            empty_tensor = torch.empty(0, device=self.device)
            return empty_tensor, empty_tensor, empty_tensor, empty_tensor, torch.tensor([0], device=self.device)

        if self.max_episodes_per_worker > 0:
            assigned_episodes = assigned_episodes[: self.max_episodes_per_worker]

        stats: Dict[str, List[float]] = {"success": [], "spl": [], "soft_spl": [], "distance_to_goal": []}

        try:
            rgb_sensor_cfg = getattr(self.sim_sensors_config, "rgb_sensor", None)
            intrinsic_matrix = (
                self.get_intrinsic_matrix(rgb_sensor_cfg) if rgb_sensor_cfg is not None else np.eye(4)
            )
            axis_align_matrix = self.axis_align_matrix

            for episode in assigned_episodes:
                env.current_episode = episode
                self.model.reset_for_env(idx)
                observations = env.reset()

                scene_id_raw = getattr(episode, "scene_id", "")
                scene_name = os.path.splitext(os.path.basename(scene_id_raw))[0] if scene_id_raw else f"scene_{idx}"
                episode_id = getattr(episode, "episode_id", len(stats["success"]))
                episode_id_str = str(episode_id)
                vis_frames: List[np.ndarray] = []
                video_basename = f"{scene_name}_{episode_id_str}"

                if self.save_video and self.output_path:
                    check_dir = os.path.join(self.output_path, f"check_sim_{self.epoch}")
                    os.makedirs(check_dir, exist_ok=True)
                    Image.fromarray(observations["rgb"]).save(
                        os.path.join(check_dir, f"rgb_{idx}.jpg")
                    )
                    vis_dir = os.path.join(self.output_path, f"vis_{self.epoch}")
                    os.makedirs(vis_dir, exist_ok=True)

                initial_height = env.sim.get_agent_state().position[1]

                rgb_list: List[torch.Tensor] = []
                depth_list: List[torch.Tensor] = []
                pose_list: List[torch.Tensor] = []
                intrinsic_list: List[torch.Tensor] = []
                time_ids: List[int] = []
                action_seq: List[int] = []
                past_key_values = None
                output_ids = None
                step_id = 0
                goal_description = getattr(episode, "object_category", None)

                while not env.episode_over:
                    self.model.eval()
                    time_ids.append(step_id)

                    rgb = observations["rgb"]
                    depth_obs = observations.get("depth")
                    gps_obs = np.asarray(observations.get("gps", np.zeros(2)), dtype=np.float32)
                    if gps_obs.size >= 2:
                        x, y = float(gps_obs[0]), float(gps_obs[1])
                    else:
                        x = y = 0.0
                    compass_obs = observations.get("compass")
                    compass_arr = (
                        np.asarray(compass_obs, dtype=np.float32)
                        if compass_obs is not None
                        else None
                    )
                    camera_yaw = float(compass_arr[0]) if compass_arr is not None and compass_arr.size > 0 else 0.0

                    depth_tensor: torch.Tensor
                    resize_shape = (
                        self.image_processor.crop_size["width"],
                        self.image_processor.crop_size["height"],
                    )
                    if depth_obs is not None:
                        depth_np = np.asarray(depth_obs)
                        if depth_np.ndim == 3:
                            depth_np = depth_np[..., 0]
                        depth_np = filter_depth(depth_np, blur_type=None)
                        if self._max_depth is not None and self._min_depth is not None:
                            depth_np = depth_np * (self._max_depth - self._min_depth) + self._min_depth
                        depth_mm = (depth_np * 1000.0).astype(np.uint16)
                        depth_tensor, resize_shape = self.preprocess_depth_image(
                            Image.fromarray(depth_mm, mode="I;16"), do_depth_scale=True
                        )
                    else:
                        depth_tensor = torch.zeros(
                            (
                                self.image_processor.crop_size["height"],
                                self.image_processor.crop_size["width"],
                            ),
                            dtype=torch.float32,
                        )

                    agent_state = env.sim.get_agent_state()
                    height_delta = agent_state.position[1] - initial_height
                    camera_position = np.array([x, -y, self._camera_height + height_delta])
                    tf_camera_to_episodic = self.xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

                    image = Image.fromarray(rgb).convert("RGB")
                    image_size = image.size
                    image_tensor = self.image_processor.preprocess(images=image, return_tensors="pt")["pixel_values"][0]

                    intrinsic_np = self.preprocess_intrinsic(intrinsic_matrix, image_size, resize_shape)
                    intrinsic_tensor = torch.from_numpy(intrinsic_np).float()

                    rgb_list.append(image_tensor)
                    depth_list.append(depth_tensor.float())
                    pose_list.append(torch.from_numpy(tf_camera_to_episodic).float() @ axis_align_matrix)
                    intrinsic_list.append(intrinsic_tensor)

                    if self.save_video and self.output_path:
                        info = env.get_metrics()
                        top_down = info.get("top_down_map") if isinstance(info, dict) else None
                        if top_down is not None:
                            frame = observations_to_image({"rgb": observations["rgb"]}, info)
                            vis_frames.append(frame)

                    if len(action_seq) == 0:
                        model_history = getattr(self.model.model, "num_history", None) or 0
                        required_frames = max(1, model_history + 1)
                        if len(rgb_list) < required_frames:
                            logger.debug(
                                "Insufficient frames (%s/%s); defaulting to move_forward",
                                len(rgb_list),
                                required_frames,
                            )
                            action_seq = [1]
                        else:
                            start_index = max(0, len(rgb_list) - required_frames)
                            selected_indices = list(range(start_index, len(rgb_list)))
                            selected_rgb = [rgb_list[i] for i in selected_indices]
                            selected_depth = [depth_list[i] for i in selected_indices]
                            selected_pose = [pose_list[i] for i in selected_indices]
                            selected_intrinsics = [intrinsic_list[i] for i in selected_indices]
                            if len(time_ids) >= len(selected_indices):
                                selected_time_ids = time_ids[-len(selected_indices):]
                            else:
                                selected_time_ids = selected_indices

                            if output_ids is None:
                                sources = copy.deepcopy(self.conversation)
                                if goal_description:
                                    sources[0]["value"] += f' The goal object category is "{goal_description}".'
                                include_memory_token = model_history > 0 and start_index > 0
                                if include_memory_token:
                                    sources[0]["value"] += (
                                        f" These are your historical observations {DEFAULT_MEMORY_TOKEN}."
                                    )
                                sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN + "\n", "")
                                add_system = True
                            else:
                                sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                                add_system = False

                            input_ids, conversations = self.preprocess_qwen([sources], self.tokenizer, True, add_system=add_system)
                            # Log the assembled prompt (textual conversations) for debugging / inspection
                            # try:
                            #     logger.info(
                            #         f"[Prompt assembled] env={idx} ep={episode_id_str} step={step_id} convs={conversations}"
                            #     )
                            # except Exception as _:
                            #     logger.warning("[Prompt assembled] failed to log conversations")
                            if output_ids is not None:
                                input_ids = torch.cat([output_ids, input_ids.to(output_ids.device)], dim=1)

                            input_dict = {
                                "images": torch.stack(selected_rgb).unsqueeze(0),
                                "depths": torch.stack(selected_depth).unsqueeze(0),
                                "poses": torch.stack(selected_pose).unsqueeze(0),
                                "intrinsics": torch.stack(selected_intrinsics).unsqueeze(0),
                                "inputs": input_ids,
                                "env_id": idx,
                                "time_ids": [selected_time_ids],
                                "task_type": [0],
                            }

                            input_dict = dict_to_cuda(input_dict, self.device)
                            for key in ["images", "depths", "poses", "intrinsics"]:
                                input_dict[key] = input_dict[key].to(torch.float16)

                            outputs = self.model.generate(
                                **input_dict,
                                do_sample=False,
                                num_beams=1,
                                max_new_tokens=128,
                                use_cache=True,
                                return_dict_in_generate=True,
                                past_key_values=past_key_values,
                            )

                            output_ids = outputs.sequences
                            past_key_values = outputs.past_key_values
                            llm_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
                            action_seq = self.parse_actions(llm_outputs)
                            if len(action_seq) == 0:
                                action_seq = [0]

                    action_idx = action_seq.pop(0)
                    habitat_action = self.idx_to_action.get(action_idx, self.idx_to_action[0])
                    observations = env.step(habitat_action)
                    step_id += 1

                    if step_id % self.num_frames == 0:
                        self.model.reset_for_env(idx)
                        output_ids = None
                        past_key_values = None
                        time_ids = []

                episode_metrics = env.get_metrics()
                success_metric = self._sanitize_metric(
                    episode_metrics.get("success", 0.0),
                    "success",
                    episode_id_str,
                )
                spl_metric = self._sanitize_metric(
                    episode_metrics.get("spl", 0.0),
                    "spl",
                    episode_id_str,
                )
                soft_spl_metric = self._sanitize_metric(
                    episode_metrics.get("soft_spl", episode_metrics.get("spl", 0.0)),
                    "soft_spl",
                    episode_id_str,
                )
                distance_to_goal = self._sanitize_metric(
                    episode_metrics.get("distance_to_goal", episode_metrics.get("goal_distance", 0.0)),
                    "distance_to_goal",
                    episode_id_str,
                )

                stats["success"].append(success_metric)
                stats["spl"].append(spl_metric)
                stats["soft_spl"].append(soft_spl_metric)
                stats["distance_to_goal"].append(distance_to_goal)

                if self.save_video and self.output_path and vis_frames:
                    images_to_video(
                        vis_frames,
                        os.path.join(self.output_path, f"vis_{self.epoch}"),
                        video_basename,
                        fps=6,
                        quality=9,
                    )
                    vis_frames.clear()

                completed_episodes = len(stats["success"])
                running_success = float(np.mean(stats["success"])) if completed_episodes > 0 else 0.0
                logger.info(
                    "[Rank %s] Episode %s/%s running success rate: %.3f",
                    get_rank(),
                    completed_episodes,
                    len(assigned_episodes),
                    running_success,
                )
        finally:
            env.close()

        success_tensor = torch.tensor(stats["success"], device=self.device, dtype=torch.float32)
        spl_tensor = torch.tensor(stats["spl"], device=self.device, dtype=torch.float32)
        soft_spl_tensor = torch.tensor(stats["soft_spl"], device=self.device, dtype=torch.float32)
        distance_to_goal_tensor = torch.tensor(stats["distance_to_goal"], device=self.device, dtype=torch.float32)
        ep_count = torch.tensor([len(stats["success"])], device=self.device)
        return success_tensor, spl_tensor, soft_spl_tensor, distance_to_goal_tensor, ep_count


def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default="config/objectnav_hm3d.yaml")
    parser.add_argument("--eval_split", type=str, default="val")
    parser.add_argument("--output_path", type=str, default="./results/objectnav/hm3d")
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--port", default="1111")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--render", action="store_true", default=False)
    args = parser.parse_args()

    init_distributed_mode(args)
    local_rank = args.local_rank

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path, model_max_length=args.model_max_length, padding_side="right"
    )
    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model = StreamVLNForCausalLM.from_pretrained(
                args.model_path,
                attn_implementation="sdpa",
                torch_dtype=torch.float16,
                config=config,
                low_cpu_mem_usage=False,
                )
    model.model.num_history = args.num_history
    model.requires_grad_(False)
    model.to(local_rank)

    evaluate(model, tokenizer, args)


def _compute_finite_mean(tensor: torch.Tensor, metric_name: str) -> float:
    """Return the mean of finite entries, logging and skipping NaN/Inf values."""
    if tensor.numel() == 0:
        return 0.0
    finite_mask = torch.isfinite(tensor)
    if not torch.all(finite_mask):
        dropped = int((~finite_mask).sum().item())
        logger.warning(
            "[Metric aggregate] Dropping %d non-finite entries from %s before averaging",
            dropped,
            metric_name,
        )
        tensor = tensor[finite_mask]
    if tensor.numel() == 0:
        return 0.0
    return tensor.mean().item()


def evaluate(model, tokenizer, args):
    model.eval()
    world_size = get_world_size()
    model.reset(world_size)

    evaluator = ObjectNavEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        tokenizer=tokenizer,
        epoch=0,
        args=args,
    )
    success, spl, soft_spl, distance_to_goal, ep_count = evaluator.eval_action(get_rank())

    counts_all = [torch.zeros_like(ep_count) for _ in range(world_size)]
    dist.all_gather(counts_all, ep_count)

    success_all = [torch.zeros(counts_all[i], device=success.device) for i in range(world_size)]
    spl_all = [torch.zeros(counts_all[i], device=spl.device) for i in range(world_size)]
    soft_spl_all = [torch.zeros(counts_all[i], device=soft_spl.device) for i in range(world_size)]
    distance_to_goal_all = [torch.zeros(counts_all[i], device=distance_to_goal.device) for i in range(world_size)]

    dist.barrier()
    dist.all_gather(success_all, success)
    dist.all_gather(spl_all, spl)
    dist.all_gather(soft_spl_all, soft_spl)
    dist.all_gather(distance_to_goal_all, distance_to_goal)
    dist.barrier()

    success_all = torch.cat(success_all, dim=0)
    spl_all = torch.cat(spl_all, dim=0)
    soft_spl_all = torch.cat(soft_spl_all, dim=0)
    distance_to_goal_all = torch.cat(distance_to_goal_all, dim=0)

    results = {
        "success": _compute_finite_mean(success_all, "success"),
        "spl": _compute_finite_mean(spl_all, "spl"),
        "soft_spl": _compute_finite_mean(soft_spl_all, "soft_spl"),
        "goal_distance": _compute_finite_mean(distance_to_goal_all, "goal_distance"),
        "episodes": int(counts_all[0].sum().item()),
    }
    print(results)
    if get_rank() == 0 and args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        with open(os.path.join(args.output_path, "objectnav_metrics.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    eval()