import os
import sys
import torch
import json
import argparse
import transformers

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.dist import *
# from utils.dagger import DaggerCollector
from streamvln_eval import VLNEvaluator
from model.stream_video_vln import StreamVLNForCausalLM

import os
import random
import numpy as np
import torch
import tqdm
import copy
import json
import random
import habitat
import time

from PIL import Image
from omegaconf import OmegaConf
from typing import List, Dict
from PIL import Image

from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.config import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import images_to_video, observations_to_image, append_text_underneath_image
from depth_camera_filtering import filter_depth

from utils.dist import *
from utils.utils import dict_to_cuda
from utils.utils import DEFAULT_MEMORY_TOKEN, DEFAULT_VIDEO_TOKEN
from habitat_extensions.maps import image_resize

DATASET = "rxr"
CONFIG_PATH = "./config/vln_r2r.yaml"
OUTPUT_PATH = "./generated_data"

DEFAULT_EPISODE_LENGTH = 60
MIDGOAL_RADIUS = 0.5
GOAL_RADIUS = 0.25
RELATIVE_PATH_LENGTH_THRESHOLD = 0.93
SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD = 0.85

class StreamVLNDAggerCollector:
    def __init__(self, args, rank, world_size):
        self.device = torch.device("cuda")
        self.args = args
        self.rank = rank
        self.world_size = world_size

        self.dataset = self.args.dagger_dataset.lower()
        self.output_path = self.args.dagger_output_path
        self.data_path = self.args.dagger_data_path
        self.config = get_habitat_config(args.habitat_config_path)
        # print(OmegaConf.to_yaml(self.config))

        with open(self.args.dagger_gt_annotations_path, "r") as f:
            self.gt_annotations = json.load(f)
        
        with read_write(self.config):
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        # dagger config
        self.dagger_config = OmegaConf.create({
            "p": self.args.dagger_p,
            "update_size": self.args.dagger_update_size,
            "commit_freq": self.args.dagger_commit_freq,
        })
        print("This is the dagger config:")
        print(self.dagger_config)
        
        sim_sensors_cfg = self.config.habitat.simulator.agents.main_agent.sim_sensors
        self._camera_height = sim_sensors_cfg.rgb_sensor.position[1]
        self._min_depth = sim_sensors_cfg.depth_sensor.min_depth
        self._max_depth = sim_sensors_cfg.depth_sensor.max_depth
        camera_fov_rad = np.deg2rad(sim_sensors_cfg.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = sim_sensors_cfg.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))
        # self.R = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

    def config_env(self, scene=None) -> habitat.Env:

        # This is the palce to update the data path
        if self.data_path is not None:
            with read_write(self.config):
                self.config.habitat.dataset.data_path = self.data_path
        print(OmegaConf.to_yaml(self.config))
        return habitat.Env(config=self.config)
    
    def get_intrinsic_matrix(self, sensor_cfg):
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx  # Assuming square pixels (fx = fy)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        intrinsic_matrix = np.array([
            [fx,  0.0, cx, 0.0],
            [ 0.0, fy, cy, 0.0],
            [ 0.0,  0.0,  1.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0]
        ])
        return intrinsic_matrix
    
    def get_axis_align_matrix(self):
        # ma = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # ma = torch.tensor([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        ma = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        return ma
    
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
    
    def generate(
        self,
        env: habitat.Env,
        evaluator = None,
        save_video: bool = True,
        force_expert: bool = False,
    ) -> Dict:
        
        beta = 0 if self.dagger_config.p == 0 else self.dagger_config.p ** self.args.dagger_data_it

        os.makedirs(os.path.join(self.output_path), exist_ok=True)

        episode = env.current_episode
        agent = ShortestPathFollower(sim=env.sim, goal_radius=1.8, return_one_hot=False)
        scene_id = episode.scene_id.split('/')[-2]
        episode_id = int(episode.episode_id)
        trajectory_id = episode.trajectory_id
        instructions = episode.instruction.instruction_text

        # This path includes the goal point
        ref_path = episode.reference_path

        observation = env.reset()
        annotation = []
        rgb_data_list = []
        depth_data_list = []
        step_id = 0
        actions = [-1]
        next_waypoint_id = 1

        if save_video:
            os.makedirs(os.path.join(self.output_path, 'videos'), exist_ok=True)

        initial_height = env.sim.get_agent_state().position[1]
        intrinsic_matrix = self.get_intrinsic_matrix(self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor)
        intrinsic_matrix = np.around(intrinsic_matrix, decimals=4)
        
        # main loop
        mem_ids=[]
        vis_frames = []
        left_expert_actions_num = 0
        from_expert = True if force_expert else False
        force_episode_end = False
        model_success = True
        action_seq, action_mask = [], []
        rgb_list, depth_list, pose_list, intrinsic_list, time_ids = [], [], [], [], []
        past_key_values, output_ids = None, None
        metrics = None
        accumulated_error = 0 

        # Here the gt annotations are the data collected by the streamvln, saved in trajectory_data folder, id refers to the episode id
        # The so called gt annotations are just to provide the actual number of the actions
        ref_actions_len = next((len(annot["actions"]) for annot in self.gt_annotations if int(episode_id) == annot["id"]), DEFAULT_EPISODE_LENGTH)
        print(f"reference_actions_len: {ref_actions_len}")
        
        if evaluator is not None:
            evaluator.model.eval()

        # entering the action loop
        while not env.episode_over:
            time_ids.append(step_id)
            rgb = observation["rgb"]
            depth = observation["depth"]
            x, y = observation["gps"]
            camera_yaw = observation["compass"][0]
            depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
            filled_depth = depth * (self._max_depth - self._min_depth) + self._min_depth
            depth = filled_depth * 1000

            # Habitat GPS makes west negative, so flip y
            height = env.sim.get_agent_state().position[1] - initial_height
            camera_position = np.array([x, -y, self._camera_height + height])
            tf_camera_to_episodic = self.xyz_yaw_to_tf_matrix(camera_position, camera_yaw)
            tf_camera_to_episodic = tf_camera_to_episodic @ self.get_axis_align_matrix()
            tf_camera_to_episodic = np.around(tf_camera_to_episodic, decimals=4)

            rgb_path = os.path.join(self.output_path, "images", f"{scene_id}_{self.dataset}_{episode_id:06d}", "rgb", f"{step_id:03d}.jpg")
            depth_path = os.path.join(self.output_path, "images", f"{scene_id}_{self.dataset}_{episode_id:06d}", "depth", f"{step_id:03d}.png")
            
            
            rgb_data_list.append((rgb, rgb_path))
            depth_data_list.append((depth, depth_path))

            # prepare for model inputs
            if evaluator is not None:
                image = Image.fromarray(rgb).convert('RGB')
                image_size = image.size
                image = evaluator.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
                depth_image, resize_shape = evaluator.preprocess_depth_image(Image.fromarray(depth.astype(np.uint16), mode='I;16'), do_depth_scale=True)

                intrinsic = evaluator.preprocess_instrinsic(intrinsic_matrix, image_size, resize_shape)
                intrinsic = torch.from_numpy(intrinsic).float()

                rgb_list.append(image)
                depth_list.append(torch.from_numpy(depth_image).float())
                pose_list.append(torch.from_numpy(tf_camera_to_episodic))
                intrinsic_list.append(intrinsic)     

                # get next action with mixed policy
                # This is the place that the param DAGGER_P is used! 
                # If DAGGER_P is 0, then the model will always make decisions by model
                # If DAGGER_P is 1, then the model will always follow the expert's instructions
                # If DAGGER_P is between 0 and 1, then the model will make decisions with a probability of DAGGER_P
                if len(action_seq) == 0 and left_expert_actions_num == 0:
                    from_expert = True if force_expert else random.random() < beta
                
                if len(action_seq) == 0:

                    # expert action inference!
                    if left_expert_actions_num > 0:
                        action = agent.get_next_action(ref_path[next_waypoint_id])
                        action_seq = [action]
                        left_expert_actions_num -= 1
                    else:
                        # expert action inference!
                        if from_expert:
                            action = agent.get_next_action(ref_path[next_waypoint_id])
                            action_seq = [action]
                            # generate future 4 steps
                            left_expert_actions_num = self.args.num_future_steps - 1 # HARDCODED
                        else:
                            # model action inference!
                            # This is the typical model inference process
                            if output_ids is None:
                                sources = copy.deepcopy(evaluator.conversation)
                                sources[0]["value"] = sources[0]["value"].replace(' Where should you go next to stay on track?', f' Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.')
                                if step_id != 0 :
                                    sources[0]["value"] += f' These are your historical observations {DEFAULT_MEMORY_TOKEN}.'
                                sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN+'\n', '')
                                sources[0]["value"] = sources[0]["value"].replace('<instruction>.', 
                                                                                episode.instruction.instruction_text if isinstance(episode.instruction.instruction_text, str) else episode.instruction.instruction_text[0])
                                add_system = True
                            else:
                                sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                                add_system = False

                            input_ids, conversations = evaluator.preprocess_qwen([sources], evaluator.tokenizer, True, add_system=add_system)
                            if output_ids is not None:
                                input_ids = torch.cat([output_ids,input_ids.to(output_ids.device)], dim=1)

                            images = rgb_list[-1:]
                            depths = depth_list[-1:]
                            poses = pose_list[-1:]
                            intrinsics = intrinsic_list[-1:]

                            add_mem_or_not = False
                            mem_ids.append(step_id)
                            # This is for the double check
                            if len(mem_ids)>1:
                                add_mem_or_not = ((mem_ids[-1]//evaluator.num_frames) -(mem_ids[-2] //evaluator.num_frames) >=1)
                            if step_id != 0 and (step_id % evaluator.num_frames == 0 or add_mem_or_not):
                                # import ipdb; ipdb.set_trace()
                                if evaluator.num_history is None:
                                    history_ids = slice(0, time_ids[0], evaluator.num_future_steps)
                                else:
                                    history_ids = slice(0, time_ids[0], ((time_ids[0]) // evaluator.num_history))
                                images = rgb_list[history_ids] + images
                                depths = depth_list[history_ids] + depths
                                poses = pose_list[history_ids] + poses
                                intrinsics = intrinsic_list[history_ids] + intrinsics
                                    
                            input_dict = {'images':torch.stack(images).unsqueeze(0), 'depths':torch.stack(depths).unsqueeze(0), \
                                        'poses':torch.stack(poses).unsqueeze(0), 'intrinsics':torch.stack(intrinsics).unsqueeze(0), 'inputs':input_ids, 'env_id':self.rank, 'time_ids':[time_ids],'task_type':[0]}
                            
                            input_dict = dict_to_cuda(input_dict, self.device)
    
                            for key, value in input_dict.items():
                                if key in ['images', 'depths', 'poses', 'intrinsics']:
                                    input_dict[key] = input_dict[key].to(torch.bfloat16)

                            outputs = evaluator.model.generate(**input_dict, do_sample=False, num_beams=1, max_new_tokens=10000, use_cache=True, return_dict_in_generate=True, past_key_values=past_key_values)
                            output_ids = outputs.sequences
                            past_key_values = outputs.past_key_values
                            llm_outputs = evaluator.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                            # print(llm_outputs, flush=True)
                            action_seq = evaluator.parse_actions(llm_outputs)
                            # print(action_seq, flush=True)
            else:
                action = agent.get_next_action(ref_path[next_waypoint_id])  
                action_seq = [action]
                pass
            
            # ==== Finish the action generation process, and here we get the action sequnces.
            # The debug information
            # DEBUG
            # print(f"step {step_id}, from_expert: {from_expert}, force_expert: {force_expert}, left_expert_actions_num: {left_expert_actions_num}", flush=True)
            # print(f"next waypoint id: {next_waypoint_id}, action_seq: {action_seq}, accumulated_error: {accumulated_error}, agent_get_next_action: {agent.get_next_action(ref_path[next_waypoint_id])}", flush=True)
            # import ipdb; ipdb.set_trace()

            action_source = "expert" if from_expert else "model"
            # print(f"action from {action_source}", flush=True)
            # print(f"action_seq: {action_seq}", flush=True)

            # TODO: deal with empty action sequence output
            if len(action_seq) == 0:
                # print(f"empty action, force stop", flush=True)
                action_seq = [0]

            action = action_seq.pop(0)

            # If the inferenced action is not the same as the expert's action, then accumulate the error
            if action != agent.get_next_action(ref_path[next_waypoint_id]):
                accumulated_error += 1
            
            # when reach a waypoint, free the model to make decision
            # If we get the right path --> the waypoint, we will let the model to make further decision
            # If we trigger the expert into the path guidance, then this is the only way to stop expert -- heading to the next waypoint
            
            # Also, if we are using the model actions, and go by the waypoint, we will also refresh the next waypoint.
            # once the expert online, it will garantee you can successfully going to the next waypoint

            # BUT if the model actions can successfully reach the waypoint, should the system clear the accumulated_error?
            while agent.get_next_action(ref_path[next_waypoint_id]) == 0:
                next_waypoint_id += 1
                force_expert = False
                left_expert_actions_num = 0
                if next_waypoint_id == len(ref_path) - 1:
                    # force_expert = True
                    # change the goal distance tolerance
                    agent = ShortestPathFollower(sim=env.sim, goal_radius=GOAL_RADIUS, return_one_hot=False)
                if next_waypoint_id >= len(ref_path):
                    force_episode_end = True
                    action = 0
                    action_source = "expert"
                    break
            
            # force expert to take action if the model make an error
            metrics = env.get_metrics()
            wp_id_available = next_waypoint_id < len(ref_path)

            # This is a failure detector, we will end the episode if the model get lost 
            # 1. model stop and there is far away from the goal
            # 2. relative error --> if the failure over the 90% of the common action steps between waypoints
            # from here we can also know that the accumulated_error is the error times between the waypoints
            # 3. abosulute error
            error_not_toleranted = ((from_expert == False and action == 0 and metrics["distance_to_goal"] >= 3.0) or (accumulated_error/max(1,int(ref_actions_len/(len(ref_path)-1))) > 0.8) or accumulated_error > 12)
            
            # if we find the error, we will use the expert to correct the error
            if wp_id_available and error_not_toleranted:
                model_success = False
                force_expert = True
                accumulated_error = 0
                action = agent.get_next_action(ref_path[next_waypoint_id])
                action_source = "expert"
            
            # action check
            # this is the recover for the last expert period
            # because the condition to jump out of the expert mode is action == 0
            # so if is not the end of the episode, keep moving
            if action == 0 and not force_episode_end:
                action = agent.get_next_action(ref_path[next_waypoint_id])
            
            # update env here we execute the action
            # action execuation!
            observation = env.step(action)
            metrics = env.get_metrics()

            # video
            if save_video:
                metrics = env.get_metrics()
                if metrics['top_down_map'] is not None:
                    resized_rgb = np.array(image_resize(img=observation['rgb'],
                                                        size=(int(observation['rgb'].shape[0] * 1.6), int(observation['rgb'].shape[1] * 1.6)),
                                                        channels_last=True))
                    frame = observations_to_image({'rgb': resized_rgb}, metrics)
                    frame = append_text_underneath_image(frame, episode.instruction.instruction_text if isinstance(episode.instruction.instruction_text, str) else episode.instruction.instruction_text[0])
                    # frame = overlay_text_to_image(frame, [action_source], font_size=1)
                    frame = append_text_underneath_image(frame, action_source)
                    frame = append_text_underneath_image(frame, f"force_expert is {force_expert}")
                    frame = append_text_underneath_image(frame, f"step: {step_id}")
                    frame = append_text_underneath_image(frame, f"next wp id: {next_waypoint_id} / {len(ref_path) - 1}")
                    frame = append_text_underneath_image(frame, f"action : {next_waypoint_id} / {len(ref_path) - 1}")
                    # add action_seq infomation
                    frame = append_text_underneath_image(frame, f"action_seq: {action_seq}")
                    frame = append_text_underneath_image(frame, f"current action: {action}")
                    frame = append_text_underneath_image(frame, f"accmulated_error: {accumulated_error}")
                    
                    vis_frames.append(frame)

            if env.episode_over or force_episode_end:         
                break

            actions.append(action)
            step_id += 1

            # this is for the model inference
            if step_id % evaluator.num_frames == 0:
                evaluator.model.reset_for_env(self.rank)
                output_ids = None
                past_key_values = None
                time_ids = []

        # import ipdb; ipdb.set_trace()

        # check action length
        assert len(rgb_data_list) == len(actions), f"Length of rgbs and actions mismatch, rgb_data_list: {len(rgb_data_list)}, actions: {(actions)}"

        # This is the exactly the format of the annotations!
        annotation.append({
            "id": episode_id,
            "video": os.path.join("images", f"{scene_id}_{self.dataset}_{episode_id:06d}"),
            "instructions": instructions if isinstance(instructions, list) else [instructions],
            "actions": actions,
        })

        # determine whether to save the episode
        # 这行代码是一个数据过滤器。它的目的是在一次任务（episode）结束后，根据任务的最终结果和过程指标，
        # 来判断这条刚刚采集到的轨迹是否有价值、是否应该被保存到最终的DAgger数据集中
        # A and (B or C)
        # A: 必须到达目标点附近，这是必要条件
        # B: model出现失误，有expert参与，走的路程也相对来说比较磕磕绊绊
        # C: model几乎没有失误，但是走的路程十分磕磕绊绊
        # 本质上就是保留那些瞎吉儿走, 但是最后能到终点的那些episode
        # 有点问题，如果走的很烂但这样不是和instruction没有那么align么
        ## 这个步骤很关键本质上来说


        episode_save = metrics["distance_to_goal"] < MIDGOAL_RADIUS and (((not model_success) and (metrics["rpl"] < RELATIVE_PATH_LENGTH_THRESHOLD)) or (metrics["rpl"] < SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD))
        
        # DEBUG print the metrics
        print(f"distance_to_goal: {metrics['distance_to_goal']}, relative path length: {metrics['rpl']}")

        if episode_save:
            # assert len(rgb_data_list) == len(depth_data_list), f"Length of rgbs and depths mismatch, rgb_data_list: {len(rgb_data_list)}, depth_data_list: {len(depth_data_list)}"
            # assert len(init_rgb_data_list) == len(init_depth_data_list), f"Length of rgbs and depths mismatch, init_rgb_data_list: {len(init_rgb_data_list)}, init_depth_data_list: {len(init_depth_data_list)}"
            os.makedirs(os.path.join(self.output_path, "images", f"{scene_id}_{self.dataset}_{episode_id:06d}", "rgb"), exist_ok=True)
            # os.makedirs(os.path.join(self.output_data_path, f'{scene_id}_{episode_id:06d}_{self.args.dagger_data_it}', 'depth_images'), exist_ok=True)

            for rgb, rgb_path in rgb_data_list:
                Image.fromarray(rgb).convert("RGB").save(rgb_path)
            # for depth, depth_path in depth_data_list:
            #     Image.fromarray(depth.astype(np.uint16), mode='I;16').save(depth_path)

        if save_video:
            if episode_save:
                images_to_video(vis_frames, os.path.join(self.output_path, 'videos'), f'save_{scene_id}_{self.dataset}_{episode_id:06d}', fps=6, quality=10)
                vis_frames.clear()
            else:
                images_to_video(vis_frames, os.path.join(self.output_path, 'videos'), f'notsave_{scene_id}_{self.dataset}_{episode_id:06d}', fps=6, quality=10)
                vis_frames.clear()

    
        metrics.update({
            "step_id": step_id,
            "ref_actions_len": ref_actions_len,
            "accumulated_error": accumulated_error,
            "save": int(episode_save),
            "model_success": model_success,
            "force_episode_end": force_episode_end,
            }
        )        
        
        episode_dict = dict(
            anno=annotation,
            metrics=metrics,
        )

        return episode_dict

    def update_dataset(self, evaluator, dataset=None):
        '''Update dataset with the collected data.'''
        
        # 在分布式环境下，为每个进程设置不同种子，可以避免所有进程都去采集完全相同的任务序列
        seed = self.rank
        random.seed(seed)
        np.random.seed(seed)

        if evaluator is None:
            self.args.force_expert = True

        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        
        # categorize episodes to be collected by scene
        env = self.config_env()
        scene_episode_dict = {}
        episode_uuids = []
        start = time.time()
        for episode in env.episodes:
            episode_uuid = (episode.scene_id, episode.episode_id, episode.trajectory_id)
            episode_uuids.append(episode_uuid)
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)
        
        # 在这里可以实现对任务序列的整体采样
        # 比如说：sampled_episodes_uuids = random.sample(episode_uuids, 100) 
        # DEBUG的地方！要是使用DEBUG的话，这边就采用比较少的episode就行了
        sampled_episodes_uuids = random.sample(episode_uuids, 2) 
        # sampled_episodes_uuids = episode_uuids
        sampled_episodes_by_scene = {}
        for scene_id in sorted(scene_episode_dict.keys()):
            sampled_episodes_traj_ids = [(episode_uuid[1], episode_uuid[2]) for episode_uuid in sampled_episodes_uuids if episode_uuid[0] == scene_id]
            sampled_episodes_by_scene[scene_id] = [ep for ep in scene_episode_dict[scene_id] if (ep.episode_id, ep.trajectory_id) in sampled_episodes_traj_ids]

        # main
        num_collect_episodes = 0
        start_id = 0
        annotations = []

        # import ipdb; ipdb.set_trace()

        with tqdm.tqdm(total=min(self.dagger_config.update_size, len(sampled_episodes_uuids)) // self.world_size, dynamic_ncols=True) as pbar, \
            torch.no_grad():
            # 遍历每个场景，获取采样后的任务序列
            for scene_id in sorted(scene_episode_dict.keys()):
                episodes = sampled_episodes_by_scene[scene_id]
                if len(episodes) == 0:
                    continue
                print(f"scene_id: {scene_id}, len of episodes: {len(episodes)}")
                # 每一个GPU拿到属于自己的任务序列
                # traverse the episodes that each GPU should collect
                for episode in episodes[self.rank::self.world_size]:  
                    assert scene_id == episode.scene_id, f"scene mismatch: {scene_id} vs {episode.scene_id}"          
                    scan = episode.scene_id.split('/')[-2]
                    env.current_episode = episode
                    env.current_episode.goals[0].radius = MIDGOAL_RADIUS # HARDCODED
                    if evaluator is not None:
                        evaluator.model.reset_for_env(self.rank)
                    episode_dagger = self.generate(
                        env=env,
                        evaluator=evaluator,
                        save_video=self.args.dagger_save_video,
                        force_expert=self.args.force_expert
                    )

                    with open(os.path.join(self.output_path, f"result.json"), "a") as f:
                        result = {"scene": scan, 
                                  "episode_id": episode.episode_id, 
                                  "trajectory_id": episode.trajectory_id, 
                                  "save": episode_dagger["metrics"]["save"],
                                  "model_success": episode_dagger["metrics"]["model_success"], 
                                  "success": episode_dagger["metrics"]["success"], 
                                  "relative_pl": episode_dagger["metrics"]["rpl"],
                                  "step_id": episode_dagger["metrics"]["step_id"],
                                  "ref_actions": episode_dagger["metrics"]["ref_actions_len"],
                                  "accumulated_error": episode_dagger["metrics"]["accumulated_error"],
                                  "force_episode_end": episode_dagger["metrics"]["force_episode_end"],
                                  }
                        f.write(json.dumps(result) + "\n")
                    
                    if not episode_dagger["metrics"]["save"]:
                        continue 

                    for k,v in episode_dagger.items():
                        if isinstance(v, torch.Tensor):
                            episode_dagger[k] = v.numpy()

                    # cache data
                    print(f"model_success = {episode_dagger['metrics']['model_success']}, scene {scan} id {episode.episode_id} trajectory {episode.trajectory_id}")

                    annotations.extend(episode_dagger['anno'])
                    pbar.update()
                    num_collect_episodes += 1


                    if num_collect_episodes % self.dagger_config.commit_freq == 0:
                        tgt_anno_path = os.path.join(self.output_path, f"annotations_{self.rank}.json")

                        # -------
                        if os.path.exists(tgt_anno_path):
                            merged_anno = json.load(open(tgt_anno_path))
                        else:
                            merged_anno = []
                        with open(tgt_anno_path, "w") as json_file:
                            merged_anno.extend(annotations)
                            anno_videos = set()
                            for item in merged_anno:
                                anno_videos.add(item["video"])
                            temp_anno = []
                            for item in merged_anno:
                                if item["video"] in anno_videos:
                                    temp_anno.append(item)
                                    anno_videos.remove(item["video"])
                            merged_anno = temp_anno
                            json_data = json.dumps(merged_anno, indent=4)
                            json_file.write(json_data) 
                        # -------

                    if num_collect_episodes >= self.dagger_config.update_size:
                        break
                if num_collect_episodes >= self.dagger_config.update_size:
                    break

            # -------
            tgt_anno_path = os.path.join(self.output_path, f"annotations_{self.rank}.json")

            if os.path.exists(tgt_anno_path):
                merged_anno = json.load(open(tgt_anno_path))
            else:
                merged_anno = []
            with open(tgt_anno_path, "w") as json_file:
                merged_anno.extend(annotations)
                anno_videos = set()
                for item in merged_anno:
                    anno_videos.add(item["video"])
                temp_anno = []
                for item in merged_anno:
                    if item["video"] in anno_videos:
                        temp_anno.append(item)
                        anno_videos.remove(item["video"])
                merged_anno = temp_anno
                json_data = json.dumps(merged_anno, indent=4)
                json_file.write(json_data) 
            
            if get_rank() == 0:
                tgt_anno_path = os.path.join(self.output_path, f"annotations.json")
                merged_anno = []
                sub_tgt_anno_list = [
                    os.path.join(self.output_path, f)
                    for f in os.listdir(self.output_path)
                    if f.startswith('annotations_') and f.endswith('.json')
                ]
                for sub_tgt_anno_path in sub_tgt_anno_list:
                    if os.path.exists(sub_tgt_anno_path):
                        merged_anno.extend(json.load(open(sub_tgt_anno_path)))
                merged_anno = sorted(merged_anno, key=lambda x: x['id'])
                with open(tgt_anno_path, "w") as json_file:
                    anno_videos = set()
                    for item in merged_anno:
                        anno_videos.add(item["video"])
                    temp_anno = []
                    for item in merged_anno:
                        if item["video"] in anno_videos:
                            temp_anno.append(item)
                            anno_videos.remove(item["video"])
                    merged_anno = temp_anno
                    json_data = json.dumps(merged_anno, indent=4)
                    json_file.write(json_data)
            # -------
            print(f"save scene_id {scene_id} with total episodes {num_collect_episodes} time cost {time.time() - start}")


if __name__ == "__main__":

    global local_rank
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_dagger.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/streamvln')
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help= "Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    
    parser.add_argument("--dagger_p",type=float, default=0.9)
    parser.add_argument("--dagger_update_size", type=int, default=1)
    parser.add_argument("--dagger_commit_freq",type=int, default=1)
    parser.add_argument("--dagger_dataset", type=str, default=DATASET)
    parser.add_argument("--force_expert", action="store_true", default=False)
    parser.add_argument("--dagger_data_it", type=int, default=0)
    parser.add_argument("--dagger_output_path",type=str, default="data/dagger")
    parser.add_argument("--dagger_data_path", type=str, default="data/datasets/vln_datasets/{split}.json.gz")
    parser.add_argument("--dagger_gt_annotations_path", type=str, default="data/datasets/vln_datasets/annotations.json")
    parser.add_argument("--dagger_save_video", action="store_true", default=False, help="whether to save video during dagger collection")

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='rank')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu')
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    args = parser.parse_args()
    init_distributed_mode(args)
    local_rank = args.local_rank

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path,
                                                        model_max_length=args.model_max_length,
                                                        padding_side="right")
    
    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model = StreamVLNForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=False,
    )
    model.model.num_history = args.num_history
    model.requires_grad_(False)
    model.to(local_rank)
        
    model.eval()
    rank = get_rank()
    world_size = get_world_size()

    model.reset(world_size)
    # node_id = os.environ['SLURM_NODEID']
    # node_list = os.environ['SLURM_NODELIST']


    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        tokenizer=tokenizer,
        epoch=0,
        args=args
    )
    
    collector = StreamVLNDAggerCollector(args=args, rank=rank, world_size=world_size)
    collector.update_dataset(evaluator=evaluator)