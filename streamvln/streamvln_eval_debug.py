import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import tqdm
import torch
import copy
import json
import random
import argparse
import itertools
import quaternion
import transformers
import numpy as np
import time

from typing import Any
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
from depth_camera_filtering import filter_depth
from transformers.image_utils import to_numpy_array

import habitat
from habitat import logger, Env
from habitat_extensions import measures
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video, observations_to_image

from model.stream_video_vln import StreamVLNForCausalLM
from utils.utils import dict_to_cuda
from utils.dist import *
from utils.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX, DEFAULT_VIDEO_TOKEN

from utils.time_utils import timing_context

class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        tokenizer: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device('cuda')
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        self.epoch = epoch
        self.config_path = config_path

        # here we load the habitat config and agent config
        # setting up all configs for the habitat environment
        self.config = get_habitat_config(config_path)
        # get the agent config from the habitat config
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        # get sensors config from the agent config
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        with habitat.config.read_write(self.config):
            # self.config.habitat.task.measurements.success.success_distance=3.0
            self.config.habitat.dataset.split = self.split
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

        print(f"config = {type(self.config)}")
        # pdb.set_trace()

        # print all the habitat configs
        print(OmegaConf.to_yaml(self.config))

        # setting up the camera parameters
        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))

        # setting up the model and tokenizer
        self.image_processor = model.get_vision_tower().image_processor
        self.model = model
        self.tokenizer = tokenizer

        # the prompt template for the model
        prompt = f"<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""

        # conversation format
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

        # mapping
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "↑": [1],
            "←": [2],
            "→": [3]
        })

        # conjunctions for the prompt
        self.conjunctions = [
                                'you can see ',
                                'in front of you is ',
                                'there is ',
                                'you can spot ',
                                'you are toward the ',
                                'ahead of you is ',
                                'in your sight is '
                            ]

        # paramters for the model for evaluation
        # I think this part is related to the trainning of the model
        self.num_frames = args.num_frames
        self.num_future_steps = args.num_future_steps
        self.num_history = args.num_history
        

    def preprocess_depth_image(self, depth_image, do_depth_scale=True, depth_scale=1000):
        target_height = self.image_processor.crop_size['height']  # 384
        target_width  = self.image_processor.crop_size['width']  # 384
        resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)
        
        img = to_numpy_array(resized_depth_image)
        if do_depth_scale:
            img = img / depth_scale
    
        return img, (target_width, target_height)
    
    def get_intrinsic_matrix(self, sensor_cfg) -> np.ndarray:
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
    
    def preprocess_instrinsic(self, intrinsic, ori_size, target_size):  # (V, 4, 4) (resize_shape) (h, w)
        intrinsic = copy.deepcopy(intrinsic)
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[None, :, :]  # (1, 4, 4) or (B, 4, 4)
        
        intrinsic[:, 0] /= ori_size[0] / target_size[0]  # width
        intrinsic[:, 1] /= ori_size[1] / target_size[1]  # height

        # for crop transform
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2

        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)

        return intrinsic
    
    def get_axis_align_matrix(self):
        # 这个mat目的是为了将habitat 坐标系变换成为通用的世界坐标系
        # habitat坐标系：+Y 向上 +Z 向前 +X 向右
        # 通用世界坐标系： +Z 向上 +X 向前 +Y 向左

        # ma = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        ma = torch.tensor([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).double()
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

    def config_env(self) -> Env:
        env = Env(config=self.config)
        
        # DEBUG: Using only one episode for debugging
        # env.episodes = env.episodes[0:1]

        return env

    def eval_action(self, idx) -> None:

        # Building up the habitat environment based on the configuration
        env = self.config_env()

        # Get all the episodes rearranged by the scene_id
        # I guess it is for loading the glb for once
        # episode.scene_id is a string that represents the DIR PATH of the scene glb
        # e.g. /data/home/jiangjiajun/data/scene_datasets/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)
        
        # filtered_scene_episode_dict = {
        #     k: v for k, v in scene_episode_dict.items() if "QUCTc6BB5sX" in k
        # }

        # Get intrinsic Mat
        intrinsic_matrix = self.get_intrinsic_matrix(self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor)
        

        # Setup Result metrics
        # Also record the previously done episodes
        # so that we can skip the already done episodes
        sucs, spls, oss, ones = [], [], [], []

        done_res = []
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'),'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])

                    # only record the metrics in the main process
                    # to avoid multi processes writing to the same file
                    if get_rank() == 0:
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        ones.append(res['ne'])

        for scene in sorted(scene_episode_dict.keys()):
            
            # scene is the DIR PATH of the scene glb
            # e.g. /data/home/jiangjiajun/data/scene_datasets/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb

            # get the episodes for the target scene
            episodes = scene_episode_dict[scene]

            # get the scene_id from the scene path
            # e.g. zsNo4HB9uLZ
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")


            # episode_id = 0
            # 对episodes进行切片，将episodes按照env_num进行分割，不同GPU对应的idx获取相应的episodes
            # 所以就是在这里进行了episodes的分割啊，然后每一个GPU就分配到了他对应的episode的数量
            # idx::self.env_num 代表从第 idx 个环境开始，每隔 self.env_num 取一个，所以拿到的就是 某个环境对应的所有 episode
            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"scene {scene_id}")

            # Iterate through the episodes for the current scene
            # 每一个GPU处理自己的idx对应的episodes
            # Here the iteration in episodes
            for episode in episodes[idx::self.env_num]:

                # 如果是objectnav任务，则使用episode.object_category作为指令
                # 否则使用episode.instruction.instruction_text作为指令
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                print("episode start, instruction is: ",episode_instruction)

                # 获取episode独有的id
                episode_id = episode.episode_id


                print(f"scene_episode {scene_id}_{episode_id}")

                # debug: if the episode id is not 1003, skip it
                # if episode_id != 1003:
                #     continue
                
                # So the system do have the ability to resume the running of the episode
                # 这里也就是告诉我们，这个代码可以进行resume
                if [scene_id, episode_id, episode_instruction] in done_res:
                    print(f"scene_episode {scene_id}_{episode_id} already done, skip")
                    continue


                self.model.reset_for_env(idx)

                # 指定当前的episode
                env.current_episode = episode

                # based on the current episode, reset the environment
                observations = env.reset()

                # save the initial rgb image for debugging
                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
                Image.fromarray(observations['rgb']).save(os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{idx}.jpg'))
                
                vis_frames = []
                step_id = 0
                
                if self.save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}_{episode_id}'), exist_ok=True)
                
                # Get the initial height of the agent, 
                initial_height = env.sim.get_agent_state().position[1]

                rgb_list = []
                depth_list = []
                depth_images_list = []
                pose_list = []
                intrinsic_list = []


                time_ids = []
                action_seq = []
                past_key_values = None
                output_ids = None

                # set time calculation
                episode_start_time = time.time()

                # 当前的episode执行，退出条件就是当前episode执行完毕
                # 在这边就是持续在当前的episode下面进行action操作
                while not env.episode_over:
                    

                    with timing_context('input_preprocess', self, 'timing_results'):
                        self.model.eval()
                        time_ids.append(step_id)
                        
                        ###########
                        # 处理habitat的observation，model输入预处理
                        rgb = observations["rgb"]
                        depth = observations["depth"]
                        x, y = observations["gps"]
                        camera_yaw = observations["compass"][0]
                        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                        depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                        depth = depth * 1000

                        agent_state = env.sim.get_agent_state()
                        height = agent_state.position[1] - initial_height # Habitat GPS makes west negative, so flip y
                        camera_position = np.array([x, -y, self._camera_height + height])
                        robot_xy = camera_position[:2]

                        # 获取T，这应该是给depth pruning用的
                        tf_camera_to_episodic = self.xyz_yaw_to_tf_matrix(camera_position, camera_yaw)
                        
                        # Pose: Agent to world
                        rotation = agent_state.rotation
                        translation = agent_state.position
                        rotation_matrix = quaternion.as_rotation_matrix(rotation)
                        transformation_matrix = np.eye(4)
                        transformation_matrix[:3, :3] = rotation_matrix
                        transformation_matrix[:3, 3] = translation
                        
                        image = Image.fromarray(rgb).convert('RGB')
                        image_size = image.size
                        # image = self.image_processor.preprocess(images=image, do_rescale=True, do_normalize=True, return_tensors='pt')['pixel_values'][0]
                        image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
                        depth_image, resize_shape = self.preprocess_depth_image(Image.fromarray(depth.astype(np.uint16), mode='I;16'), do_depth_scale=True)
                        
                        # 因为resize了depth的shape，所以这里intrinsics也需要重新计算
                        intrinsic = self.preprocess_instrinsic(intrinsic_matrix, image_size, resize_shape)
                        intrinsic = torch.from_numpy(intrinsic).float()
        
                        rgb_list.append(image)
                        depth_list.append(torch.from_numpy(depth_image).float())

                        # TODO: 此处乘法存疑，这里的pose处理存疑
                        # 之后参考这里：https://github.com/Eku127/habitat-data-collector/blob/fc986ababcc57b361b7f640f5824cf083415796e/habitat_data_collector/utils/ros_data_collector.py#L157C13-L157C29
                        # 之后可以参考我自己的habitat data collector的代码来看这个pose的变换关系
                        pose_list.append(torch.from_numpy(tf_camera_to_episodic) @ self.get_axis_align_matrix())

                        intrinsic_list.append(intrinsic)
                        
                        info = env.get_metrics()

                        # 构造一个vis_frames的图像用来视频可视化
                        if info['top_down_map'] is not None:
                            frame = observations_to_image({'rgb':observations['rgb']}, info)
                            vis_frames.append(frame)

                    # ==============================================================
                    # 以上就是处理当前step的现有的状态信息，到达这里之后就是真正的开始进行inference输出action了

                    # import ipdb; ipdb.set_trace()

                    # 只有action seq是空的情况下才会进行模型的generate
                    # 如果action seq还保留上次generate出来的actions，那么直接跳过去执行action去
                    # 因为我们现在有的模型还是一次inference输出四个动作的形式来去做的
                    if len(action_seq) == 0:
                        # 如果是第一次，very beginning
                        if output_ids is None:
                            # 构建conversation
                            sources = copy.deepcopy(self.conversation)
                            sources[0]["value"] = sources[0]["value"].replace(' Where should you go next to stay on track?', f' Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.')
                            # 如果短期记忆被清理了，这边需要增加上memory的表示
                            if step_id != 0 :
                                sources[0]["value"] += f' These are your historical observations {DEFAULT_MEMORY_TOKEN}.'
                            
                            # 这里把<video>\n去掉
                            sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN+'\n', '')
                            
                            # 这里就把episode的instruction给扔进去了
                            # 放的位置就是human的位置
                            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', episode.instruction.instruction_text)
                            
                            # 然后add system设置了
                            # 这个 add system 就是增加一个system的token： you are a help assistant
                            # 如果说第一次的话就加上这个，如果不是第一次就不要加了
                            add_system = True
                            print(step_id, sources[0]["value"])
                        else:
                            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                            add_system = False
                        
                        # 这里是处理成qwen所需要的输入
                        # 这里处理对话的prompt，将prompt变成token id作为input id
                        # 然后将prompt中需要嵌入image和memory的地方替换成相对应的default tokens
                        with timing_context('template_tokenization', self, 'timing_results'):
                            input_ids, conversations = self.preprocess_qwen([sources], self.tokenizer, True, add_system=add_system)
                        
                        
                            if output_ids is not None:
                                input_ids = torch.cat([output_ids,input_ids.to(output_ids.device)], dim=1)

                        with timing_context('model_input_preprocess', self, 'timing_results'):
                            # 获取最新的一帧rgb的list
                            images = rgb_list[-1:]
                            depths = depth_list[-1:]
                            poses = pose_list[-1:]
                            intrinsics = intrinsic_list[-1:]

                            # import ipdb; ipdb.set_trace()

                            # 如果是第一次，那么step_id == 0 下面的判断不会进入
                            # num_frames是32
                            if step_id != 0 and step_id % self.num_frames == 0:
                                if self.num_history is None:
                                    history_ids = slice(0, time_ids[0], self.num_future_steps)
                                else:
                                    history_ids = slice(0, time_ids[0], (time_ids[0] // self.num_history))
                                images = rgb_list[history_ids] + images
                                depths = depth_list[history_ids] + depths
                                poses = pose_list[history_ids] + poses
                                intrinsics = intrinsic_list[history_ids] + intrinsics
                            
                            # 输入mock成为dict
                            input_dict = {'images':torch.stack(images).unsqueeze(0), 'depths':torch.stack(depths).unsqueeze(0), \
                                            'poses':torch.stack(poses).unsqueeze(0), 'intrinsics':torch.stack(intrinsics).unsqueeze(0), 'inputs':input_ids, 'env_id':idx, 'time_ids':[time_ids],'task_type':[0]}
                                
                            input_dict = dict_to_cuda(input_dict, self.device)
                            
                            # 将输入的图像和深度图像转换为bfloat16张量格式
                            for key, value in input_dict.items():
                                if key in ['images', 'depths', 'poses', 'intrinsics']:
                                    input_dict[key] = input_dict[key].to(torch.bfloat16)
                        
                        ## TODO: 进一步去分析generate里面的设计
                        ## 此处就是使用已经训练好的模型进行generate
                        # with timing_context('model_generate', self, 'timing_results'):
                        #     outputs = self.model.generate(**input_dict, do_sample=False, num_beams=1, max_new_tokens=10000, use_cache=True, return_dict_in_generate=True, past_key_values=past_key_values)

                        if step_id == 0 or step_id % self.num_frames == 0:
                            with timing_context('model_generate_long', self, 'timing_results'):
                                outputs = self.model.generate(**input_dict, do_sample=False, num_beams=1, max_new_tokens=10000, use_cache=True, return_dict_in_generate=True, past_key_values=past_key_values)
                        else:
                            with timing_context('model_generate_short', self, 'timing_results'):
                                outputs = self.model.generate(**input_dict, do_sample=False, num_beams=1, max_new_tokens=10000, use_cache=True, return_dict_in_generate=True, past_key_values=past_key_values)

                        # 也就是他们经过训练之后的大模型才会吐出这样的输出
                        output_ids = outputs.sequences

                        # past_key_values 是 Transformer 模型中用于 加速生成推理（generation）过程 的一种缓存机制。
                        # 它的作用是在 多轮生成（如 .generate() 推理中）复用前一轮计算出的注意力信息，避免重复计算。
                        # 这里取出来是为了在下一轮生成（generate）时继续使用之前的注意力信息
                        past_key_values = outputs.past_key_values

                        # 使用tokenizer反解成文本
                        ## output_ids: tensor([[151644,  77091,    198,  51018,  76286,  71858,  76286, 151645]], device='cuda:0')
                        ## llm_outputs: '<|im_start|>assistant\n→↑←↑<|im_end|>'
                        with timing_context('decode', self, 'timing_results'):
                            llm_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
                        print(llm_outputs, flush=True)

                        # 将输出重置生成的action序列 list
                        # 或者说，将输出的序列规范成habitat能听懂的动作序列
                        ## action_seq: [3, 1, 2, 1]
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)
                        if len(action_seq) == 0: ## if generated llm without Specific values
                            action_seq = [0]
                        
                        # 下面都是输出的fix
                        if len(action_seq) < 4 and 0 not in action_seq:
                            action_seq_original = action_seq
                            print(f"action_seq is too short: {len(action_seq)}, fill with 2, 3", flush=True)
                            with open(os.path.join(self.output_path, f'check_sim_{self.epoch}', f'{scene_id}_{episode_id}.txt'), 'w') as f:
                                f.write(' '.join(str(a) for a in action_seq_original))
                            if len(action_seq) == 0:
                                action_seq = [2, 3, 2, 3]  
                            elif len(action_seq) == 1:
                                action_seq += [2, 2, 3]     
                            elif len(action_seq) == 2:
                                action_seq += [2, 3]       
                            elif len(action_seq) == 3:
                                action_seq += [2]
                        
                        if len(action_seq) > 4:
                            print(f"action_seq is too long: {len(action_seq)}, truncate to 4", flush=True)
                            action_seq_original = action_seq
                            action_seq = action_seq[:4]
                            # export and save the "scene id" into the self.output_path as a txt
                            # os.path.join(self.output_path, f'check_sim_{self.epoch}' save in this directory
                            # named as f"{scene_id}_{episode_id}_action_seq.txt"
                            with open(os.path.join(self.output_path, f'check_sim_{self.epoch}', f'{scene_id}_{episode_id}.txt'), 'w') as f:
                                f.write(' '.join(str(a) for a in action_seq_original))
                    # 到上面为止就是使用VLM输出动作序列的部分了，下面要干的事情就是，使用生成的动作序列在场景中进行运动
                    # 也就是在habitat sim中去执行这些actions
                    # 每一次生成都会出来四个action动作，然后进行执行
                    ################################################################


                    with timing_context('env_step', self, 'timing_results'):
                        # 从这里可以看出，不管输出了啥，这边都会执行
                        # 所以每一次generate都会生成未来四次的动作序列
                        action = action_seq.pop(0)
                        
                        # 场景执行action
                        observations = env.step(action)

                    # 所以每一次模型进行generate，都会产生四个step id，然后执行操作
                    step_id += 1

                    # 然后在执行32次action，模型generate 8次 之后 对整体进行一次reset
                    # 然后在执行的过程中，每次的图像都会存下来
                    if step_id % self.num_frames == 0:
                        # TODO: 这里的reset_for_env是什么意思？是否清空了历史的信息？
                        self.model.reset_for_env(idx)
                        output_ids = None
                        past_key_values = None
                        time_ids = []

                # set time end of one episode
                episode_end_time = time.time()

                # get the total test time
                episode_all_time = episode_end_time - episode_start_time
                print(f"Total inference time for episode {episode_id}: {episode_all_time:.2f} seconds")

                # 当前的episode执行完毕，process bar往前走一步，然后开始统计这次episode的结果        
                process_bar.update(1)
                # episode_id += 1
                metrics = env.get_metrics()
                if self.save_video:
                    images_to_video(
                        vis_frames, os.path.join(self.output_path, f'vis_{self.epoch}'), f'{scene_id}_{episode_id}', fps=6, quality=9
                    )
                vis_frames.clear()
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])
                print(f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}")
                
                # 统计所有的耗时情况
                time_summary = {
                    key: {
                        "sum": sum(times),
                        "avg": sum(times) / len(times) if times else 0
                    }
                    for key, times in self.timing_results.items()
                }

                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    # "ndtw": metrics["ndtw"],
                    "steps": step_id,
                    "t_total": episode_all_time,
                    "t_input_preprocess": time_summary["input_preprocess"]["sum"],
                    "t_model_encode_sum": time_summary["template_tokenization"]["sum"],
                    "t_model_input_preprocess": time_summary["model_input_preprocess"]["sum"],
                    "t_model_long_sum": time_summary["model_generate_long"]["sum"],
                    "t_model_short_sum": time_summary["model_generate_short"]["sum"],
                    "t_model_decode_sum": time_summary["decode"]["sum"],
                    "t_env_step_sum": time_summary["env_step"]["sum"],
                    "episode_instruction": episode_instruction
                }

                # clear the timing results
                self.timing_results = {key: [] for key in self.timing_results.keys()}
                
                with open(os.path.join(self.output_path, f'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")

        env.close()
        return torch.tensor(sucs).to(self.device), torch.tensor(spls).to(self.device), torch.tensor(oss).to(self.device), torch.tensor(ones).to(self.device), torch.tensor(len(sucs)).to(self.device)   

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)



    def preprocess_qwen(self, sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.",add_system: bool = False):
        # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
        roles = {"human": "user", "gpt": "assistant"}
        # import ipdb; ipdb.set_trace()
        # Add image tokens to tokenizer as a special tokens
        # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
        tokenizer = copy.deepcopy(tokenizer)
        # When there is actually an image, we add the image tokens as a special token
        if has_image:
            # 将 "<image>" 这个字符串，注册为一个特殊 token（不是普通新词），并加入到 tokenizer 的词表（vocab）中
            # 如果你不先注册，tokenizer.tokenize("<image>") 会这样：['<', 'image', '>']
            # 但注册之后，会变成：['<image>']
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            tokenizer.add_tokens(["<memory>"], special_tokens=True)

        # 将特殊 token "<image>" 转换为其在 tokenizer 中对应的 token ID。
        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")

        # ['<|im_start|>', '<|im_end|>'] [151643, 151644]
        # 这些是 Qwen 默认注册的两个特殊 token
        im_start, im_end = tokenizer.additional_special_tokens_ids

        # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
        unmask_tokens_idx =  [198, im_start, im_end]

        # Qwen中换行的token id是 198
        nl_tokens = tokenizer("\n").input_ids

        # Reset Qwen chat templates so that it won't include system message every time we apply
        # chat template的作用就是将刚刚的conversation格式化成Qwen所需要的输入字符串
        # 这里streamvln干的事情就是把qwen原本的template中的You are a helpful assistant.这个给删掉了
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        # _system = tokenizer("system").input_ids + nl_tokens
        # _user = tokenizer("user").input_ids + nl_tokens
        # _assistant = tokenizer("assistant").input_ids + nl_tokens

        # Apply prompt templates
        conversations = []
        input_ids = []

        # 这边就是开始组织要输入的信息和prompt了
        ## sources: 
        ## [[{'from': 'human', 'value': 'You are an autonomous navigation assistant. Your task is..'}, {'from': 'gpt', 'value': ''}]]
        for i, source in enumerate(sources):
            # 选择一个随机的连接词，并添加默认的图像标记
            ## prompt
            ## 'you can spot <image>'
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN

            # pdb.set_trace()

            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else: 
                source[0]["value"] = f"{prompt}."
            
            if roles[source[0]["from"]] != roles["human"]:
                # Skip the first one if it is not from human
                source = source[1:]

            # 这样就得到了组合上conjunction的prompt
            ## source[0]["value"]：
            ## 'You are an autonomous navigation assistant. Your task is ... you can spot <image>.'
            input_id, target = [], []

            # import ipdb; ipdb.set_trace()

            # New version, use apply chat template
            # Build system message for each sentence
            # 如果是true的话，那么会增加上system的token，也就是you are a helpful assistant.
            # 这个token会被添加到每个句子的开头 if add_system is True
            if add_system:
                input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])

            # 对prompt进行tokenize化
            for conv in source:

                # 所以这里就是进行格式转换，转换成Qwen所需要的格式形式
                # Make sure llava data can load
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]

                ## role: 'human'
                ## roles: {'human': 'user', 'gpt': 'assistant'}
                role =  roles.get(role, role)

                ## role: 'user'
                conv = [{"role" : role, "content" : content}]

                ## 第一次的conv:
                ## [{'role': 'user', 'content': 'You are an autonomous navigation assistant. Your task is ... you can spot <image>.'}]
                ## [{'role': 'assistant', 'content': ''}]
                # import ipdb; ipdb.set_trace()
                conversations.append(content)
                encode_id = tokenizer.apply_chat_template(conv)
                input_id += encode_id
            
            # 所以最后encode之后的input_id本质上就是上面的部分
            # assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

            # 遍历所有的token， 把特定的token替换成image_token_index和memory_token_index
            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
            
            # 完成遍历后，-200和-300便嵌入

            # 将当前的obs的prompt输入到input ids中去
            input_ids.append(input_id)

        # 将list of list转换成为int64的张量数组
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return input_ids,  conversations # tensor(bs x seq_len)

def pad_tensors(tensors, lens=None, max_len=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
        if len(lens) == 1 and lens[0] == max_len:
            return tensors
    if max_len is None:
        max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[1:]
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, *hid, dtype=dtype).to(tensors[0].device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output
   
def eval():
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/streamvln-R2Rv1-slurm-eval')
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help= "Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    
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

    # Setting up tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path,
                                                        model_max_length=args.model_max_length,
                                                        padding_side="right")
    
    # setting up model config from the pretrained model
    config = transformers.AutoConfig.from_pretrained(args.model_path)
    
    # Setting up the model
    # StreamVLNForCausalLM is a custom model that inherits from transformers.PreTrainedModel
    # We will discuss the details of this model when we entering the trainning process
    model = StreamVLNForCausalLM.from_pretrained(
                args.model_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                config=config,
                low_cpu_mem_usage=False,
                )
    
    # Custom settings for the model, which is the number of frames, future steps and history
    model.model.num_history = args.num_history

    # because we are in eval mode, we don't need to set the gradient to True
    model.requires_grad_(False)
    model.to(local_rank)
    print(f"[Rank {local_rank}] Model on device:", next(model.parameters()).device)

    # This is the main evaluation function
    evaluate(model, tokenizer, args)



def evaluate(model, tokenizer, args):

    # Set the model to evaluation mode
    model.eval()
    
    # Get the process number for current distributed calculation
    world_size = get_world_size()

    # The model functions, I guess split maybe also in this function
    model.reset(world_size)

    
    # This block is the main evaluation process
    # First is the initalization of the evaluator
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
    # Then we call the eval_action function to get the evaluation results
    sucs, spls, oss, ones, ep_num = evaluator.eval_action(get_rank()) 



    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)
    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]
    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(ones_all, ones)
    dist.barrier()
    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    ones_all = torch.cat(ones_all, dim=0)
    result_all = {
                    "sucs_all": (sum(sucs_all)/len(sucs_all)).item(),
                    "spls_all": (sum(spls_all)/len(spls_all)).item(),
                    "oss_all": (sum(oss_all)/len(oss_all)).item(),
                    "ones_all": (sum(ones_all)/len(ones_all)).item(),
                    'length': len(sucs_all)
                }
    
    print(result_all)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))

if __name__ == "__main__":
    eval()
