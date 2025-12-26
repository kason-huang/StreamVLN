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
from transformers import BitsAndBytesConfig
import transformers
import numpy as np

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
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
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
        print(OmegaConf.to_yaml(self.config))

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))
        self.image_processor = model.get_vision_tower().image_processor
        self.model = model
        self.tokenizer = tokenizer
        prompt = f"<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "↑": [1],
            "←": [2],
            "→": [3]
        })
        self.conjunctions = [
                                'you can see ',
                                'in front of you is ',
                                'there is ',
                                'you can spot ',
                                'you are toward the ',
                                'ahead of you is ',
                                'in your sight is '
                            ]
        self.num_frames = args.num_frames
        self.num_future_steps = args.num_future_steps
        self.num_history = args.num_history
    
    
    def config_env(self) -> Env:
        env = Env(config=self.config)
        # env.episodes = env.episodes[0:1]
        return env

    def eval_action(self, idx) -> None:
        # 初始化和环境配置模块
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        # 初始化性能指标列表（success rate, spl, oracle_success, navigation error）
        sucs, spls, oss, ones = [], [], [], []
        # 已处理列表
        done_res = []
        # 断点续传模块
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'),'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        ones.append(res['ne'])
        
        # 场景循环处理
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            # episode_id = 0
            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"scene {scene_id}")
            for episode in episodes[idx::self.env_num]:
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                print("episode start",episode_instruction)
                episode_id = episode.episode_id
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue

                # episode初始化
                self.model.reset_for_env(idx)
                env.current_episode = episode
                observations = env.reset()
                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
                Image.fromarray(observations['rgb']).save(os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{idx}.jpg'))
                
                vis_frames = []
                step_id = 0
                
                if self.save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}_{episode_id}'), exist_ok=True)
                initial_height = env.sim.get_agent_state().position[1]

                rgb_list = []
                time_ids = []
                action_seq = []
                past_key_values = None
                output_ids = None

                # 导航主循环模块
                while not env.episode_over:
                    self.model.eval()
                    time_ids.append(step_id)
                    rgb = observations["rgb"]

                    image = Image.fromarray(rgb).convert('RGB')
                    image_size = image.size
                    # image = self.image_processor.preprocess(images=image, do_rescale=True, do_normalize=True, return_tensors='pt')['pixel_values'][0]
                    image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
                    
                    rgb_list.append(image)
                    
                    # 生成俯视图可视化帧,就是那个左边observation，右边地图缩略图还有一个agent的那个图
                    info = env.get_metrics()
                    if info['top_down_map'] is not None:
                        frame = observations_to_image({'rgb':observations['rgb']}, info)
                        vis_frames.append(frame)

                    # 模型推理模块
                    # import ipdb; ipdb.set_trace()
                    if len(action_seq) == 0:
                        # 构建对话prompt，包括导航指令
                        if output_ids is None:
                            sources = copy.deepcopy(self.conversation)
                            sources[0]["value"] = sources[0]["value"].replace(' Where should you go next to stay on track?', f' Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.')
                            if step_id != 0 :
                                sources[0]["value"] += f' These are your historical observations {DEFAULT_MEMORY_TOKEN}.'
                            sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN+'\n', '')
                            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', episode.instruction.instruction_text)
                            add_system = True
                            print(step_id, sources[0]["value"])
                        else:
                            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                            add_system = False
                            
                        input_ids, conversations = self.preprocess_qwen([sources], self.tokenizer, True, add_system=add_system)
                        if output_ids is not None:
                            input_ids = torch.cat([output_ids,input_ids.to(output_ids.device)], dim=1)

                        # 处理历史记忆token，准备输入数据
                        images = rgb_list[-1:]
                        # import ipdb; ipdb.set_trace()
                        if step_id != 0 and step_id % self.num_frames == 0:
                            if self.num_history is None:
                                history_ids = slice(0, time_ids[0], self.num_future_steps)
                            else:
                                history_ids = slice(0, time_ids[0], (time_ids[0] // self.num_history))
                            images = rgb_list[history_ids] + images

                        # 输入字典        
                        input_dict = {'images':torch.stack(images).unsqueeze(0), 'depths':None, \
                                        'poses':None, 'intrinsics':None, 'inputs':input_ids, 'env_id':idx, 'time_ids':[time_ids],'task_type':[0]}
                            
                        input_dict = dict_to_cuda(input_dict, self.device)
                        
                        for key, value in input_dict.items():
                            if key in ['images']:
                                #input_dict[key] = input_dict[key].to(torch.bfloat16) # 这里应该是什么呢？没搞懂
                                input_dict[key] = input_dict[key].to(torch.bfloat16) # 这里应该是什么呢？没搞懂
                        
                        #outputs = self.model.generate(**input_dict, do_sample=False, num_beams=1, max_new_tokens=10000, use_cache=True, return_dict_in_generate=True, past_key_values=past_key_values)
                        # 调用模型生成动作序列
                        outputs = self.model.generate(**input_dict, do_sample=False, num_beams=1, max_new_tokens=256, use_cache=True, return_dict_in_generate=True, past_key_values=past_key_values)

                        # 解析LLM输出位具体动作
                        output_ids = outputs.sequences
                        past_key_values = outputs.past_key_values
                        llm_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
                        print(llm_outputs, flush=True)
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)
                        if len(action_seq) == 0: ## if generated llm without Specific values
                            action_seq = [0]

                        # bugfix: https://github.com/InternRobotics/StreamVLN/issues/3 下面都是输出的fix
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


                    action = action_seq.pop(0)
                    
                    # 执行动作并获取新观测
                    observations = env.step(action)
                    step_id += 1
                    # 定期重置模型状态
                    if step_id % self.num_frames == 0:
                        self.model.reset_for_env(idx)
                        output_ids = None
                        past_key_values = None
                        time_ids = []
                
                # episode结束处理模块
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
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction
                }
                
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
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            tokenizer.add_tokens(["<memory>"], special_tokens=True)

        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        im_start, im_end = tokenizer.additional_special_tokens_ids
        # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
        unmask_tokens_idx =  [198, im_start, im_end]
        nl_tokens = tokenizer("\n").input_ids

        # Reset Qwen chat templates so that it won't include system message every time we apply
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        # _system = tokenizer("system").input_ids + nl_tokens
        # _user = tokenizer("user").input_ids + nl_tokens
        # _assistant = tokenizer("assistant").input_ids + nl_tokens

        # Apply prompt templates
        conversations = []
        input_ids = []
        for i, source in enumerate(sources):
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else: 
                source[0]["value"] = f"{prompt}."
            if roles[source[0]["from"]] != roles["human"]:
                # Skip the first one if it is not from human
                source = source[1:]

            input_id, target = [], []

            # import ipdb; ipdb.set_trace()
            # New version, use apply chat template
            # Build system message for each sentence
            if add_system:
                input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])

            for conv in source:
                # Make sure llava data can load
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]

                role =  roles.get(role, role)
                
                conv = [{"role" : role, "content" : content}]
                # import ipdb; ipdb.set_trace()
                conversations.append(content)
                encode_id = tokenizer.apply_chat_template(conv)
                input_id += encode_id
            

            # assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
                    
            input_ids.append(input_id)
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
   
def detect_best_attention_implementation():
    """
    Automatically detect the best available attention implementation with runtime compatibility checks.
    
    Returns:
        str: Best available attention implementation ('flash_attention_2', 'sdpa', or 'eager')
    
    Priority order:
    1. flash_attention_2 (best performance, requires Ampere GPU or newer)
    2. sdpa (good performance, PyTorch 2.0+, with runtime validation)
    3. eager (fallback, highest memory usage)
    """
    
    # Check if current GPU supports FlashAttention (Ampere or newer)
    def _is_ampere_or_newer():
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    
    # Try Flash Attention 2 with runtime compatibility check
    try:
        import flash_attn
        
        if hasattr(flash_attn, 'flash_attn_func') and _is_ampere_or_newer():
            # Test with small tensors
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                q = torch.randn(1, 32, 64, device=f'cuda:{device}', dtype=torch.float16)
                k = torch.randn(1, 32, 64, device=f'cuda:{device}', dtype=torch.float16)
                v = torch.randn(1, 32, 64, device=f'cuda:{device}', dtype=torch.float16)
                
                flash_attn.flash_attn_func(q, k, v)
                print("✓ Flash Attention 2 available and compatible")
                return 'flash_attention_2'
                
    except (ImportError, RuntimeError) as e:
        if "FlashAttention only supports Ampere GPUs" in str(e):
            capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
            if capability:
                major, minor = capability
                print(f"⚠ GPU compute capability {major}.{minor} incompatible with FlashAttention (requires >= 8.0)")
        else:
            print(f"✗ Flash Attention unavailable: {type(e).__name__}")
    
    # Check PyTorch SDPA with runtime test and configure backends only when using SDPA
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        try:
            # Configure SDPA backends to avoid cutlass issues
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)  # Disable cutlass
            
            # Test SDPA with small tensors
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                batch_size, num_heads, seq_len, head_dim = 1, 8, 32, 64
                
                q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=f'cuda:{device}', dtype=torch.float16)
                k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=f'cuda:{device}', dtype=torch.float16)
                v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=f'cuda:{device}', dtype=torch.float16)
                
                torch.nn.functional.scaled_dot_product_attention(q, k, v)
                print("✓ PyTorch SDPA available and compatible (cutlass disabled)")
                return 'sdpa'
                
        except RuntimeError as e:
            if "cutlassF" in str(e) or "no kernel found" in str(e):
                print(f"✗ SDPA cutlass kernel error: {e}")
            else:
                print(f"✗ SDPA runtime error: {e}")
        except Exception as e:
            print(f"✗ SDPA configuration failed: {e}")
    
    # Final fallback
    print("⚠ Using eager attention (may be slower)")
    return 'eager'

def eval():
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/streamvln')
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
    parser.add_argument('--vision_tower_path', type=str, default=None,
            help='Path to vision tower model (e.g., checkpoints/google/siglip-so400m-patch14-384)')
    parser.add_argument("--quantization_bits", type=int,
            help="Quantization bits. 4 for 4-bit, 8 for 8-bit.", default=-1)
    
    args = parser.parse_args()
    init_distributed_mode(args)
    local_rank = args.local_rank

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path,
                                                        model_max_length=args.model_max_length,
                                                        padding_side="right")
    
    quantization_bits = args.quantization_bits
    # 传递这个道siglip里面使用
    if quantization_bits is not None:
      os.environ['QUANTIZATION_BITS'] = str(quantization_bits)


    if quantization_bits == 4:
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
        print(f"Using 4-bit quantization")
    elif quantization_bits == 8:
        qconf = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
        print(f"Using 8-bit quantization")
    else:
        print(f"\033[91mWarning: Unsupported quantization bits: {quantization_bits}, falling back to no quantization\033[0m")

    attn_implementation = detect_best_attention_implementation()

    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model = StreamVLNForCausalLM.from_pretrained(
                args.model_path,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                config=config,
                low_cpu_mem_usage=True,
                device_map="auto",
                quantization_config=qconf,
                )
    model.model.num_history = args.num_history
    model.requires_grad_(False)
    # model.to(local_rank)
    evaluate(model, tokenizer, args)



def evaluate(model, tokenizer, args):
    model.eval()
    
    world_size = get_world_size()
    model.reset(world_size)
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
