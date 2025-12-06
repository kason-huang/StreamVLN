import os
import torch
import json
import copy
import random
import tokenizers
import numpy as np
import transformers
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from typing import Dict, Optional, Sequence, List
from PIL import Image
from packaging import version

from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token
from llava.model import *

from streamvln.utils.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX
from streamvln.args import DataArguments

# 复用必要的常量和函数
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")

# 从VLNActionDataset复用的工具函数
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

def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # 只检查单张图片
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

            # 清理噪声数据
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources

def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    # 复用tokenizer处理逻辑
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

    # 应用prompt模板
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # 构建系统消息
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)

            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)[1:]  # 去掉bos token
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

def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """ObjectNav专用的预处理函数"""
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # 可以根据需要添加其他模型的预处理函数
    else:
        # 默认使用llama3的预处理
        return preprocess_llama3(sources, tokenizer, has_image=has_image)

def pad_tensors(tensors, lens=None, max_len=None, pad=0):
    """填充tensor序列"""
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


class ObjNavActionDataset(Dataset):
    """
    独立的Object Navigation Action Dataset
    专门为Object Navigation任务设计，不继承VLNActionDataset
    """

    def __init__(
        self,
        tokenizer,
        data_args,
        task_id="objectnav"
    ):
        super(ObjNavActionDataset, self).__init__()

        self.task_id = task_id
        self.image_size = data_args.image_size
        self.tokenizer = tokenizer
        self.transforms = data_args.transform_train
        self.image_processor = SigLipImageProcessor()

        # ObjectNav特定的超参数
        self.num_frames = getattr(data_args, 'num_frames', 4)
        self.num_history = getattr(data_args, 'num_history', 4)
        self.num_future_steps = getattr(data_args, 'num_future_steps', 4)
        self.remove_init_turns = getattr(data_args, 'remove_init_turns', False)

        # ObjectNav数据路径
        self.video_folder = getattr(data_args, 'objnav_video_folder', '').split(',')

        # 加载ObjectNav数据
        self.nav_data = self.load_objectnav_data()

        # 生成训练样本
        self.data_list = []
        self.generate_data_samples()

        # ObjectNav动作映射
        self.idx2actions = {
            '0': 'STOP',
            '1': "↑",
            '2': "←",
            '3': "→",
        }

        # ObjectNav特有的对话模板
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is '
        ]

        self.act_conjunctions = [
            'and then ',
            'after that ',
            'next ',
            'the next action is ',
            'followed by ',
            'leading to ',
            'continuing ',
            'subsequently ',
            'proceeding to '
        ]

        # ObjectNav任务特定的prompt
        self.conversations = self.create_objectnav_conversations()

    def create_objectnav_conversations(self):
        """创建ObjectNav特定的对话模板"""
        prompt = "You are an object finding assistant. Your task is to <instruction>. Devise an action sequence using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        return [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

    def load_objectnav_data(self):
        """加载ObjectNav格式的数据"""
        nav_data = []

        if not self.video_folder or not self.video_folder[0]:
            print("Warning: No ObjectNav video folder specified")
            return nav_data

        for vf in self.video_folder:
            if not vf.strip():
                continue

            anno_path = os.path.join(vf, 'annotations.json')
            if not os.path.exists(anno_path):
                print(f"Warning: ObjectNav annotation file not found: {anno_path}")
                continue

            try:
                with open(anno_path, 'r') as f:
                    objectnav_data = json.load(f)

                print(f"Loading {len(objectnav_data)} ObjectNav episodes from {vf}")

                # 转换数据格式
                for item in objectnav_data:
                    converted_item = self.convert_objectnav_format(item, vf)
                    if converted_item:
                        nav_data.append(converted_item)

            except Exception as e:
                print(f"Error loading ObjectNav data from {vf}: {e}")

        print(f"Successfully loaded {len(nav_data)} ObjectNav episodes")
        return nav_data

    def convert_objectnav_format(self, item, base_path):
        """转换ObjectNav数据为内部格式"""
        try:
            # 确保必要字段存在
            required_fields = ['id', 'video', 'actions', 'object_category']
            for field in required_fields:
                if field not in item:
                    print(f"Missing required field '{field}' in item {item.get('id', 'unknown')}")
                    return None

            # 动态生成多样化指令
            object_category = item.get("object_category", "object")
            instructions = self.generate_objectnav_instructions(object_category)

            return {
                "id": item["id"],
                "video": os.path.join(base_path, item["video"]),
                "instructions": instructions,  # 动态生成的指令
                "actions": item["actions"],  # 用户提供的actions
                "object_category": object_category
            }
        except Exception as e:
            print(f"Error converting ObjectNav item {item.get('id', 'unknown')}: {e}")
            return None

    def generate_objectnav_instructions(self, object_category):
        """生成多样化的ObjectNav指令"""
        templates = [
            f"Navigate to the {object_category}.",
            f"Find and move to the {object_category}.",
            f"Go to the {object_category}.",
            f"Walk towards the {object_category}.",
            f"Find the {object_category} and stop there.",
            f"Move to where the {object_category} is located.",
            f"Navigate to find the {object_category}."
        ]

        # 随机选择3-5个不同的指令
        return random.sample(templates, min(len(templates), random.randint(3, 5)))

    def generate_data_samples(self):
        """生成训练样本"""
        for ep_id, item in enumerate(self.nav_data):
            instructions = item['instructions']
            actions = item['actions']
            actions_len = len(actions)

            # 过滤过短的轨迹
            if actions_len < 4:
                continue

            if not isinstance(instructions, list):
                instructions = [instructions]

            # 为每个指令生成样本
            for ins_id in range(len(instructions)):
                valid_idx = 0
                if self.remove_init_turns:
                    valid_idx = self.clean_initial_rotations(instructions[ins_id], actions)

                if actions_len - valid_idx < 4:
                    continue

                # 生成多轮对话样本
                num_rounds = (actions_len - valid_idx) // self.num_frames
                for n in range(num_rounds + 1):
                    if n * self.num_frames == actions_len - valid_idx:
                        continue
                    self.data_list.append((ep_id, ins_id, n * self.num_frames, valid_idx))

        print(f"Generated {len(self.data_list)} training samples from ObjectNav data")

    def clean_initial_rotations(self, instruction, actions):
        """清理初始旋转动作"""
        valid_idx = 0
        for i, action in enumerate(actions[:10]):  # 只检查前10个动作
            if action in [2, 3]:  # 只旋转，没有前进
                valid_idx = i + 1
            else:
                break
        return valid_idx

    def __len__(self):
        return len(self.data_list)

    @property
    def task(self):
        return self.task_id

    def actions2text(self, actions):
        """将动作序列转换为文本表示"""
        converted_sequence = []
        for action in actions:
            act_text = self.idx2actions.get(str(action), 'STOP')
            if type(act_text) == list:
                act_text = random.choice(act_text)
            converted_sequence.append(act_text)

        return ''.join(converted_sequence)

    def prepare_objectnav_conversation(self, conversation, instruction, actions):
        """准备ObjectNav特定的对话格式"""
        i = 0
        sources = []

        while i < len(actions):
            source = copy.deepcopy(conversation)
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            step_actions = actions[i:i+self.num_future_steps]
            answer = self.actions2text(step_actions)

            if i == 0:
                # 第一轮对话，替换<instruction>占位符
                source[0]["value"] = source[0]["value"].replace("<instruction>.", instruction)
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."

            source[1]["value"] = answer
            i += len(step_actions)
            sources.extend(source)

        return sources

    def __getitem__(self, i):
        """获取单个训练样本"""
        ep_id, ins_id, start_idx, valid_idx = self.data_list[i]
        data = self.nav_data[ep_id]

        # 加载视频帧
        video_path = data['video']
        rgb_folder = os.path.join(video_path, 'rgb')

        if not os.path.exists(rgb_folder):
            print(f"Warning: RGB folder not found: {rgb_folder}")
            # 返回空的占位数据
            return self._get_empty_sample()

        video_frames = sorted([f for f in os.listdir(rgb_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])

        if len(video_frames) == 0:
            print(f"Warning: No video frames found in {rgb_folder}")
            return self._get_empty_sample()

        instructions = data.get("instructions", None)
        if not isinstance(instructions, list):
            instructions = [instructions]

        actions = data['actions'][1+valid_idx:] + [0]  # 添加STOP动作
        actions_len = len(actions)
        time_ids = np.arange(start_idx, min(start_idx + self.num_frames, actions_len))

        if len(time_ids) == 0:
            return self._get_empty_sample()

        actions = np.array(actions)[time_ids]

        # 采样当前步骤的帧
        start_idx_adj, end_idx, interval = time_ids[0]+valid_idx, time_ids[-1]+1+valid_idx, self.num_future_steps
        sample_step_ids = np.arange(start_idx_adj, min(end_idx, len(video_frames)), interval, dtype=np.int32)

        # 防止索引越界
        sample_step_ids = sample_step_ids[sample_step_ids < len(video_frames)]
        if len(sample_step_ids) == 0:
            sample_step_ids = np.array([min(start_idx_adj, len(video_frames)-1)])

        sample_frames = [os.path.join(rgb_folder, video_frames[i]) for i in sample_step_ids]

        # 采样历史帧
        if time_ids[0] != 0:
            # 使用与VLN相同的逻辑：从起点到当前时间点的完整历史
            history_step_ids = np.arange(0+valid_idx, time_ids[0]+valid_idx, max(time_ids[0] // self.num_history, 1))
            history_step_ids = history_step_ids[history_step_ids < len(video_frames)]
            history_frames = [os.path.join(rgb_folder, video_frames[i]) for i in history_step_ids]
        else:
            history_frames = []

        # 处理图像
        images = []
        for image_file in history_frames + sample_frames:
            try:
                image = Image.open(image_file).convert('RGB')
                if self.transforms is not None:
                    image = self.transforms(image)

                image_tensor = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
                images.append(image_tensor)
            except Exception as e:
                print(f"Error processing image {image_file}: {e}")
                # 创建一个空的tensor占位
                images.append(torch.zeros(3, self.image_size, self.image_size))

        if len(images) == 0:
            return self._get_empty_sample()

        images = torch.stack(images)

        # 准备对话
        sources = copy.deepcopy(self.conversations)

        # 从生成的指令中选择一个（模仿VLN的模式）
        instructions = data.get("instructions", [])
        instruction = instructions[ins_id % len(instructions)]  # 使用对应的指令

        if start_idx != 0:
            sources[0]["value"] += f' These are your historical observations: {DEFAULT_MEMORY_TOKEN}.'

        # 生成ObjectNav特定的对话
        interleave_sources = self.prepare_objectnav_conversation(sources, instruction, list(actions))

        # 预处理文本
        try:
            data_dict = preprocess([interleave_sources], self.tokenizer, True)

            return (
                data_dict["input_ids"][0],
                data_dict["labels"][0],
                images,
                torch.tensor(time_ids),
                self.task
            )
        except Exception as e:
            print(f"Error in text preprocessing: {e}")
            return self._get_empty_sample()

    def _get_empty_sample(self):
        """返回空的占位样本"""
        # 创建最小的有效样本
        input_ids = torch.tensor([self.tokenizer.pad_token_id] * 10)
        labels = torch.tensor([IGNORE_INDEX] * 10)
        images = torch.zeros(1, 3, self.image_size, self.image_size)
        time_ids = torch.tensor([0])

        return input_ids, labels, images, time_ids, self.task


def objectnav_collate_fn(batch, tokenizer):
    """ObjectNav专用的batch整理函数"""
    # 过滤掉空样本
    valid_batch = [item for item in batch if item is not None]

    if len(valid_batch) == 0:
        # 如果没有有效样本，返回空的batch
        return {
            'images': torch.zeros(1, 1, 3, 224, 224),
            'time_ids': torch.zeros(1, 1, dtype=torch.long),
            'attention_mask': torch.zeros(1, 1, dtype=torch.bool),
            'input_ids': torch.zeros(1, 1, dtype=torch.long),
            'labels': torch.zeros(1, 1, dtype=torch.long),
            'task_type': ['objectnav']
        }

    input_ids_batch, labels_batch, image_batch, time_ids_batch, task_type_batch = zip(*valid_batch)

    input_ids_batch = pad_sequence(input_ids_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_batch = pad_sequence(labels_batch, batch_first=True, padding_value=IGNORE_INDEX)

    # 限制最大长度
    max_len = min(tokenizer.model_max_length, input_ids_batch.size(1))
    input_ids_batch = input_ids_batch[:, :max_len]
    labels_batch = labels_batch[:, :max_len]

    attention_mask = input_ids_batch.ne(tokenizer.pad_token_id)

    img_lens = np.array([i.size(0) for i in image_batch])

    if time_ids_batch[0] is not None:
        time_ids_batch = pad_sequence(time_ids_batch, batch_first=True, padding_value=-1)
    else:
        time_ids_batch = None

    image_batch = pad_tensors(image_batch, img_lens)

    return {
        'images': image_batch,
        'time_ids': time_ids_batch,
        'attention_mask': attention_mask,
        'input_ids': input_ids_batch,
        'labels': labels_batch,
        'task_type': task_type_batch
    }