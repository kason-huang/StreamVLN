#!/usr/bin/env python3
"""
StreamVLN联合训练阶段输入和标签处理分析
"""

def analyze_co_training_data_flow():
    """分析联合训练的数据流"""

    print("=== StreamVLN联合训练阶段输入和标签处理详解 ===\n")

    print("## 1. 联合训练数据集结构")
    print("联合训练包含4个不同类型的数据集：")

    datasets = {
        "VLN导航任务": {
            "task_id": 0,
            "dataset_class": "VLNActionDataset",
            "data_format": "轨迹数据 (视频+动作序列)",
            "sample_count": "数万个轨迹样本",
            "input_type": "多帧RGB图像 + 动作序列",
            "target": "预测导航动作序列"
        },
        "视频QA任务": {
            "task_id": 1,
            "dataset_class": "LazySupervisedDataset",
            "data_format": "LLaVA-Video-178K",
            "sample_count": "约14万个视频QA样本",
            "input_type": "视频帧 + 问答对话",
            "target": "回答视频相关问题"
        },
        "3D扫描QA": {
            "task_id": 2,
            "dataset_class": "LazySupervisedDataset",
            "data_format": "ScanNet扫描数据",
            "sample_count": "数万个3D场景QA样本",
            "input_type": "RGB-D图像序列 + 问答对话",
            "target": "回答3D场景相关问题"
        },
        "图文配对": {
            "task_id": 3,
            "dataset_class": "LazyMMC4Dataset",
            "data_format": "MMC4大规模图文数据",
            "sample_count": "约1400个样本",
            "input_type": "网页图像 + 文本描述",
            "target": "理解图文对应关系"
        }
    }

    for name, info in datasets.items():
        print(f"\n### {name}")
        print(f"- 任务ID: {info['task_id']}")
        print(f"- 数据集类: {info['dataset_class']}")
        print(f"- 数据格式: {info['data_format']}")
        print(f"- 样本数量: {info['sample_count']}")
        print(f"- 输入类型: {info['input_type']}")
        print(f"- 目标任务: {info['target']}")

    print("\n## 2. 数据集混合策略")
    print("### CombineDataset实现:")
    print("```python")
    print("class CombineDataset(Dataset):")
    print("    def __init__(self, datasets: List[Dataset]):")
    print("        self.datasets = datasets")
    print("        self.lengths = [len(dataset) for dataset in datasets]")
    print("        self.cum_lengths = np.cumsum(self.lengths)")
    print("    ")
    print("    def __len__(self):")
    print("        return self.cum_lengths[-1]  # 总样本数")
    print("    ")
    print("    def __getitem__(self, i):")
    print("        # 根据索引找到对应的数据集")
    print("        for idx, cum_len in enumerate(self.cum_lengths):")
    print("            if i < cum_len:")
    print("                return self.datasets[idx][i - cum_len + self.lengths[idx]]")
    print("```")

    print("\n### 混合特点:")
    print("- **顺序混合**: 按数据集顺序拼接，不是随机采样")
    print("- **索引映射**: 全局索引映射到具体数据集和局部索引")
    print("- **任务标识**: 每个样本携带task_id标识任务类型")
    print("- **统一接口**: 所有数据集实现相同的__getitem__接口")

    print("\n## 3. 多任务输入组织")
    print("### 不同任务类型的输入结构:")

    input_examples = {
        "VLN任务": {
            "input_ids": "token化的导航指令和对话",
            "images": "[batch, frames, 3, H, W] 多帧导航观测",
            "depths": "[batch, frames, H, W] 深度信息",
            "poses": "[batch, frames, 4, 4] 相机位姿",
            "intrinsics": "[batch, frames, 4, 4] 相机内参",
            "time_ids": "[batch, frames] 时间步标识",
            "task_type": "[batch] 全为0"
        },
        "视频QA": {
            "input_ids": "token化的视频问答对话",
            "images": "[batch, 1, 3, H, W] 单帧或多帧视频",
            "depths": "None (无深度信息)",
            "poses": "None (无位姿信息)",
            "intrinsics": "None (无内参信息)",
            "time_ids": "None",
            "task_type": "[batch] 全为1"
        },
        "3D扫描QA": {
            "input_ids": "token化的3D场景问答对话",
            "images": "[batch, frames, 3, H, W] 3D扫描图像序列",
            "depths": "None",
            "poses": "None",
            "intrinsics": "None",
            "time_ids": "None",
            "task_type": "[batch] 全为2"
        },
        "图文配对": {
            "input_ids": "token化的图文描述对话",
            "images": "[batch, 1, 3, H, W] 网页图像",
            "depths": "None",
            "poses": "None",
            "intrinsics": "None",
            "time_ids": "None",
            "task_type": "[batch] 全为3"
        }
    }

    for task, example in input_examples.items():
        print(f"\n#### {task}:")
        for key, value in example.items():
            print(f"  - {key}: {value}")

    print("\n## 4. 统一Collate函数处理")
    print("### VLNCollate函数:")
    print("```python")
    print("def collate_fn(batch, tokenizer):")
    print("    # 解包批次数据")
    print("    input_ids_batch, labels_batch, image_batch, time_ids_batch, task_type_batch = zip(*batch)")
    print("    ")
    print("    # 序列填充")
    print("    input_ids_batch = pad_sequence(input_ids_batch, batch_first=True, ")
    print("                             padding_value=tokenizer.pad_token_id)")
    print("    labels_batch = pad_sequence(labels_batch, batch_first=True, ")
    print("                           padding_value=IGNORE_INDEX)")
    print("    ")
    print("    # 注意力掩码")
    print("    attention_mask = input_ids_batch.ne(tokenizer.pad_token_id)")
    print("    ")
    print("    # 图像批次处理")
    print("    img_lens = np.array([i.size(0) for i in image_batch])")
    print("    image_batch = pad_tensors(image_batch, img_lens)")
    print("    ")
    print("    return {")
    print("        'input_ids': input_ids_batch,")
    print("        'labels': labels_batch,")
    print("        'attention_mask': attention_mask,")
    print("        'images': image_batch,")
    print("        'task_type': task_type_batch  # 关键：任务类型标识")
    print("    }")
    print("```")

    print("\n## 5. 标签处理和损失计算")
    print("### 任务特定的标签掩码:")

    label_examples = {
        "VLN任务": {
            "对话示例": "Human: 你可以看到<image>。 Assistant: ↑↑→",
            "标签处理": "[IGNORE, IGNORE, ..., ↑, ↑, →, IGNORE]",
            "损失计算": "只计算动作token (↑↑→) 的损失"
        },
        "视频QA": {
            "对话示例": "Human: 视频中发生了什么？<image> Assistant: 一个人在走路",
            "标签处理": "[IGNORE, IGNORE, ..., 一, 个, 人, 在, 走, 路, IGNORE]",
            "损失计算": "只计算回答部分 (一个人在走路) 的损失"
        },
        "3D扫描QA": {
            "对话示例": "Human: 桌子在哪里？<image> Assistant: 在房间中央",
            "标签处理": "[IGNORE, IGNORE, ..., 在, 房, 间, 中, 央, IGNORE]",
            "损失计算": "只计算回答部分 (在房间中央) 的损失"
        },
        "图文配对": {
            "对话示例": "Human: 描述图片<image> Assistant: 一只猫在睡觉",
            "标签处理": "[IGNORE, IGNORE, ..., 一, 只, 猫, 在, 睡, 觉, IGNORE]",
            "损失计算": "只计算描述部分 (一只猫在睡觉) 的损失"
        }
    }

    for task, example in label_examples.items():
        print(f"\n#### {task}:")
        for key, value in example.items():
            print(f"  - {key}: {value}")

    print("\n## 6. 模型输入处理")
    print("### 模型前向传播:")
    print("```python")
    print("def forward(self, input_ids, images, task_type, ...):")
    print("    # 1. 多模态输入处理")
    print("    inputs_embeds = self.prepare_inputs_labels_for_multimodal(")
    print("        input_ids, images, task_type=task_type)")
    print("    ")
    print("    # 2. 根据任务类型调整处理")
    print("    if task_type == 0:  # VLN任务")
    print("        # 处理深度、位姿、内参等几何信息")
    print("        # 使用SlowFast机制")
    print("    else:  # QA等其他任务")
    print("        # 标准视觉-语言处理")
    print("    ")
    print("    # 3. 统一的模型前向传播")
    print("    outputs = self.model(")
    print("        input_ids=input_ids,")
    print("        inputs_embeds=inputs_embeds,")
    print("        task_type=task_type,")
    print("        ...)")
    print("    ")
    print("    return outputs")
    print("```")

    print("\n## 7. 训练优势")
    print("### 多任务学习的好处:")
    print("1. **知识迁移**: 不同任务间的视觉理解能力互相促进")
    print("2. **泛化能力**: 在多任务上训练的模型具有更好的泛化性")
    print("3. **数据效率**: 充分利用大规模多模态数据")
    print("4. **鲁棒性**: 避免在单一任务上过拟合")
    print("5. **能力互补**: QA任务提升语言理解，VLN任务提升空间推理")

    print("\n### 挑战和解决方案:")
    print("1. **任务平衡**: 通过数据采样比例平衡各任务的影响")
    print("2. **输入格式统一**: 统一的collate函数处理不同数据格式")
    print("3. **梯度冲突**: 任务标识帮助模型区分不同任务")
    print("4. **计算效率**: 批次处理不同任务，提高GPU利用率")

if __name__ == "__main__":
    analyze_co_training_data_flow()