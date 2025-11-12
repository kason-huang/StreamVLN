#!/usr/bin/env python3
"""
详细分析StreamVLN联合训练阶段的数据格式
"""

def analyze_co_training_data_format():
    """分析联合训练的具体数据格式和Dataset管理"""

    print("=== StreamVLN联合训练阶段数据格式详解 ===\n")

    print("## 1. CombineDataset的cum_lengths机制解析")
    print("### cum_lengths的作用:")
    print("```python")
    print("class CombineDataset(Dataset):")
    print("    def __init__(self, datasets: List[Dataset]):")
    print("        self.datasets = datasets  # [VLN_Dataset, QA_Dataset, ScanQA_Dataset, MMC4_Dataset]")
    print("        self.lengths = [len(dataset) for dataset in datasets]  # [10000, 50000, 20000, 1400]")
    print("        self.cum_lengths = np.cumsum(self.lengths)             # [10000, 60000, 80000, 81400]")
    print("    ")
    print("    def __len__(self):")
    print("        return self.cum_lengths[-1]  # 返回总样本数: 81400")
    print("    ")
    print("    def __getitem__(self, i):")
    print("        for idx, cum_len in enumerate(self.cum_lengths):")
    print("            if i < cum_len:")
    print("                # 找到对应的数据集")
    print("                local_idx = i - cum_len + self.lengths[idx]")
    print("                return self.datasets[idx][local_idx]")
    print("```")

    print("\n### 索引映射示例:")
    print("- 全局索引 0-9999 → VLN数据集 (task_id=0)")
    print("- 全局索引 10000-59999 → QA数据集 (task_id=1)")
    print("- 全局索引 60000-79999 → ScanQA数据集 (task_id=2)")
    print("- 全局索引 80000-81399 → MMC4数据集 (task_id=3)")

    print("\n### cum_lengths的数学原理:")
    print("```python")
    print("# 假设有4个数据集，长度分别为 [10000, 50000, 20000, 1400]")
    print("lengths = [10000, 50000, 20000, 1400]")
    print("cum_lengths = np.cumsum(lengths)  # [10000, 60000, 80000, 81400]")
    print("    ")
    print("# 索引映射算法:")
    print("def map_global_to_local(global_idx):")
    print("    for dataset_idx, cum_len in enumerate(cum_lengths):")
    print("        if global_idx < cum_len:")
    print("            local_idx = global_idx - (cum_len - lengths[dataset_idx])")
    print("            return dataset_idx, local_idx")
    print("    ")
    print("# 示例:")
    print("# global_idx=15000 → 找到 dataset_idx=1 (因为 15000 < 60000)")
    print("# local_idx = 15000 - (60000-50000) = 5000")
    print("# 结果: QA数据集的第5000个样本")
    print("```")

    print("\n## 2. 各个Dataset类的返回格式对比")
    print("### 不同Dataset类的返回格式差异:")

    dataset_formats = {
        "VLNActionDataset": {
            "task_id": 0,
            "返回格式": "tuple",
            "返回内容": "(input_ids, labels, images, time_ids, task_type)",
            "特点": [
                "返回5元组结构",
                "包含多帧导航观测图像",
                "包含时间步信息",
                "包含深度、位姿、内参等VLN特有信息"
            ]
        },
        "LazySupervisedDataset": {
            "task_id": [1, 2],
            "返回格式": "dict",
            "返回内容": "{'input_ids': ..., 'labels': ..., 'image': ...}",
            "特点": [
                "返回字典格式",
                "处理视频QA和3D扫描QA",
                "支持懒加载",
                "统一处理不同类型的视觉数据"
            ]
        },
        "LazyMMC4Dataset": {
            "task_id": 3,
            "返回格式": "dict",
            "返回内容": "{'input_ids': ..., 'labels': ..., 'image': ...}",
            "特点": [
                "专注于图文配对任务",
                "处理大规模网页数据",
                "返回单张图像"
            ]
        }
    }

    for name, info in dataset_formats.items():
        print(f"\n#### {name}:")
        print(f"- 任务ID: {info['task_id']}")
        print(f"- 返回格式: {info['返回格式']}")
        print(f"- 返回内容: {info['返回内容']}")
        print("- 特点:")
        for feature in info['特点']:
            print(f"  * {feature}")

    print("\n## 3. Collate函数的统一处理")
    print("### 不同返回格式的统一处理:")
    print("```python")
    print("def collate_fn(batch, tokenizer):")
    print("    # 批次数据可能是混合格式")
    print("    # VLN: (input_ids, labels, images, time_ids, task_type)")
    print("    # QA:   {'input_ids': ..., 'labels': ..., 'image': ..., task_type}")
    print("    ")
    print("    # 统一解包处理")
    print("    if isinstance(batch[0], tuple):  # VLN格式")
    print("        input_ids_batch, labels_batch, image_batch, time_ids_batch, task_type_batch = zip(*batch)")
    print("    else:  # QA格式 (字典)")
    print("        input_ids_batch = [item['input_ids'] for item in batch]")
    print("        labels_batch = [item['labels'] for item in batch]")
    print("        image_batch = [item['image'] for item in batch]")
    print("        time_ids_batch = [item.get('time_ids', None) for item in batch]")
    print("        task_type_batch = [item.get('task_type', 0) for item in batch]")
    print("    ")
    print("    # 统一的批次处理")
    print("    return {")
    print("        'input_ids': input_ids_batch,")
    print("        'labels': labels_batch,")
    print("        'images': image_batch,")
    print("        'time_ids': time_ids_batch,")
    print("        'task_type': task_type_batch")
    print("    }")
    print("```")

    print("\n## 4. 实际训练数据的示例")
    print("### CombineDataset管理的4种数据类型:")

    data_examples = {
        "VLN导航任务": {
            "全局索引范围": "0-9999",
            "数据来源": "VLNActionDataset",
            "原始数据": "轨迹观测: [rgb_frames, depth_frames, poses, actions]",
            "处理后数据": "多轮对话: [观测图像] → [动作序列]",
            "数据维度": "input_ids: [seq_len], images: [8, 3, 384, 384], time_ids: [8]"
        },
        "视频QA任务": {
            "全局索引范围": "10000-59999",
            "数据来源": "LazySupervisedDataset",
            "原始数据": "视频文件: [video_frames, qa_pairs]",
            "处理后数据": "问答对话: [视频帧] → [问题回答]",
            "数据维度": "input_ids: [seq_len], images: [1, 3, 384, 384], time_ids: None"
        },
        "3D扫描QA": {
            "全局索引范围": "60000-79999",
            "数据来源": "LazySupervisedDataset",
            "原始数据": "3D扫描: [posed_images, scene_qa]",
            "处理后数据": "场景问答: [扫描图像] → [场景理解]",
            "数据维度": "input_ids: [seq_len], images: [5, 3, 384, 384], time_ids: None"
        },
        "图文配对": {
            "全局索引范围": "80000-81399",
            "数据来源": "LazyMMC4Dataset",
            "原始数据": "网页数据: [web_image, text_description]",
            "处理后数据": "图文理解: [网页图像] → [文本描述]",
            "数据维度": "input_ids: [seq_len], images: [1, 3, 384, 384], time_ids: None"
        }
    }

    for task, info in data_examples.items():
        print(f"\n#### {task}:")
        print(f"- 全局索引范围: {info['全局索引范围']}")
        print(f"- 数据来源: {info['数据来源']}")
        print(f"- 原始数据: {info['原始数据']}")
        print(f"- 处理后数据: {info['处理后数据']}")
        print(f"- 数据维度: {info['数据维度']}")

    print("\n## 5. Dataset类的具体职责")
    print("### 各Dataset类的详细职责分工:")

    responsibilities = {
        "CombineDataset": {
            "核心职责": "统一数据集管理器",
            "具体功能": [
                "维护多个子数据集的索引映射关系",
                "通过cum_lengths实现O(1)时间复杂度的索引查找",
                "提供统一的__getitem__接口给DataLoader",
                "负责不同数据集的顺序混合",
                "处理数据集的动态组合"
            ],
            "不负责": "具体的数据加载和预处理",
            "技术优势": "内存高效，无需复制数据，只需索引转发"
        },
        "VLNActionDataset": {
            "核心职责": "VLN专用数据处理",
            "具体功能": [
                "加载轨迹视频文件和多帧图像",
                "处理导航动作序列和指令文本",
                "生成多轮导航对话数据",
                "管理历史帧和当前帧的采样",
                "处理深度图、相机位姿、内参等几何信息",
                "实现SlowFast机制的数据准备"
            ],
            "特殊处理": "时间相关的序列数据，支持流式导航",
            "返回格式": "5元组 (input_ids, labels, images, time_ids, task_type)"
        },
        "LazySupervisedDataset": {
            "核心职责": "通用多模态数据处理",
            "具体功能": [
                "支持多种数据格式 (JSON/JSONL/YAML)",
                "实现多种采样策略 (first/end/random)",
                "处理视频帧的时间采样和空间预处理",
                "支持大规模数据集的懒加载",
                "统一处理视频QA和3D扫描QA数据",
                "管理数据源标识和子集采样"
            ],
            "特殊处理": "支持复杂的数据配置和采样策略",
            "返回格式": "字典格式 {'input_ids': ..., 'labels': ..., 'image': ...}"
        },
        "LazyMMC4Dataset": {
            "核心职责": "图文配对数据专用处理",
            "具体功能": [
                "处理网页图像和文本描述的对应关系",
                "管理大规模图文数据集的索引",
                "支持图像的多种预处理方式",
                "处理文本描述的token化",
                "优化大规模网页数据的加载效率"
            ],
            "特殊处理": "专注于图文理解任务的特殊需求",
            "返回格式": "字典格式，专门为图文配对优化"
        }
    }

    for name, info in responsibilities.items():
        print(f"\n### {name}:")
        print(f"**核心职责**: {info['核心职责']}")
        print("**具体功能**:")
        for func in info['具体功能']:
            print(f"- {func}")
        print("**不负责**: {info.get('不负责', 'N/A')}")
        print("**特殊处理**: {info['特殊处理']}")
        print("**返回格式**: {info['返回格式']}")

    print("\n## 6. 训练数据流转的完整过程")
    print("### 从CombineDataset到模型的数据流转:")
    print("```python")
    print("# 1. 初始化阶段")
    print("datasets = [")
    print("    VLNActionDataset(tokenizer, vln_args, task_id=0),      # 10000个样本")
    print("    LazySupervisedDataset(tokenizer, qa_args, task_id=1),  # 50000个样本")
    print("    LazySupervisedDataset(tokenizer, scanqa_args, task_id=2), # 20000个样本")
    print("    LazyMMC4Dataset(tokenizer, mmc4_args, task_id=3)        # 1400个样本")
    print("]")
    print("combined_dataset = CombineDataset(datasets)  # 总计81400个样本")
    print("    ")
    print("# 2. DataLoader初始化")
    print("dataloader = DataLoader(")
    print("    dataset=combined_dataset,")
    print("    batch_size=8,")
    print("    collate_fn=collate_fn,")
    print("    shuffle=True")
    print(")")
    print("    ")
    print("# 3. 训练循环中的数据获取")
    print("for epoch in range(num_epochs):")
    print("    for batch_idx, batch in enumerate(dataloader):")
    print("        # CombineDataset.__getitem__(random_idx) 被调用")
    print("        # 内部流程:")
    print("        # 1. 根据random_idx找到对应的数据集")
    print("        # 2. 计算局部索引")
    print("        # 3. 调用子数据集.__getitem__(local_idx)")
    print("        # 4. collate_fn统一处理批次数据")
    print("        ")
    print("        # 4. 模型训练")
    print("        outputs = model(")
    print("            input_ids=batch['input_ids'],")
    print("            images=batch['images'],")
    print("            task_type=batch['task_type'],")
    print("            labels=batch['labels']")
    print("        )")
    print("        loss = outputs.loss")
    print("        loss.backward()")
    print("        optimizer.step()")
    print("```")

    print("\n### 关键技术实现细节:")
    print("1. **索引映射算法**: cum_lengths + lengths实现O(1)查找")
    print("2. **格式统一**: collate_fn处理tuple和dict两种返回格式")
    print("3. **内存优化**: 懒加载避免大内存占用")
    print("4. **任务区分**: task_type让模型能区分不同任务")
    print("5. **数据混合**: 顺序拼接保证数据分布的稳定性")

    print("\n### 性能优化策略:")
    print("- **缓存机制**: 索引映射表预计算，避免重复计算")
    print("- **懒加载**: 大数据集按需加载，减少内存压力")
    print("- **批次对齐**: 不同长度的序列智能填充")
    print("- **多线程**: DataLoader支持多进程数据加载")
    print("- **预取机制**: 提前加载下一批次数据")

if __name__ == "__main__":
    analyze_co_training_data_format()