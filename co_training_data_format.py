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

    print("\n## 2. 各个Dataset类的返回格式")
    print("### VLNActionDataset (task_id=0) 返回格式:")
    print("```python")
    print("# VLNActionDataset.__getitem__() 返回")
    print("return (")
    print("    input_ids: torch.tensor([1, 2, 3, ..., 100]),      # 形状: [seq_len]")
    print("    labels: torch.tensor([-100, -100, 100, 101, 102]), # 形状: [seq_len]")
    print("    images: torch.tensor([[[frame1], [frame2], ...]]),  # 形状: [num_frames, 3, H, W]")
    print("    time_ids: torch.tensor([0, 4, 8, 12, 16, 20]),    # 形状: [num_frames]")
    print("    task_type: 0                                        # VLN任务标识")
    print(")")
    print("```")

    print("\n### LazySupervisedDataset (task_id=1,2) 返回格式:")
    print("```python")
    print("# LazySupervisedDataset.__getitem__() 返回")
    print("return {")
    print("    'input_ids': torch.tensor([1, 2, 3, ..., 200]),     # 形状: [seq_len]")
    print("    'labels': torch.tensor([-100, -100, 150, 151, ...]), # 形状: [seq_len]")
    print("    'image': [(tensor_image, image_size, 'video')],      # 单个视频帧")
    print("    # 注意: time_ids 为 None")
    print("}")
    print("```")

    print("\n### LazyMMC4Dataset (task_id=3) 返回格式:")
    print("```python")
    print("# LazyMMC4Dataset.__getitem__() 返回")
    print("return {")
    print("    'input_ids': torch.tensor([1, 2, 3, ..., 180]),     # 形状: [seq_len]")
    print("    'labels': torch.tensor([-100, -100, 160, 161, ...]), # 形状: [seq_len]")
    print("    'image': [(tensor_image, image_size, 'image')],      # 单个图像")
    print("    # 注意: time_ids 为 None")
    print("}")
    print("```")

    print("\n## 3. Collate函数的统一处理")
    print("### collate_fn的输入示例:")
    print("```python")
    print("# batch = [sample1, sample2, sample3, sample4]")
    print("# 其中sample1来自VLN, sample2来自QA, sample3来自ScanQA, sample4来自MMC4")
    print("batch = [")
    print("    (input_ids_vln, labels_vln, images_vln, time_ids_vln, 0),     # VLN样本")
    print("    (input_ids_qa, labels_qa, images_qa, None, 1),                # QA样本")
    print("    (input_ids_scanqa, labels_scanqa, images_scanqa, None, 2),     # ScanQA样本")
    print("    (input_ids_mmc4, labels_mmc4, images_mmc4, None, 3)             # MMC4样本")
    print("]")
    print("```")

    print("\n### collate_fn的输出格式:")
    print("```python")
    print("return {")
    print("    'input_ids': torch.tensor([[vln_seq], [qa_seq], [scanqa_seq], [mmc4_seq]]),  # [4, max_seq_len]")
    print("    'labels': torch.tensor([[vln_labels], [qa_labels], [scanqa_labels], [mmc4_labels]]), # [4, max_seq_len]")
    print("    'attention_mask': torch.tensor([[vln_mask], [qa_mask], [scanqa_mask], [mmc4_mask]]), # [4, max_seq_len]")
    print("    'images': torch.tensor([padded_images]),                                      # [4, max_frames, 3, H, W]")
    print("    'time_ids': torch.tensor([[vln_time_ids], [-1], [-1], [-1]]),                # [4, max_frames]")
    print("    'task_type': torch.tensor([0, 1, 2, 3])                                      # [4]")
    print("}")
    print("```")

    print("\n## 4. 具体的训练数据示例")
    print("### 实际的训练批次数据:")

    # 模拟一个真实的训练批次
    sample_batch = {
        'input_ids': [
            # VLN样本 (task_id=0)
            [1, 2, 3, 4, 5, 32000, 10, 11, 12, 13, 14, 100, 101, 102, 103, 2],
            # QA样本 (task_id=1)
            [1, 2, 3, 4, 5, 32000, 20, 21, 22, 23, 150, 151, 152, 153, 2],
            # ScanQA样本 (task_id=2)
            [1, 2, 3, 4, 5, 32000, 30, 31, 32, 33, 200, 201, 202, 2],
            # MMC4样本 (task_id=3)
            [1, 2, 3, 4, 5, 32000, 40, 41, 42, 43, 44, 45, 300, 301, 2]
        ],
        'labels': [
            # VLN标签: 只计算助手回复的损失
            [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 100, 101, 102, 103, -100],
            # QA标签
            [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 150, 151, 152, 153, -100],
            # ScanQA标签
            [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 200, 201, 202, -100],
            # MMC4标签
            [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 300, 301, -100]
        ],
        'attention_mask': [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ],
        'images': [
            # VLN: 8帧图像
            torch.randn(8, 3, 384, 384),
            # QA: 1帧图像
            torch.randn(1, 3, 384, 384),
            # ScanQA: 5帧图像
            torch.randn(5, 3, 384, 384),
            # MMC4: 1帧图像
            torch.randn(1, 3, 384, 384)
        ],
        'time_ids': [
            # VLN: 时间步信息
            torch.tensor([0, 4, 8, 12, 16, 20, 24, 28]),
            # 其他任务: 填充-1
            torch.tensor([-1]),
            torch.tensor([-1]),
            torch.tensor([-1])
        ],
        'task_type': [0, 1, 2, 3]
    }

    print("### 数据解读:")
    print(f"- 批次大小: {len(sample_batch['input_ids'])}")
    print(f"- 输入序列形状: {[len(seq) for seq in sample_batch['input_ids']]}")
    print(f"- 图像序列形状: {[img.shape if hasattr(img, 'shape') else 'scalar' for img in sample_batch['images']]}")
    print(f"- 任务类型分布: {sample_batch['task_type']}")
    print(f"- 有效token数量: {[sum(1 for t in labels if t != -100) for labels in sample_batch['labels']]}")

    print("\n## 5. Dataset类的职责分工")
    print("### 各Dataset类的具体职责:")

    datasets_info = {
        "CombineDataset": {
            "职责": "统一的数据集管理器",
            "功能": [
                "管理多个子数据集的索引映射",
                "通过cum_lengths实现全局索引到局部索引的转换",
                "提供统一的__getitem__接口",
                "负责数据的顺序混合"
            ],
            "特点": "不直接处理数据，只负责索引转发"
        },
        "VLNActionDataset": {
            "职责": "VLN导航任务数据管理",
            "功能": [
                "处理轨迹视频和动作序列",
                "生成多轮导航对话",
                "管理历史帧和当前帧",
                "处理深度、位姿、内参等几何信息"
            ],
            "特点": "返回5元组：(input_ids, labels, images, time_ids, task_type)"
        },
        "LazySupervisedDataset": {
            "职责": "通用视觉-问答数据管理",
            "功能": [
                "处理多种格式的视觉数据（图像/视频）",
                "支持多种采样策略",
                "处理视频QA和3D扫描QA数据",
                "管理大规模数据集的懒加载"
            ],
            "特点": "返回字典格式，包含input_ids, labels, image等字段"
        },
        "LazyMMC4Dataset": {
            "职责": "图文配对数据管理",
            "功能": [
                "处理网页图像和文本描述",
                "支持大规模图文数据集",
                "管理图文对应关系"
            ],
            "特点": "专注于图文理解任务"
        }
    }

    for name, info in datasets_info.items():
        print(f"\n#### {name}:")
        print(f"- 职责: {info['职责']}")
        print("- 功能:")
        for func in info['功能']:
            print(f"  * {func}")
        print(f"- 特点: {info['特点']}")

    print("\n## 6. 数据流转的完整过程")
    print("### 从Dataset到模型的完整流程:")
    print("```python")
    print("# 1. 数据加载")
    print("dataloader = DataLoader(")
    print("    dataset=CombineDataset([vln_dataset, qa_dataset, scanqa_dataset, mmc4_dataset]),")
    print("    batch_size=8,")
    print("    collate_fn=collate_fn,")
    print("    shuffle=True")
    print(")")
    print("    ")
    print("# 2. 批次获取")
    print("for batch in dataloader:")
    print("    # CombineDataset.__getitem__ 调用链:")
    print("    # → 找到对应子数据集")
    print("    # → 调用子数据集.__getitem__")
    print("    # → 返回统一格式的数据")
    print("    ")
    print("    # 3. 数据处理")
    print("    outputs = model(")
    print("        input_ids=batch['input_ids'],")
    print("        images=batch['images'],")
    print("        task_type=batch['task_type'],")
    print("        labels=batch['labels']")
    print("    )")
    print("    ")
    print("    # 4. 损失计算")
    print("    loss = outputs.loss")
    print("    loss.backward()")
    print("    optimizer.step()")
    print("```")

    print("\n### 关键技术点:")
    print("1. **索引映射**: cum_lengths实现高效的全局到局部索引转换")
    print("2. **格式统一**: collate_fn将不同格式的数据统一为批次张量")
    print("3. **任务标识**: task_type让模型能够区分不同任务")
    print("4. **懒加载**: 大规模数据集支持按需加载")
    print("5. **内存优化**: 不同任务的数据格式差异得到妥善处理")

if __name__ == "__main__":
    analyze_co_training_data_format()