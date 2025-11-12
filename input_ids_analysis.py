#!/usr/bin/env python3
"""
详细分析三个Dataset类中input_ids的组织方式
"""

def analyze_input_ids_organization():
    """分析三个Dataset类中input_ids的具体组织方式"""

    print("=== 三个Dataset类中input_ids组织方式详解 ===\n")

    print("## 1. VLNActionDataset中input_ids的组织方式")
    print("### VLN任务的多轮对话结构:")
    print("VLNActionDataset生成的是**多轮导航对话**，模拟真实的流式导航过程:")
    print()

    print("### 对话构建过程:")
    print("```python")
    print("# 1. 基础对话模板")
    print("prompt = 'You are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence...'")
    print("answer = ''")
    print("conversations = [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': answer}]")
    print("    ")
    print("# 2. 添加历史记忆 (如果不是第一步)")
    print("if start_idx != 0:")
    print("    sources[0]['value'] += f' These are your historical observations: {MEMORY_TOKEN}.'")
    print("    ")
    print("# 3. 填充具体指令")
    print("sources[0]['value'] = sources[0]['value'].replace('<instruction>.', 'go to the kitchen')")
    print("    ")
    print("# 4. 生成多轮对话 (每轮预测4个动作)")
    print("interleave_sources = prepare_conversation(sources, list(actions))")
    print("```")

    print("\n### 多轮对话的具体示例:")
    print("假设动作序列: [1, 1, 3, 1, 2, 1, 0, 1, 3, 2, 0] (↑↑→↑←↑STOP↑→←STOP)")
    print("num_future_steps = 4, 预测每轮对话的步数")
    print()

    print("#### 第1轮对话:")
    print("```python")
    print("Human: 'You are an autonomous navigation assistant. Your task is to go to the kitchen... you can see <image>.'")
    print("GPT: '↑↑→↑'  # 预测前4个动作")
    print("```")

    print("#### 第2轮对话:")
    print("```python")
    print("Human: 'in front of you is <image>.'")
    print("GPT: '←↑'    # 预测接下来的2个动作")
    print("```")

    print("#### 第3轮对话:")
    print("```python")
    print("Human: 'there is <image>.'")
    print("GPT: '↑→←'  # 预测最后3个动作")
    print("```")

    print("\n### preprocess_qwen处理后的input_ids结构:")
    print("```python")
    print("# 经过preprocess_qwen处理后，多轮对话被合并为一个长序列")
    print("input_ids = [")
    print("    # 系统消息")
    print("    [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 3619, 351, 1139, 1141, 3126, 1023],  # '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>'")
    print("    # 第1轮用户输入 (包含<image> token)")
    print("    [151644, 872, 198, 264, 1095, 3745, 319, 310, 1423, 3225, 264, 6385, 310, 278, 32000, 13],  # '<|im_start|>user\\n...you can see <image>.<|im_end|>'")
    print("    # 第1轮助手回复 (动作序列)")
    print("    [151644, 77091, 198, 3125, 3135, 3136, 1023],  # '<|im_start|>assistant\\n↑↑→↑<|im_end|>'")
    print("    # 第2轮用户输入")
    print("    [151644, 872, 198, 264, 1095, 3745, 310, 278, 32000, 13],  # '<|im_start|>user\\nin front of you is <image>.<|im_end|>'")
    print("    # 第2轮助手回复")
    print("    [151644, 77091, 198, 3125, 3136, 1023],  # '<|im_start|>assistant\\n←↑<|im_end|>'")
    print("    # 第3轮用户输入")
    print("    [151644, 872, 198, 264, 1095, 3745, 310, 278, 32000, 13],  # '<|im_start|>user\\nthere is <image>.<|im_end|>'")
    print("    # 第3轮助手回复")
    print("    [151644, 77091, 198, 3125, 3135, 3136, 1023]   # '<|im_start|>assistant\\n↑→←<|im_end|>'")
    print("]")
    print("")

    print("### 对应的labels结构 (损失计算):")
    print("```python")
    print("labels = [")
    print("    # 系统消息 - 不计损失")
    print("    [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],")
    print("    # 第1轮用户输入 - 不计损失")
    print("    [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],")
    print("    # 第1轮助手回复 - **计算损失**")
    print("    [-100, -100, -100, 3125, 3135, 3136, -100],  # ↑↑→↑")
    print("    # 第2轮用户输入 - 不计损失")
    print("    [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],")
    print("    # 第2轮助手回复 - **计算损失**")
    print("    [-100, -100, -100, 3125, 3136, -100],  # ←↑")
    print("    # 第3轮用户输入 - 不计损失")
    print("    [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],")
    print("    # 第3轮助手回复 - **计算损失**")
    print("    [-100, -100, -100, 3125, 3135, 3136, -100]   # ↑→←")
    print("]")
    print("```")

    print("\n## 2. LazySupervisedDataset中input_ids的组织方式")
    print("### QA任务的对话结构:")
    print("LazySupervisedDataset处理的是**单轮问答对话**，结构相对简单:")
    print()

    print("### 数据加载和处理过程:")
    print("```python")
    print("# 1. 从JSON文件加载数据")
    print("sample = {")
    print("    'conversations': [")
    print("        {'from': 'human', 'value': 'What is happening in the video? <image>'},")
    print("        {'from': 'gpt', 'value': 'A person is walking in the park.'}")
    print("    ],")
    print("    'video': 'path/to/video.mp4'")
    print("}")
    print("    ")
    print("# 2. 视频帧处理")
    print("video = process_video_with_decord(video_file)  # 采样10帧")
    print("image = processor.preprocess(video, return_tensors='pt')['pixel_values']")
    print("    ")
    print("# 3. 对话预处理")
    print("sources = preprocess_multimodal(copy.deepcopy([sample['conversations']]), data_args)")
    print("data_dict = preprocess(sources, tokenizer, has_image=True)")
    print("```")

    print("\n### 预处理后的input_ids结构:")
    print("```python")
    print("input_ids = [")
    print("    # 系统消息")
    print("    [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 3619, 351, 1139, 1141, 3126, 1023],")
    print("    # 用户问题 (包含<image> token)")
    print("    [151644, 872, 198, 264, 1095, 3745, 310, 278, 32000, 13, 8426, 278, 310, 337, 3748, 311, 278, 368, 349, 314, 279, 277, 310, 384, 298, 313, 281, 285, 13],  # 'What is happening...<image>?<|im_end|>'")
    print("    # 助手回答")
    print("    [151644, 77091, 198, 264, 1095, 3745, 310, 278, 263, 8192, 29914, 29879, 29335, 310, 278, 311, 279, 263, 10603, 29973, 29300, 28423, 13]  # 'A person is walking...<|im_end|>'")
    print("]")
    print("")

    print("### 对应的labels结构:")
    print("```python")
    print("labels = [")
    print("    # 系统消息 - 不计损失")
    print("    [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],")
    print("    # 用户问题 - 不计损失")
    print("    [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],")
    print("    # 助手回答 - **计算损失**")
    print("    [-100, -100, -100, 264, 1095, 3745, 310, 278, 263, 8192, 29914, 29879, 29335, 310, 278, 311, 279, 263, 10603, 29973, 29300, 28423, -100]  # 'A person is walking...'")
    print("]")
    print("```")

    print("\n## 3. LazyMMC4Dataset中input_ids的组织方式")
    print("### 图文配对任务的对话结构:")
    print("LazyMMC4Dataset处理的是**网页图文配对**，主要是图像描述任务:")
    print()

    print("### 数据格式示例:")
    print("```python")
    print("sample = {")
    print("    'conversations': [")
    print("        {'from': 'human', 'value': 'Describe this image: <image>'},")
    print("        {'from': 'gpt', 'value': 'This is a photo of a cat sitting on a windowsill, looking outside.'}")
    print("    ],")
    print("    'image': 'path/to/image.jpg'")
    print("}")
    print("```")

    print("\n### 预处理后的input_ids结构:")
    print("```python")
    print("input_ids = [")
    print("    # 系统消息")
    print("    [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 3619, 351, 1139, 1141, 3126, 1023],")
    print("    # 用户请求")
    print("    [151644, 872, 198, 264, 1095, 3745, 310, 278, 32000, 13, 4154, 29914, 29879, 29871, 278, 312, 279, 277, 310, 13],  # 'Describe this image: <image><|im_end|>'")
    print("    # 图像描述")
    print("    [151644, 77091, 198, 264, 1095, 3745, 310, 278, 263, 6279, 29962, 29929, 29345, 310, 278, 311, 279, 263, 198, 279, 263, 6325, 29896, 29345, 310, 278, 417, 311, 279, 28423, 29933, 279, 263, 10603, 29973, 311, 279, 277, 310, 278, 263, 5087, 310, 279, 277, 13]  # 'This is a photo of a cat...'")
    print("]")
    print("")

    print("### 对应的labels结构:")
    print("```python")
    print("labels = [")
    print("    # 系统消息 - 不计损失")
    print("    [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],")
    print("    # 用户请求 - 不计损失")
    print("    [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],")
    print("    # 图像描述 - **计算损失**")
    print("    [-100, -100, -100, 264, 1095, 3745, 310, 278, 263, 6279, 29962, 29929, 29345, 310, 278, 311, 279, 263, 198, 279, 263, 6325, 29896, 29345, 310, 278, 417, 311, 279, 28423, 29933, 279, 263, 10603, 29973, 311, 279, 277, 310, 278, 263, 5087, 310, 279, 277, -100]  # 完整的描述文本")
    print("]")
    print("```")

    print("\n## 4. 三个Dataset中input_ids组织方式的对比")
    print("### 关键差异总结:")

    comparison = {
        "对话结构": {
            "VLNActionDataset": "多轮对话 (3-4轮交互式导航)",
            "LazySupervisedDataset": "单轮问答 (问题-回答对)",
            "LazyMMC4Dataset": "单轮描述 (图像-描述对)"
        },
        "内容特点": {
            "VLNActionDataset": "动作序列 (↑↑→↑←↑)",
            "LazySupervisedDataset": "自然语言回答 (句子/段落)",
            "LazyMMC4Dataset": "图像描述 (详细描述文本)"
        },
        "序列长度": {
            "VLNActionDataset": "中等长度 (多轮叠加)",
            "LazySupervisedDataset": "中等长度 (问题+回答)",
            "LazyMMC4Dataset": "可变长度 (描述长短不一)"
        },
        "任务目标": {
            "VLNActionDataset": "预测具体导航动作",
            "LazySupervisedDataset": "回答视觉理解问题",
            "LazyMMC4Dataset": "生成图像描述"
        },
        "特殊token": {
            "VLNActionDataset": "IMAGE_TOKEN + MEMORY_TOKEN",
            "LazySupervisedDataset": "IMAGE_TOKEN",
            "LazyMMC4Dataset": "IMAGE_TOKEN"
        }
    }

    print("| 对比维度 | VLNActionDataset | LazySupervisedDataset | LazyMMC4Dataset |")
    print("|---------|------------------|----------------------|------------------|")
    for aspect, values in comparison.items():
        print(f"| {aspect} | {values['VLNActionDataset']} | {values['LazySupervisedDataset']} | {values['LazyMMC4Dataset']} |")

    print("\n### labels与input_ids的对应关系:")
    print("所有三个Dataset都遵循**相同的损失计算策略**:")
    print("1. **系统消息**: labels = [-100, -100, ...] (IGNORE_INDEX)")
    print("2. **用户输入**: labels = [-100, -100, ...] (IGNORE_INDEX)")
    print("3. **特殊token**: labels = [-100] (IMAGE_TOKEN, MEMORY_TOKEN等)")
    print("4. **助手回复**: labels = [真实token_ids] (计算CrossEntropyLoss)")

    print("\n### 输入输出映射关系:")
    mapping_examples = {
        "VLN导航": {
            "输入": "导航指令 + 历史观测 + 当前图像",
            "输出": "动作序列 (↑↑→↑←↑)",
            "映射": "视觉理解 → 动作决策"
        },
        "视频QA": {
            "输入": "视频帧 + 问题",
            "输出": "自然语言回答",
            "映射": "视觉理解 → 文本生成"
        },
        "图文配对": {
            "输入": "网页图像",
            "输出": "详细图像描述",
            "映射": "视觉理解 → 文本描述"
        }
    }

    for task, mapping in mapping_examples.items():
        print(f"#### {task}:")
        print(f"- 输入: {mapping['输入']}")
        print(f"- 输出: {mapping['输出']}")
        print(f"- 映射: {mapping['映射']}")

    print("\n### 技术实现细节:")
    print("1. **统一接口**: 三个Dataset都返回(input_ids, labels)的基本结构")
    print("2. **格式差异**: VLN返回tuple，其他返回dict，但都统一为相同的input_ids格式")
    print("3. **预处理统一**: 都使用preprocess_qwen进行token化和标签生成")
    print("4. **损失计算统一**: 都使用IGNORE_INDEX掩码用户输入和系统消息")
    print("5. **特殊token统一**: IMAGE_TOKEN_INDEX=32000 在所有Dataset中保持一致")

    print("\n### 训练效果差异:")
    print("- **VLNActionDataset**: 学习动作决策能力，支持连续的导航行为")
    print("- **LazySupervisedDataset**: 提升视觉理解和文本生成能力")
    print("- **LazyMMC4Dataset**: 增强图像理解和描述能力")
    print("- **联合训练**: 三种能力互补，提升整体多模态理解水平")

if __name__ == "__main__":
    analyze_input_ids_organization()