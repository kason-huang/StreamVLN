import gzip
import json
import glob
import os

# 目录路径
data_dir = 'data/datasets/cloudrobo/val_train'
pattern = os.path.join(data_dir, '*.json.gz')
files = sorted(glob.glob(pattern))

print(f"找到 {len(files)} 个文件:")
for f in files:
    print(f"  - {os.path.basename(f)}")

all_episodes = []
instruction_vocab = None

# 遍历所有文件
for i, file_path in enumerate(files):
    with gzip.open(file_path, 'rt') as f:
        data = json.load(f)

    # 第一个文件保留 instruction_vocab
    if i == 0:
        instruction_vocab = data['instruction_vocab']

    # 收集 episodes
    all_episodes.extend(data['episodes'])
    print(f"{os.path.basename(file_path)}: {len(data['episodes'])} episodes")

# 限制最大 1800 条
if len(all_episodes) > 1800:
    all_episodes = all_episodes[:1800]

# 构建合并后的数据
merged_data = {
    'instruction_vocab': instruction_vocab,
    'episodes': all_episodes
}

# 保存
output_path = 'data/datasets/cloudrobo/val_train/val_train_merged.json'
with open(output_path, 'w') as f:
    json.dump(merged_data, f, indent=2)

print(f"\n合并完成:")
print(f"  总 episodes 数: {len(all_episodes)}")
print(f"  输出文件: {output_path}")
