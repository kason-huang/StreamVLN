import gzip
import json

# 读取 train.json.gz
with gzip.open('data/datasets/r2r/train/train.json.gz', 'rt') as f:
    train_data = json.load(f)

# 读取 annotations.json
with open('data/trajectory_data/R2R/annotations.json', 'r') as f:
    annotations_data = json.load(f)

# 创建 episode_id 到 episode 的映射字典
episode_map = {ep['episode_id']: ep for ep in train_data['episodes']}

# 按 annotations.json 的顺序提取 episodes，只保留前1800条
subset_episodes = []
missing_ids = []
for anno in annotations_data[:1800]:
    anno_id = anno['id']
    if anno_id in episode_map:
        subset_episodes.append(episode_map[anno_id])
    else:
        missing_ids.append(anno_id)

# 构建子集数据
subset_data = {
    'instruction_vocab': train_data['instruction_vocab'],
    'episodes': subset_episodes
}

# 输出统计信息
print(f"原始 train.json.gz episodes 长度: {len(train_data['episodes'])}")
print(f"annotations.json 长度: {len(annotations_data)}")
print(f"提取的子集 episodes 长度: {len(subset_episodes)}")
print(f"缺失的 ID 数量: {len(missing_ids)}")
if missing_ids:
    print(f"缺失的 ID (前10个): {missing_ids[:10]}")

# 保存子集到新文件
output_path = 'data/datasets/r2r/train/train_subset.json'
with open(output_path, 'w') as f:
    json.dump(subset_data, f, indent=2)
print(f"\n子集已保存到: {output_path}")