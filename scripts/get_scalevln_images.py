import os
import json
import shutil
import habitat
from tqdm import tqdm
import PIL.Image as Image

from habitat_baselines.config.default import get_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

CONFIG_PATH = "config/vln_r2r_scalevln.yaml"
ANNOT_PATH = "/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/ScaleVLN/annotations.json"
GOAL_RADIUS = 0.25

env = habitat.Env(config=get_config(CONFIG_PATH))
annotations = json.load(open(ANNOT_PATH, "r"))

for episode in tqdm(env.episodes, desc="Processing episodes"):
    annotation = next(annot for annot in annotations if annot["id"] == int(episode.episode_id))
    video_id = annotation["video"]
    rgb_dir = f"/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/ScaleVLN/{video_id}/rgb"

    expected_frame_count = len(annotation["actions"])  # 每一步动作 → 一帧图像
    if os.path.exists(rgb_dir):
        existing_frames = [f for f in os.listdir(rgb_dir) if f.endswith(".jpg")]
        if len(existing_frames) == expected_frame_count:
            print(f"Skipping episode {episode.episode_id} as it already has {len(existing_frames)} frames.")
            continue  # ✅ 跳过已完整生成的 episode
        else:
            # ⚠️ 不完整，清空目录重新生成
            shutil.rmtree(rgb_dir)

    env.current_episode = episode
    agent = ShortestPathFollower(sim=env.sim, goal_radius=GOAL_RADIUS, return_one_hot=False)
    observation = env.reset()

    reference_actions = annotation["actions"][1:] + [0]  # Skip dummy, add STOP
    step_id = 0

    os.makedirs(rgb_dir, exist_ok=True)

    while not env.episode_over:
        rgb = observation["rgb"]
        Image.fromarray(rgb).convert("RGB").save(os.path.join(rgb_dir, f"{step_id:03d}.jpg"))

        action = reference_actions.pop(0)
        observation = env.step(action)
        step_id += 1

env.close()
