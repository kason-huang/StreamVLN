import os
import json
import sys
import habitat

from habitat_baselines.config.default import get_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from streamvln.habitat_extensions import measures


def save_rgb(annot_path):
    CONFIG_PATH = "config/vln_r2r.yaml"  # Path to the Habitat config file
    ANNOT_PATH = annot_path  # Path to the annotations file
    GOAL_RADIUS = 0.25  # Radius for the goal in meters. not used if get actions from annotations

    env = habitat.Env(config=get_config(CONFIG_PATH))
    annotations = json.load(open(ANNOT_PATH, "r"))

    for episode in env.episodes:
        env.current_episode = episode
        agent = ShortestPathFollower(sim=env.sim, goal_radius=GOAL_RADIUS, return_one_hot=False)
        observation = env.reset()

        annotation = next(annot for annot in annotations if annot["id"] == int(episode.episode_id))  # Get annotation for current episode
        reference_actions = annotation["actions"][1:] + [0]  # Pop the dummy action at the beginning and add stop action at the end
        step_id = 0  # Initialize step ID

        while not env.episode_over:
            rgb = observation["rgb"]  # Get the current rgb observation
            
            # TODO: Save RGB frame (customize as needed)
            # --------------------------------------------------------
            import PIL.Image as Image
            video_id = annotation["video"]  # Get the video ID from the annotation
            rgb_dir = f"data/trajectory_data/EnvDrop/{video_id}/rgb"
            os.makedirs(rgb_dir, exist_ok=True)
            Image.fromarray(rgb).convert("RGB").save(os.path.join(rgb_dir, f"{step_id:03d}.jpg"))
            # --------------------------------------------------------

            action = reference_actions.pop(0)  # Get next action from our annotation
            observation = env.step(action)  # Update observation
            step_id += 1

    env.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        annot_path = sys.argv[1]  # 获取第一个参数

    else:
       annot_path = "data/trajectory_data/EnvDrop/annotations.json" 
