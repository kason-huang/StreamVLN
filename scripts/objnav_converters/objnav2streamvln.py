import os
import json
import habitat

from habitat_baselines.config.default import get_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
# 导入ObjectNav-V1 dataset
from streamvln.habitat_extensions import objectnav_dataset, sensor, config
# from streamvln.habitat_extensions import measures
from streamvln.habitat_extensions.config import HabitatConfigPlugin

CONFIG_PATH = "config/objnav_image.yaml"  # Path to the Habitat config file
ANNOT_PATH = "data/trajectory_data/objectnav/cloudrobo_v1_l3mvn/annotations.json"  # Path to the annotations file
GOAL_RADIUS = 0.25  # Radius for the goal in meters. not used if get actions from annotations


from habitat.config.default_structured_configs import register_hydra_plugin
register_hydra_plugin(HabitatConfigPlugin)


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
        # rgb = observation["rgb"]  # Get the current rgb observation
        rgb = observation["gs_rgb"]  # Get the current rgb observation

        # TODO: Save RGB frame (customize as needed)
        # --------------------------------------------------------
        import PIL.Image as Image
        video_id = annotation["video"]  # Get the video ID from the annotation
        rgb_dir = f"data/trajectory_data/objectnav/cloudrobo_v1_l3mvn/{video_id}/rgb"
        os.makedirs(rgb_dir, exist_ok=True)
        Image.fromarray(rgb).convert("RGB").save(os.path.join(rgb_dir, f"{step_id:03d}.jpg"))
        # --------------------------------------------------------

        action = reference_actions.pop(0)  # Get next action from our annotation
        observation = env.step(action)  # Update observation
        step_id += 1

env.close()