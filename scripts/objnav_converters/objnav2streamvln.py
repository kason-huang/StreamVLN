import os
import json
import argparse
import habitat
from tqdm import tqdm

from habitat_baselines.config.default import get_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
# 导入ObjectNav-V1 dataset
from streamvln.habitat_extensions import objectnav_dataset, sensor, config
# from streamvln.habitat_extensions import measures
from streamvln.habitat_extensions.config import HabitatConfigPlugin

# Parse command line arguments
parser = argparse.ArgumentParser(description='Convert ObjectNav to StreamVLN format')
parser.add_argument('--annot-path',
                    default='data/trajectory_data/objectnav/cloudrobo_v1_l3mvn/annotations.json',
                    help='Path to the annotations file')
parser.add_argument(
    'opts',
    default=None,
    nargs=argparse.REMAINDER,
    help='Modify config options from command line, e.g., habitat.dataset.data_path="custom/path"'
)
args = parser.parse_args()

# objnav_image.yaml是给gs场景准备的，如果是hm3d场景的话，直接用vln_r2r.yaml即可
CONFIG_PATH = "config/objnav_image.yaml"  # Path to the Habitat config file
ANNOT_PATH = args.annot_path  # Path to the annotations file from command line
BASE_DIR = os.path.dirname(ANNOT_PATH)  # Base directory for rgb frames
GOAL_RADIUS = 0.25  # Radius for the goal in meters. not used if get actions from annotations


from habitat.config.default_structured_configs import register_hydra_plugin
register_hydra_plugin(HabitatConfigPlugin)


config = get_config(CONFIG_PATH, args.opts)
env = habitat.Env(config=config)
annotations = json.load(open(ANNOT_PATH, "r"))

# Build episode lookup dictionary for fast access
episode_dict = {ep.episode_id: ep for ep in env.episodes}

for annotation in tqdm(annotations, desc="Processing annotations"):
    # annotation["id"] is int, but episode_dict keys are str, need to convert
    episode_id = str(annotation["id"])
    episode = episode_dict.get(episode_id)

    if episode is None:
        tqdm.write(f"Warning: Episode {episode_id} not found in env.episodes, skipping")
        continue

    env.current_episode = episode
    agent = ShortestPathFollower(sim=env.sim, goal_radius=GOAL_RADIUS, return_one_hot=False)
    observation = env.reset()
    reference_actions = annotation["actions"][1:] + [0]  # Pop the dummy action at the beginning and add stop action at the end
    step_id = 0  # Initialize step ID

    # Display current episode info
    episode_info = f"Episode {episode_id} ({annotation['video']})"

    # Check if rgb directory already exists and has correct number of images
    video_id = annotation["video"]
    rgb_dir = os.path.join(BASE_DIR, video_id, "rgb")
    expected_frames = len(reference_actions)

    if os.path.exists(rgb_dir):
        # Count existing jpg images in the directory
        existing_frames = len([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])

        if existing_frames == expected_frames:
            tqdm.write(f"Skipping {episode_info}: {expected_frames} frames already exist")
            continue
        else:
            tqdm.write(f"Re-processing {episode_info}: expected {expected_frames} frames, found {existing_frames}")

    tqdm.write(f"Processing {episode_info} with {len(reference_actions)} steps")

    while not env.episode_over:
        # rgb = observation["rgb"]  # Get the current rgb observation
        rgb = observation["gs_rgb"]  # Get the current rgb observation

        # TODO: Save RGB frame (customize as needed)
        # --------------------------------------------------------
        import PIL.Image as Image
        video_id = annotation["video"]  # Get the video ID from the annotation
        rgb_dir = os.path.join(BASE_DIR, video_id, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        frame_path = os.path.join(rgb_dir, f"{step_id:03d}.jpg")
        Image.fromarray(rgb).convert("RGB").save(frame_path)
        # tqdm.write(f"[{episode_info}] Saved frame {step_id}: {frame_path}")
        # --------------------------------------------------------

        action = reference_actions.pop(0)  # Get next action from our annotation
        observation = env.step(action)  # Update observation
        step_id += 1

env.close()