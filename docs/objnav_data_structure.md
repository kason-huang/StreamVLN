# ObjectNav Data Structure Analysis

**Date:** 2026-02-26
**File Analyzed:** `data/trajectory_data_hm3d_format/objectnav/cloudrobo_v1_l3mvn/train/content/suzhou-room-shengwei-metacam-2025-07-09_01-13-22.json.gz`
**File Size:** 16.23 MB
**Episodes:** 1657

## Overview

The ObjectNav trajectory data is stored in gzipped JSON format with a hierarchical structure containing episode definitions, category mappings, and goal information.

## Top-Level Structure

```json
{
  "episodes": [/* 1657 episode objects */],
  "category_to_task_category_id": {/* mapping */},
  "category_to_scene_annotation_category_id": {/* mapping */},
  "goals_by_category": {/* mapping */}
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `episodes` | list | Array of episode objects (1657 total) |
| `category_to_task_category_id` | dict | Maps object categories to task category IDs |
| `category_to_scene_annotation_category_id` | dict | Maps categories to scene annotation IDs |
| `goals_by_category` | dict | Organizes available goals by object category |

## Episode Structure

Each episode in the `episodes` array contains **15 fields**:

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `episode_id` | str | Unique episode identifier | `"0"`, `"5324"` |
| `scene_id` | str | Path to scene GLB file | `"data/scene_datasets/cloudrobo_v1/train/..."` |
| `start_position` | list[float] | Initial agent position [x, y, z] | `[5.10, 0.12, 1.66]` |
| `start_rotation` | list[float] | Initial rotation quaternion [w, x, y, z] | `[0.0, 0.0, 0.0, 1.0]` |
| `object_category` | str | Target object category to find | `"sofa"`, `"chair"`, `"bed"` |
| `reference_replay` | list | Action sequence for reference trajectory | See below |
| `goals` | list | Goal object(s) with positions and view points | See below |

### Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `info` | dict | Episode metrics (`geodesic_distance`, `euclidean_distance`, `closest_goal_object_id`) |
| `is_thda` | bool | Whether episode uses THDA (Habitat) |
| `scene_dataset` | str | Scene dataset configuration path |
| `scene_dataset_config` | str | Scene config file path |
| `additional_obj_config_paths` | list | Additional object configuration paths |
| `attempts` | int | Number of attempts for this episode |
| `start_room` | null | Start room identifier (empty) |
| `shortest_paths` | null | Pre-computed shortest paths (empty) |

## Reference Replay Structure

The `reference_replay` field contains the action sequence demonstrating how to navigate to the goal. Each step in the replay contains:

```json
{
  "action": "MOVE_FORWARD",  // or "TURN_LEFT", "TURN_RIGHT", "LOOK_DOWN", "STOP"
  "agent_state": {
    "position": [x, y, z],    // Agent position after action
    "rotation": [w, x, y, z]  // Agent rotation as quaternion after action
  }
}
```

### Action Types

| Action | Description |
|--------|-------------|
| `MOVE_FORWARD` | Move agent forward by configured step size |
| `TURN_LEFT` | Rotate agent left by configured angle |
| `TURN_RIGHT` | Rotate agent right by configured angle |
| `LOOK_DOWN` | Look downward (camera pitch) |
| `STOP` | Stop and end episode |

**Note:** The first step typically has only `action: "STOP"` without agent_state.

### Example Replay Sequence

```
Step 0: {action: "STOP"}
Step 1: {action: "LOOK_DOWN", agent_state: {...}}
Step 2: {action: "MOVE_FORWARD", agent_state: {...}}
Step 3: {action: "TURN_RIGHT", agent_state: {...}}
...
Step 151: {action: "STOP", agent_state: {...}}
```

## Goals Structure

Each episode has one or more goal objects to find:

```json
{
  "position": [x, y, z],           // Goal position in scene
  "radius": null,                  // Goal acceptance radius (usually null)
  "object_id": 6,                  // Unique object ID in scene
  "object_name": "sofa_6",         // Object instance name
  "object_name_id": null,          // Object name ID
  "object_category": "sofa",       // Object category
  "room_id": null,                 // Room identifier
  "room_name": null,               // Room name
  "view_points": [                 // Valid viewing positions
    {
      "agent_state": {
        "position": [x, y, z],
        "rotation": [w, x, y, z]
      },
      "iou": 0.09                  // Intersection over Union metric
    }
  ]
}
```

## Info Field Structure

```json
{
  "geodesic_distance": 0.035,      // Shortest path distance to goal
  "euclidean_distance": 1.57,      // Straight-line distance to goal
  "closest_goal_object_id": 6      // ID of nearest goal object
}
```

## Data Format Notes

1. **Rotations** are stored as quaternions [w, x, y, z] in Habitat format
2. **Positions** are in 3D world coordinates [x, y, z] where y is height
3. **Episode IDs** are stored as strings, not integers
4. **Empty fields** (`start_room`, `shortest_paths`) are typically `null`
5. **Reference replays** can vary significantly in length (from ~50 to 200+ steps)

## Usage Example

```python
import gzip
import json

# Load the data
with gzip.open('path/to/file.json.gz', 'rt') as f:
    data = json.load(f)

# Access episodes
for episode in data['episodes']:
    episode_id = episode['episode_id']
    object_category = episode['object_category']
    scene_id = episode['scene_id']

    # Get reference trajectory
    for step in episode['reference_replay']:
        action = step['action']
        if 'agent_state' in step:
            position = step['agent_state']['position']
            rotation = step['agent_state']['rotation']

    # Get goal info
    for goal in episode['goals']:
        goal_position = goal['position']
        view_points = goal['view_points']
```

## Related Files

- Analysis script: `scripts/objnav_converters/analyze_json_structure.py`
- Converter script: `scripts/objnav_converters/objnav2streamvln.py`
- Config: `config/objnav_image.yaml`

## Category Mappings

The `category_to_task_category_id` and `category_to_scene_annotation_category_id` fields provide mappings between different category ID systems used in the Habitat simulator and dataset annotations.

Example categories include:
- `sofa`
- `chair`
- `bed`
- `toilet`
- `tv`
- And other household objects

---

# Reference Replay Field Analysis

**Purpose:** Analyze the `reference_replay` field for converting ObjectNav data to StreamVLN format (similar to `objnav2streamvln.py`)

## Detailed Statistics

Based on analysis of 1,657 episodes:

| Metric | Value |
|--------|-------|
| Total steps | 215,766 |
| Steps with agent_state | 214,109 (99.2%) |
| Min replay length | 6 steps |
| Max replay length | 501 steps |
| Mean replay length | 130.2 steps |
| Median replay length | 128 steps |

### Action Distribution

| Action | Count | Percentage |
|--------|-------|------------|
| `MOVE_FORWARD` | 88,831 | 41.2% |
| `TURN_RIGHT` | 50,742 | 23.5% |
| `TURN_LEFT` | 48,844 | 22.6% |
| `LOOK_DOWN` | 12,041 | 5.6% |
| `LOOK_UP` | 11,994 | 5.6% |
| `STOP` | 3,314 | 1.5% |

## Key Differences from EnvDrop Format

| Aspect | EnvDrop | ObjectNav |
|--------|---------|-----------|
| **Action storage** | Numeric IDs: `[-1, 1, 1, 2, 0]` | String names: `["STOP", "MOVE_FORWARD", ...]` |
| **Action source** | Shortest path follower | Pre-recorded demonstrations |
| **Agent state** | Not stored | Stored per step (position + rotation) |
| **First action** | Dummy action (usually -1) | `"STOP"` action without agent_state |
| **Images/Observations** | Pre-rendered RGB frames exist | **NOT stored** - must render via Habitat |
| **Episode ID** | Integer | String |

## Action Mapping for Converter

To convert ObjectNav actions to Habitat action IDs:

| ObjectNav Action | Habitat ID | Habitat Name |
|------------------|------------|--------------|
| `STOP` | 0 | `"stop"` |
| `MOVE_FORWARD` | 1 | `"move_forward"` |
| `TURN_LEFT` | 2 | `"turn_left"` |
| `TURN_RIGHT` | 3 | `"turn_right"` |
| `LOOK_UP` | 4 | `"look_up"` |
| `LOOK_DOWN` | 5 | `"look_down"` |

**Important:** The current `config/objnav_image.yaml` only defines 4 actions (move_forward, stop, turn_left, turn_right). You may need to add `look_up` and `look_down` actions to support the full ObjectNav replay.

## Critical Considerations for Converter

### 1. No Pre-rendered Images

ObjectNav format does **NOT** contain RGB/depth observations. You must use Habitat simulator to render observations on-the-fly while replaying actions.

### 2. First Step Handling

The first step (index 0) typically only contains `{"action": "STOP"}` without `agent_state`. You should skip this step when rendering.

### 3. Complete Action Set

ObjectNav uses 6 actions including `LOOK_UP` and `LOOK_DOWN` which may not be in the current Habitat config. Verify these are available or handle them appropriately.

### 4. Agent State Validation

Use stored `agent_state` for validation:

```python
stored_pos = step['agent_state']['position']
actual_pos = env.sim.get_agent(0).state.position
assert np.allclose(stored_pos, actual_pos, atol=0.01), "Position mismatch!"
```

## Converter Workflow

```python
# Action string to ID mapping
ACTION_MAP = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
    "LOOK_UP": 4,
    "LOOK_DOWN": 5
}

# Load ObjectNav data
with gzip.open('content.json.gz', 'rt') as f:
    data = json.load(f)

for episode in data['episodes']:
    # Setup Habitat with correct scene
    config.DATASET.SCENE = episode['scene_id']
    env = habitat.Env(config=config)
    observation = env.reset()

    # Replay actions (skip first "STOP" step)
    for step in episode['reference_replay'][1:]:
        # Capture observation
        rgb = observation['rgb']
        # Save: Image.fromarray(rgb).save(f"{rgb_dir}/{step_id:03d}.jpg")

        # Execute action
        action_id = ACTION_MAP[step['action']]
        observation = env.step(action_id)

        # Stop on final STOP
        if step['action'] == "STOP":
            break
```

## Comparison with objnav2streamvln.py

| Aspect | objnav2streamvln.py (EnvDrop) | ObjectNav Converter |
|--------|-------------------------------|---------------------|
| Data source | `annotations.json` | `content.json.gz` |
| Actions | Numeric IDs directly | String names → ID mapping |
| Episode ID | Integer (`annotation["id"]`) | String (`episode["episode_id"]`) |
| Replay loop | `while not env.episode_over` | Iterate through `reference_replay` |
| First action | Pop dummy: `actions[1:]` | Skip index 0: `replay[1:]` |
