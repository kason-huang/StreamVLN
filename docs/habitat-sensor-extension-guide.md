# Habitat Sensor 扩展指南：gs_rgb 实现分析

## 概述

本文档详细分析了 `GaussianSplattingRGBSensor` (gs_rgb) 在 Habitat 中的扩展实现流程，包括完整的代码架构、注册机制和数据流向。

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    1. YAML 配置层                             │
│  config/objnav_image.yaml                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ lab_sensors:                                         │    │
│  │   gaussian_splatting_rgb_sensor:                     │    │
│  │     reconstruction_scene_assets_dir: "reconstruction"│    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              2. Hydra Config 注册层                           │
│  streamvln/habitat_extensions/config.py                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ GaussianSplattingRGBSensorConfig                      │    │
│  │   type: "GaussianSplattingRGBSensor"                  │    │
│  │   width/height/hfov 等参数                            │    │
│  │   注册到 Hydra ConfigStore                            │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                3. Sensor 实现层                              │
│  streamvln/habitat_extensions/sensor.py                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ @registry.register_sensor                             │    │
│  │ class GaussianSplattingRGBSensor(Sensor):             │    │
│  │   cls_uuid: str = "gs_rgb"                           │    │
│  │   实现观测空间定义和核心渲染逻辑                       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              4. 使用层                                       │
│  scripts/objnav_converters/objnav2streamvln.py              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ register_hydra_plugin(HabitatConfigPlugin)            │    │
│  │ rgb = observation["gs_rgb"]                           │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 关键组件详解

### 1. Sensor 类定义

**文件**: `streamvln/habitat_extensions/sensor.py:31-32`

```python
@registry.register_sensor  # ← 装饰器注册到 Habitat
class GaussianSplattingRGBSensor(Sensor):
    cls_uuid: str = "gs_rgb"  # ← 唯一标识符，用于访问
```

**关键点**:
- `@registry.register_sensor`: 将 Sensor 注册到 Habitat 的全局注册表
- `cls_uuid = "gs_rgb"`: 这是你在 observation 字典中访问时用的 key
- 继承自 `habitat.core.simulator.Sensor`

---

### 2. 观测空间定义

**文件**: `streamvln/habitat_extensions/sensor.py:57-67`

```python
def _get_observation_space(self, *args, **kwargs) -> Space:
    height = getattr(self.config, 'height', 480)
    width = getattr(self.config, 'width', 640)
    return spaces.Box(
        low=0,
        high=255,
        shape=(height, width, 3),  # RGB 图像: H x W x C
        dtype=np.uint8,
    )
```

**作用**: 定义返回数据的形状和范围，Habitat 用此初始化环境。

---

### 3. 核心观测逻辑

**文件**: `streamvln/habitat_extensions/sensor.py:140-167`

```python
def get_observation(self, observations, episode: NavigationEpisode, *args, **kwargs):
    # 1. 获取场景名
    scene_id = episode.scene_id
    scene_name = os.path.basename(scene_id).replace('.glb', '')
    self._set_scene(scene_name)

    # 2. 获取 agent 状态（位置、旋转等）
    agent_state = self._sim.get_agent_state().sensor_states["rgb"]

    # 3. 构建 3DGS 相机视角
    vpc = construct_viewpoint_cam_from_agent_state(
        agent_state=agent_state,
        cam_setting=self.current_gs_sim.default_cam_setting,
        gs_habitat_transform=self.current_gs_sim.habitat_transform,
        gs_floor_transform=self.current_gs_sim.floor_transform
    )

    # 4. 渲染 3DGS 图像
    gs_image = self.current_gs_sim.get_observations(
        vpc,
        request_semantic=False,
        request_semantic_rgb=False,
        request_instance=False,
        request_instance_rgb=False,
    )

    return np.array(gs_image["rgb"])  # 返回 numpy 数组
```

**数据流**:
1. 从 episode 获取场景 ID
2. 加载对应的 3DGS 模型（带缓存机制）
3. 获取当前 agent 的相机状态
4. 将 Habitat 坐标系转换为 3DGS 坐标系
5. 渲染 RGB 图像并返回

---

### 4. 配置类定义

**文件**: `streamvln/habitat_extensions/config.py:26-33`

```python
@dataclass
class GaussianSplattingRGBSensorConfig(LabSensorConfig):
    type: str = "GaussianSplattingRGBSensor"  # ← 对应 Sensor 类名
    width: int = 640
    height: int = 480
    hfov: int = 79
    position: List[float] = field(default_factory=lambda: [0, 0.88, 0])
    reconstruction_scene_assets_dir: str = "data/reconstruction_scene_assets/"
```

**作用**: 定义 Sensor 的默认参数，可以被 YAML 配置覆盖。

---

### 5. Hydra 注册

**文件**: `streamvln/habitat_extensions/config.py:40-45`

```python
cs.store(
    package="habitat.task.lab_sensors.gaussian_splatting_rgb_sensor",
    group="habitat/task/lab_sensors",
    name="gaussian_splatting_rgb_sensor",  # ← YAML 中引用的名称
    node=GaussianSplattingRGBSensorConfig,
)
```

**关键点**:
- `name`: 在 YAML 配置中引用的名称
- `node`: 配置类
- `group`: 配置组的路径

---

### 6. YAML 配置

**文件**: `config/objnav_image.yaml:8-9, 37-39`

```yaml
# 在 defaults 中引入
defaults:
  - /habitat/task/lab_sensors:
    - gaussian_splatting_rgb_sensor

# 配置参数
habitat:
  task:
    lab_sensors:
      gaussian_splatting_rgb_sensor:
        reconstruction_scene_assets_dir: "reconstruction"
```

**作用**: 告诉 Habitat 加载这个 Sensor，并设置参数。

---

### 7. 插件注册

**文件**: `scripts/objnav_converters/objnav2streamvln.py:11, 18-19`

```python
from streamvln.habitat_extensions.config import HabitatConfigPlugin

from habitat.config.default_structured_configs import register_hydra_plugin
register_hydra_plugin(HabitatConfigPlugin)  # ← 让 Hydra 能找到你的配置
```

**作用**: 注册 Hydra 搜索路径插件，使 Habitat 能找到自定义配置。

---

## 三种名称的对应关系

| 位置 | 名称 | 用途 | 示例 |
|------|------|------|------|
| `cls_uuid` | `"gs_rgb"` | 代码中访问 observation | `observation["gs_rgb"]` |
| `cs.store(name=...)` | `"gaussian_splatting_rgb_sensor"` | YAML 中引用的名称 | `lab_sensors.gaussian_splatting_rgb_sensor` |
| `type` | `"GaussianSplattingRGBSensor"` | Sensor 类名（用于实例化） | 对应 Python 类名 |

---

## 完整数据流向

```
env.step(action)
    ↓
Habitat 遍历所有已注册的 sensors
    ↓
GaussianSplattingRGBSensor.get_observation()
    ↓
    ├─ 1. 获取场景名: episode.scene_id
    ├─ 2. 设置场景: _set_scene(scene_name)
    │   └─ _get_scene_gs_sim() [带缓存]
    │       └─ 加载 3DGS 模型文件
    ├─ 3. 获取 agent 状态
    ├─ 4. 构建相机视角
    └─ 5. 渲染图像: current_gs_sim.get_observations()
    ↓
返回 RGB numpy array (480, 640, 3)
    ↓
observation["gs_rgb"] = array
    ↓
你的脚本: rgb = observation["gs_rgb"]
```

---

## 3DGS 场景缓存机制

**文件**: `streamvln/habitat_extensions/sensor.py:70-135`

```python
def _set_scene(self, scene_name: str):
    """设置当前场景（使用缓存）"""
    if self.current_scene_name == scene_name:
        return  # 场景未改变，直接返回

    self.current_gs_sim = self._get_scene_gs_sim(scene_name)
    self.current_scene_name = scene_name

def _get_scene_gs_sim(self, scene_name: str):
    """获取指定场景的GS模拟器实例(带缓存)"""
    if scene_name in self.gs_sim_cache:
        return self.gs_sim_cache[scene_name]  # 从缓存读取

    # 加载新场景
    gs_sim = GsSimulator(
        splat_ply_path=...,
        habitat_transform_path=...,
        floor_transform_path=...,
        anno_res_dir=...
    )
    self.gs_sim_cache[scene_name] = gs_sim  # 存入缓存
    return gs_sim
```

**优化**: 同一场景的多个 episode 共享同一个 GsSimulator 实例，避免重复加载模型。

---

## 如何扩展新的 Sensor

### 步骤 1: 定义 Sensor 类

```python
# streamvln/habitat_extensions/sensor.py
from habitat.core.registry import registry
from habitat.core.simulator import Sensor

@registry.register_sensor
class MyCustomSensor(Sensor):
    cls_uuid: str = "my_custom_obs"  # 观测字典中的 key

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(...)

    def get_observation(self, observations, episode, *args, **kwargs):
        # 实现你的观测逻辑
        return np.array(...)
```

### 步骤 2: 定义配置类

```python
# streamvln/habitat_extensions/config.py
@dataclass
class MyCustomSensorConfig(LabSensorConfig):
    type: str = "MyCustomSensor"
    # 添加你的参数
```

### 步骤 3: 注册到 Hydra

```python
cs.store(
    name="my_custom_sensor",
    group="habitat/task/lab_sensors",
    node=MyCustomSensorConfig,
)
```

### 步骤 4: 在 YAML 中配置

```yaml
defaults:
  - /habitat/task/lab_sensors:
    - my_custom_sensor

habitat:
  task:
    lab_sensors:
      my_custom_sensor:
        param1: value1
```

### 步骤 5: 注册插件并使用

```python
from streamvln.habitat_extensions.config import HabitatConfigPlugin
from habitat.config.default_structured_configs import register_hydra_plugin

register_hydra_plugin(HabitatConfigPlugin)

# 使用
observation = env.reset()
custom_obs = observation["my_custom_obs"]
```

---

## 常见问题

### Q1: 为什么 observation 中的 key 是 `gs_rgb` 而不是 `gaussian_splatting_rgb_sensor`?

**A**: `cls_uuid` 定义了在 observation 字典中的 key，而 YAML 中的名称用于配置引用。这是 Habitat 的设计，将配置名称和运行时标识分离。

### Q2: 如何调试 Sensor 的输出?

**A**: 在 `get_observation` 中添加日志或断点：

```python
def get_observation(self, observations, episode, *args, **kwargs):
    result = super().get_observation(...)
    logger.info(f"gs_rgb shape: {result.shape}")
    return result
```

### Q3: Sensor 如何访问其他信息（如深度、语义）?

**A**: 通过 `observations` 参数或 `self._sim`:

```python
def get_observation(self, observations, episode, *args, **kwargs):
    depth = observations["depth"]  # 其他 sensor 的输出
    sim = self._sim  # 直接访问模拟器
    ...
```

---

## 相关文件

- **Sensor 实现**: `streamvln/habitat_extensions/sensor.py`
- **配置定义**: `streamvln/habitat_extensions/config.py`
- **YAML 配置**: `config/objnav_image.yaml`
- **使用示例**: `scripts/objnav_converters/objnav2streamvln.py`
- **3DGS 渲染器**: `panoptic_gs/gs_simulator/`

---

## 参考文档

- [Habitat Lab Sensors 官方文档](https://aihabitat.org/docs/habitat-lab/sensors)
- [Hydra Configuration 文档](https://hydra.cc/docs/tutorials/structured_config/intro)
- 本项目文档: `docs/objnav-3dgs-rendering-analysis.md`
