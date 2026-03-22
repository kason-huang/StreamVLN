from dataclasses import dataclass, field
from typing import Optional, List

from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    HabitatConfig,
    LabSensorConfig,
    MeasurementConfig,
    SimulatorConfig,
)

from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesBaseConfig,
    HabitatBaselinesRLConfig,
    PolicyConfig,
    RLConfig,
    EvalConfig,
)
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.plugins.search_path_plugin import SearchPathPlugin

cs = ConfigStore.instance()


@dataclass
class GaussianSplattingRGBSensorConfig(LabSensorConfig):
    type: str = "GaussianSplattingRGBSensor"
    width: int = 640
    height: int = 480
    hfov: int = 79
    position: List[float] = field(default_factory=lambda: [0, 0.88, 0])
    reconstruction_scene_assets_dir: str = "data/scene_datasets/reconstruction"


# -----------------------------------------------------------------------------
# Register configs in the Hydra ConfigStore
# -----------------------------------------------------------------------------

cs.store(
    package="habitat.task.lab_sensors.gaussian_splatting_rgb_sensor",
    group="habitat/task/lab_sensors",
    name="gaussian_splatting_rgb_sensor",
    node=GaussianSplattingRGBSensorConfig,
)

class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="habitat",
            path="pkg://config/tasks/",
        )
        search_path.append(
            provider="habitat_baselines",
            path="pkg://habitat_baselines/config/",
        )