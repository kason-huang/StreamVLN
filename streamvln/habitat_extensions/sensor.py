from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from gym import Space, spaces
from habitat.core.registry import registry
from habitat.core.simulator import RGBSensor, Sensor, SensorTypes, Simulator, Observations
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.run import execute_exp

import os 
import sys

import panoptic_gs
# panopatic_gs里面的路径要以panopatics_gs为基础
# sys.path.insert(0, os.path.dirname(os.path.abspath(panoptic_gs.__file__)))
sys.path.insert(0, panoptic_gs.__path__[0])

from panoptic_gs.gs_simulator import GsSimulator
from panoptic_gs.gs_simulator.sim_camera.habitat_cam import (
    habitat_sensor_to_CamSettings,
    construct_viewpoint_cam_from_agent_state
)
from habitat.tasks.nav.nav import NavigationEpisode

if TYPE_CHECKING:
    from omegaconf import DictConfig


import time
@registry.register_sensor
class GaussianSplattingRGBSensor(Sensor):
    cls_uuid: str = "gs_rgb"

    def __init__(self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any):
        super().__init__(config=config)

        # 存储基础配置路径，而不是具体的场景路径
        self._sim = sim
        self.gs_sim = None
        self.cam_setting = None
        self.current_gs_sim = None
        self.current_scene_name = None
        self.gs_sim_cache = {}

        # 初始化相机设置（使用配置中的参数）
        width = getattr(config, 'width', 640)
        height = getattr(config, 'height', 480)
        hfov = getattr(config, 'hfov', 79)
        self.cam_setting = habitat_sensor_to_CamSettings(width, height, hfov)

        self.scene_base_path = getattr(config, 'reconstruction_scene_assets_dir', 'data/scene_datasets/reconstruction')

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args, **kwargs) -> Space:
        # 返回观测空间定义
        # 假设是640x480的RGB图像
        height = getattr(self.config, 'height', 480)
        width = getattr(self.config, 'width', 640)
        return spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 3),
            dtype=np.uint8,
        )

    
    def _set_scene(self, scene_name: str):
        """设置当前场景（使用缓存）"""
        if self.current_scene_name == scene_name:
            return  # 场景未改变
        
        self.current_gs_sim = self._get_scene_gs_sim(scene_name)
        self.current_scene_name = scene_name
   
        
    def _get_scene_gs_sim(self, scene_name: str):
        """获取指定场景的GS模拟器实例(带缓存)"""
        if scene_name in self.gs_sim_cache:
            # 场景已在缓存中
            return self.gs_sim_cache[scene_name]

        print("Loading new 3dgs scene:", scene_name)

        splat_ply_path = os.path.join(
            # self.scene_base_path,
            "data/scene_datasets/reconstruction",
            scene_name,
            "semantic",
            "splat.semantic.ply"
        )
        if not os.path.exists(splat_ply_path):
            raise FileNotFoundError(f"PLY file not found: {splat_ply_path}")

        habitat_transform_path = os.path.join(
            # self.scene_base_path,
            "data/scene_datasets/reconstruction",
            scene_name,
            "anno_res",
            "to_habitat.txt"
        )
        if not os.path.exists(habitat_transform_path):
            raise FileNotFoundError(f"To habitat file not found: {habitat_transform_path}")

        floor_transform_path = os.path.join(
            # self.scene_base_path,
            "data/scene_datasets/reconstruction",
            scene_name,
            "anno_res",
            "floor_transform.txt"
        )
        if not os.path.exists(floor_transform_path):
            raise FileNotFoundError(f"Floor transform file not found: {floor_transform_path}")

        anno_res_dir = os.path.join(
            # self.scene_base_path,
            "data/scene_datasets/reconstruction",
            scene_name,
            "anno_res"
        )
        if not os.path.exists(anno_res_dir):
            raise FileNotFoundError(f"Anno file not found: {anno_res_dir}")

        gs_sim = GsSimulator(
            splat_ply_path=splat_ply_path,
            habitat_transform_path=habitat_transform_path,
            floor_transform_path=floor_transform_path,
            anno_res_dir=anno_res_dir
        )

        gs_sim.set_default_cam_setting(self.cam_setting)
        self.gs_sim_cache[scene_name] = gs_sim
        return gs_sim
        
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(
        self, 
        observations, 
        episode: NavigationEpisode, 
        *args: Any, 
        **kwargs: Any
    ):
        scene_id  = episode.scene_id
        scene_name = os.path.basename(scene_id).replace('.glb', '')
        self._set_scene(scene_name)
        
        agent_state = self._sim.get_agent_state().sensor_states["rgb"]
        vpc = construct_viewpoint_cam_from_agent_state(
            agent_state = agent_state,
            cam_setting = self.current_gs_sim.default_cam_setting,
            gs_habitat_transform = self.current_gs_sim.habitat_transform,
            gs_floor_transform = self.current_gs_sim.floor_transform
        )

        gs_image = self.current_gs_sim.get_observations(
                    vpc,
                    request_semantic=False,
                    request_semantic_rgb=False,
                    request_instance=False,
                    request_instance_rgb=False,
                )

        return np.array(gs_image["rgb"])