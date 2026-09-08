from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Location:
    """仿真世界中的 XYZ 坐标。"""

    X: float
    Y: float
    Z: float


@dataclass
class Rotation:
    """由滚转角、偏航角和俯仰角组成的旋转。"""

    roll: float
    yaw: float
    pitch: float


class TongSimInterface(ABC):
    """面向客户端的 TongSim 感知与动作接口。

    动作接口接收的物体 ID 均为服务端分配的映射 ID，
    本接口不会向客户端暴露 TongSim SDK 的原始 ID。
    """

    @abstractmethod
    def acquire_first_person_perception(
        self,
        character_id,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        """获取第一人称组合图和映射后的物体信息。

        ``width`` 和 ``height`` 表示最终组合图的宽高，必须同时传入。
        默认不缩放：宽度为摄像机宽度的 2 倍，高度等于摄像机高度。
        按 ``spawn_character`` 默认摄像机 720×1000 计算，默认组合图为 1440×1000。
        ``VLMAgent`` 默认使用 1280×720 摄像机，对应组合图为 2560×720。
        """
        raise NotImplementedError()

    @abstractmethod
    def spawn_character(
        self,
        loc,
        rot,
        desired_name,
        fov: float = 120.0,
        width: int = 720,
        height: int = 1000,
        camera_name_suffix: str | None = None,
        camera_stream_id: str | None = None,
        spawn_extra_camera: bool = False,
    ) -> str:
        raise NotImplementedError()

    @abstractmethod
    def destory_character(self, character_id=None):
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def look_at_location(
        self, character_id, target_location, is_cancel: bool = False, execute_immediately: bool = False
    ):
        raise NotImplementedError()

    @abstractmethod
    def look_at_object(self, character_id, object_id: str, is_cancel: bool = False):
        raise NotImplementedError()

    @abstractmethod
    def point_at_object(self, character_id, object_id: str, is_cancel: bool = False, which_hand: int = 0):
        raise NotImplementedError()

    @abstractmethod
    def move_and_take_object(
        self,
        character_id,
        object_id: str,
        which_hand: int = 0,
        movable_object_ids: list[str] | None = None,
    ):
        raise NotImplementedError()

    @abstractmethod
    def move_and_take_puzzle_piece(self, character_id, piece_object_id: str, which_hand: int = 0):
        raise NotImplementedError()

    @abstractmethod
    def move_to_location(self, character_id, target_location, stop_distance: float = 0.5):
        raise NotImplementedError()

    @abstractmethod
    def move_forward(self, character_id, distance: float):
        raise NotImplementedError()

    @abstractmethod
    def put_down_sth(
        self,
        character_id,
        target_location,
        target_rotation: Rotation | None = None,
        auto_rotate: bool = False,
        force_locate: bool = False,
    ):
        """将手中物体放到指定位置；未指定旋转时由服务端自动调整朝向。"""
        raise NotImplementedError()

    @abstractmethod
    def turn_in_degree(self, character_id, degree):
        raise NotImplementedError()

    @abstractmethod
    def move_to_object(self, character_id, object_id: str):
        raise NotImplementedError()

    @abstractmethod
    def move_to_npc(self, character_id, name: str):
        raise NotImplementedError()

    @abstractmethod
    def pour_water(self, character_id, object_id: str, location, which_hand: int = 0):
        raise NotImplementedError()

    @abstractmethod
    def sit_down_to_object(self, character_id, object_id: str):
        raise NotImplementedError()

    @abstractmethod
    def slice_food(self, character_id, object_id: str, location):
        raise NotImplementedError()

    @abstractmethod
    def wash_hands(self, character_id, faucet_object_id: str):
        raise NotImplementedError()

    @abstractmethod
    def wash_object_in_hand(self, character_id, faucet_object_id: str):
        raise NotImplementedError()

    @abstractmethod
    def mop_floor(self, character_id, dirt_id: str):
        raise NotImplementedError()

    @abstractmethod
    def rest(self, character_id):
        raise NotImplementedError()

    @abstractmethod
    def speak_to_npc(self, character_id, target: str, content: str):
        raise NotImplementedError()

    @abstractmethod
    def move_and_put_down_object_in_container(self, character_id, which_hand: int = 0):
        raise NotImplementedError()

    @abstractmethod
    def move_and_put_down(
        self,
        character_id,
        move_target_location,
        put_target_location,
        which_hand: int = 0,
        put_rotation: Rotation | None = None,
    ):
        raise NotImplementedError()

    @abstractmethod
    def has_object_in_hand(self, character_id) -> tuple[bool, int | None]:
        """返回手中是否存在物体以及被占用的手部索引。"""
        raise NotImplementedError()
