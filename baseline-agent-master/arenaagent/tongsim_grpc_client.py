from __future__ import annotations

import contextlib
import threading
import uuid
from typing import Any

import grpc
from loguru import logger

from arenaagent.agent_base import pack_data_to_struct, parse_struct_to_data
from arenaagent.generated.tongsim.tongsim_service_pb2_grpc import TongSimServiceStub
from arenaagent.tongsim_interface import Rotation, TongSimInterface

_MAX_MSG_BYTES = 50 * 1024 * 1024
_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", _MAX_MSG_BYTES),
    ("grpc.max_receive_message_length", _MAX_MSG_BYTES),
]
_CLIENT_ID_METADATA_KEY = "x-tongsim-client-id"


class TongSimGrpcClient(TongSimInterface):
    """通过远程 TongSimService 代理调用的 TongSimInterface 实现。"""

    def __init__(self, endpoint: str = "127.0.0.1:50060", heartbeat_interval_secs: float = 2.0) -> None:
        self._channel = grpc.insecure_channel(endpoint, options=_GRPC_OPTIONS)
        self._stub = TongSimServiceStub(self._channel)
        self._endpoint = endpoint
        self._client_id = uuid.uuid4().hex
        self._metadata = ((_CLIENT_ID_METADATA_KEY, self._client_id),)
        self._heartbeat_interval_secs = max(float(heartbeat_interval_secs), 0.5)
        self._heartbeat_stop = threading.Event()
        self._heartbeat_supported = True
        self._heartbeat_compat_logged = False
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"tongsim-heartbeat-{self._client_id[:8]}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _call(self, method_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        rpc = getattr(self._stub, method_name)
        return parse_struct_to_data(rpc(pack_data_to_struct(payload), metadata=self._metadata))

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.is_set():
            try:
                self.heartbeat()
            except grpc.RpcError as exc:
                if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
                    self._heartbeat_supported = False
                    if not self._heartbeat_compat_logged:
                        logger.warning(
                            "TongSim server {} does not implement heartbeat RPC; disabling heartbeat for client {}. "
                            "Disconnect-triggered auto cleanup requires the upgraded tongsim-server.",
                            self._endpoint,
                            self._client_id,
                        )
                        self._heartbeat_compat_logged = True
                    return
                logger.debug("TongSim heartbeat failed for client {}: {}", self._client_id, exc)
            except Exception as exc:  # pragma: no cover - 运行时保护
                logger.debug("TongSim heartbeat failed for client {}: {}", self._client_id, exc)

            if self._heartbeat_stop.wait(self._heartbeat_interval_secs):
                return

    def heartbeat(self) -> None:
        if not self._heartbeat_supported:
            return
        self._call("heartbeat", {})

    # ------------------------------------------------------------------ #
    # 角色生命周期
    # ------------------------------------------------------------------ #

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
        result = self._call(
            "spawn_character",
            {
                "loc": list(loc),
                "rot": list(rot),
                "desired_name": desired_name,
                "fov": fov,
                "width": width,
                "height": height,
                "camera_name_suffix": camera_name_suffix,
                "camera_stream_id": camera_stream_id,
                "spawn_extra_camera": spawn_extra_camera,
            },
        )
        return result.get("character_id", "")

    def destory_character(self, character_id=None) -> dict:
        return self._call("destory_character", {"character_id": str(character_id) if character_id is not None else ""})

    def close(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=self._heartbeat_interval_secs + 1.0)

        with contextlib.suppress(Exception):
            self._call("close", {})
        self._channel.close()

    # ------------------------------------------------------------------ #
    # 感知
    # ------------------------------------------------------------------ #

    def acquire_first_person_perception(
        self,
        character_id,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        """获取感知结果，并可指定最终组合图的宽高。

        默认不缩放；按默认摄像机 720×1000 计算，组合图为 1440×1000。
        VLMAgent 默认使用 1280×720 摄像机，对应组合图为 2560×720。
        """
        payload: dict[str, Any] = {"character_id": str(character_id)}
        if width is not None:
            payload["width"] = width
        if height is not None:
            payload["height"] = height
        return self._call("acquire_first_person_perception", payload)

    def has_object_in_hand(self, character_id) -> tuple[bool, int | None]:
        result = self._call("has_object_in_hand", {"character_id": str(character_id)})
        has_object = bool(result.get("has_object", False))
        hand_idx = result.get("hand_idx")
        return has_object, int(hand_idx) if hand_idx is not None else None

    # ------------------------------------------------------------------ #
    # 视角控制
    # ------------------------------------------------------------------ #

    def look_at_location(
        self, character_id, target_location, is_cancel: bool = False, execute_immediately: bool = False
    ):
        return self._call(
            "look_at_location",
            {
                "character_id": str(character_id),
                "target_location": target_location,
                "is_cancel": is_cancel,
                "execute_immediately": execute_immediately,
            },
        )

    def look_at_object(self, character_id, object_id: str, is_cancel: bool = False):
        return self._call(
            "look_at_object",
            {
                "character_id": str(character_id),
                "object_id": object_id,
                "is_cancel": is_cancel,
            },
        )

    def point_at_object(self, character_id, object_id: str, is_cancel: bool = False, which_hand: int = 0):
        return self._call(
            "point_at_object",
            {
                "character_id": str(character_id),
                "object_id": object_id,
                "is_cancel": is_cancel,
                "which_hand": which_hand,
            },
        )

    # ------------------------------------------------------------------ #
    # 移动
    # ------------------------------------------------------------------ #

    def move_to_location(self, character_id, target_location, stop_distance: float = 0.5):
        return self._call(
            "move_to_location",
            {
                "character_id": str(character_id),
                "target_location": target_location,
                "stop_distance": stop_distance,
            },
        )

    def move_forward(self, character_id, distance: float):
        return self._call(
            "move_forward",
            {
                "character_id": str(character_id),
                "distance": float(distance),
            },
        )

    def move_to_object(self, character_id, object_id: str):
        return self._call(
            "move_to_object",
            {
                "character_id": str(character_id),
                "object_id": object_id,
            },
        )

    def move_to_npc(self, character_id, name: str):
        return self._call(
            "move_to_npc",
            {
                "character_id": str(character_id),
                "name": name,
            },
        )

    def move_and_take_object(
        self,
        character_id,
        object_id: str,
        which_hand: int = 0,
        movable_object_ids: list[str] | None = None,
    ):
        return self._call(
            "move_and_take_object",
            {
                "character_id": str(character_id),
                "object_id": object_id,
                "which_hand": which_hand,
                "movable_object_ids": movable_object_ids,
            },
        )

    def move_and_take_puzzle_piece(self, character_id, piece_object_id: str, which_hand: int = 0):
        return self._call(
            "move_and_take_puzzle_piece",
            {
                "character_id": str(character_id),
                "piece_object_id": piece_object_id,
                "which_hand": which_hand,
            },
        )

    def turn_in_degree(self, character_id, degree):
        return self._call(
            "turn_in_degree",
            {
                "character_id": str(character_id),
                "degree": float(degree),
            },
        )

    # ------------------------------------------------------------------ #
    # 物体操作
    # ------------------------------------------------------------------ #

    def put_down_sth(
        self,
        character_id,
        target_location,
        target_rotation: Rotation | None = None,
        auto_rotate: bool = False,
        force_locate: bool = False,
    ):
        """放下手中物体；具体使用哪只手由服务端自动判断。"""
        rot_dict = None
        if target_rotation is not None:
            rot_dict = {
                "roll": target_rotation.roll,
                "yaw": target_rotation.yaw,
                "pitch": target_rotation.pitch,
            }
        return self._call(
            "put_down_sth",
            {
                "character_id": str(character_id),
                "target_location": target_location,
                "target_rotation": rot_dict,
                "auto_rotate": auto_rotate,
                "force_locate": force_locate,
            },
        )

    def pour_water(self, character_id, object_id: str, location, which_hand: int = 0):
        return self._call(
            "pour_water",
            {
                "character_id": str(character_id),
                "object_id": object_id,
                "location": location,
                "which_hand": which_hand,
            },
        )

    def slice_food(self, character_id, object_id: str, location):
        return self._call(
            "slice_food",
            {
                "character_id": str(character_id),
                "object_id": object_id,
                "location": location,
            },
        )

    def wash_hands(self, character_id, faucet_object_id: str):
        return self._call(
            "wash_hands",
            {
                "character_id": str(character_id),
                "faucet_object_id": faucet_object_id,
            },
        )

    def wash_object_in_hand(self, character_id, faucet_object_id: str):
        return self._call(
            "wash_object_in_hand",
            {
                "character_id": str(character_id),
                "faucet_object_id": faucet_object_id,
            },
        )

    def move_and_put_down_object_in_container(self, character_id, which_hand: int = 0):
        return self._call(
            "move_and_put_down_object_in_container",
            {
                "character_id": str(character_id),
                "which_hand": which_hand,
            },
        )

    def move_and_put_down(
        self,
        character_id,
        move_target_location,
        put_target_location,
        which_hand: int = 0,
        put_rotation: Rotation | None = None,
    ):
        rot_dict = None
        if put_rotation is not None:
            rot_dict = {"roll": put_rotation.roll, "yaw": put_rotation.yaw, "pitch": put_rotation.pitch}
        return self._call(
            "move_and_put_down",
            {
                "character_id": str(character_id),
                "move_target_location": move_target_location,
                "put_target_location": put_target_location,
                "which_hand": which_hand,
                "put_rotation": rot_dict,
            },
        )

    # ------------------------------------------------------------------ #
    # 交互
    # ------------------------------------------------------------------ #
    def sit_down_to_object(self, character_id, object_id: str):
        return self._call(
            "sit_down_to_object",
            {
                "character_id": str(character_id),
                "object_id": object_id,
            },
        )

    def mop_floor(self, character_id, dirt_id: str):
        return self._call(
            "mop_floor",
            {
                "character_id": str(character_id),
                "dirt_id": dirt_id,
            },
        )

    def rest(self, character_id):
        return self._call("rest", {"character_id": str(character_id)})

    def speak_to_npc(self, character_id, target: str, content: str):
        return self._call(
            "speak_to_npc",
            {
                "character_id": str(character_id),
                "target": target,
                "content": content,
            },
        )
