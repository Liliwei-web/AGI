"""智能体主体：把"看一眼 -> 求解 -> 按顺序放"接成赛场的 agent 循环。

依赖赛场 SDK（`arenaagent`），只保留解题路径，去掉了探针脚本里所有侦察分支。
与探针脚本（2400 行）的关系：这里约 150 行是它的解题内核。

一次完整执行只有三种状态：
    start -> solve -> done
真正花时间的都在 solve：3 次 move_and_take_object + 3 次 put_down_sth。
"""

from __future__ import annotations

import os
import time
from typing import Any

from loguru import logger

from arenaagent.agent_base import AgentBase, AgentCfg
from arenaagent.builder import Register
from arenaagent.tongsim_grpc_client import TongSimGrpcClient

from . import planner, scene as scene_mod, solver


class JigsawAgentCfg(AgentCfg):
    name: str = "jigsaw_agent"
    sleep_between_steps: float = 0.1  # 步间等待：RPC 是同步的，这里只留最小缓冲
    log_dir: str = "logs"
    tongsim_server_endpoint: str = "127.0.0.1:50060"


@Register("jigsaw_agent")
class JigsawAgent(AgentBase):
    """把三块拼图放回 3x3 板面的三个空槽。"""

    def __init__(self, stub, channel, cfg=None, sleep_between_steps: float = 1.0) -> None:
        super().__init__(stub=stub, channel=channel, cfg=cfg or JigsawAgentCfg(),
                         sleep_between_steps=sleep_between_steps)
        self._tongsim: TongSimGrpcClient | None = None
        self._character_id: str | None = None
        self._spawn_loc: list[float] = []
        self._scene: scene_mod.Scene | None = None
        self._todo: list[str] = []          # 待取放的块 id，已按最短走行排序
        self._plan: dict[str, tuple[float, float]] = {}
        self._in_hand: str | None = None
        self._state = "start"
        self._action_wait = float(os.environ.get("JIGSAW_ACTION_WAIT", "0.15"))

    # ---------------------------------------------------------------- #
    # 生命周期
    # ---------------------------------------------------------------- #

    def init(self, opt: dict[str, Any]) -> None:
        self._tongsim = TongSimGrpcClient(opt.get("tongsim_server_endpoint", "127.0.0.1:50060"))
        self._tongsim.connect()
        self._character_id = opt.get("character_id")

    def run_step(self, subject, task_response) -> dict[str, Any]:
        try:
            if self._state == "start":
                return self._start(subject, task_response)
            if self._state == "solve":
                return self._solve_step()
            return self._finish("done")
        except Exception as exc:
            logger.opt(exception=True).error("jigsaw step crashed: {}", exc)
            return self._finish("exception: " + str(exc))

    # ---------------------------------------------------------------- #
    # 阶段一：看一帧，还原场景，定方案
    # ---------------------------------------------------------------- #

    def _start(self, subject, task_response) -> dict[str, Any]:
        self._spawn_loc = list(self._tongsim.get_character_location(self._character_id) or [])
        frame = self._tongsim.acquire_first_person_perception(self._character_id) or {}
        self._scene = scene_mod.analyze(frame, spawn_loc=self._spawn_loc)
        if self._scene is None:
            # 现场读不出来就什么都不做：错放的分数（42）低于留空（49）
            logger.warning("scene analysis failed; leaving board untouched")
            return self._finish("analysis failed")

        self._plan = solver.solve(self._scene)
        if not self._plan:
            logger.warning("no verified plan; leaving board untouched")
            return self._finish("no plan")

        self._todo = planner.order_by_walk(self._scene, self._plan)
        logger.info("plan: {}", {pid: self._plan[pid] for pid in self._todo})
        self._state = "solve"
        return self._ok("start done")

    # ---------------------------------------------------------------- #
    # 阶段二：逐块 取 -> 放
    # ---------------------------------------------------------------- #

    def _solve_step(self) -> dict[str, Any]:
        if not self._todo:
            return self._finish("solve done")
        piece = self._todo[0]

        if self._in_hand != piece:
            result = self._tongsim.move_and_take_object(self._character_id, piece, which_hand=0) or {}
            self._record("take", piece, result)
            if self._ok_result(result):
                self._in_hand = piece
                return self._ok("taken")
            self._todo.pop(0)  # 取不到就跳过，不阻塞剩余两块
            return self._ok("take failed")

        target = self._plan[piece]
        result = self._tongsim.put_down_sth(
            self._character_id,
            target_location=self._scene.slot_loc([target[0], target[1]]),
            target_rotation=None,
            auto_rotate=True,   # 让服务端把块转到板面朝向：消掉"转多少度"这个自由度
            force_locate=True,
        ) or {}
        self._record("put", piece, result)
        if self._ok_result(result):
            self._in_hand = None
            self._todo.pop(0)
        return self._ok("placed")

    # ---------------------------------------------------------------- #
    # 工具
    # ---------------------------------------------------------------- #

    def _record(self, action: str, piece: str, result: dict) -> None:
        time.sleep(self._action_wait)  # RPC 同步返回，这里只是给动画留最小缓冲
        logger.debug("{} {} -> {}", action, piece, result.get("result"))

    @staticmethod
    def _ok_result(result: dict | None) -> bool:
        return str((result or {}).get("result", "")).lower() in ("success", "ok", "true")

    def _ok(self, msg: str) -> dict[str, Any]:
        return {"status": "continue", "message": msg}

    def _finish(self, reason: str) -> dict[str, Any]:
        self._state = "done"
        return {"status": "finish", "message": reason}