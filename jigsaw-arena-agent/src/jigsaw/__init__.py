"""拼图任务智能体（解题版）。

    scene.analyze()  一帧物体列表 -> 题目结构（rows/cols/anchors/movables/empty_slots）
    solver.solve()   题目结构 -> {块 id: 目标格位}（ID 反解 + 自校验，失败则保守留空）
    planner          {块 id: 目标格位} -> 最短走行的取放顺序
    agent            把上面三步接成赛场 agent 循环（取块 -> auto_rotate 放置）

这个包是从赛场探针脚本 `jigsaw_probe_agent.py`（约 2400 行，含十余个侦察分支）
里抽取出来的解题内核，依赖赛场 SDK `arenaagent`，不能独立运行。
完整推导过程见仓库根目录 `docs/`。

`scene` / `solver` / `planner` 三个模块**不依赖赛场 SDK**，可以离线单独导入与测试
（见 `tools/selftest_solver.py`）；只有 `agent` 需要 SDK，所以这里用惰性导入，
避免"只想跑一下 solver"却被 SDK 的 import 拦住。
"""

from typing import Any

from .scene import Scene, analyze
from .solver import solve, solve_by_id_map

__all__ = ["JigsawAgent", "JigsawAgentCfg", "Scene", "analyze", "solve", "solve_by_id_map"]


def __getattr__(name: str) -> Any:  # PEP 562：只有真正用到时才导入 agent
    if name in ("JigsawAgent", "JigsawAgentCfg"):
        from .agent import JigsawAgent, JigsawAgentCfg

        return {"JigsawAgent": JigsawAgent, "JigsawAgentCfg": JigsawAgentCfg}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")