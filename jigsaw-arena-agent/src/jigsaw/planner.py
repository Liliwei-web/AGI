"""走行排序：把 3 次取放排成一个总路程最短的顺序。

板面与货架在同一面墙（X≈837）上，角色只能沿 Z 方向移动，所以两点距离近似为 |ΔZ|。
三块共 3! = 6 种顺序，直接穷举即可。

实测收益只有约 0~1 秒（见 docs/08-performance.md）：每段行走的耗时主要由
起步/停下的动画构成，与距离几乎无关。这个模块保留下来主要是因为它零成本、
而且让路径在视觉上更"顺"。
"""

from __future__ import annotations

import itertools

from .scene import Scene


def order_by_walk(scene: Scene, plan: dict[str, tuple[float, float]]) -> list[str]:
    """返回待放块的取放顺序（块 id 列表）；解析信息不足时按原顺序返回。"""
    pieces = [pid for pid in scene.movables if pid in plan]
    if len(pieces) != 3 or len(scene.movable_locs) != 3:
        return pieces
    try:
        start_z = float(scene.spawn_loc[2])
    except (IndexError, TypeError, ValueError):
        return pieces

    def shelf_z(pid: str) -> float:
        return float(scene.movable_locs[pid][1])

    def target_z(pid: str) -> float:
        return float(plan[pid][1])

    best, best_cost = None, None
    for perm in itertools.permutations(pieces):
        cost, cur = 0.0, start_z
        for pid in perm:
            cost += abs(shelf_z(pid) - cur) + abs(target_z(pid) - shelf_z(pid))
            cur = target_z(pid)
        if best_cost is None or cost < best_cost:
            best, best_cost = perm, cost
    return list(best) if best else pieces