"""场景解析：从一帧物体列表里还原出 3x3 板面的结构信息。

这个模块只做一件事——把"黑盒给我们的原始物体列表"翻译成"题目结构"：

    rows / cols        板面 3 行 3 列的世界坐标
    anchors            6 个已放块的 {id: (Y, Z)}
    movables           3 个待放块（按出生行 Z 升序 = 货架从左到右）
    empty_slots        3 个空槽的 (Y, Z)
    target_yaw         本局板面基准朝向（所有拼图块 yaw 的众数）

注意这里不认识"哪块该放哪格"——那是 solver 的事。场景解析只负责如实还原现场。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# 板面与货架的物理位置。这些值来自侦察阶段实测，多个题目间一致。
BOARD_X = 837.0
BOARD_Y_RANGE = (120.0, 225.0)
BOARD_Z_RANGE = (60.0, 125.0)
SPAWN_ROW_Y = 209.0
ROW_SPACING = 11.0
DEFAULT_ROWS = (155.0, 166.0, 177.0)
DEFAULT_COLS = (88.0, 99.0, 110.0)


@dataclass
class Scene:
    """一局题目的结构化描述。"""

    rows: list[float]
    cols: list[float]
    anchors: dict[str, tuple[float, float]]
    movables: list[str]
    movable_locs: dict[str, tuple[float, float]]
    empty_slots: list[list[float]]
    target_yaw: float
    spawn_loc: list[float] = field(default_factory=list)

    @property
    def empty_set(self) -> set[tuple[float, float]]:
        return {(round(y, 1), round(z, 1)) for y, z in self.empty_slots}

    def slot_loc(self, slot: list[float]) -> dict[str, float]:
        """把 (Y, Z) 格位翻译成 put_down_sth 需要的目标坐标。"""
        return {"X": BOARD_X, "Y": float(slot[0]), "Z": float(slot[1])}


def is_jigsaw_block(obj: dict) -> bool:
    """按位置 + 包围盒识别拼图块，排除墙面参考大图（object 16）与家具。"""
    loc = obj.get("place_location") or {}
    x, y, z = loc.get("X"), loc.get("Y"), loc.get("Z")
    if x is None or y is None or z is None:
        return False
    if abs(float(x) - BOARD_X) > 2.0:
        return False
    if not (BOARD_Y_RANGE[0] <= float(y) <= BOARD_Y_RANGE[1]):
        return False
    if not (BOARD_Z_RANGE[0] <= float(z) <= BOARD_Z_RANGE[1]):
        return False
    aabb = obj.get("world_aabb") or {}
    mini, maxi = aabb.get("min") or {}, aabb.get("max") or {}
    if mini and maxi:
        x_ext = abs(float(maxi.get("x", 0.0)) - float(mini.get("x", 0.0)))
        y_ext = abs(float(maxi.get("y", 0.0)) - float(mini.get("y", 0.0)))
        z_ext = abs(float(maxi.get("z", 0.0)) - float(mini.get("z", 0.0)))
        if x_ext > 4.0 or y_ext > 30.0 or z_ext > 30.0:
            return False
        if y_ext < 5.0 or z_ext < 5.0:
            return False
    return True


def complete_axis(values: list[float], prefer: tuple[float, ...]) -> list[float] | None:
    """把板上实际观测到的坐标补成完整的 3 条。

    6 个已放块不一定同时覆盖 3 行和 3 列：如果某个行/列整列都是空的，
    那这一条坐标就不会出现在已放块里，只能按已知格距推出来。这时用
    `prefer`（板面默认坐标）从两个候选里挑一个，避免"补错方向"。
    """
    vals = sorted({round(float(v), 1) for v in values})
    if len(vals) == 3:
        return vals
    if len(vals) != 2:
        return None
    gap = round(vals[1] - vals[0], 1)
    if abs(gap - 22.0) < 0.51:
        # 观测到的是第 1、3 条，中间那条直接插值
        return [vals[0], round((vals[0] + vals[1]) / 2.0, 1), vals[1]]
    if abs(gap - 11.0) < 0.51:
        # 观测到的是相邻两条，缺的那条在两端之一
        candidates = [round(vals[0] - 11.0, 1), round(vals[1] + 11.0, 1)]
        for cand in prefer:
            if cand in candidates:
                return sorted(vals + [cand])
    return None


def analyze(frame: dict, spawn_loc: list[float] | None = None) -> Scene | None:
    """把一帧物体列表解析成 Scene；解析不出来时返回 None（调用方应保守收尾）。"""
    blocks: list[dict] = []
    for obj in frame.get("objects", []):
        if not is_jigsaw_block(obj):
            continue
        loc = obj.get("place_location") or {}
        yaw = (obj.get("rotation") or {}).get("yaw")
        blocks.append(
            {
                "id": str(obj.get("object_id")),
                "loc": [float(loc["Y"]), float(loc["Z"])],
                "yaw": float(yaw) if yaw is not None else 0.0,
            }
        )
    if len(blocks) < 9:
        return None

    # 板面基准朝向 = 所有拼图块 yaw 的众数（同一局内 9 块一致）
    yaw_counter = Counter(round(b["yaw"], 1) for b in blocks)
    target_yaw = float(yaw_counter.most_common(1)[0][0])

    # 待放块出生在 Y=209 的货架行，其余是板面已放块
    spawn_rows = sorted({round(b["loc"][0], 1) for b in blocks if abs(b["loc"][0] - SPAWN_ROW_Y) <= 5.0})
    if len(spawn_rows) != 1:
        return None
    spawn_row = spawn_rows[0]
    board_blocks = [b for b in blocks if abs(b["loc"][0] - spawn_row) > 5.0]
    movables = sorted(
        (b for b in blocks if abs(b["loc"][0] - spawn_row) <= 5.0),
        key=lambda b: b["loc"][1],  # 按 Z 升序 = 货架从左到右
    )
    if len(board_blocks) != 6 or len(movables) != 3:
        return None

    rows = complete_axis([b["loc"][0] for b in board_blocks], DEFAULT_ROWS)
    cols = complete_axis([b["loc"][1] for b in board_blocks], DEFAULT_COLS)
    if rows is None or cols is None:
        return None

    filled = {(round(b["loc"][0], 1), round(b["loc"][1], 1)) for b in board_blocks}
    empty_slots = [
        [row, col] for row in rows for col in cols if (round(row, 1), round(col, 1)) not in filled
    ]
    empty_slots.sort(key=lambda c: (c[0], c[1]))
    if len(empty_slots) != 3:
        return None

    return Scene(
        rows=rows,
        cols=cols,
        anchors={b["id"]: (round(b["loc"][0], 1), round(b["loc"][1], 1)) for b in board_blocks},
        movables=[b["id"] for b in movables],
        movable_locs={b["id"]: (float(b["loc"][0]), float(b["loc"][1])) for b in movables},
        empty_slots=empty_slots,
        target_yaw=target_yaw,
        spawn_loc=list(spawn_loc or []),
    )