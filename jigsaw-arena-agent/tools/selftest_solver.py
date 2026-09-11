"""solver / scene 的离线自测：不需要赛场 SDK，也不连仿真器。

用 3 道合成题目检验 ID 反解是否真的在"反解"而不是"背表"：
  - 训练题（空槽 = 上中 / 中左 / 中右）
  - 随机题 A（空槽 = 中列整列）
  - 随机题 B（空槽 = 中中 / 中右 / 下右）
  - 反例（锚点与空槽不一致 -> 必须拒绝落子）

运行：
    uv run --no-project python tools/selftest_solver.py
或任意 Python 3.10+ 环境：
    python tools/selftest_solver.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jigsaw.scene as scene_mod  # noqa: E402
import jigsaw.solver as solver_mod  # noqa: E402

ROWS = [155.0, 166.0, 177.0]
COLS = [88.0, 99.0, 110.0]
# id = 7 + 3*k + j，k 对应 Z 列 110/99/88，j 对应 Y 行 155/166/177
CELLS = {
    7: (155.0, 110.0), 8: (166.0, 110.0), 9: (177.0, 110.0),
    10: (155.0, 99.0), 11: (166.0, 99.0), 12: (177.0, 99.0),
    13: (155.0, 88.0), 14: (166.0, 88.0), 15: (177.0, 88.0),
}


# 货架上的出生 Z 与目标格位的 Z 是两回事（训练题实测：8->82 / 10->96 / 14->110），
# 所以货架排序不能拿 CELLS 的 Z 来代替。只有训练题这三个 id 有实测值。
SPAWN_Z = {8: 82.0, 10: 96.0, 14: 110.0}


def make_scene(empty_ids: list[int]):
    anchors = {str(i): CELLS[i] for i in CELLS if i not in empty_ids}
    movables = [str(i) for i in sorted(empty_ids, key=lambda i: SPAWN_Z.get(i, CELLS[i][1]))]  # 出生 Z 升序 = 货架从左到右
    empty_slots = [list(CELLS[i]) for i in empty_ids]
    return scene_mod.Scene(
        rows=ROWS, cols=COLS, anchors=anchors, movables=movables,
        movable_locs={str(i): (209.0, SPAWN_Z.get(i, CELLS[i][1])) for i in empty_ids},
        empty_slots=empty_slots, target_yaw=104.0,
        spawn_loc=[800.0, 209.0, 60.0],
    )


def check(name: str, empty_ids: list[int], expect: dict[str, tuple[float, float]]):
    sc = make_scene(empty_ids)
    got = solver_mod.solve(sc, force_rule=True)  # 强制绕过查表，只考验反解
    signed = {k: (round(v[0], 1), round(v[1], 1)) for k, v in got.items()}
    want = {k: (round(v[0], 1), round(v[1], 1)) for k, v in expect.items()}
    ok = signed == want
    print(("PASS" if ok else "FAIL") + f"  {name}")
    print(f"      empty={empty_ids} movables={sc.movables}")
    print(f"      got ={signed}")
    if not ok:
        print(f"      want={want}")
    return ok


def check_reject(name: str):
    """锚点与空槽不匹配（人为篡改一个锚点）-> 反解必须失败，且 solve() 不给出任何放置。"""
    sc = make_scene([8, 10, 14])
    first = next(iter(sc.anchors))
    sc.anchors[first] = (999.0, 999.0)
    plan = solver_mod.solve(sc, force_rule=True, allow_mirror=False)
    ok = plan == {}
    print(("PASS" if ok else "FAIL") + f"  {name}")
    print(f"      plan={plan}")
    return ok


def main() -> int:
    results = [
        check("训练题：空槽 = 上中/中左/中右", [8, 10, 14],
              {"8": (166.0, 110.0), "10": (155.0, 99.0), "14": (166.0, 88.0)}),
        check("随机题 A：空槽 = 中列整列", [10, 11, 12],
              {"10": (155.0, 99.0), "11": (166.0, 99.0), "12": (177.0, 99.0)}),
        check("随机题 B：空槽 = 中中/中右/下右", [8, 9, 11],
              {"11": (166.0, 99.0), "8": (166.0, 110.0), "9": (177.0, 110.0)}),
        check("随机题 C：空槽 = 左上/下左/下右", [7, 13, 15],
              {"7": (155.0, 110.0), "13": (155.0, 88.0), "15": (177.0, 88.0)}),
        check_reject("反例：锚点不一致时必须拒绝落子"),
    ]
    n_ok = sum(1 for r in results if r)
    print(f"\n{n_ok}/{len(results)} passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())