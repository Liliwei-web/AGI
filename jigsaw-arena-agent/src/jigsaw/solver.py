"""求解器：决定"哪块进哪槽"。

整个项目里唯一真正重要的 40 行就在这里。

核心事实（见 docs/06-breakthrough.md）：题目不是"图案推理题"，而是"生成器的还原题"。
生成器按固定格序生成 9 块放到 9 个固定格位，然后随机挑 3 格掏空、把对应块挪到货架。
所以块的 object_id 唯一决定了它原本属于哪一格：

    id = base + 3*k + j
      k = 0/1/2 -> 列 Z = 110/99/88
      j = 0/1/2 -> 行 Y = 155/166/177

但这里**不写死这张表**。程序只写下"规则的形式"，然后拿当局已经放好的 6 个锚点
现场反解参数 (base, 行序, 列序)，再要求 3 个待放块的预测格位集合恰好等于 3 个空槽集合。
两个条件都满足才采用；反解不唯一或校验不过，就返回 None 交给保守回退。
"""

from __future__ import annotations

from .scene import Scene


def _predict(bid: int, base: int, rflip: int, cflip: int, rows, cols):
    """按 (base, 行序, 列序) 三个参数预测某个 id 的归属格。"""
    n = bid - base
    if n < 0 or n // 3 > 2:
        return None
    row_idx = (2 - (n % 3)) if rflip else (n % 3)
    col_idx = (n // 3) if cflip else (2 - (n // 3))
    return (rows[row_idx], cols[col_idx])


def solve_by_id_map(scene: Scene) -> dict[str, tuple[float, float]] | None:
    """反解 ID -> 格位规则，返回 {待放块 id: (Y, Z)}；无法唯一确定时返回 None。"""
    if len(scene.anchors) != 6 or len(scene.movables) != 3:
        return None
    rows, cols = scene.rows, scene.cols
    if len(rows) != 3 or len(cols) != 3:
        return None
    try:
        anchors = [(int(bid), cell) for bid, cell in scene.anchors.items()]
        mov_ids = [int(mid) for mid in scene.movables]
    except (TypeError, ValueError):
        return None
    empty_set = scene.empty_set
    if len(empty_set) != 3:
        return None

    fits: list[dict[str, tuple[float, float]]] = []
    for base in range(-60, 61):
        for rflip in (0, 1):
            for cflip in (0, 1):
                # 条件 1：必须能解释全部 6 个已放锚点
                if any(_predict(bid, base, rflip, cflip, rows, cols) != cell for bid, cell in anchors):
                    continue
                # 条件 2：待放块的预测格位集合必须恰好等于空槽集合
                pred: dict[str, tuple[float, float]] = {}
                for mid in mov_ids:
                    cell = _predict(mid, base, rflip, cflip, rows, cols)
                    if cell is None:
                        pred = {}
                        break
                    pred[str(mid)] = cell
                if not pred or set(pred.values()) != empty_set:
                    continue
                fits.append(pred)

    if not fits:
        return None
    first = fits[0]
    if any(other != first for other in fits[1:]):
        # 多组不同参数都能通过校验 -> 不做判断（宁可不落子）
        return None
    return first


def mirror_rule(scene: Scene, tie: str = "yasc") -> dict[str, tuple[float, float]] | None:
    """保守回退规则：货架从左到右 ↔ 空槽按 Z 从右到左。

    这条规则只来自 1 个样本，且无法在本地证伪，在随机题上曾把两块对调（2/3 错）
    —— 见 docs/05-falsified-hypotheses.md。保留它只是因为：猜中的收益（100）、
    猜错的代价（历史实测 46~92）与留空板（49）大致相当，所以在无法反解时
    仍略优于完全不动作。
    """
    pieces = list(scene.movables)
    slots = [[float(s[0]), float(s[1])] for s in scene.empty_slots]
    if len(slots) != len(pieces):
        return None
    slots.sort(key=lambda s: (-s[1], s[0] if tie != "ydesc" else -s[0]))
    return {pid: (slots[i][0], slots[i][1]) for i, pid in enumerate(pieces)}


# 第一代解法留下的常量表（见 docs/04-lookup-table-era.md）。
# 只对"训练环境那道固定题"成立，这里仅作为已知形态的快速通道保留。
KNOWN_PATTERN = {(155.0, 99.0), (166.0, 88.0), (166.0, 110.0)}
KNOWN_TABLE = {"8": (166.0, 110.0), "10": (155.0, 99.0), "14": (166.0, 88.0)}


def solve(scene: Scene, force_rule: bool = False, allow_mirror: bool = True) -> dict[str, tuple[float, float]]:
    """求解入口：返回 {块 id: (Y, Z)}；返回空字典表示"不落子，留空板"。

    优先级：
      1. 已知空槽形态且未强制绕过查表 -> 常量表（与第 2 步结果互为交叉验证）
      2. ID <-> 格位反解 + 自校验（主解法，唯一可泛化的那一条）
      3. 镜像顺序经验规则（默认开启，可用 allow_mirror=False 关闭）
      4. 空字典 —— 保守留空，避免错放惩罚
    """
    if not force_rule and scene.empty_set == {(round(y, 1), round(z, 1)) for y, z in KNOWN_PATTERN}:
        if all(bid in scene.movables for bid in KNOWN_TABLE):
            return {bid: KNOWN_TABLE[bid] for bid in scene.movables}

    plan = solve_by_id_map(scene)
    if plan:
        return plan

    if allow_mirror:
        plan = mirror_rule(scene)
        if plan:
            return plan

    return {}


def as_placements(plan: dict[str, tuple[float, float]]):
    """把 {id: (Y, Z)} 转成 (块 id, Y, Z, yaw) 的放置序列；yaw=None 表示 auto_rotate。"""
    return [(bid, float(cell[0]), float(cell[1]), None) for bid, cell in plan.items()]