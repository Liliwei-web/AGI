"""从 results/scores.csv 反推实测评分公式（纯标准库 + 最小二乘）。

假设评分是"完成度分 + 时间惩罚"的线性形式：

    总分 = 100 - a * (100 - jigsaw_score) - b * 用时(秒)

待定参数只有 a、b。移项后是一次最小二乘：

    100 - 总分 = a * (100 - jigsaw_score) + b * 用时

只使用"完成度通过"的运行（jigsaw_score >= 60），因为完成度不通过时总分另有规则。
脚本会打印拟合出的 a、b，以及逐条回代残差——这就是 docs/08 里那条公式的来源。

运行：python tools/fit_score_model.py [scores.csv]
"""

from __future__ import annotations

import csv
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results" / "scores.csv"
MIN_JIGSAW = 60.0


def solve2(a11, a12, a22, b1, b2):
    """2x2 对称正定线性方程组：[[a11,a12],[a12,a22]] [x,y]^T = [b1,b2]^T"""
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        raise SystemExit("拟合失败：数据退化（完成度与用时不独立）")
    return (b1 * a22 - b2 * a12) / det, (a11 * b2 - a12 * b1) / det


def main() -> int:
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    rows = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw_j, raw_t, raw_s = row["jigsaw_score"].strip(), row["elapsed_sec"].strip(), row["total_score"].strip()
            if not (raw_j and raw_t and raw_s):
                continue
            jigsaw, elapsed, total = float(raw_j), float(raw_t), float(raw_s)
            if jigsaw < MIN_JIGSAW:
                continue
            rows.append((row["run_at"], elapsed, jigsaw, total))

    if len(rows) < 3:
        raise SystemExit(f"样本不足：只有 {len(rows)} 条可用运行")

    # 法方程
    a11 = sum((100.0 - j) ** 2 for _, _, j, _ in rows)
    a12 = sum((100.0 - j) * t for _, t, j, _ in rows)
    a22 = sum(t * t for _, t, _, _ in rows)
    b1 = sum((100.0 - j) * (100.0 - s) for _, _, j, s in rows)
    b2 = sum(t * (100.0 - s) for _, t, _, s in rows)
    a, b = solve2(a11, a12, a22, b1, b2)

    print(f"samples            : {len(rows)}")
    print(f"总分 = 100 - {a:.4f} x (100 - jigsaw) - {b:.4f} x 用时")
    print()
    print(f"{'run_at':<17} {'用时':>7} {'jigsaw':>7} {'实测总分':>9} {'回代':>8} {'残差':>7}")
    worst = 0.0
    for run_at, elapsed, jigsaw, total in rows:
        pred = 100.0 - a * (100.0 - jigsaw) - b * elapsed
        resid = pred - total
        worst = max(worst, abs(resid))
        print(f"{run_at:<17} {elapsed:>6.1f}s {jigsaw:>7.1f} {total:>9.2f} {pred:>8.2f} {resid:>+7.3f}")
    print()
    print(f"最大绝对残差        : {worst:.3f} 分")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())