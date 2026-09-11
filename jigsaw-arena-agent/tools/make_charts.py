"""从 results/scores.csv 生成 docs/assets/score_history.svg（纯标准库，无需第三方依赖）。

左图：每次历史运行的拼图完成度（jigsaw_score）随时间推进的变化 —— 能看到 46 -> 78 -> 92 -> 100 的阶梯。
右图：总分 vs 单局用时散点，并画出完成度满分时的评分参考线（总分 = 100 - 0.0301 * 秒）。

运行：python tools/make_charts.py
"""

from __future__ import annotations

import csv
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "scores.csv"
OUT_PATH = ROOT / "docs" / "assets" / "score_history.svg"

W, H = 980, 430
PANEL1 = (70, 70, 430, 330)   # x, y, w, h
PANEL2 = (580, 70, 340, 330)
PAD_TOP, PAD_BOTTOM = 70, 100


def load_rows():
    rows = []
    with CSV_PATH.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            def num(key):
                raw = (row.get(key) or "").strip()
                return float(raw) if raw else None
            rows.append(
                {
                    "run_at": row.get("run_at", ""),
                    "elapsed": num("elapsed_sec"),
                    "jigsaw": num("jigsaw_score"),
                    "total": num("total_score"),
                }
            )
    return rows


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def line(x1, y1, x2, y2, stroke="#c8ccd4", width=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{d}/>'


def text(x, y, s, size=12, anchor="middle", fill="#3c4149", weight="normal"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Segoe UI,Helvetica,Arial,sans-serif" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>'
    )


def circle(x, y, r, fill, opacity=0.85):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" fill-opacity="{opacity}"/>'


def build() -> str:
    rows = load_rows()
    graded = [r for r in rows if r["jigsaw"] is not None]
    scored = [r for r in rows if r["total"] is not None and r["elapsed"] is not None]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        text(W / 2, 30, "TongSim 拼图任务 · 64 次历史运行", size=17, weight="600"),
        text(W / 2, 50, "左：拼图完成度的迭代过程　右：总分与单局用时的关系", size=12, fill="#6b7280"),
    ]

    # ---------------- 左图：jigsaw_score 随运行推进 ----------------
    px, py, pw, ph = PANEL1
    parts.append(text(px + pw / 2, py - 16, "拼图完成度（按运行先后）", size=13, weight="600"))
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = py + ph - frac * ph
        parts.append(line(px, y, px + pw, y, dash="3,3"))
        parts.append(text(px - 10, y + 4, f"{int(frac * 100)}", size=11, anchor="end", fill="#6b7280"))
    parts.append(line(px, py, px, py + ph, stroke="#9aa0a6"))
    parts.append(line(px, py + ph, px + pw, py + ph, stroke="#9aa0a6"))
    n = max(len(graded), 1)
    for i, r in enumerate(graded):
        x = px + (i / max(n - 1, 1)) * pw
        y = py + ph - (r["jigsaw"] / 100.0) * ph
        colour = "#2f9e44" if r["jigsaw"] >= 100 else "#f08c00" if r["jigsaw"] >= 80 else "#c92a2a"
        parts.append(circle(x, y, 4.5, colour))
    for value, label in ((100.0, "100"), (92.0, "92"), (78.0, "78"), (46.0, "46")):
        y = py + ph - (value / 100.0) * ph
        parts.append(text(px + pw - 4, y - 6, label, size=10, anchor="end", fill="#868e96"))
    parts.append(text(px + pw / 2, py + ph + 26, "运行次序（早 → 晚）", size=11.5, fill="#6b7280"))

    # ---------------- 右图：总分 vs 用时 ----------------
    qx, qy, qw, qh = PANEL2
    parts.append(text(qx + qw / 2, qy - 16, "总分 vs 单局用时", size=13, weight="600"))
    max_t = max((r["elapsed"] for r in scored), default=100.0)
    x_hi = 10.0
    while x_hi < max_t:
        x_hi += 10.0
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = qy + qh - frac * qh
        parts.append(line(qx, y, qx + qw, y, dash="3,3"))
    for sec in range(0, int(x_hi) + 1, 10):
        x = qx + (sec / x_hi) * qw
        parts.append(line(x, qy, x, qy + qh, dash="2,3", stroke="#eceff3"))
        parts.append(text(x, qy + qh + 18, str(sec), size=10.5, fill="#6b7280"))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = qy + qh - frac * qh
        parts.append(text(qx - 10, y + 4, f"{int(frac * 100)}", size=11, anchor="end", fill="#6b7280"))
    parts.append(line(qx, qy, qx, qy + qh, stroke="#9aa0a6"))
    parts.append(line(qx, qy + qh, qx + qw, qy + qh, stroke="#9aa0a6"))

    # 满分参考线：总分 = 100 - 0.0301 * 秒
    pts = []
    for sec in (0.0, x_hi):
        score = max(0.0, 100.0 - 0.0301 * sec)
        pts.append(f"{qx + (sec / x_hi) * qw:.1f},{qy + qh - (score / 100.0) * qh:.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#1c7ed6" stroke-width="2" stroke-dasharray="6,4"/>')

    for r in scored:
        x = qx + (min(r["elapsed"], x_hi) / x_hi) * qw
        y = qy + qh - (r["total"] / 100.0) * qh
        colour = "#2f9e44" if r["jigsaw"] >= 100 else "#f08c00" if r["jigsaw"] >= 80 else "#c92a2a"
        parts.append(circle(x, y, 5.0, colour, opacity=0.8))
    parts.append(text(qx + qw / 2, qy + qh + 40, "单局用时（秒）", size=11.5, fill="#6b7280"))

    legend_y = H - 42
    parts.append(circle(140, legend_y - 4, 5, "#2f9e44"))
    parts.append(text(152, legend_y, "jigsaw = 100", size=11.5, anchor="start"))
    parts.append(circle(300, legend_y - 4, 5, "#f08c00"))
    parts.append(text(312, legend_y, "80–99", size=11.5, anchor="start"))
    parts.append(circle(420, legend_y - 4, 5, "#c92a2a"))
    parts.append(text(432, legend_y, "< 80", size=11.5, anchor="start"))
    parts.append(line(560, legend_y - 4, 592, legend_y - 4, stroke="#1c7ed6", width=2, dash="6,4"))
    parts.append(text(600, legend_y, "满分参考线：总分 = 100 − 0.0301 × 秒", size=11.5, anchor="start"))
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()