# tools · 离线自测与复盘脚本

这个目录收集的是**不需要仿真器也能跑**的脚本：它们要么在验证求解器本身，要么在复现文档里的某个结论。
真实实验还需要主办方的仿真客户端与赛题系统，那些运行脚本属于归档性质（见文末）。

## 能直接跑的

| 脚本 | 作用 | 复现文档里的哪条结论 | 依赖 |
| --- | --- | --- | --- |
| `selftest_solver.py` | 用 4 道合成题 + 1 个反例检验 ID 反解 | `06-breakthrough.md`、`07-implementation.md`：反解与查表给出同一答案；不一致输入必须拒绝落子 | 无（标准库） |
| `make_charts.py` | 从 `results/scores.csv` 生成 `docs/assets/score_history.svg` | `README.md` 的结果可视化 | 无（标准库） |
| `fit_score_model.py` | 对 `results/scores.csv` 做最小二乘，反推评分公式 | `08-performance.md`：总分 = 100 − 0.8006×(100−jigsaw) − 0.0302×用时，最大残差 0.047 | 无（标准库） |
| `collect_scores.py` | 从赛场 arena 日志里抽取 `(用时, jigsaw_score, 总分)` 生成 CSV | `results/scores.csv` 的来源 | 无（标准库） |

```bash
python tools/selftest_solver.py
python tools/fit_score_model.py
python tools/make_charts.py
```

## 侦察阶段的度量脚本（复盘用）

这些脚本需要 **numpy + Pillow**，并且要指向当年的侦察产物目录（大量 `.jpg` 与 `.jsonl`）。
它们不是"解题代码"，而是"当初怎么得出否定结论"的原始度量方式——正因为记录在案，那些否定结论才可复核。

| 脚本 | 度量什么 | 对应结论 |
| --- | --- | --- |
| `recon_aim_check.py` | 同站位、不同瞄准点连拍之间的逐像素差 | `02-recon.md`：`look_at_*` 不改变取景（同站位 2–3，换站位 29） |
| `recon_turn_scan.py` | 转身扫描的两两帧差矩阵 | `02-recon.md`：`turn_in_degree` 不可复现（同一名义角度 15–21） |
| `recon_seam_metric.py` | 正确排列 A / 错误排列 B 的接缝跳变剖线 | `05-falsified-hypotheses.md`：一个方向 3.3× 分离，另一个方向无分离 |
| `vlm_judge_probe.py` | 把裁剪图交给视觉大模型判"哪组连贯" | `05-falsified-hypotheses.md`：反复误判，不可靠 |

运行前设置输入目录（默认值见各脚本文件头）：

```powershell
$env:RECON_DIR   = "logs/recon"        # 当年的侦察产物目录
$env:CROP_DIR    = "crops"             # VLM 探针的裁剪图目录
$env:ARENA_LOGS  = "arena_offline/logs" # arena 日志目录
$env:DEEPSEEK_API_KEY = "..."          # 仅 vlm_judge_probe.py 需要，凭据不入库
```

`vlm_judge_probe.py` 里的 API key 在整理入库时已被替换成 `os.environ["DEEPSEEK_API_KEY"]`——原脚本曾把 key 明文写在文件里，这也是整理这个仓库时做过的一轮脱敏。

## 归档：需要真实环境的运行脚本

`run_train_solve.ps1` 是当年驱动解法的脚本（启动仿真服务器 → 等就绪 → 跑 agent → 读本地评分），保留下来是为了说明"一次运行到底包含哪些步骤"。它需要主办方环境，无法独立运行。

## 这些脚本的取舍

原始实验工作区里有 33 个 `.ps1` 与 88 个 `.py`，绝大多数是一次性的补丁脚本（`patch_*.py`、`_add_*.py`）与调试脚本。
这里只保留了**能说明结论是怎么得来的**那一小部分：验证求解器的、复现评分的、以及当初用来证伪视觉路线的度量方法。