# 任务级 Prompt（五个任务五个分支）

初赛五个任务各有一个独立 prompt 目录，运行时按 `subject["task_type"]` 自动加载
`prompts/tasks/<task_type>/task_spec.txt`。

| task_type | 任务 |
| --- | --- |
| `tidyroom` | 整理房间任务 |
| `counting` | 分类计数任务 |
| `npc` | NPC 对话任务 |
| `raven` | 瑞文测试任务 |
| `jigsaw` | 拼图任务 |

规则：
- 只把该任务专属的提示写进对应目录；通用规则留在 `arenaagent/vlm_agent/prompts/`。
- 每个任务的 prompt 迭代独立、一次只改一处并单独 commit，方便回溯版本。
- 目录或文件缺失时按“无任务提示”（空字符串）处理，不会报错。
