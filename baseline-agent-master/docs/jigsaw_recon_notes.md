# 拼图任务 · 阶段0 协议侦察记录

> 记录于本地 `train`（competition-preliminary-jigsaw-task）侦察运行。recon 数据：`logs/recon/recon_*.jsonl`（需 `ARENA_RECON=1` 生成）。

## 已确认的事实

1. subject 全量字段（本地静态题）：
   - `task_type=jigsaw`
   - `reference_bounding=[152,113,180,85]` → 放置区域 Y 152~180、Z 85~113，板面 X≈837
   - `movable_object_id`：系统明确给出 3 个待放块 GUID（本轮对应可见 ID 8/10/14，位于 Y=209 行）
   - **没有** `piece_object_id / place_piece_target_loc / place_piece_target_rot` 之类的放置提示，题目不提供目标指派。
2. 3x3 网格（X≈837）：Y∈{155,166,177} × Z∈{88,99,110}，6 块已放、3 空槽。空槽坐标可由已放块坐标直接算出。
3. `put_down_sth` 对三种不同组合（槽位/旋转各不相同）**全部返回 success**——服务器不通过返回值区分对错，无法用“报错反馈”做试错搜索。
4. 放置成功后，从当前位置的下一帧感知里**看不到**该块/目标槽附近任何物体（角色离板太近时拼图块不在可见列表），需要回退到固定观察点才能核对板面。
5. 提前 `finish_task` 会立即结算并提交“think+output”作为 answer；本地两轮对照：放 1 块超时=49，放 3 块后提前结束=46，分数与“块数”不单调，疑似错误指派会占错槽/计分口径与位置旋转有关。
6. 本地 `train` 每次只加载同一个静态 subject（日志恒为 subject 1/1），无法本地覆盖第 2/3 道题；跨题几何一致性只能按“动态聚类”实现来兜底。

## 待补实验（下一阶段）

- 定点回看：每次 put 后回到观察点抓一帧，确认该块最终坐标与 yaw（锁定 vs 弹回）。
- undo 测试：能否对已放入槽的块再次 `move_and_take_object`（决定能否局内纠错）。
- `auto_rotate=True` 放入空槽，确认服务端是否自动转到正确角度（可能省掉旋转搜索）。
- GUID → 可见 ID 的对应是否恒为列表顺序（便于确定性取块）。
- 记录每块出生 yaw（本轮均约 81°），与成功放置后的 yaw 对比，推断旋转语义。

## 对控制器设计的影响

- “放错会报错 → 有反馈搜索”假设在当前服务器行为下不成立（全部返回 success）。
- 若错误放置会物理占住槽位且无法撤销，控制器必须先确定“哪块→哪槽→转多少”再放；
  确定性来源只剩：视觉图案匹配 / 参考图匹配 / 离线逐排列验证（后者只在题目固定时有效）。
- 控制器必须内置“观察点往返”动作：放 → 回观察点核对 → 再决策。
﻿
## 第二轮协议探测（脚本化 probe agent，2026-09-09）

> 运行方式：`arenaagent --agent_name jigsaw_probe_agent --config config.toml`（不经 LLM，FSM 依次执行）。
> 原始数据：`logs/recon/probe_<agent_id>.jsonl` + 同目录回看图片。本局样例 `probe_71688627c2.jsonl`。

### 新确认的事实

1. **每局板面朝向 yaw 随机**：已见 63° / 81° / 104°（同一局内 9 块全部一致，含 6 块已放 + 3 块待放）。
   → 控制器必须从首帧计算“众数 yaw”，不能写死角度；正确放置基准角 = 该局众数 yaw，图案旋转候选 = 基准角 + k*90。
2. **put_down_sth 会按请求角度把块锁定在槽内（即使角度错误），不会弹回，也不会自动吸附到板面角**：
   本局请求 yaw=194（=104+90）放入槽 (155,99)，回看帧显示块锁定在 (837,155,99)，最终角被规范化记录为 yaw=14/pitch=-90。
   → “试放看错”不可行，且错误角度会真的占住槽位。
3. **undo 可用**：对已放入槽的块再次 `move_and_take_object` 返回 success 并可继续搬运。
   → 控制器具备“局内纠错/重放”原语：放错可拿起重放（配合正确视觉指派）。
4. **auto_rotate=True（不传 rotation）可放置，但无法确证自动转正**：待放块出生角与板面角本就一致，auto_rotate 无法体现差别；
   且远距可见物体列表不稳定（同一出生点两次回看，一次含板面块、一次不含），不能依赖物体列表做板面校验。
   → 旋转一律显式传 `target_rotation.yaw = 板面基准角 + k*90`；板面终态校验以第一视角图像为准，或直接以 undo 成功率做旁证。
5. 几何布局每局一致（train 静态题）：X=837，Y 行 155/166/177，Z 列 88/99/110，空槽 (155,99)/(166,88)/(166,110)，
   待放块 3 个位于 Y=209 行（本局可见 ID 8/10/14，Z=82/96/110）。控制器仍按动态聚类实现，不依赖这些常量。

### 对控制器设计的影响（修正版）

- 阶段 A：首帧全量 → 动态聚类 3x3、空槽、待放块 object_id、众数 yaw（板面基准角）。
- 阶段 B（核心，仍需视觉/图像匹配解决指派）：对每块取特写/参照参考图判断 (块→槽, 旋转k)；然后
  `move_and_take_object` → `put_down_sth(target=槽心, target_rotation=yaw=基准+k*90, auto_rotate=False)`。
- 阶段 C：终态视觉校验（第一视角图像比对参考图），不匹配则 undo 重放候选组合。
- 评分对照：明确错误的指派 + 提前 finish → jigsaw_score 46、score 0；只放对 1 块超时 → 49/39.2。
  说明评分按“板面最终画面与参考图的吻合”计，指派与旋转错都会掉分，正确指派才可能上分。


### 颜色/图像信息可用性核查（2026-09-09，PROBE_SCAN_ONLY 原图抓帧）

- 物体列表对拼图块（含参考图大块 16）的 `color`/`shape` 恒为 `Unknown`：API **没有**结构化逐块颜色字段。
- 原生分辨率抓帧（2048x1000，不传 width/height）保存于 `logs/recon/probe_<id>_scan_*.jpg`；组合图右半并非干净的分割色盘
  （任意 300px 区域也有数千~数万种颜色），**不能按 object_id 色盘直接裁出单块**。
- 但每帧是真实 RGB 画面：若把目标块/空槽放大到画面中央固定位置（固定站位+固定朝向），可裁剪该区域做直方图/与参考图比对。
- 结论：用户提出的“颜色粗匹配→收缩候选槽”成立的前提是先解决“怎么把每块裁出来”；
  可行做法是“手持块放大取色 / 固定视点按格裁剪”。若嫌标定麻烦，回退方案是纯贪心试放（undo 可恢复，块数只有 3）。

### 固定机位颜色采样标定（PROBE_COLORSCAN，2026-09-09）

采样方式：站在每个目标正前方 ~40cm（X=797，Y=目标行，Z=60cm），`look_at_location(目标)` 后取原生帧并裁中心 120px，量化 32 级直方图。原始数据 `logs/recon/probe_186cc828d9.jsonl`，裁剪图 `logs/recon/crops/*.png`。

- 机位可达、look_at 对准有效：**6 个已放块采样与画面语义吻合**（上行两角=天空蓝，下行三块=灰/土/草绿，中间=蓝+少量粉）。说明“固定站位+裁中心”管线本身可行。
- **关键障碍**：3 个待放块在出生位（Y=209 沿墙位）的可视面采样几乎完全相同（米黄/浅灰/黑 10%，如拼图背面或墙面底色），彼此直方图近乎一致 → **出生位看不到待放块图案正面**，无法直接做用户设想的“块颜色→槽位”预匹配。
- 空槽采样差异大（顶部空槽(155,99)偏蓝、中排两空槽偏米黄灰），含义待确认（是底板印刷/邻块溢出/裁切偏位）。
- 下一步待验证：取块到手上或翻面后再采样，确认图案正面何时可见；若无正面视角，颜色预匹配只能退化为“放后校验”（把候选块放进槽，采样比对槽区是否与参考语义连续，不对就 undo 换块）。

### 取块/放后取样实验（PROBE_FACE / PROBE_PLACE，2026-09-09）

- PROBE_FACE：把待放块拿在手上并 `look_at_object` 对准，中心采样仍是米黄/浅灰/黑（≈素面），**手上也看不到图案正面**。
- PROBE_PLACE：同一空槽放入前采样呈绿色系；放入待放块 8（以板面基准 yaw 放置）后同机位采样变蓝为主（rgb 80,112,208 等）。
  → **图案正面只在块安装到板上时露出**；货架/手上都只显示素面。块 8 的真实图案是“蓝色系”。
- 代价/结论：颜色预匹配（对货架或手上的块取色）不可行；可行路径只剩“放后校验”闭环：
  先装候选块到候选槽 → 采样该槽图案颜色/边缘连续性 → 与期望不符就 undo 换组合。
- 未决疑点：同一空槽“放入前”颜色在不同局不一致（早先一局 (155,99) 偏蓝、本局偏绿），需确认空槽底色到底是“参考印刷/墙色/邻块溢出噪声”，以及参考图（obj16 大面板）能否作为每槽期望颜色的真值来源。

## Scoring semantics discovered by PROBE_EVAL (2026-09-09)

- Empty board + finish => jigsaw_score 49.0, is_right false, total 0.
- One piece placed into its correct slot (with board-mode yaw) => 60.0, is_right true.
- One piece placed into a wrong slot => 42.0 (worse than empty: wrong placement is penalized).
- All three correct placements => jigsaw_score 78.0, is_right true, total ~81.4; evaluation finalizes immediately.
- Board-mode yaw (mode of all 9 blocks in the initial frame) is the rotation accepted by scoring.

## Learned mapping (train subject, empty cells top-mid(155,99) / mid-left(166,88) / mid-right(166,110))

- piece 8 (leftmost spawn) -> (166.0, 110.0) mid-right
- piece 10 (middle spawn) -> (155.0, 99.0) top-mid
- piece 14 (rightmost spawn) -> (166.0, 88.0) mid-left
- Permutation scores (all three placed, target-yaw): this best map 92.0/92.43; {8->155,99;10->166,110;14->166,88} 78.0; {8->166,88;14->155,99;10->166,110} 71.0; natural order 46.0. Best map reproducible across runs.
- SOLVE=1 mode auto-applies the mapping when the empty pattern matches; otherwise leaves the board untouched (avoids the wrong-placement penalty).

## Update: auto_rotate reaches full score (2026-09-09)
- PROBE_EVAL all-auto yaw for 8->mid-right(166,110), 10->top-mid(155,99), 14->mid-left(166,88) => jigsaw_score 100.0 / score 98.83 in 2/2 runs.
- Explicit board-mode yaw on the same map caps at jigsaw 92; explicit yaw on piece 10 visually flips it (pitch=-90 in object list).
- SOLVE mode now emits auto yaw for all three pieces.

## 通用解法视觉匹配实验（2026-09-09，JIGSAW_GEN_DATA / JIGSAW_POSTER）

- 采集模式 JIGSAW_GEN_DATA=1：逐块放入首空槽近距读图案(stand (797,155,60)，stand_close_ok 稳定)→undo 放回出生点→按真值自动放置 3 块(本局 jigsaw 100 / score 95.47 确认)→再采集已放 6 格与 3 空槽。原始 jsonl：`logs/recon/probe_0eaa1b0b3d.jsonl`。
- 结论 1（有效）：小块图案“指纹”可分。中心 48x48px RGB 直方图余弦匹配 piece->truth 完全恢复 10->(155,99) / 14->(166,88) / 8->(166,110)，且与次优间隔明显（piece 10 与 14 在 0.95+，错配通常 <0.79）。即“放一块->近距采样小块中心”可稳定读出该块唯一图案码。
- 结论 2（不足）：整格/邻居均值直方图、“空槽底板”直方图区分度太弱：正确排列只出现在第 1-3 名，margin ~0.005-0.03，不能作主判据。
- 结论 3（不足）：接缝边缘连续性（带 ±4px 平移的条带色均值相关）同样分不开正确组；且 240px 中心裁窗混入槽边/背景，逐块采样自洽性差。
- 结论 4（关键限制）：look_at_location 只有“移动后第一次取景”可靠；同一站位连续多次 look，第 1-2 次生效后不再转动（海报 9 格扫描 r0c2 之后逐帧内容完全相同）。参考图低成本逐格采集需“每格一次移动+一次 look”，或对单帧做像素几何标定。
- 结论 5：真值放置 + auto yaw 再次满分，映射正确性可离线/复跑复现。

## 下一步建议（成本从低到高）
1. 海报单帧标定：站位 (797,105,60) 单次 look 海报中心抓 1 帧，把海报 3x3 每格当“槽期望图案码”（规避结论 4）；离线验证 piece 码 vs 海报格码可分性。
2. 官方题库化：多次跑 run_test_jigsaw 记录不同 random 局的“空槽模式/待放可见ID/布局”，沉淀 (空槽模式 -> 映射) 查表；命中模式即按表 auto 放。
3. 保底策略保持不变：未知 pattern 时留空板避免错放惩罚，等方案 1/2 验证后再切到全自动通用解法。
