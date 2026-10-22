# Task: Lab 1 Stage 2 —— 去除无效数据 + 按时间断层拆分 event

## 输入
Stage 1 输出：`data/preprocess/stage_1/driver_1.csv`、`driver_2.csv`
列：`driver,role,t_from_start,x,y,heading,velocity`；文件内 follower 段在上、leader 段在下，各段内 t 升序。

## 处理步骤（新建脚本 `src/preprocess_stage2.py`，仅用标准库，沿用 stage1 的幂等/原子写入/落盘校验风格）

### 1. 无效起点/终点裁剪（per driver, per role）
判定"未启动段"和"终点停车段"，规则：

- velocity==0 视为停车采样点；**孤立的 ≤3 个连续 0 采样点视为行驶中丢值，忽略不计**；
- **起点**：从序列头部开始，找到"持续运动开始前最后一个非孤立的 velocity==0"，该点及其之前的行全部删除；
- **终点**：同理，从尾部向前，若存在持续停车段（>3 个连续 0），则从最后一个运动采样点之后的行全部删除（即"最后一个 0 速为终点"语义：停车段不属于有效行驶数据）；
- 起点裁剪后，**以 follower 的有效起点时间为基准，把 leader 中 t_from_start 早于该时间的行也一并删除**（保证两车起点对齐）。若某 role 本身没有触发裁剪则不动。

预期结果（用于校验，实际以逻辑计算为准并打印判定依据）：
- driver_1 follower：删除前 29 行（索引 0–28，最后一个非孤立 0 在 t≈3.62s）；leader 删除 t<3.62s 的行；136/137/140 的孤立 0 保留。
- driver_2：不触发任何裁剪。

### 2. 按时间断层拆分 event
- 在每个 role 序列内，相邻采样点 `Δt > 1.0 s` 视为断层（正常采样间隔约 0.1–0.2s，转弯断层 6–8s），断层两侧属于不同 event；
- 每段连续直路 = 一个 event，按时间顺序编号 event_1, event_2, …；
- follower 与 leader 的 event **按直线段出现顺序（序号）配对**：由于跟随车与头车在路线上有位置差，同一节直路两车经过的时刻不同，时间区间可能不重叠（driver_2 的第 1 段就是如此：follower t=[4.63,6.42]，leader t=[-0.05,0.35]），因此**不能用时间重叠配对**；某车多出的段为 unmatched。

### 3. 输出
输出到 `data/preprocess/stage_2/`，目录每次运行前重建：
- 每个配对 event 一个文件：`{driver}_{event_k}.csv`（如 `driver_1_event_1.csv`），包含该 event 内 follower 和 leader 两行集合；
- 列与 stage1 相同：`driver,role,t_from_start,x,y,heading,velocity`，文件内排序沿用 follower 段在前、leader 段在后、各段内 t 升序；
- 另输出一个 `stage2_summary.json`：每个 driver/role 删除的起止行数与时间、event 数量、每个 event 的行数与 t 区间、unmatched event 列表。

## 验收（实际运行并在 build-report.md 记录）
1. driver_2 无裁剪：event 数 = follower 3 段、leader 3 段，配对后 3 个 event（行数 18+4、230+225、105+145，即 event_1=22, event_2=455, event_3=250）。
2. driver_1：follower 删 29 行、leader 按 t<3.62s 同步裁剪；follower 3 段、leader 裁剪后剩 3 个有配对段（末尾两个短段若无 follower 配对应进 unmatched）；报告各 event 行数与 t 区间。
3. 打印裁剪判定依据（被忽略的孤立 0 索引、起点/终点边界索引与 t）。
4. 幂等：连跑两次输出哈希一致；无 .tmp 残留。
