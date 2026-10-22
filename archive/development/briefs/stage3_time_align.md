# Task: Lab 1 Stage 3 —— leader/follower 时间对齐

## 输入
`data/preprocess/stage_2/` 下 6 个 event CSV：列 `driver,role,t_from_start,x,y,heading,velocity`，follower/leader 已按直线段序号配对。

## 背景事实（已探查）
- 两车采样独立，各约 10 Hz，但时间戳不一致、进入/离开同一节直路的时刻也不同。
- 各 event t 区间：
  - driver_1 e1: follower [3.72,22.93], leader [3.66,17.69]
  - driver_1 e2: follower [30.06,36.70], leader [25.64,32.70]
  - driver_1 e3: follower [43.24,57.25], leader [40.46,51.46]
  - driver_2 e1: follower [4.63,6.42], leader [-0.05,0.35] —— **完全不重叠**
  - driver_2 e2: follower [12.44,38.59], leader [8.40,33.87]
  - driver_2 e3: follower [46.05,58.33], leader [41.82,58.34]

## 要求（新建 `src/preprocess_stage3.py`，仅标准库，沿用幂等/原子写入/落盘校验/不写时间戳）

### 1. 取重叠区间
- 对每个 event 计算两车 t 的交集 [max(f.min, l.min), min(f.max, l.max)]。
- 若交集为空或长度 < 1.0s：整个 event 删除（输出摘要中记录原因 `no_overlap`）。
- 非重叠部分的行一律删除。

### 2. 公共时间轴 + 插值
- 在重叠区间上生成 dt = 0.1s 的等间隔时间栅格（起点为区间左端点，右端点若不足一格则包含端点）。
- 每个 role 的 x、y、velocity 对栅格时间做**线性插值**，禁止区间外外推。
- heading 为角度（0–360），先按相邻采样点解卷（unwrap，跨 0/360 处理）再线性插值，结果归一化回 [0,360)。

### 3. 不连贯点删除
- 若某个栅格点的插值跨距（该 role 两侧最近源采样点的 Δt）> 0.3s，视为该 role 在该处不连贯，**该时间点两车行一起删除**（栅格可能因此断开，输出时按保留下来的时间点排列，不强制连续）。

### 4. 输出
输出到 `data/preprocess/stage_3/`，目录运行时重建：
- 每个保留 event：`{driver}_{event_k}.csv`，列与输入相同：`driver,role,t_from_start,x,y,heading,velocity`（t_from_start 写栅格时间）；follower 段在前、leader 段在后，各自按 t 升序，同一栅格时刻两车各一行。
- `stage3_summary.json`：每个 event 的重叠区间、栅格点数、因不连贯删除的点数与时间、被删除 event 及原因、输出行数。

## 验收（实际运行，更新 build-report.md）
- driver_2_event_1 被删除（no_overlap）；其余 5 个 event 有输出。
- 每个输出文件 follower 与 leader 行数相等且时间戳逐一相同。
- 报告各 event 输出行数；点数守恒可在 summary 核对。
- 幂等：连跑两次哈希一致；无 .tmp 残留。
