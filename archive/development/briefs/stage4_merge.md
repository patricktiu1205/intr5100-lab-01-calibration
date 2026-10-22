# Task: Lab 1 Stage 4 —— 合并为标定用表 (time, fv, lv, dhw)

## 输入
`data/preprocess/stage_3/` 下 5 个 event CSV：列 `driver,role,t_from_start,x,y,heading,velocity`，follower/leader 时间戳已对齐（0.1s 栅格、逐点配对）。

## 已验证的事实
- heading 约定：0°→+x，90°→+y，180°→−x，270°→−y（位移方向与 heading 一致，已 40/40 验证）。
- 上报的 velocity 与位置差分速度有约 3 倍差异，**本阶段 velocity 列丢弃**，速度一律由位置差分计算。

## 处理（新建 `src/preprocess_stage4.py`，仅标准库；沿用幂等/原子写入/落盘校验/不写时间戳/目录重建）

对每个 event：

1. **时间**：t_from_start 四舍五入到 0.1s（数据已在栅格上，仅做显示精度归一化）。
2. **速度**（fv 对 follower，lv 对 leader）：由相邻 (x,y) 位移距离 / Δt 得到；内部点用中心差分（t_{i+1} 与 t_{i-1} 两点间位移除以 2Δt；若栅格不连续则用实际 Δt），首末点用单侧差分。单位=坐标单位/s。
3. **dhw**（距离头车的前向距离，**不减车长**）：
   `dhw = (x_l - x_f)*cos(h_f) + (y_l - y_f)*sin(h_f)`
   其中 h_f 为 follower heading（弧度）。
4. 输出列：`time,fv,lv,dhw`，每行一个时间点。
5. 若出现 dhw ≤ 0 的点（投影异常/超车），删除并在摘要记录。

## 输出
`data/preprocess/stage_4/`，目录重建：
- 5 个文件：`{driver}_{event_k}.csv`（沿用 stage3 的 event 编号：driver_1_event_1/2/3，driver_2_event_2/3）。
- `stage4_summary.json`：每个文件行数、时间范围、dhw 范围、删除的异常点。

## 验收（实际运行，更新 build-report.md）
- driver_1 e1/e2/e3：138/28/79 行；driver_2 e2/e3：211/124 行。
- 抽查：driver_1 e1 中段 dhw ≈ 730（=480+250 量级），fv ≈ 95–115（坐标单位/s）。
- 幂等：连跑哈希一致；无 .tmp 残留。
