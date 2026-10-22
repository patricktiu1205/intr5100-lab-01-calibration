# Task: Lab 1 Stage 1 预处理脚本（需求修订）

## 背景
课程 INTR5100 Lab Session I（IDM 标定），项目根目录 = 本目录。

原始数据：
- `data/raw/collection_20260917_144220/straight_trajectory.csv` → driver_1
- `data/raw/collection_20260917_144342/straight_trajectory.csv` → driver_2

一律以 `straight_trajectory.csv` 为准。

## 修订后的要求（覆盖上一版）
Stage 1 的定位是**只做列裁剪，不做 role 拆分**：

1. 读取两组 `straight_trajectory.csv`。
2. **保留 role 列，不再拆成 follower/leader 文件**（后续阶段再分）。
3. 保留列为：`role, t_from_start, x, y, heading, velocity`，并在最前面加 `driver` 列（driver_1 / driver_2）。
   - 注意：velocity 是车辆上报的参考速度（与位置差分差约 3 倍，单位问题后续阶段处理），本阶段先原样保留。
4. 按 `t_from_start` 升序排序。
5. 每个驾驶员输出一个文件到 `data/preprocess/stage_1/`：
   - `driver_1.csv`（721 行）
   - `driver_2.csv`（727 行）
6. 旧的 4 个拆分文件（`driver_1_follower.csv` 等）应从 stage_1 目录中删除，避免混淆。
7. 修改现有脚本 `src/preprocess_stage1.py`（沿用其幂等、原子写入、落盘校验、缺列报错等做法；仅用标准库）。
8. 更新 `build-report.md` 记录本次修订。

## 验收
实际运行脚本，并报告：命令、每个输出文件行数（driver_1=721，driver_2=727）、role 计数（driver_1: leader 347/follower 374；driver_2: leader 374/follower 353）、表头列顺序，以及旧拆分文件已删除。
