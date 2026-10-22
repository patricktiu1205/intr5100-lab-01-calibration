# Task: Lab 1 Stage 1 数据预处理脚本

## 背景
课程 INTR5100 Lab Session I（IDM 跟驰模型标定）。工作目录即本项目根目录。

原始数据（两组，分别来自两个驾驶员，一人对应一组待标定参数）：
- `data/raw/collection_20260917_144220/straight_trajectory.csv` → driver_1
- `data/raw/collection_20260917_144342/straight_trajectory.csv` → driver_2

所有处理一律以各采集目录下的 `straight_trajectory.csv` 为准（不要用 follower_/leader_ 分文件）。

## 要求
编写一个**可复用、可重复运行**的 Python 预处理脚本（建议放 `src/preprocess_stage1.py` 或你认为合理的位置），完成：

1. 读取上述两组 `straight_trajectory.csv`。
2. 按 `role` 列拆分为 follower 和 leader 两个数据集。
3. 每个数据集只保留列：`t_from_start, x, y, heading`（另加一列 `driver`，值为 driver_1 / driver_2）。
4. 按 `t_from_start` 升序排序。
5. 输出 4 个 CSV 到 `data/preprocess/stage_1/`：
   - `driver_1_follower.csv`、`driver_1_leader.csv`
   - `driver_2_follower.csv`、`driver_2_leader.csv`
6. 脚本应幂等（重复运行结果一致），可用标准库 csv（环境里没有 pandas），路径基于项目根目录解析，不要写死绝对路径。
7. 脚本运行结束后打印每个输出文件的行数做校验。

## 验收
- 运行脚本，确认输出文件行数：driver_1 follower=374/leader=347，driver_2 follower=353/leader=374。
- 实际运行通过后，在回复里报告：脚本路径、命令、实际行数校验结果。
