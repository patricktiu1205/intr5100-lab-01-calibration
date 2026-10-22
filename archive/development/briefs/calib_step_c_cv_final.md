# Task: Lab 1 IDM 标定 Step C —— 交叉验证选优 + 最终参数

## 环境与前置
- 解释器：`C:/Users/ptiu1/miniconda3/envs/intr5100/python.exe`。
- 已有 5 个可复用标定方法（同一边界、同一 total_error 口径，见 idm_core）：
  M1 `m1_instant_ls.py`（瞬时最小二乘；已知在 driver_1 全量上发散）、
  M2 `m2_neldermead.py`（13 起点 NM）、M3 `m3_ga.py`、M4 `m4_pso.py`、M5 `m5_differentiable.py`。
- 输入：`data/preprocess/stage_4/`（5 event；driver_1: event_1/2/3，driver_2: event_2/3）。

## 1. Leave-one-event-out 交叉验证
新建 `src/calibrate/cross_validate.py`，对每个 driver 分别做：
- driver_1：3 折（留 e1、留 e2、留 e3）；driver_2：2 折（留 e2、留 e3）。
- 每折：用**训练折 event** 以 5 个方法各自标定（各方法的预算/种子沿用原脚本，seed 固定可复现；M1 在某折若不可行/发散，该折结果记 infeasible，不中断流程）。
- 用得到的参数对**留出 event**做纯预测：从留出 event **首点实测 s、v** 初始化，喂实测 lv，前向积分整条，计算该 event 的指标。

## 2. 统一指标
每个"方法 × driver × 折"记录留出 event 的：
- `spacing_rmse`（s=dhw−250 口径，原始坐标单位）
- `speed_rmse`（原始单位）
- `s_violations`：仿真中 s<=0 的点数
- `objective_j`：该 event 的归一化 j（同 idm_core 口径，便于对照）
- **常规回归指标**（spacing s=dhw−250 与 speed v 分别输出）：MSE、RMSE、MAE、MAPE（百分比；分母为 0 的点跳过并记录跳过数）、R²。
对每个 driver 做 CV 汇总：误差类指标（MSE/RMSE/MAE/MAPE）各折等权平均；R² 用合并口径重算（全部留出点的残差平方和 ÷ 全部留出点对其所属折均值的总平方和）；违规次数求和。

## 3. 排名与最终参数
- 综合评分：`score = mean(spacing_rmse/scale_s) + mean(speed_rmse/scale_v)`，scale 取该 driver **全部 event 的 pooled 标准差**（s 与 v 各一）；有任何 s_violations 的方法加固定罚（每次违规 +1，写入评分公式）。infeasible 方法不参与排名。
- 选 score 最小的方法为该 driver 的最优方法。
- 用**全部 event** 以该最优方法重训一次，得到最终五参数。
- 输出：
  - `results/cv_results.json`：全部折/方法的指标、per-driver 平均、综合评分与排名、各方法 infeasible 记录。
  - `results/final_params.json`：per-driver 的最优方法名、最终 θ（v0,T,s0,a,b，δ=4）、全量 total_error 与每 event RMSE。
  - 终端打印排名表（per driver：方法 / CV spacing RMSE / MAE / MAPE / R² / CV speed RMSE / MAE / MAPE / R² / violations / score）和最终参数。MSE 写入 JSON 不强制打印。

## 4. 报告（build-report.md，新内容在前，旧报告附录存档）
- 交叉验证方法说明与排名结果；
- per-driver 最终参数表及其含义解释；
- 标定集 vs CV 误差对比，说明"为什么 CV 选出来的方法不一定是标定集 J 最低的方法"；
- 简短局限性讨论（driver_1 e1 口径问题、driver_2 仅两折、参数顶边界现象）。

## 约束
- 不改 idm_core 与 M1..M5 脚本（如需把其中优化逻辑封装调用，请在脚本内 import 复用；不要复制粘贴算法代码）。
- 不装新包、不动 stage_1..4。
- 结果可复现（两次运行 JSON 一致，耗时字段除外）。
- 计算预算控制：5 方法 × 每 driver 折数 × 一次优化；若预计总耗时过长，允许对全局类方法（GA/PSO/M5 多起点）在 CV 中统一减半预算，但必须在报告中写明，并对最终重训恢复全预算。
