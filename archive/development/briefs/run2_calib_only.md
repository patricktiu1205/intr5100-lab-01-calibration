# Task: run_2 标定 + run_1/run_2 选优（预处理已完成，勿重跑 stage）

解释器：`C:/Users/ptiu1/miniconda3/envs/intr5100/python.exe`。项目根=lab1。

## 现状（已核实，不要动预处理）
- run_registry: run_1=collection_20260917_144342, run_2=collection_20260922_142956
- `data/preprocess/stage_4/` 已就绪：
  - run_1: event_2(211行), event_3(124行)
  - run_2: event_1, event_3, event_6, event_7, event_8, event_9, event_12, event_13（8 个 event）
- 5 个短段 e2/4/5/10/11 已在 stage3 因两车无时间重叠被删除，**此为确认口径，不要恢复**。
- run_1 已有标定结果（M5，J=1.214），其数值不得改变。
- **本任务不要运行 preprocess_stage*.py，直接从 stage_4 读数。**

## 1. run_2 五方法标定
M1..M5 对 run_2 的 8 个 event 独立标定，复用 idm_core/M1..5 现有算法，边界不变（v0[40,200],T[0.3,5],s0[10,400],a,b[10,200],δ=4,L=250）。
- 已知真实异常：**event_6 内 follower 在 t≈133.63–133.94s 有 0.3s 突发停车（leader 正常）**，数据保留；报告中标注并讨论其对 a/b 与 comfort/safety 的影响。

## 2. 交叉验证
- run_2：8 event leave-one-event-out（8 折）。为控时间，CV 中 M3/M4/M5 预算减半并在报告注明，M1/M2 正常；最终全量重训全预算。
- 指标照旧：MSE/RMSE/MAE/MAPE/R²、s_violations、score。

## 3. 两 run 选优（关键）
统一口径比较 run_1、run_2 两套最终参数：
- 主指标：CV 综合 score（spacing RMSE/ pooled σ_s + speed RMSE/pooled σ_v，σ 用各 run 自身 pooled）；佐证 raw mean spacing RMSE、speed RMSE、s_violations、MAPE。
- 选 score 最小的一套写入 `final_params.json` 顶层 `selected_for_validation`：来源 run、方法、θ=(v0,T,s0,a,b)、对比表、选择理由。

## 4. 输出
- 更新 `results/cv_results.json`（runs 含 run_1+run_2）、m1..m5_params.json（run_2 结果）、`final_params.json`（runs 两块 + selected_for_validation）。
- 终端打印 run_2 CV 排名表 + 两 run 选优对比 + 最终五参数。
- 更新 build-report.md（新内容在前）。
- 不生成 HTML。不装包。可复现（耗时字段外两次一致）。

## 验收回复报告
run_2 五方法 8 折 CV 排名、两 run 选优对比表、**最终 validation 参数**。
