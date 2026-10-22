# Task: run_2 快速标定 —— 仅 M1..M4（不含 M5），先出结果

解释器：`C:/Users/ptiu1/miniconda3/envs/intr5100/python.exe`。项目根=lab1。

## 现状
- stage_4 已就绪：run_2 有 8 个 event（event_1,3,6,7,8,9,12,13）。run_1 有 event_2,3。
- 不要运行 preprocess，不要动 stage 文件。不要跑 M5（用户让其以后再说）。
- 真实异常：event_6 follower t≈133.63–133.94 突发停车 0.3s，保留，报告标注。

## 要求（快速优先）
1. 对 run_2 用 M1/M2/M3/M4 各自标定（8 event 全量，复用现有算法与边界；不要复制算法代码）。
2. CV：8 折 leave-one-event-out。**为快速出结果：CV 中 M3/M4 预算减半；M1/M2 正常**。每方法出 8 折汇总指标 MSE/RMSE/MAE/MAPE/R²、s_violations、score。
3. 每方法全量重训（M3/M4 全预算）得最终 θ。
4. 更新结果文件（保留 run_1；run_2 只写 M1..M4 四块；M5 字段保留 run_1 既有值）：
   `results/cv_results.json`、`m1..m4_params.json`、`final_params.json`（runs 中 run_2 先记 M1..M4，best_method 暂取 M1..M4 中 score 最小者，标注 M5 pending）。
5. **跨 run 选优先做 M1..M4 可比版本**：run_1、run_2 用各自 M1..M4 中的最优 score 比较，选综合 score 最小者写入 `selected_for_validation`，并注明 "preliminary, M5 for run_2 pending"。
6. 终端打印 run_2 M1..M4 CV 排名、两 run 对比、初步 validation 参数。更新 build-report.md。

## 验收回复报告
run_2 M1..M4 八折 CV 排名表、两 run 初步选优、待 M5 补跑后的更新方式。
