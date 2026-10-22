# Task: Lab 1 run_2 接入 —— 全链路标定 + run_1/run_2 选优

## 环境
解释器：`C:/Users/ptiu1/miniconda3/envs/intr5100/python.exe`。项目根=lab1。

## 输入
新 collection 已放入：`data/raw/collection_20260922_142956/`（3435 行，follower 1787/leader 1648，304s，12 个断层→13 个直路段）。
`run_registry.json` 应自动把它登记为 **run_2**（run_1=collection_20260917_144342 保持不变）。

## 1. run_2 全链路
- stage1→4 重跑（仅新增/重建 run_2 产物；run_1 既有产物不变，其 sha256 应保持）。
- stage2：13 event；event 内 follower/leader 按序号配对（注意极短 event 或无重叠 event 的处理沿用既有规则：stage3 无重叠删除，stage4 只保留 time,fv,lv,dhw；若某 event 因短被删，在 summary 记录）。
- 五方法 M1..M5 对 run_2 独立标定（边界/口径不变：v0[40,200],T[0.3,5],s0[10,400],a,b[10,200],δ=4,L=250）。
- CV：13 个有效 event 的 leave-one-event-out。**计算预算**：为控制总耗时，CV 中全局类方法（M3 GA/M4 PSO/M5 多起点）预算可减半（沿用既有允许条款并在报告注明），M1/M2 正常；最终全量重训恢复全预算。
- 全部指标照出（MSE/RMSE/MAE/MAPE/R²、s_violations、score）。

## 2. run_1 / run_2 最终选优（关键交付）
- 两个 run 各自已有最优方法与全量参数。比较口径必须统一：用**同尺度指标**比较两套参数，建议主用
  `mean per-event normalized score`（CV 汇总 score：spacing RMSE/pooled σ_s + speed RMSE/pooled σ_v；σ 用各 run 自身 pooled 尺度——由于同一实验平台、同一单位，这样跨 run 可比）；
  同时列出 raw 的 mean spacing RMSE、mean speed RMSE、s_violations、MAPE 做佐证。
- 选综合 score 最小的一套为 **validation 最终参数**。
- 输出：
  - 更新 `results/cv_results.json`（含 run_2）、`results/final_params.json`（run_1+run_2 两块 + 新增顶层 `selected_for_validation`：来源 run、方法、θ、对比表、选择理由）。
  - 终端打印两 run 的 CV 排名摘要 + 选优对比表 + 最终五参数。

## 3. 报告
更新 build-report.md（新内容在前）：run_2 event 清单（各 event 行数/时长）、run_2 五方法 CV 排名、两 run 选优证据与最终参数；局限性里写明：仿真器无真实速度感/过弯侧倾→驾驶风格难一致、数据质量受设备限制（不归因驾驶员）；13 event 中若存在异常/被删段如实列出。

## 约束
- 不改算法与边界；不装新包；run_1 产物数值不得变。
- 结果可复现（两次运行 JSON 一致，耗时字段除外）。
- 不生成 HTML 全报告（等 validation 数据到位后再跑），但确认字段适配。

## 验收回复报告
registry 最终内容、run_2 各 stage/event 数（含被删 event 及原因）、run_2 CV 排名、run_1 vs run_2 选优对比表、**最终提交 validation 的参数**。
