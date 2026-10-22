# Task: 生成 Lab 1 全阶段 HTML 报告（含两次 validation 对比）

解释器：`C:/Users/ptiu1/miniconda3/envs/intr5100/python.exe`。项目根=lab1。

## 数据（全部已到位）
- Human run（标定源）：`data/raw/collection_20260922_142956/`（run_2，实际采用的标定数据；raw 中 144342 为早期 run_1，报告中作为历史提及即可，不参与最终）
- 标定产物：`data/preprocess/stage_1..4/`（run_2 共 8 个有效 event：e1,3,6,7,8,9,12,13；5 个短段 e2/4/5/10/11 在 stage3 因无时间重叠删除）
- 标定参数：`results/m1..m5_params.json`（最终选定 run_2 的 M5 参数：v0=123.36,T=2.091,s0=28.72,a=13.97,b=185.16，δ=4,L=250）
- Validation 数据（两次）：
  - `data/validation/idm_test_20260922_152115/`：原始速度参数上车（失败 run）
  - `data/validation/idm_test_20260922_155015/`：v0÷3=41.12 上车（成功 run）
  两目录结构同 raw（straight_trajectory.csv/metadata.json/commands.csv 等）

## 关键分析点（必须写入报告）
1. **单位不一致根因**：标定速度来自位置差分（坐标单位），实验车速度读表为物理单位（≈坐标速度的 1/3）。原始参数上车后 IDM 自由项认为车辆远未达期望速度、持续追赶 → 逼近 leader 触发防撞。v0÷3 后行为正常。
2. 实测安全对比（已核算，请用数据处理脚本重新独立计算核实）：
   - 152115：s_min=61（低于防撞阈值80），32/867 点 s<100，t≈128s 最危险
   - 155015：s_min=138，0/757 点 s<100，零防撞触发
3. 两次 validation 都要做：预处理（stage1–4 口径，validation 数据分别输出到 data/preprocess/validation_152115、validation_155015）、模型仿真回放（用各自实际上车参数，见 metadata.json idm_params）、全部指标。

## HTML 报告要求
单个自包含文件 `results/lab1_full_report.html`（英文，纯 CSS，图全部 matplotlib base64 内嵌，无外部依赖）。章节：
1. Experiment & IDM：IDM 公式、参数定义、单位说明、SC²I 平台
2. Data processing：stage1→4 流程与行数、event 清单、短段删除原因（表格）
3. Calibration：5 方法说明，标定集对比表（J/ spacing RMSE/speed RMSE，run_2 全量数据），最终 M5 参数表与解释
4. **Validation（重点）**：
   - 两次 run 的实验设置差异（参数表对比）
   - 每 run 每 event：实测 spacing/speed 时间序列图（双面板）；152115 用红色标注防撞触发区间
   - 两次 run 的模型回放 vs 实测指标表（MSE/RMSE/MAE/MAPE/R²，spacing 与 speed）
   - 两次 run 安全/效率指标直接对比表 + 柱状图（s_min、s<100 点数、防撞次数等）
5. Driving style：对 human、152115、155015 三组分别算 efficiency（均速、v/v0 达成率、stop 占比）、comfort（RMS a、jerk RMS/max）、safety（s_min、TTC_min、低 TTC 占比、违规次数），三方对比表 + 雷达图（三条线）+ 自动文字结论
6. Discussion & limitations：单位一致性教训（作为部署 calib 的核心发现）、仿真器无真实速度感/侧倾导致风格难一致（设备局限，不归因驾驶员）、event_6 中途停车真实异常、参数 a/b 边界问题、样本与 event 数量限制
报告顶部写明复现命令。

## 复用与约束
- 复用 src/run_validation_pipeline.py、src/analysis/metrics.py、src/analysis/build_html_report.py 的现有逻辑，按需要扩展支持双 validation 输入；如旧 demo 产物（data/preprocess/validation_driver_*）残留请清理。
- 不装新包；不改 M1..5 算法。结果可复现（两次运行 HTML/ JSON 一致，耗时字段除外）。
- 更新 build-report.md。
- 最终实际生成 HTML 并核验：图片数、表格数、base64 可解码、无 placeholder。

## 验收回复报告
HTML 路径、章节与图表数量、两次 validation 的核心指标对比表、复现命令。
