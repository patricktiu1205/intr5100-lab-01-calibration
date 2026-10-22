# Task: Lab 1 数据重组 —— 按 run 组织 + run_1 全链路重跑

## 背景变更
之前误以为两个 collection 是两个驾驶员。现已确认：**两次数据来自同一个人**——
- `data/raw/collection_20260917_144220`：试跑（water-test），**作废**
- `data/raw/collection_20260917_144342`：实际实验，编号 **run_1**
- 用户稍后会自己再跑一次新的 human run，放进 `data/raw/` 后编号 **run_2**
今后每个 run **独立标定一套 IDM 参数**（不再有 driver 概念）。

## 1. 数据归档与清理
- 把 `data/raw/collection_20260917_144220/` 整体移动到 `archive/collection_20260917_144220/`（连同文件；不删除）。
- 删除所有由旧结构生成的中间产物：
  - `data/preprocess/stage_1..4` 目录全部重建（旧的 driver_1/driver_2 标签文件不得残留）
  - `data/preprocess/validation_driver_1/**`（demo 树）删除
  - `results/` 下旧 m1..m5_params.json、cv_results.json、final_params.json、validation_driver_1.json、lab1_full_report_demo.html、demo 相关产物先移到 `archive/old_results/`（不要直接删，保留可追溯）

## 2. 管线改为 run 驱动
修改 stage 脚本/配置（保持各 stage 处理逻辑与口径不变）：
- 不再硬编码 collection→driver 映射；改为**自动发现 `data/raw/collection_*` 目录并按时间顺序编号 run_1, run_2, …**（映射可写入一个 `data/run_registry.json`：run 标签 ↔ collection 目录名）。
- 标签列从 `driver` 改为 `run`（stage_1 输出等）；下游 stage、标定、analysis 全部同步改为 run 口径（import 名/字段名一致替换，算法不动）。
- 新 run 数据放入后，重跑管线应自动处理 run_1+run_2 两个 run，互不混合；标定与 CV **per run 独立**。
- `idm_core.load_events` 等 loader 改为按 run 分组返回。

## 3. run_1 全链路重跑（目前只有 144342 一个有效 collection）
- stage1→4 全部重跑并校验：
  - stage_2：run_1 有 3 个 event（原 driver_2 结构，行数 event=22/455/250；注意 event_1 两车时间不重叠）
  - stage_3：event_1 删除(no_overlap)，event_2=211 对、event_3=124 对
  - stage_4：event_2=211 行、event_3=124 行（time,fv,lv,dhw）
- M1..M5 五方法对 run_1 重新标定；CV 做 2 折 leave-one-event-out（event_2/event_3）；指标口径全部沿用（含 MSE/RMSE/MAE/MAPE/R²、s_violations、score 公式）。
- 产出 `results/final_params.json`：结构改为 per-run（当前仅 run_1：最优方法 + 最终 θ）。
- 结果可复现（两次运行 JSON 一致，耗时字段除外）。

## 4. 暂不做的
- HTML 报告本轮**不重生成**（等 run_2 数据齐全后再跑），但请确认 build_html_report 已能适配 run 字段；若有硬编码 driver 处一并改掉。
- run_2 数据由用户稍后放入，届时运行同一套命令即可，不需要再改代码。请在 build-report.md 顶部写明"接入新 run 的操作命令"。

## 约束
- 不装新包；除字段/映射重构外不改算法与边界口径。
- 更新 build-report.md（新内容在前），报告 run_1 的 CV 排名与最终参数、归档清单。

## 验收回复报告
归档/清理清单、run_registry 内容、run_1 各 stage 行数、五方法标定与 2 折 CV 排名表、run_1 最终参数、接入 run_2 的操作命令。
