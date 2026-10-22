# Task: Lab 1 Validation 管线 + 驾驶风格指标 + 全阶段 HTML 报告

## 背景
Lab Session I 流程：人工驾驶数据 → IDM 标定 → TA 用标定参数跑 validation run → 拿到 validation 数据后做对比分析。
**目前没有 validation 数据**，但格式与 human run 的 `data/raw/collection_xxx/` 完全一致（straight_trajectory.csv 等同一结构）。所以本任务目标是：**预先写好全部分析代码，数据到位后一条命令产出完整 HTML 报告**。

项目根 = lab1 目录。解释器：`C:/Users/ptiu1/miniconda3/envs/intr5100/python.exe`（numpy/scipy/pandas/matplotlib/torch 可用）。

## 已有资产（必须复用，不要重写算法）
- 预处理：`src/preprocess_stage1.py` … `src/preprocess_stage4.py`（幂等，参数/边界见各脚本与 build-report.md）
- 标定：`src/calibrate/idm_core.py`、M1..M5、`cross_validate.py`
- 标定结果：`results/final_params.json`（per-driver 最优方法=M5 与最终 θ）
- 驾驶风格相关技能可参考：用户要求 efficiency / comfort / safety 三大类。

## 交付 1：Validation 运行管线 `src/run_validation_pipeline.py`
- 输入：一个新的 raw collection 目录（validation run）+ 对应的 human driver 标识（driver_1/driver_2，命令行参数指定，如 `--validation-raw data/raw/validation_xxx --driver driver_1`）。
- 依次调用/复用现有 stage 1→4 处理函数（import 调用，不要 subprocess 复制），把 validation 数据处理成与 stage_4 同构的标定表（time,fv,lv,dhw），输出到 `data/preprocess/validation_{driver}/stage_1..4` 对应位置（目录命名你可定，但要与 human 数据隔离）。
- 用该 driver 的最终 IDM 参数（final_params.json）对 validation event 做模型仿真（同 idm_core.simulate_event 口径，首点实测初始化）。
- 无 validation 数据时脚本可用 `--dry-run` 检查全部 import/路径/参数加载是否正常。

## 交付 2：指标模块 `src/analysis/metrics.py`
对每个 event（human 与 validation 都可算）输出两套指标：

### A. 基础预测指标（模型仿真 vs 实测）
spacing(s=dhw−250) 与 speed 的 MSE/RMSE/MAE/MAPE/R²（口径同 CV）。

### B. 驾驶风格指标（对实测轨迹计算；human 与 validation 分别算）
- **Efficiency**：平均速度、速度相对 v0 的达成率 mean(v)/v0、行驶总时间（event 时长）、stop 比例（v≈0 采样占比，若无则 0）。
- **Comfort**：加速度（速度差分）的 mean/std、RMS acceleration、纵向 |a| 超阈值（阈值请定义为坐标单位下的合理值并在报告注明换算依据，或直接用相对阈值如 >2σ_a）比例；jerk（a 的差分）RMS 与 max|jerk|。
- **Safety**：最小 spacing s_min、最小 TTC（dv>0 时 s/dv，dv≤0 记 inf）、TTC 低于阈值（定义阈值并注明）的时间占比、spacing 违规(s<=0)次数。
- 所有阈值集中为常量配置，报告中列出。

## 交付 3：HTML 报告生成器 `src/analysis/build_html_report.py`
数据到位后生成**单个自包含 HTML 文件**（输出 `results/lab1_full_report.html`），包含：
1. 实验概述与 IDM 公式、参数定义
2. 数据处理：stage1→4 流程说明、每 stage 行数/event 信息（从各 summary json 读）
3. 标定：5 方法说明与 CV 排名（含 MSE/RMSE/MAE/MAPE/R² 表格）、最终参数表与参数解释
4. Validation：实测 vs IDM 仿真对比（spacing/time、speed/time 折线图，matplotlib 渲染为 base64 内嵌；每 event 一图），基础指标表
5. 驾驶风格：human vs validation 的 efficiency/comfort/safety 对比表+雷达图或柱状图（matplotlib base64），并给简短自动文字结论（基于数值比较，不要空话套话）
6. 局限性讨论（driver_1 一致性差、参数顶边界、event 数量少、单位为坐标单位）
风格：简洁学术风，表格用样式，单文件可离线打开（图全部内嵌，不依赖 CDN 外的资源；可用纯 CSS）。
报告默认语言：**英文**（课程为英文学术报告）。

## 交付 4：可验证性
- 当前无 validation 数据：用现有 9/17 human 数据**模拟跑一遍全管线**（可以把某一 driver 的 human collection 当作伪 validation 输入），确认：管线跑通、HTML 成功生成且可打开、图表齐全。生成的演示 HTML 输出到 `results/lab1_full_report_demo.html`（明确标注 DEMO）。
- 真实 validation 数据到位后，用户只需运行：
  `python src/run_validation_pipeline.py --validation-raw <dir> --driver <driver_1|driver_2>`
  然后
  `python src/analysis/build_html_report.py --driver <...>`（或合并为一条命令，请设计清楚并在报告顶部写明）。
- 不装新包、不改已有 stage/M1..5 脚本（如必须小改以支持 import 复用，保持其原有输出行为不变）。
- 结果可复现；更新 build-report.md（markdown 开发报告）。

## 验收回复请报告
脚本路径、dry-run 结果、demo HTML 路径、真实数据到位后的操作命令清单、报告包含的章节与图表数量。
