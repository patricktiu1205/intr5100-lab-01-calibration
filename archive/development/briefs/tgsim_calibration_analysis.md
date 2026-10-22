# 任务：TGSIM 真实数据 IDM 标定与 AV–HV 分析（Lab I 延伸，独立于评分）

## 背景
Lab Session I 主体（SC²I 缩微台 IDM 标定）已完成。本任务是报告最后一部分 "Comparison with real-world IDM"：
用 TGSIM I-90/I-94 Moving Trajectories 数据中的 HV–HV 与 HV–AV 跟驰轨迹，以**同一标定方法**（可微 IDM + Adam）各标定一套参数，
并做误差与驾驶风格分析。**这部分只是 IDM 标定的延伸，与 Lab 实验本身无直接关联。**

## 已有基础（先读、复用，不要重写）
1. `TGSIM_data/HV-HV.csv`、`TGSIM_data/HV-AV.csv` — 原始数据，10Hz。
   列：`ID,time,xloc_kf,yloc_kf,lane_kf,speed_kf,acceleration_kf,length_smoothed,width_smoothed,type_most_common,av,run_index,role`
2. `src/tgsim/preprocess_events.py` — **已跑通**：按停车波（follower speed<0.3 持续≥1s）切 event，
   输出 `data/tgsim/<pair>/event_*.csv`（列 time,fv,lv,dhw；s、m/s、m），及 `data/tgsim/manifest.json`。
   现状：HV-HV 10 events（4251 点），HV-AV 8 events（4370 点）。
3. `src/tgsim/calibrate_events.py` — 已有草稿（torch 可微 IDM，Euler rollout、sigmoid 入盒、Adam、pooled-scale loss），
   **尚未验证通过**，请审查、修正并跑通。
4. 缩微台标定参考：`src/calibrate/idm_core.py`、`m5_differentiable.py`（口径一致，但单位/搜索框不同，不要混用常量）。

## 已核实的数据事实（可直接采信，仍建议复核）
- 单位为公制：车长 4.5–4.8 m；speed_kf 与位置差分速度严格 1:1（中位比值 1.0000）。
- spacing = 欧氏投影 gap − leader length，无负值。xloc 为前保险杠位置。
- 时间连续（Δt=0.1s）、无车道变换，所以按停车波切 event 即可。
- **统一单位，不需要任何倍数换算。**
- 建议上网再次确认 TGSIM Moving Trajectories 数据单位（公制 SI），结论写进报告/JSON 说明。

## 要完成的工作
### 1. 标定（HV-HV、HV-AV 各一套）
- 方法：可微 IDM（torch 计算图，Euler 前向积分，参数 sigmoid 映射入搜索盒，Adam，多起点）。
- 首点初始化：每 event 首点实测 (s,v)；leader 速度为输入；loss 用 pooled 公共尺度。
- 实单位搜索盒（可按数据调整并说明依据）：
  v0∈[5,35] m/s，T∈[0.3,5] s，s0∈[0.2,10] m，a∈[0.5,6] m/s²，b∈[0.5,8] m/s²；δ=4。
- 交付前用 numpy（统一口径）重算误差与 torch 结果对照。
- 产物：`results/tgsim_calibration.json`（两组参数、cost、每 event 指标），原子写入、幂等。

### 2. 分析（三部分）
**Part 1：标定误差对比**
- HV-HV vs HV-AV：spacing / speed 的 RMSE、MAE（可加 MAPE）；总 cost；每 event 表 + 聚合表。
**Part 2：AV 与 HV 的跟驰风格差异**
- 参数差异：v0, T, s0, a, b；结合 efficiency / comfort / safety 指标：
  平均速度、停车占比、a/jerk RMS、s_min、TTC 分布（s/Δv，Δv>0）。
- 说明差异对应的跟驰特性（如 AV 是否更保守/更平滑）。
**Part 3：补充 metrics（自行判断选取，建议）**
- 平衡态 spacing–speed 关系/散点；leader–follower 加速度相关性（string stability 近似）；
  TTC 分布；参数相关性；与 SC²I 参数对比时用**无量纲组合**（v0/√(b·s0)、a/b、T√(b/s0)），
  因两套数据单位不同、不可直接比数值。

### 3. 图表与落盘
- 出图（带 human/实测对照）：标定轨迹叠加、误差、风格雷达/条形、TTC、spacing–speed 等；
  图存 `results/figures/tgsim/`，脚本在 `src/tgsim/`，全部可重跑。
- 所有脚本：标准库 + intr5100 环境（torch/numpy/pandas/matplotlib），幂等、`.tmp`+`os.replace`、落盘自校验。

## 约束
- 装包 / push / 迁移 / 改其他项目文件：先问。
- 运行 torch 脚本若报 OMP Error #15，设 `KMP_DUPLICATE_LIB_OK=TRUE`。
- 用环境解释器绝对路径：`C:/Users/ptiu1/miniconda3/envs/intr5100/python.exe`。
- 完成后写 build-report：列出文件、真实运行输出、关键数字。
- 领域/口径疑问可让主控咨询 intr5100-tutor；你负责实现与自测。
