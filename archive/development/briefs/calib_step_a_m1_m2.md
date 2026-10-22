# Task: Lab 1 IDM 标定框架 + M1/M2（第 1 步，共 3 步）

## 环境
- conda 环境 `intr5100`（Python 3.11，有 numpy/scipy/pandas/matplotlib/torch 2.14 CPU）。
- 运行方式：`C:/Users/ptiu1/miniconda3/envs/intr5100/python.exe src/calibrate/xxx.py`（git-bash 下 conda run 有问题，直接用解释器绝对路径）。
- 项目根：lab1 目录（本目录）。

## 输入数据
`data/preprocess/stage_4/` 下 5 张标定表：
- `driver_1_event_1.csv` 138 行, `driver_1_event_2.csv` 27 行, `driver_1_event_3.csv` 79 行
- `driver_2_event_2.csv` 211 行, `driver_2_event_3.csv` 124 行
列：`time,fv,lv,dhw`（速度=坐标单位/s，dhw=位置单位，未减车长）。
DHW → bumper spacing: `s = dhw - L_veh, L_veh=250`。

## IDM（δ=4 固定）
a_idm(s,v,dv) = a [1 - (v/v0)^δ - (s*/s)^2]
s* = s0 + max(0, vT + v dv / (2 sqrt(a b)))
dynamics: ds/dt = vL - v, dv/dt = a_idm。
待标定 θ = [v0, T, s0, a, b]。
参数边界（坐标单位）：
v0 ∈ [40,200], T ∈ [0.3,5.0], s0 ∈ [10,400], a ∈ [10,200], b ∈ [10,200]。
（优化中对边界外参数用大罚值；初始点取边界范围里的合理值，如 v0=120,T=1.2,s0=100,a=80,b=100。）

## 本步交付

### 1. 公共模块 `src/calibrate/idm_core.py`
- `idm_acceleration(s, v, dv, theta)`：支持标量与 numpy 数组。
- `simulate_event(event_df, theta)`：给定实测 lv 序列，从**首点实测 s、fv** 初始化，按 time 列前向积分（先用显式欧拉，步长=实际 Δt；若你判断 0.1s 下精度不足，可用 RK4，但 vL 在积分点上线性插值）。返回仿真 s、v 数组（长度与输入相同）。加入数值保护：s<=0 或爆炸时返回 inf 代价。
- `trajectory_error(event_df, theta)`：归一化组合误差
  J_e = mean((s_sim-s_obs)^2)/σ_s^2 + mean((v_sim-v_obs)^2)/σ_v^2
  σ_s、σ_v 取**该 event 实测 s、v 的标准差**（每个 event 自带尺度）；另加间距违规惩罚（仿真 s<=0 次数 *大常数）。
- `total_error(events, theta)`：多 event **等权平均**各 event 的 J_e（先各 event 平均再平均，不要先拼接）。
- 提供读取 stage_4 的 loader（返回按 driver 分组的 dict: driver -> list of (event_name, df)）。

### 2. M1 —— 瞬时加速度最小二乘 `src/calibrate/m1_instant_ls.py`
- 对每个 event：实测加速度用 fv 的中心差分（首末单侧），dv = fv-lv 用同点值，s=dhw-250。
- 最小化 sum((a_idm - a_obs)^2)（各 event 残差拼接前按 a_obs 标准差归一化），用 scipy.optimize.least_squares（带 bounds）。
- 输出 per driver 一组参数。

### 3. M2 —— Nelder-Mead 轨迹法 `src/calibrate/m2_neldermead.py`
- 目标 = total_error；用 scipy.optimize.minimize(method='Nelder-Mead')，**≥10 个多起点**（拉丁超立方/随机覆盖边界域，固定随机种子保证可复现），取最优。
- 输出 per driver 一组参数。

### 4. 输出与校验
- 所有标定结果输出到 `results/`（新建）：`m1_params.json`、`m2_params.json`，内容为 per driver 的参数 + total_error + 每 event 的 spacing/speed RMSE（未归一化，原始单位）。
- 每个脚本可独立运行并打印上述结果。
- 数值合理性自检：参数在边界内；driver_1/driver_2 分别得到参数；用最优参数对 driver_1 e1 仿真，spacing RMSE 应明显优于"常数零模型"（若没有，打印警告但不报错）。
- 更新 build-report.md：命令、结果参数、RMSE、自检证据。

## 注意
- 不要动 stage_1..4 的脚本与数据；只读。
- 不要安装新包（现有 numpy/scipy 已够）。
- 本步之后还有 M3(GA)/M4(PSO)/M5(可微IDM) 和交叉验证选优，请把接口设计成可复用模块。
