# Task: Lab 1 IDM 标定 Step B —— M3 GA / M4 PSO / M5 可微IDM

## 环境与前置
- 解释器：`C:/Users/ptiu1/miniconda3/envs/intr5100/python.exe`（numpy/scipy/torch 2.14 CPU）。
- 复用 `src/calibrate/idm_core.py`：`load_events / simulate_event / trajectory_error / total_error / event_metrics / build_results_payload`，参数边界与 M2 完全一致（v0[40,200],T[0.3,5],s0[10,400],a[10,200],b[10,200]）。**本步不改目标口径，不改 idm_core**（driver_1 J 偏高的已知问题用户已知悉，后续统一处理）。
- 输入：`data/preprocess/stage_4/`（5 event）。输出沿用 `results/`，JSON 结构对齐 m2_params.json（per-driver: theta、total_error、每 event j_spacing/j_speed、spacing/speed RMSE、收敛信息）。

## M3 遗传算法 `src/calibrate/m3_ga.py`（自己实现 GA，不调 DE）
- 实数编码（不要二进制编码），种群 60，代数 80，固定 seed=20260921，per driver 独立运行。
- 操作：锦标赛选择（k=3）、**模拟二进制交叉 SBX**（η=15）、多项式变异（η=20，每维变异概率 1/5）、精英保留（每代最优 2 个直接进下一代）、参数在边界内裁剪。
- 适应度 = total_error（越小越好；不可行轨迹的 inf 按最差个体处理）。
- 收敛信息记录：best cost 随代数曲线（每代一个值）、总评估次数、耗时、末代与初始代 best。

## M4 粒子群 `src/calibrate/m4_pso.py`
- 粒子 40，迭代 80，固定 seed；惯性权重 w 从 0.9 线性衰减到 0.4，c1=c2=1.5；速度限幅 ±0.2×边界宽度。
- 标准 gbest PSO，边界处粒子裁剪位置、速度反弹或置零均可（请固定一种并注释）；记录 best 曲线、评估次数、耗时。

## M5 可微 IDM + Adam `src/calibrate/m5_differentiable.py`
- 用 **torch** 把 IDM 逐点递推写成可微计算图（不要用 numpy 仿真再插）：状态 (s,v) 从首点实测值初始化，每步用实测 lv（torch 常量张量），欧拉积分（Δt 用实际 time 差），IDM 各项公式严格对应 idm_core，sqrt 处加 eps（如 1e-6）防 NaN。
- 参数用归一化到 [0,1] 的无约束变量（sigmoid 映射到边界），Adam lr=0.05，300 步；**10 个多起点**（与 M2 同样的种子策略），取最优。
- 记录每起点的最终 cost、best 训练曲线、耗时。
- 结果一致性校验：M5 每组最优参数用 **idm_core.simulate_event（numpy/RK4 口径）重算** total_error，与 torch 内 cost 的差异打印出来（欧拉 vs RK4 会有小差，差异 >10% 则 WARNING）。

## 通用要求
1. 三个脚本各自独立可运行，全部实际运行并保存 `results/m3_params.json`、`m4_params.json`、`m5_params.json`。
2. 每个脚本打印 per-driver 参数、J、spacing/speed RMSE、耗时。
3. 计算预算：若单方法总耗时预计 >5 分钟，先小幅试跑确认评估速度再放量；GA/PSO 向量化或缓存均可，但结果必须可复现（两次运行 JSON 一致）。
4. 更新 build-report.md（新结果在前，旧报告附录存档），包含 5 方法结果的横向小表（M1..M5 per driver 的 J）。
5. 不装新包、不动 stage_1..4 数据与脚本。

## 验收要点
- 每个方法 per driver 各一套边界内参数。
- GA/PSO 的 best 曲线总体下降（打印首尾值）。
- M5 能收敛；多起点结果离散度合理。
- M2 同口径下 J=3.18(d1)/1.20(d2) 作为对照，GA/PSO 应达到相近或更优（GA/PSO 若明显更差请在报告中分析原因，不要调参硬凑）。
