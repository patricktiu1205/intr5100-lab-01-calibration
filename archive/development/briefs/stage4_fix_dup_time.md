# Task: Stage 4 小修 —— 时间取整重复点处理

## 背景
`src/preprocess_stage4.py` 当前把时间四舍五入到 0.1s。已知 `driver_1_event_2.csv` 末两行（原始 t=32.657770 与 32.698555，后者为 stage3 右端点收尾点）取整后 time 都是 32.7。

## 修改要求
1. 在 stage4 输出前增加规则：按时间顺序扫描，**取整后 time 与上一行重复时，丢弃后一个点（即保留栅格上的前一点，删除收尾点）**。该规则对所有 event 通用，不是只针对 driver_1_event_2 的硬编码。
2. 更新 `stage4_summary.json`：记录被删点的原始时间、所在 event（替换原有的 time_rounding_collisions 记录方式，明确标注为 dropped）。
3. 预期结果：`driver_1_event_2.csv` = 27 行；其余 4 个 event 行数不变（138/79/211/124）；总计 579 行。
4. 输出文件中 time 列必须严格唯一、单调。
5. 沿用幂等/原子写入/落盘校验；实际运行后更新 build-report.md。
