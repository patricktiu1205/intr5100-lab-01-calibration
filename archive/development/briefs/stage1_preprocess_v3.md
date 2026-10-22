# Task: Lab 1 Stage 1 脚本微调（排序规则）

## 当前状态
`src/preprocess_stage1.py` 已将每个驾驶员输出为一个文件：
`driver,role,t_from_start,x,y,heading,velocity`，当前按 `t_from_start` 全局升序（follower/leader 行交错）。

## 修改要求
排序规则改为两级：
1. 先按 role 分组：**follower 全部在上半部，leader 全部在下半部**；
2. 每个 role 内部仍按 `t_from_start` 升序。

即排序键为 `(role 组序: follower=0, leader=1, t_from_start)`。其余一切不变（列、driver 标签、行数、幂等、原子写入、落盘校验、缺列/未知 role 报错；旧拆分文件保持删除状态）。

## 验收
实际运行 `python src/preprocess_stage1.py`，报告：
- 行数不变（driver_1=721，driver_2=727），role 计数不变；
- 验证每个文件中最后一个 follower 行的行号 = follower 行数（即 follower 段连续位于文件前部，之后全是 leader，无交错）；
- follower 段、leader 段内部 t_from_start 均严格升序；
- 更新 build-report.md。
