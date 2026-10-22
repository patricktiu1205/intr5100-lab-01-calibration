把第一阶段（Lab I stage 2，人驾数据采集）导出的文件放在这里。

需要（每组至少其一）：
  follower_straight_trajectory.csv   本车轨迹
  leader_straight_trajectory.csv     前车轨迹
  straight_trajectory.csv            合并导出（含 role 列时一个文件即可）

建议同时放入：
  metadata.json        路长、车长常数、车辆标识
  leader_events.csv    前车事件（本工具暂不读取，留作核对）
  commands.csv         控制指令（同上）

文件名必须与上面完全一致；改过名的文件不会被自动识别。
本程序只读取这里的文件，不会修改或删除任何东西。
