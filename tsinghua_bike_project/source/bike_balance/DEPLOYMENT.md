# 实车部署验收清单

按此顺序逐项验证，**不要跳过，不要一上来就闭合平衡环**。

参考：tutorial 06 §7、tutorial 07 §8。

---

## 第 1 步 · 确认协议与坐标系

**目标**：确认 IMU 四元数顺序、角速度单位、世界坐标定义与代码一致。

- [ ] 查 IMU 文档：四元数是 `wxyz` 还是 `xyzw`？代码假设 `xyzw`（`sensor_msgs/Imu.orientation` 标准）
- [ ] 查 IMU 文档：世界坐标系是 ENU（东-北-天）还是 NED？
- [ ] 查 IMU 文档：角速度单位是 rad/s 还是 deg/s？代码假设 rad/s
- [ ] 确认转向电机编码器分辨率（GIM 电机默认 16384 counts/rev）
- [ ] 确认后轮电机编码器分辨率
- [ ] 确认 CAN 帧格式（代码发送 `#C1<4bytes-LE>` = 速度模式）

**验证**：
```bash
# 查看原始 IMU 消息
ros2 topic echo /imu/data_raw --once
```

---

## 第 2 步 · 确认轴方向与符号

**目标**：手动晃动自行车，确认代码中每个轴的正方向与实际物理方向一致。

启动 `sign_check` 模式（**不发送任何电机指令**）：

```bash
ros2 run bike_balance balance_node --ros-args -p mode:=sign_check
```

按以下动作逐项检查终端输出和 `/bike_balance/debug` 中的数据：

| 动作 | 期望看到 | 验证通过？ |
|------|---------|-----------|
| 车身向**左**滚转 | `roll` 为正值增加 | [ ] |
| 车身向**右**滚转 | `roll` 为负值增加 | [ ] |
| 车身**左滚转过程中** | `body_p` (x 轴角速度) 为正值 | [ ] |
| 车身**右滚转过程中** | `body_p` 为负值 | [ ] |
| 抬**车头**（前轮离地）| `pitch` 为正值增加 | [ ] |
| 压**车尾** | `pitch` 为负值增加 | [ ] |
| 车头向**左**转 | `yaw` 为正值增加 / 转向角正 | [ ] |
| 车头向**右**转 | `yaw` 为负值增加 / 转向角负 | [ ] |

**如果符号不对**：
- 可能需要在 `sensor_to_bike_rpy_deg` 中加 180° 偏移
- 或者检查 IMU 安装方向（侧装？倒装？）
- 转向角符号可能和编码器方向有关

---

## 第 3 步 · 静止标定 — 陀螺零偏

**目标**：估计 IMU 的陀螺零偏 $b_S$。

```bash
# 保持自行车完全静止 → 运行 15 秒
ros2 run bike_balance calibrate_imu --ros-args -p duration_s:=15.0
```

将输出的 `gyro_bias_dps` 值复制到 `config/balance_params.yaml`：
```yaml
gyro_bias_dps: [bx, by, bz]    # 从校准结果填入
```

**重复**：温度变化明显时应重新标定。

---

## 第 4 步 · 安装标定 — $R_{BS}$

**目标**：估计 IMU 坐标系 S 到自行车坐标系 B 的固定旋转 $R_{BS}$。

需要多轴标定动作（至少两个不共线方向，最好三个轴都有激励）：

1. 自行车放在水平地面
2. 启动 `sign_check` 模式记录数据
3. 分别做以下动作各 5-10 秒：
   - 纯滚转：左右晃动车身（保持直线、不转向）
   - 纯俯仰：抬起/压下车头
   - 纯偏航：水平旋转车体
4. 用采集的 `body_rates` 和原始 `gyro` 数据求解 Wahba 问题（`Rotation.align_vectors`）
5. 将标定得到的 Euler 角填入 `sensor_to_bike_rpy_deg`

> **注意**：如果 IMU 安装方向与自行车坐标系基本一致（误差 < 2°），此步可跳过，`sensor_to_bike_rpy_deg` 保持 `[0,0,0]`。

---

## 第 5 步 · 离线回放检查

**目标**：用标定参数处理一段记录数据，确认修正后交叉轴耦合明显减小。

用 `sign_check` 模式采集的数据，离线绘制：
- 修正前后的滚转角对比
- 修正前后的 `body_p` (滚转角速度) 对比
- `body_q`、`body_r` 不应该有大幅滚转耦合

```python
# 示例：对比直接用 gyro_x vs 修正后的 body_p
```

---

## 第 6 步 · 确认电机驱动模式

**目标**：确认电机接受的是速度指令还是位置/电流/转矩指令。

- [ ] 查阅电机通信协议文档
- [ ] 当前代码用 `#C1` 前缀 = **速度模式**。如果电机需要转矩模式，需要在 `_send_can` 中修改前缀
- [ ] 确认 `steer_speed_scale` / `drive_speed_scale` 的单位转换是否正确
- [ ] 在安全条件下（车轮架空），发送小的测试指令验证电机转动方向

---

## 第 7 步 · 验证转向内环（PI）

**目标**：在闭合平衡外环之前，先验证转向电机能跟踪速度指令。

架空自行车（后轮离地、用支撑架稳住车身），启动 `inner_loop` 模式：

```bash
ros2 run bike_balance balance_node --ros-args -p mode:=inner_loop
ros2 topic pub /bike_balance/enable std_msgs/Bool "data: true"
```

手动发送转向速度指令：
```bash
# 慢速正转
ros2 topic pub /bike_balance/steer_rate_ref std_msgs/Float32MultiArray "data: [1.0]"
# 停止
ros2 topic pub /bike_balance/steer_rate_ref std_msgs/Float32MultiArray "data: [0.0]"
# 慢速反转
ros2 topic pub /bike_balance/steer_rate_ref std_msgs/Float32MultiArray "data: [-1.0]"
```

验证：
- [ ] 电机转动方向符合预期
- [ ] `steer_rate` (实际) 能跟踪 `steer_rate_ref` (目标)
- [ ] PI 无持续震荡
- [ ] 饱和标志 `sat_inner` 在合理范围内

如跟踪不好，调整 `steer_kp` / `steer_ki`。

---

## 第 8 步 · 测量控制周期抖动

**目标**：确认 50 Hz 控制周期实际波动在可接受范围内。

查看 `/bike_balance/status` 中的 `jitter` 字段：
```bash
ros2 topic echo /bike_balance/status | grep jitter
```

- [ ] `dt_mean_ms` ≈ 20 ms
- [ ] `dt_std_ms` < 2 ms（抖动不应超过 10%）
- [ ] `dt_max_ms` < 40 ms

如果抖动过大，检查：
- Python GIL 是否被其他进程阻塞
- CAN 通信是否有竞争
- 是否有大量日志输出

---

## 第 9 步 · 低增益闭合平衡外环

**目标**：首次闭合完整平衡环，从低风险开始。

```bash
ros2 run bike_balance balance_node --ros-args \
  -p mode:=full \
  -p method:=pole \
  -p pole_wc:=-2.0 \
  -p target_speed:=1.0 \
  -p steer_kp:=0.002 \
  -p steer_ki:=0.002
```

- [ ] **用支撑架**！不要让自行车自由站立
- [ ] 先不使能（`enable_on_start: false`）
- [ ] 手动用遥控器推到一个稳定的低速
- [ ] 再使能平衡：`ros2 topic pub /bike_balance/enable std_msgs/Bool "data: true"`
- [ ] 观察 `roll_deg` 是否收敛（不一定是 0，有稳态偏差是正常的）
- [ ] 如果出现震荡，逐步降低 `pole_wc`（更慢的极点，更低的增益）
- [ ] 逐步提高增益：`pole_wc` 从 -2.0 → -3.0 → -5.0
- [ ] 逐步提高速度：`target_speed` 从 1.0 → 2.0 → 2.7

**紧急情况**：`|roll| > max_roll_angle_deg` 会自动急停。手动急停：
```bash
ros2 topic pub /bike_balance/enable std_msgs/Bool "data: false"
```

---

## 第 10 步 · 标定自行车物理参数

**目标**：把名义模型参数替换为真实测量值。

- [ ] 测量总质量 → `bike_mass`
- [ ] 测量轴距 → `bike_wheelbase`
- [ ] 测量后轮接地点到质心距离 → `bike_rear_to_com`
- [ ] 测量质心离地高度 → `bike_com_height`
- [ ] 测量拖曳距 → `bike_trail`
- [ ] 测量转向轴倾角 → `bike_steering_axis_angle`
- [ ] 测量车轮半径 → `wheel_radius`

将测量值填入 `config/balance_params.yaml`。

---

## 第 11 步 · 记录所有诊断量

**目标**：确保所有状态、指令、饱和标志和时间戳都被记录，方便事后分析失败原因。

数据来源：
- `/bike_balance/status` — JSON，2 Hz，含所有控制状态 + 增益 + 饱和标志 + jitter
- `/bike_balance/debug` — Float32MultiArray，2 Hz，13 通道数值

```bash
# 记录到文件
ros2 topic echo /bike_balance/status --csv > balance_log.csv
```

检查 `/bike_balance/status` 中是否包含：
- [ ] `sat_outer` / `sat_inner` / `sat_rear` — 三环各自的饱和标志
- [ ] `K` — 当前反馈增益
- [ ] `jitter` — 控制周期统计
- [ ] `steer_cmd` / `drive_cmd` — 实际 CAN 发送的值
- [ ] `body_p_dps` / `body_q_dps` / `body_r_dps` — 修正后的三轴角速度

---

## 总结：不要做的事

- ❌ 不要在完成步骤 1-6 之前就闭合平衡外环
- ❌ 不要在没有支撑架的情况下首次测试
- ❌ 不要直接把 MuJoCo 仿真中的成功等同于实车安全
- ❌ 不要在未标定 IMU 零偏的情况下依赖角速度数据
- ❌ 不要在确认符号方向之前调增益
