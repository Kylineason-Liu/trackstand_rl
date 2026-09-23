# bike_balance — 运行指南

## 概述

自行车平衡系统由以下节点组成：

| 节点 | 来源 | 作用 |
|------|------|------|
| `imu_publish.py` | `source/imu_publish.py`（独立脚本） | 读取 IMU，发布 `/imu/data_raw`、`/imu/euler`、`/imu/mag` |
| `calibrate_imu` | `bike_balance` 包 | 静止采集陀螺数据，估算零偏 |
| `balance_node` | `bike_balance` 包 | 平衡控制主节点，支持三种模式 |

`balance_node` 一旦启用即接管转向和后轮两个电机（通过 CAN），因此 **运行期间不要同时跑 `motor_controller`**。

---

## 前置条件（每次上电后执行一次）

### 步骤 0：使能 CAN 接口

```bash
sudo ip link set can0 up type can bitrate 1000000
```

验证：

```bash
ip link show can0          # state 应为 UP
```

### 步骤 0b：启动 IMU 发布

```bash
python3 source/imu_publish.py
```

这是单独的脚本，不属于 `bike_balance` 包。所有后续节点依赖它发布的 `/imu/data_raw` 和 `/imu/euler`。**这个终端在测试期间需要一直运行。**

---

## 整体顺序（严格按此顺序）

```
步骤0  使能CAN + 启动IMU
  ↓
步骤1  校准陀螺零偏       → calibrate_imu
  ↓
步骤2  验证轴方向与符号   → balance_node mode:=sign_check
  ↓
步骤3  测试转向速度内环   → balance_node mode:=inner_loop
  ↓
步骤4  低增益首次平衡     → balance_node mode:=full pole_wc:=-2.0
  ↓
步骤5  调参 → 最终平衡    → balance_node mode:=full pole_wc:=-5.0
```

---

## 步骤 1：标定 IMU 陀螺零偏

**目的**：静止状态下陀螺仪不为零，需要在 IMU 坐标系中减去这个固定偏差。

**前提**：IMU 已经在上电并稳定运行。`imu_publish.py` 必须正在运行。

```bash
ros2 run bike_balance calibrate_imu --ros-args -p duration_s:=15.0
```

- 全程保持自行车**完全静止**
- 终端会打印结果，形如：
  ```
  gyro_bias_dps: [0.0234, -0.0156, 0.0082]
  ```
- 复制这三值，填入 `config/balance_params.yaml` 的 `gyro_bias_dps` 字段

> 如果更换 IMU 安装方式、环境温度显著变化、或重新上电后零偏明显漂移，需重新标定。

---

## 步骤 2：验证轴方向与符号

**目的**：确认代码中 roll/pitch/yaw 的正方向与你物理晃动自行车时一致。
**此步骤不发送任何电机指令**，完全安全。

```bash
ros2 run bike_balance balance_node --ros-args -p mode:=sign_check
```

终端每 2 秒打印一行修正后的 IMU 数据。按以下动作逐项检查：

| 动作 | 期望看到的输出变化 | 你的观察 |
|------|-------------------|----------|
| 车身向**左**滚转（你站车后看，向左倾） | `roll` 正值增加 |  |
| 车身向**右**滚转 | `roll` 负值增加 |  |
| 滚转过程中 | `p` 的符号与 roll 变化方向一致 |  |
| 抬车**头**（前轮离地、后轮着地） | `pitch` 正值增加 |  |
| 压车**尾** | `pitch` 负值增加 |  |
| 车头向**左**转 | `steer` 或 `yaw` 正值增加 |  |
| 车头向**右**转 | `steer` 或 `yaw` 负值增加 |  |

**如果任何一项的符号反了 → 不要继续。** 调试方向：
- IMU 是否侧装/倒装 → 调整 `sensor_to_bike_rpy_deg`（加 180° 偏移）
- IMU 驱动输出的坐标系是否和代码假设一致（代码假设 ENU + rad/s）
- 电机编码器正反向是否和转向正负一致

完成后 Ctrl+C 停止此节点。

---

## 步骤 3：测试转向速度内环

**目的**：在车轮架空、车身有支撑的情况下，验证转向电机能正确跟踪速度指令。

**前提**：自行车用支撑架稳住（后轮离地），确保即使电机转动也不会翻倒。

```bash
# 终端 1：启动 inner_loop 模式
ros2 run bike_balance balance_node --ros-args -p mode:=inner_loop
```

```bash
# 终端 2：使能
ros2 topic pub /bike_balance/enable std_msgs/Bool "data: true"
```

测试指令：

```bash
# 慢速正转
ros2 topic pub /bike_balance/steer_rate_ref std_msgs/Float32MultiArray "data: [1.0]"
# 停止
ros2 topic pub /bike_balance/steer_rate_ref std_msgs/Float32MultiArray "data: [0.0]"
# 慢速反转
ros2 topic pub /bike_balance/steer_rate_ref std_msgs/Float32MultiArray "data: [-1.0]"
# 快速正转
ros2 topic pub /bike_balance/steer_rate_ref std_msgs/Float32MultiArray "data: [5.0]"
```

验证：
- 电机转向方向正确（正 = 左转 / 负 = 右转，取决于你的约定）
- 转速大致与指令值成正比
- 到达限位角 `±steer_angle_limit_deg` 后自动停止同向运动
- 发送反方向指令后可以回正

```bash
# 监视状态
ros2 topic echo /bike_balance/status
```

观察 `steer_angle_deg` 和 `steer_rate_dps` 确认跟踪正常。

---

## 步骤 4：低增益首次闭合平衡外环

**目的**：第一次让平衡外环控制转向电机。从超低增益开始，用支撑架保护。

**❗ 必须用支撑架！不要让自行车真正独立平衡！**

```bash
ros2 run bike_balance balance_node --ros-args \
  -p mode:=full \
  -p method:=pole \
  -p pole_wc:=-2.0 \
  -p target_speed:=1.0
```

启动后：
1. 确认 `enable_on_start: false`（默认），此时电机保持静止
2. 用遥控器或手动推车至一个稳定的低速（约 0.5~1 m/s）
3. 然后使能平衡：

```bash
ros2 topic pub /bike_balance/enable std_msgs/Bool "data: true"
```

4. 观察 `/bike_balance/status` 中的 `roll_deg`：
   - 如果逐渐趋于 0° 附近震荡 → 增益方向正确
   - 如果滚转角持续放大 → **立即急停**：`ros2 topic pub /bike_balance/enable std_msgs/Bool "data: false"`，检查增益符号和轴方向

```bash
ros2 topic echo /bike_balance/status
# 关注字段：roll_deg, steer_angle_deg, speed_m_s, sat_outer, K
```

---

## 步骤 5：逐步提高增益 → 最终平衡

确认步骤 4 安全后，逐步提高 `pole_wc` 和 `target_speed`：

```bash
# 阶段 A：提高增益，速度不变
ros2 run bike_balance balance_node --ros-args \
  -p mode:=full -p method:=pole -p pole_wc:=-3.0 -p target_speed:=1.0

# 阶段 B：再提高增益
ros2 run bike_balance balance_node --ros-args \
  -p mode:=full -p method:=pole -p pole_wc:=-5.0 -p target_speed:=1.0

# 阶段 C：提高目标速度
ros2 run bike_balance balance_node --ros-args \
  -p mode:=full -p method:=pole -p pole_wc:=-5.0 -p target_speed:=2.0

# 阶段 D：最终参数（tutorial 默认值）
ros2 run bike_balance balance_node --ros-args \
  -p mode:=full -p method:=pole -p pole_wc:=-5.0 -p target_speed:=2.7
```

每个阶段先运行几秒，确认 `roll_deg` 收敛且不持续震荡，再进入下一阶段。

---

## 使用 LQR

切换到 LQR 方法（需要 pole-placement 调通之后再试）：

```bash
ros2 run bike_balance balance_node --ros-args \
  -p mode:=full -p method:=lqr -p target_speed:=2.7
```

LQR 的 Q/R 矩阵在 `config/balance_params.yaml` 中调整：

```yaml
lqr_Q_diag: [4.0, 100.0, 2.0]   # Q_delta, Q_phi, Q_phidot
lqr_R_diag: [10.0]               # R_steer_rate
```

- 增大 `Q_phi`（第二个值）→ 更重视抑制滚转角 → 转向更积极
- 增大 `R` → 更重视节能/平滑 → 转向更温和

---

## 使用 launch 文件

可以用 launch 文件替代手动 `ros2 run`：

```bash
# sign_check
ros2 launch bike_balance balance_launch.py mode:=sign_check

# 低增益首次测试
ros2 launch bike_balance balance_launch.py mode:=full method:=pole pole_wc:=-2.0 target_speed:=1.0
```

**注意**：launch 文件**不**自动启动 `imu_publish.py`（因为它是独立脚本，不属于 `bike_balance` 包）。IMU 需要单独在一个终端中启动。

---

## 紧急停止

```bash
# 软停止（清零速度，保持使能）
ros2 topic pub /bike_balance/enable std_msgs/Bool "data: false"

# 硬停止（如果 ROS2 还能响应）
# 直接断电或 Ctrl+C balance_node
```

自动急停条件：
- `|roll| > max_roll_angle_deg`（默认 30°）
- 转向 CAN 反馈丢失 > 0.5 秒

---

## 一次完整启动（日常调试用）

```bash
# Terminal 1: 使能 CAN（如果还没做）
sudo ip link set can0 up type can bitrate 1000000

# Terminal 2: IMU（全程运行）
python3 source/imu_publish.py

# Terminal 3: 平衡控制
ros2 run bike_balance balance_node --ros-args \
  -p mode:=full -p method:=pole -p pole_wc:=-5.0 -p target_speed:=2.7

# Terminal 4: 监视
ros2 topic echo /bike_balance/status

# 准备好后使能
ros2 topic pub /bike_balance/enable std_msgs/Bool "data: true"
```
