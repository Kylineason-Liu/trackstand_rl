# `nominal_bike_control`：名义自行车平衡控制基线

这个目录提供一套独立、简洁的自行车平衡控制基线：使用随车速更新的三状态横滚—转向模型，直接对测得的完整状态做极点配置或 LQR 反馈。

控制器本身不包含扰动模型、扰动观测器或扰动前馈补偿。它适合用来学习状态反馈、建立对照实验，以及验证“只依赖名义模型时能达到什么效果”。

## 在当前工作区中的位置

当前项目中各部分的关系如下：

```text
model/orange_bike/orange_bike_horizontal.xml
    └── MuJoCo 非线性多刚体被控对象
        ├── 转向电机：力矩输入
        └── 后轮电机：力矩输入

nominal_bike_control/bicycle_model.py
    └── 控制器内部使用的三状态线性名义模型

nominal_bike_control/controller.py
    └── 极点配置 / LQR / 一致平衡点求解 / 指令限幅

tutorials/07_MuJoCo实战_50Hz双环自行车平衡.ipynb
    └── 50 Hz 外环状态反馈 + 转向速度 PI 内环 + MuJoCo 实车模型

tests/test_nominal_bike_control.py
    └── 控制包单元测试
```

从项目根目录启动 Python 或 Jupyter，即可直接导入本包；当前目录没有单独的安装脚本。

核心依赖是 `numpy` 和 `scipy`。运行第 7 讲的完整仿真还需要 `mujoco`、`matplotlib` 和 `mediapy`。

## 名义自行车模型

模型状态定义为

```text
x[0] = delta     转向角，单位 rad
x[1] = phi       车身滚转角，单位 rad
x[2] = phi_dot   车身滚转角速度，单位 rad/s
```

控制输入为

```text
u = delta_dot    期望转向角速度，单位 rad/s
```

连续时间模型为

```text
x_dot = A x + Bu u
```

其中

```text
A = [[0,  0,  0],
     [0,  0,  1],
     [a2, a4, 0]]

Bu = [[1],
      [0],
      [a1]]
```

三个速度相关系数为

```text
a1 = m a h v cos(lambda) / (b Ib)
a2 = (m v^2 h - m a c g) cos(lambda) / (b Ib)
a4 = m g h / Ib
```

参数含义：

- `a`：后轮接地点到整车质心的水平距离；
- `b`：前后轮轴距；
- `c`：拖曳距；
- `h`：质心相对地面的高度；
- `m`：总质量；
- `Ib`：绕地面纵向轴的等效横滚惯量；
- `lambda`：转向轴倾角；
- `v`：前进速度。

`LinearBicycleModel` 会根据前进速度重新计算 `A` 和 `Bu`。默认把小于 `1 m/s` 的正向速度按 `1 m/s` 调度，避免起步时模型退化；负速度保留负号。这个模型主要用于正向行驶和平衡控制。

## 与 MuJoCo XML 的关系

两者不是同一个模型的两种文件格式：

- XML 是带自由基座、刚体惯量、轮子旋转、地面接触和摩擦的非线性被控对象；
- `bicycle_model.py` 是控制器内部的三状态、小角度、定车速近似；
- XML 的转向执行器接收力矩，名义模型的输入则是转向角速度。

当前默认 `BicycleParameters` 沿用了早期名义参数，并不是从现有 XML 严格线性化得到的。当前版本的主要参数对照如下：

| 参数 | 默认名义模型 | 当前 XML 粗略提取值 |
|---|---:|---:|
| 总质量 | `5.000 kg` | `1.601 kg` |
| 轴距 | `0.271 m` | `0.27903 m` |
| 后轮到质心距离 | `0.117 m` | 约 `0.13348 m` |
| 质心高度 | `0.292 m` | 约 `0.12550 m` |
| 等效横滚惯量 | `0.56843 kg·m²` | 约 `0.03282 kg·m²` |

因此，默认模型适合作为控制方法基线，但不能称为当前 XML 的精确线性化模型。若希望先用当前 XML 的几何和惯量构造一个更接近的初值，可以显式传入参数：

```python
from nominal_bike_control import BicycleParameters, ScaleBikeModel

xml_approx_parameters = BicycleParameters(
    rear_contact_to_com=0.13348,
    wheelbase=0.27903,
    trail=0.0,
    com_height=0.12550,
    mass=1.601,
    roll_inertia=0.03282,
    steering_axis_angle=0.0,
)

bike = ScaleBikeModel(
    dt=0.02,
    parameters=xml_approx_parameters,
    forward_speed=2.0,
)
```

这些数值只是从当前 XML 聚合得到的降阶初值，仍未包含轮胎接触、轮子陀螺效应和转向电机动态。需要高精度模型时，应围绕不同直线行驶速度做 MuJoCo 小扰动辨识或数值线性化。

## 最小使用示例：50 Hz 直线平衡

下面的例子使用默认名义参数、重复极点配置和零转向角参考：

```python
import numpy as np

from nominal_bike_control import NominalBikeController, ScaleBikeModel

control_dt = 0.02  # 50 Hz

bike = ScaleBikeModel(
    dt=control_dt,
    track_roll=False,
    min_forward_speed=1.0,
    forward_speed=2.0,
)

controller = NominalBikeController(
    bike.sys,
    method="place_multiple_poles",
    wc=-5.0,
    u_min=-10.0,
    u_max=10.0,
)

# 每个控制周期从传感器获得完整状态。
measured_state = np.array([
    [steering_angle],
    [roll_angle],
    [roll_rate],
])

# 随实测前进速度更新模型和反馈增益。
bike.update_system_parameters(
    forward_speed,
    min_forward_speed=1.0,
)
controller.update_system_and_gain(bike.sys)

# track_roll=False 时，被控输出是转向角。
steering_reference = np.array([[0.0]])
steering_velocity_command = controller.step(
    steering_reference,
    measured_state,
)[0, 0]
```

输入和输出的数组形状如下：

| 变量 | 形状 | 含义 |
|---|---:|---|
| `measured_state` | `(3, 1)` | `[转向角, 滚转角, 滚转角速度]` |
| `steering_reference` | `(1, 1)` | 被控输出参考，不是三维状态参考 |
| 控制器返回值 | `(1, 1)` | 期望转向角速度 |

`u_min` 和 `u_max` 限制的是外环转向角速度指令，单位为 `rad/s`，不是 XML 中的转向电机力矩上限。

## 参考量与动力学一致平衡点

控制律为

```text
u = u_eq + K (x_eq - x)
```

其中平衡状态和稳态输入由以下方程共同确定：

```text
A x_eq + Bu u_eq = 0
Co x_eq = output_reference
```

`step()` 和 `stepAndGetControl()` 接收的是 `Co @ x` 对应的被控输出参考。默认 `track_roll=False`，所以 `Co=[1, 0, 0]`，参考量表示转向角。

非零稳态转向角通常需要相应的车身倾角。因此，直接指定完整状态 `[delta_ref, 0, 0]` 往往不符合动力学。可以先求解一致平衡点：

```python
reference = np.array([[0.1]])
equilibrium = controller.solve_equilibrium(reference)

print(equilibrium.state)
print(equilibrium.control)
print(equilibrium.residual_norm)
```

如果确实要传入完整三维状态参考，应使用 `step_with_state_reference()`；控制器会验证该参考是否满足稳态动力学：

```python
command = controller.step_with_state_reference(
    equilibrium.state,
    measured_state,
    control_reference=equilibrium.control,
)
```

不一致的完整状态参考会抛出 `ValueError`，而不是被静默当成合法平衡点。

## 三种反馈增益方法

### 重复极点配置

```python
controller = NominalBikeController(
    bike.sys,
    method="place_multiple_poles",
    wc=-5.0,
)
```

三个闭环极点都配置在 `wc`。当前实现针对单输入系统使用 Ackermann 公式，因此支持重复极点。

### 不同极点配置

```python
controller = NominalBikeController(
    bike.sys,
    method="place_distinct_poles",
    poles_ctr=[-3.0, -5.0, -7.0],
)
```

`poles_ctr` 的元素数量必须等于状态数。

### 连续时间 LQR

```python
controller = NominalBikeController(
    bike.sys,
    method="lqr",
    Qc=np.diag([2.0, 40.0, 4.0]),
    Rc=np.array([[1.0]]),
)
```

- 增大 `Qc[1,1]`：更重视抑制滚转角；
- 增大 `Qc[2,2]`：更重视抑制滚转角速度；
- 增大 `Rc[0,0]`：减少激烈的转向速度指令。

每次调用 `bike.update_system_parameters()` 后，都应调用 `controller.update_system_and_gain()`，保证反馈增益与当前调度速度相匹配。

## 接入当前 MuJoCo 模型

当前 XML 的物理积分步长为 `0.0005 s`，教程使用 `0.02 s` 控制周期，因此每次控制更新之间约执行 40 个 MuJoCo 积分步。

控制链为

```text
转向角、滚转角、滚转角速度、前进速度
                    ↓
        NominalBikeController 外环
                    ↓
         期望转向角速度 rad/s
                    ↓
             转向速度 PI 内环
                    ↓
              转向电机力矩 N·m
                    ↓
        orange_bike_horizontal.xml
```

后轮速度由另一条 PI/P 控制链产生后轮电机力矩。XML 当前限制转向电机力矩为 `[-5, 5] N·m`，后轮电机力矩为 `[-6, 6] N·m`。

完整可运行实现位于：

```text
tutorials/07_MuJoCo实战_50Hz双环自行车平衡.ipynb
```

建议先运行第 3、4 讲理解极点配置和 LQR，再运行第 7 讲观察控制器如何接入 MuJoCo。

## 限制与预期现象

- 控制器要求完整测量三个状态，`Cm` 必须是可逆的方阵；
- 包内没有状态观测器，滚转角速度需要直接测量或在包外估计；
- IMU 姿态和角速度必须先从传感器坐标系旋转到自行车坐标系；不能在安装轴不重合时直接使用原始 `gyro_x`；
- 包内没有扰动观测器，恒定外扰和模型误差可能造成稳态偏差；
- 名义模型没有转向电机动态，因此必须由速度内环连接到 XML 的力矩执行器；
- 名义模型只在直立、小角度和给定车速附近有效；
- 低速、轮胎打滑、离地、大转角和碰撞情况下，实际 MuJoCo 动力学可能明显偏离名义模型。

IMU 安装旋转、零偏和姿态组合的完整示例见 `tutorials/06_IMU安装误差_坐标标定与状态构造.ipynb`。

## 运行测试

在项目根目录执行：

```bash
python -m unittest -v tests.test_nominal_bike_control
```

当前测试覆盖：

- 重复极点配置和随车速更新增益；
- 一致平衡点满足名义动力学；
- 拒绝不一致的完整状态参考；
- 直接使用完整状态反馈并正确执行指令限幅。
