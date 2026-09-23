# 软件延迟测量方案 — 修改说明

## 背景

为了在不启动任何电机的情况下测量 ROS2 控制管线的软件延迟，我们对 5 个文件进行了以下修改。

---

## 延迟链路总览

```
IMU 串口数据就绪
  │
  ├─[A]─ roll_publisher._tick() 读取 → 发布 /roll_info (附 publish_ts)
  │      └── 延迟来源: 定时器相位抖动 (0–5ms @ 200Hz)
  │
  ├─[B]─ ROS2 transport + executor 调度
  │      └── 延迟来源: pub→sub 传输 + 回调排队
  │
  ├─[C]─ balance_node._roll_cb() 触发 (记录 cb_ts)
  │      └── 延迟来源: 几乎为 0 (仅赋值)
  │
  ├─[D]─ 等待 balance_node._tick() 触发
  │      └── 延迟来源: 跨节点定时器相位差 (0–5ms @ 200Hz)
  │
  ├─[E]─ LQR + 内环 PD 计算
  │      └── 延迟来源: 计算耗时 (~0.1–0.2ms)
  │
  └─[F]─ 控制指令就绪 (dry_run 模式下跳过 CAN 发送，仅发布)
```

---

## 修改清单

### 1. `new_main_balance.py` — 核心修改

#### 1.1 新增 `dry_run` 参数

**位置**: `__init__` 方法

**作用**: 当 `dry_run:=true` 时，跳过所有 CAN 发送，但保持完整的控制计算和话题发布。

```python
# 参数声明
self.declare_parameter("dry_run", False)

# 读取
self._dry_run: bool = self.get_parameter("dry_run").value
```

**影响范围**: 四个 CAN 操作方法均在入口处添加了 `if self._dry_run: return`：

| 方法 | 作用 | dry_run 时行为 |
|---|---|---|
| `_send_speed()` | 发送 C1 速度指令 | 直接返回 |
| `_send_q_current()` | 发送 C0 扭矩指令 | 直接返回 |
| `_send_cf()` | 发送 CF 电机使能/失能 | 直接返回 |
| `_stop_motors()` | 停机（C1=0 + CF） | 直接返回 |

**使用方式**:
```bash
ros2 run new_balance balance_node --ros-args -p dry_run:=true -p mode:=full
```

#### 1.2 新增延迟计时字段

**位置**: `__init__` → Runtime state 区域

新增 9 个实例变量：

```python
# 从 publisher 消息中提取的发布时间戳 (float, epoch seconds)
self._roll_pub_ts: float = 0.0
self._steer_pub_ts: float = 0.0
self._rear_pub_ts: float = 0.0
self._rc_pub_ts: float = 0.0

# subscriber 回调触发时间戳
self._roll_cb_ts: float = 0.0
self._steer_cb_ts: float = 0.0
self._rear_cb_ts: float = 0.0
self._rc_cb_ts: float = 0.0

# 聚合后的延迟数据 (每 tick 更新，通过 status 发布)
self._timing: dict[str, float] = {}
```

#### 1.3 修改 subscriber 回调

**修改的回调**: `_roll_cb`, `_steer_cb`, `_rear_cb`, `_rc_cb`

每个回调现在做两件额外工作：
1. 记录回调触发时间 `_xxx_cb_ts = time.time()`
2. 从消息尾部提取发布时间戳 `_xxx_pub_ts`（兼容旧版消息，只有数据长度足够时才读取）

以 `_roll_cb` 为例：
```python
def _roll_cb(self, msg: Float32MultiArray) -> None:
    if len(msg.data) >= 7:
        self._roll_cb_ts = time.time()           # ← 新增
        # ... 原有的数据解析 ...
        self._roll_ts = self._roll_cb_ts          # ← 改用 cb_ts
        if len(msg.data) >= 8:                    # ← 新增：兼容判断
            self._roll_pub_ts = float(msg.data[7]) # ← 新增：提取 pub_ts
```

#### 1.4 新增 `_update_timing()` 方法

**位置**: `_tick` 之前

计算每对发布/接收时间戳之间的延迟，存入 `self._timing`：

| 字段 | 含义 | 单位 |
|---|---|---|
| `roll_pub_to_cb_ms` | /roll_info 发布到 balance_node 收到 | ms |
| `roll_cb_to_tick_ms` | balance_node 收到到 tick 开始计算 | ms |
| `roll_age_ms` | /roll_info 发布到 tick 开始计算 (总传输延迟) | ms |
| `steer_pub_to_cb_ms` | /steer_info 发布到收到 | ms |
| `steer_cb_to_tick_ms` | 收到到计算 | ms |
| `steer_age_ms` | /steer_info 发布到 tick 开始计算 | ms |
| `rear_pub_to_cb_ms` | /rear_info 发布到收到 | ms |
| `rear_age_ms` | /rear_info 发布到 tick 开始计算 | ms |
| `rc_pub_to_cb_ms` | /rc_info 发布到收到 (仅 vctrl=rc) | ms |
| `rc_age_ms` | /rc_info 发布到 tick 开始计算 (仅 vctrl=rc) | ms |
| `compute_us` | tick 内部计算耗时 (LQR + PD + LPF) | µs |

当时间戳缺失时（如旧版 publisher），对应值设为 `-1.0`。

#### 1.5 在三个 tick 方法中调用 `_update_timing`

- `_tick_sign_check()` → 第 1 行
- `_tick_inner_loop()` → 第 1 行
- `_tick_full()` → 第 2 行

每个 tick 方法末尾追加 `compute_us` 计算：
```python
self._timing["compute_us"] = round((time.time() - now) * 1e6, 0)
```

#### 1.6 扩展 status JSON

状态 JSON `/new_balance/status` 新增 `timing` 字段：
```json
{
  "...原有字段...": "...",
  "timing": {
    "roll_pub_to_cb_ms": 0.512,
    "roll_cb_to_tick_ms": 2.341,
    "roll_age_ms": 2.853,
    "steer_pub_to_cb_ms": 0.891,
    "steer_cb_to_tick_ms": 1.732,
    "steer_age_ms": 2.623,
    "rear_pub_to_cb_ms": 0.445,
    "rear_age_ms": 2.198,
    "rc_pub_to_cb_ms": 1.234,
    "rc_age_ms": 3.012,
    "compute_us": 156.0
  }
}
```

#### 1.7 扩展 debug 数组

`/new_balance/debug` (200 Hz) 末尾新增 3 个 float：
- 第 12 位: `roll_age_ms` — /roll_info 发布到开始计算的延迟 (ms)
- 第 13 位: `steer_age_ms` — /steer_info 发布到开始计算的延迟 (ms)
- 第 14 位: `compute_us` — 计算耗时 (µs)

---

### 2. `roll_publisher.py` — 附加发布时间戳

**修改点**:
1. 新增 `import time`
2. `msg.data` 末尾追加 `time.time()` → 数据从 7 元素变为 8 元素

```python
msg.data = [
    self._roll_f, self._pitch_f, self._yaw_f,
    self._roll_rate_f,
    self._p_f, self._q_f, self._r_f,
    time.time(),   # ← 第 8 个元素：发布时间戳
]
```

---

### 3. `steer_publisher.py` — 附加发布时间戳

**修改点**: `msg.data` 末尾追加 `time.time()` → 数据从 2 元素变为 3 元素

```python
msg.data = [float(angle), float(rate), time.time()]
```

---

### 4. `rear_publisher.py` — 附加发布时间戳

**修改点**: `msg.data` 末尾追加 `time.time()` → 数据从 1 元素变为 2 元素

```python
msg.data = [float(rate), time.time()]
```

---

### 5. `rc_publisher.py` — 附加发布时间戳

**修改点**:
1. 新增 `import time`
2. `msg.data` 末尾追加 `time.time()` → 数据从 2 元素变为 3 元素

```python
msg.data = [float(self._drive_rpm), float(self._steer_rad), time.time()]
```

---

## 兼容性

- **新旧 publisher 混用**: balance_node 的回调通过 `len(msg.data)` 判断是否有时间戳。如果旧版 publisher 不发送时间戳，对应延迟字段设为 `-1.0`，不影响原有功能。
- **dry_run 为非破坏性参数**: 默认 `false`，不影响现有启动脚本。只在显式指定 `-p dry_run:=true` 时才生效。
- **debug 数组扩展**: 在原数组末尾追加，不改变已有元素的顺序和含义。

---

## 使用示例

### 查看软件管线延迟

```bash
# 启动所有 publisher (无需 CAN 总线、无需电机)
ros2 run new_balance roll_publisher --ros-args -p port:=/dev/ttyUSB0 &
ros2 run new_balance steer_publisher --ros-args -p can_interface:=can0 &   # 需要 CAN up
ros2 run new_balance rear_publisher --ros-args -p can_interface:=can0 &    # 需要 CAN up
ros2 run new_balance rc_publisher --ros-args -p port:=/dev/ttyUSB1 &

# 启动 balance_node (dry_run 模式，不发送任何 CAN 指令)
ros2 run new_balance balance_node --ros-args -p dry_run:=true -p mode:=full

# 监视延迟数据 (2 Hz 聚合)
ros2 topic echo /new_balance/status | grep -A 15 timing

# 或查看高频 debug 数据 (200 Hz)
ros2 topic echo /new_balance/debug
```

### 不依赖 CAN 的延迟测量 (仅测 roll 管线)

如果 CAN 总线未启动，steer/rear publisher 无法工作，此时仅 roll 管线的延迟有效：

```bash
ros2 run new_balance roll_publisher --ros-args -p port:=/dev/ttyUSB0 &
ros2 run new_balance balance_node --ros-args -p dry_run:=true -p mode:=sign_check

ros2 topic echo /new_balance/status | grep -A 15 timing
```

## 延迟数据解读

| 指标 | 正常范围 | 异常含义 |
|---|---|---|
| `roll_pub_to_cb_ms` | < 1 ms | ROS2 executor 调度延迟过大，检查是否过载 |
| `roll_cb_to_tick_ms` | 0–5 ms | 跨节点定时器相位差，受轮询频率限制 |
| `roll_age_ms` | 0–6 ms | 从数据产生到开始计算的总延迟 |
| `compute_us` | 50–200 µs | LQR + PD 计算耗时，如果 > 1ms 说明异常 |
