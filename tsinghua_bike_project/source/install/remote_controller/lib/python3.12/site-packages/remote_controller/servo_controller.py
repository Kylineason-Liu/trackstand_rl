#!/usr/bin/env python3
"""
远程遥控舵机控制节点（双舵机版，速度控制模式）
- 订阅 /joy (sensor_msgs/Joy)，RC 摇杆控制舵机运动速度
- 订阅 /servo_status (std_msgs/String)，记录当前位置与零点
- 发布 /servo_command (std_msgs/String)，命令格式与 keyboard_control.py 兼容

控制逻辑:
  摇杆中位 (死区内) → 速度 = 0，不发送指令，舵机保持当前位置
  摇杆推拉         → 速度 ∝ 摇杆偏移量，累积位置并发送 m 指令
  位置限制         → 到达 zero_point ± max_deviation 时只允许反向运动

默认映射:
  RC 通道 1 (axes[1], ch2) → 舵机 1
  RC 通道 2 (axes[2], ch3) → 舵机 2
"""
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import String


class ServoConfig:
    """单个舵机的运行时状态"""

    def __init__(self, servo_id: int, channel_position: int, name: str = ""):
        self.servo_id = servo_id
        self.channel_position = channel_position
        self.name = name or f"servo_{servo_id}"
        self.zero_point = None      # 启动时的零点位置 (第一次收到状态时记录)
        self.current_pos = 0        # 从 servo_status 反馈的当前位置
        self.target_pos = None      # 累积的目标位置 (速度积分得到)
        self.speed = 0              # 当前指令速度


class ServoController(Node):
    def __init__(self):
        super().__init__('servo_controller')

        # ---- 每个舵机的通道映射 ----
        self.declare_parameter('channel_pos_1', 1)    # 舵机1: RC通道 (默认axes[1], ch2)
        self.declare_parameter('channel_pos_2', 2)    # 舵机2: RC通道 (默认axes[2], ch3)

        # ---- 速度范围 ----
        self.declare_parameter('max_speed', 200)       # 摇杆推到底时的舵机速度
        self.declare_parameter('default_acc', 20)

        # ---- 位置限制 ----
        self.declare_parameter('max_deviation', 512)  # zero_point 两侧允许的最大偏移

        # ---- 扭矩 ----
        self.declare_parameter('channel_torque', -1)   # 扭矩开关通道，-1 = 始终开启
        self.declare_parameter('torque_on_high', True)

        # ---- RC 校准 ----
        self.declare_parameter('rc_min', 192)
        self.declare_parameter('rc_max', 1792)
        self.declare_parameter('rc_center', 992)
        self.declare_parameter('deadband', 30)

        # ---- 发布频率 ----
        self.declare_parameter('rate', 20.0)

        # 读取参数
        ch1 = self.get_parameter('channel_pos_1').value
        ch2 = self.get_parameter('channel_pos_2').value

        self.servos = [
            ServoConfig(servo_id=1, channel_position=ch1, name="servo_1"),
            ServoConfig(servo_id=2, channel_position=ch2, name="servo_2"),
        ]

        self.max_speed = self.get_parameter('max_speed').value
        self.default_acc = self.get_parameter('default_acc').value
        self.max_deviation = self.get_parameter('max_deviation').value
        self.ch_tor = self.get_parameter('channel_torque').value
        self.torque_on_high = self.get_parameter('torque_on_high').value
        self.rc_min = self.get_parameter('rc_min').value
        self.rc_max = self.get_parameter('rc_max').value
        self.rc_center = self.get_parameter('rc_center').value
        self.deadband = self.get_parameter('deadband').value
        self.rate = self.get_parameter('rate').value

        # 全局状态
        self._latest_joy = None
        self._torque_on = True

        # 发布者
        self._cmd_pub = self.create_publisher(String, 'servo_command', 10)

        # 订阅
        self._joy_sub = self.create_subscription(Joy, 'joy', self._joy_callback, 10)
        self._status_sub = self.create_subscription(
            String, 'servo_status', self._status_callback, 10
        )

        # 定时发布
        self._timer = self.create_timer(1.0 / self.rate, self._timer_callback)

        # 启动时开启两个舵机的扭矩
        for sv in self.servos:
            self._send_torque(sv.servo_id, True)

        self.get_logger().info(
            f'Servo controller ready (speed-control mode, 2 servos):\n'
            f'  servo_1: id=1, channel={ch1}\n'
            f'  servo_2: id=2, channel={ch2}\n'
            f'  max_speed={self.max_speed}, acc={self.default_acc}, '
            f'max_deviation=±{self.max_deviation}\n'
            f'  deadband={self.deadband}, rate={self.rate} Hz\n'
            f'  torque_channel={self.ch_tor}'
        )

    # ---------- Joy 回调 ----------
    def _joy_callback(self, msg: Joy):
        self._latest_joy = msg

    # ---------- 定时器回调 (限速发送) ----------
    def _timer_callback(self):
        if self._latest_joy is None:
            return

        axes = self._latest_joy.axes
        if len(axes) < 16:
            self.get_logger().warn(f'Joy has only {len(axes)} axes, expected 16')
            return

        # --- 扭矩开关 ---
        if self.ch_tor >= 0 and self.ch_tor < len(axes):
            raw = axes[self.ch_tor]
            torque_on = (raw >= self.rc_center) if self.torque_on_high else (raw < self.rc_center)
            if torque_on != self._torque_on:
                self._torque_on = torque_on
                for sv in self.servos:
                    self._send_torque(sv.servo_id, torque_on)

        if not self._torque_on:
            return

        # --- 每个舵机独立处理 ---
        for sv in self.servos:
            self._process_servo(sv, axes)

    def _process_servo(self, sv: ServoConfig, axes: list):
        """速度控制模式：RC → 速度 → 积分位置 → 发送 m 指令"""
        if sv.channel_position < 0 or sv.channel_position >= len(axes):
            return

        raw = axes[sv.channel_position]

        # 死区：中位 → 速度 = 0，不发送
        if abs(raw - self.rc_center) < self.deadband:
            sv.speed = 0
            return

        # RC 值 → 速度 (线性映射，中位以上正，中位以下负)
        speed = self._map_rc_to_speed(raw)
        sv.speed = speed

        # 等待零点确立 (第一次收到 servo_status)
        if sv.zero_point is None:
            self.get_logger().info(
                f'{sv.name}: waiting for zero point...', throttle_duration_sec=5.0
            )
            return

        # 初始化目标位置 (从当前反馈位置开始)
        if sv.target_pos is None:
            sv.target_pos = sv.current_pos
            self.get_logger().info(f'{sv.name}: target initialized to {sv.target_pos}')

        # 位置限幅：到达 ±limit 时只允许反向运动 (与 motor_controller 一致)
        lo = sv.zero_point - self.max_deviation
        hi = sv.zero_point + self.max_deviation
        if sv.servo_id == 2:
            if sv.target_pos >= hi and speed > 0:
                speed = 0
            elif sv.target_pos <= (lo + 300) and speed < 0:
                speed = 0
        if sv.servo_id == 1:
            if sv.target_pos >= (hi - 300) and speed > 0:
                speed = 0
            elif sv.target_pos <= lo and speed < 0:
                speed = 0

        if speed == 0:
            return

        # 积分速度 → 目标位置
        dt = 1.0 / self.rate
        sv.target_pos += speed * dt

        # 钳位
        sv.target_pos = max(lo, min(hi, sv.target_pos))

        # 发送 m <id> <pos> <speed> <acc>
        abs_speed = max(1, int(abs(speed)))
        self._send_position(sv.servo_id, int(sv.target_pos), abs_speed, self.default_acc)

    # ---------- 伺服状态回调 ----------
    def _status_callback(self, msg: String):
        """从 servo_driver 接收舵机状态，记录零点与当前位置"""
        try:
            data = json.loads(msg.data)
            sid = data.get('id')
            # 只关心我们控制的舵机
            sv = next((s for s in self.servos if s.servo_id == sid), None)
            if sv is None:
                return

            sv.current_pos = data['position']

            # 第一次收到状态 → 记录为零点 (与 motor_controller 的 steer_origin 一致)
            if sv.zero_point is None:
                sv.zero_point = sv.current_pos
                self.get_logger().info(
                    f'{sv.name}: zero point set to {sv.zero_point}, '
                    f'limits=[{sv.zero_point - self.max_deviation}, '
                    f'{sv.zero_point + self.max_deviation}]'
                )

        except Exception:
            pass

    # ---------- 映射函数 ----------
    def _map_rc_to_speed(self, raw: int) -> float:
        """RC 通道值 → 舵机速度 (中位=0, 正向=正速度, 反向=负速度)"""
        offset = raw - self.rc_center
        if offset > 0:
            # 正向：映射到 [0, max_speed]
            return self._map_range(offset, self.deadband,
                                   self.rc_max - self.rc_center, 0, self.max_speed)
        else:
            # 反向：映射到 [-max_speed, 0]
            return self._map_range(offset, self.rc_min - self.rc_center,
                                   -self.deadband, -self.max_speed, 0)

    @staticmethod
    def _map_range(value, in_min, in_max, out_min, out_max) -> float:
        return (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min

    # ---------- 命令发送 ----------
    def _send_position(self, servo_id: int, pos: int, speed: int, acc: int):
        text = f'm {servo_id} {pos} {speed} {acc}'
        self._publish(text)

    def _send_torque(self, servo_id: int, enable: bool):
        text = f't {servo_id} {1 if enable else 0}'
        self._publish(text)

    def _publish(self, text: str):
        msg = String()
        msg.data = text
        self._cmd_pub.publish(msg)

    # ---------- 清理 ----------
    def destroy_node(self):
        self.get_logger().info('Shutting down, disabling servo torque...')
        for sv in self.servos:
            self._send_torque(sv.servo_id, False)
        # 刷新待发送的消息，确保失能指令能发出去
        rclpy.spin_once(self, timeout_sec=0.5)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ServoController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
