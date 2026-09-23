#!/usr/bin/env python3
import re
import struct
import subprocess
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')

        self.declare_parameter('can_interface', 'can0')
        self.declare_parameter('can_id_drive', '102')
        self.declare_parameter('can_id_steer', '101')
        self.declare_parameter('steer_feedback_id', '102')
        self.declare_parameter('max_rpm', 500.0)
        self.declare_parameter('steer_angle_limit', 30.0)

        self.can_iface = self.get_parameter('can_interface').value
        self.can_id_drive = self.get_parameter('can_id_drive').value
        self.can_id_steer = self.get_parameter('can_id_steer').value
        self.steer_feedback_id = self.get_parameter('steer_feedback_id').value
        self.max_rpm = self.get_parameter('max_rpm').value
        self.steer_angle_limit = self.get_parameter('steer_angle_limit').value

        # 转向角度追踪
        self._steer_origin = None       # 启动时确立的原点
        self._steer_angle = 0.0         # 当前绝对角度（相对原点）
        self._angle_lock = threading.Lock()

        # 启动 CAN 读取线程
        self._candump_proc = None
        self._start_can_reader()

        # 订阅 cmd_vel
        self.sub = self.create_subscription(Twist, 'cmd_vel', self._cmd_vel_cb, 10)

        self.get_logger().info(
            f'Motor controller ready: can={self.can_iface}, '
            f'drive_id={self.can_id_drive}, steer_id={self.can_id_steer}, '
            f'max_rpm={self.max_rpm}, steer_angle_limit=±{self.steer_angle_limit}°'
        )

    # ---------- CAN 读取 ----------
    def _start_can_reader(self):
        cmd = f'candump {self.can_iface}'
        self._candump_proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1
        )
        t = threading.Thread(target=self._read_loop, daemon=True)
        t.start()
        self.get_logger().info(f'CAN reader started: {cmd}')

    def _read_loop(self):
        # 解析 candump 输出: can0  102   [8]  AA BB CC DD EE FF 00 00
        pattern = re.compile(r'\S+\s+(\S+)\s+\[\d+\]\s+(.*)')
        for line in self._candump_proc.stdout:
            m = pattern.match(line)
            if not m:
                continue
            can_id = m.group(1)
            if can_id != self.steer_feedback_id:
                continue

            hex_str = m.group(2).strip().replace(' ', '')
            if len(hex_str) < 8:
                continue

            try:
                data = bytes.fromhex(hex_str[:8])
                raw = struct.unpack('<i', data)[0]
                angle = raw * 360.0 / 16384.0     # 编码器分辨率 16384 counts/rev
            except ValueError:
                continue

            with self._angle_lock:
                if self._steer_origin is None:
                    self._steer_origin = angle
                    self.get_logger().info(f'Steer origin set: {self._steer_origin:.2f}°')
                self._steer_angle = angle - self._steer_origin

    # ---------- cmd_vel 回调 ----------
    def _cmd_vel_cb(self, msg: Twist):
        # 驱动电机: linear.x = 目标转速
        drive_rpm = msg.linear.x
        drive_rpm = max(-self.max_rpm, min(self.max_rpm, drive_rpm))
        self._send_drive(int(drive_rpm))

        # 转向电机: linear.y = 转向速度
        steer_speed = msg.linear.y
        steer_speed = max(-self.max_rpm, min(self.max_rpm, steer_speed))

        # 角度限幅：到达 ±limit 时只允许反方向运动
        with self._angle_lock:
            angle = self._steer_angle
        if angle >= self.steer_angle_limit and steer_speed > 0:
            steer_speed = 0
        elif angle <= -self.steer_angle_limit and steer_speed < 0:
            steer_speed = 0

        self._send_steer(int(steer_speed))

    # ---------- CAN 发送 ----------
    def _send_drive(self, rpm: int):
        value = rpm * 100
        data_hex = struct.pack('<i', value).hex()
        cmd = f'cansend {self.can_iface} {self.can_id_drive}#C1{data_hex}'
        self.get_logger().info(f'Drive RPM={rpm} -> {cmd}')
        subprocess.run(cmd, shell=True, capture_output=True)

    def _send_steer(self, speed: int):
        value = speed * 20
        data_hex = struct.pack('<i', value).hex()
        cmd = f'cansend {self.can_iface} {self.can_id_steer}#C1{data_hex}'
        self.get_logger().info(f'Steer speed={speed} -> {cmd}')
        subprocess.run(cmd, shell=True, capture_output=True)

    def _send_stop(self, can_id: str):
        cmd = f'cansend {self.can_iface} {can_id}#CF'
        self.get_logger().info(f'Motor stop {can_id}: {cmd}')
        subprocess.run(cmd, shell=True, capture_output=True)

    def shutdown(self):
        self.get_logger().info('Shutting down: set speed=0 then stop motors...')
        self._send_drive(0)
        self._send_steer(0)
        self._send_stop(self.can_id_drive)
        self._send_stop(self.can_id_steer)
        if self._candump_proc:
            self._candump_proc.terminate()


def main(args=None):
    rclpy.init(args=args)
    node = MotorController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
