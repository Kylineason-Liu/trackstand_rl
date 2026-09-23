#!/usr/bin/env python3
"""
舵机硬件驱动节点
- 打开串口，连接舵机
- 订阅 /servo_command (std_msgs/String)，解析指令并发送给舵机
- 定时发布 /servo_status (std_msgs/String, JSON格式)
"""
import json
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sms_sts import SMS_STS

CMD_MOVE = 'm'

# 默认顺序: id position speed acceleration
DEFAULT_PARAMS = [2, 0, 500, 10]


class ServoDriver(Node):

    def __init__(self):
        super().__init__('servo_driver')

        self.declare_parameter('serial_port', '/dev/ttyAMA2')
        self.declare_parameter('baud_rate', 1000000)
        self.declare_parameter('servo_ids', [2])
        self.declare_parameter('publish_rate', 10.0)

        serial_port = self.get_parameter('serial_port').value
        baud_rate = self.get_parameter('baud_rate').value
        self._servo_ids = self.get_parameter('servo_ids').value
        publish_rate = self.get_parameter('publish_rate').value

        self._sms = SMS_STS()
        self._serial_ok = False
        try:
            if self._sms.begin(baud_rate, serial_port):
                self._serial_ok = True
                self.get_logger().info(
                    f'Serial OK: {serial_port} @ {baud_rate}'
                )
            else:
                self.get_logger().error(f'Failed to open: {serial_port}')
        except Exception as e:
            self.get_logger().error(f'Serial error: {e}')

        self._status_pub = self.create_publisher(String, 'servo_status', 10)
        self._cmd_sub = self.create_subscription(
            String, 'servo_cmd', self._cmd_callback, 10
        )

        period = 1.0 / publish_rate
        self._timer = self.create_timer(period, self._timer_callback)

        self.get_logger().info(
            f'Driver ready. IDs={self._servo_ids}, '
            f'serial_ok={self._serial_ok}'
        )

    def _timer_callback(self):
        if not self._serial_ok:
            return
        for sid in self._servo_ids:
            if self._sms.feedback(sid) == -1:
                self.get_logger().warn(
                    f'Feedback fail ID={sid}', throttle_duration_sec=5.0
                )
                continue

            data = {
                'id': sid,
                'position': self._sms.read_pos(-1),
                'speed': self._sms.read_speed(-1),
                'load': self._sms.read_load(-1),
                'voltage': self._sms.read_voltage(-1),
                'temperature': self._sms.read_temper(-1),
                'moving': self._sms.read_move(-1),
                'current': self._sms.read_current(-1),
            }
            msg = String()
            msg.data = json.dumps(data)
            self._status_pub.publish(msg)

    def _cmd_callback(self, msg):
        if not self._serial_ok:
            self.get_logger().warn(
                'No serial, ignoring command', throttle_duration_sec=5.0
            )
            return

        line = msg.data.strip()
        if not line:
            return

        self.get_logger().info(f'Got: {line}')

        parts = line.split()
        cmd = parts[0].lower()

        if cmd == CMD_MOVE:
            self._handle_move(parts)
        elif cmd == 't' and len(parts) >= 3:
            sid, enable = int(parts[1]), int(parts[2])
            self._sms.enable_torque(sid, enable)
            self.get_logger().info(f'Torque ID={sid} {"ON" if enable else "OFF"}')
        else:
            self.get_logger().warn(f'Unknown cmd: {line}')

    def _handle_move(self, parts):
        n = len(parts) - 1
        if n < 1:
            self.get_logger().warn('Move: need at least id')
            return
        params = DEFAULT_PARAMS[:]
        for i in range(min(n, 4)):
            params[i] = int(parts[i + 1])
        sid = params[0]
        pos, spd, acc = params[1:]
        if pos != -1:
            self._sms.write_pos_ex(sid, pos, spd, acc)
            self.get_logger().info(
                f'Move ID={sid} pos={pos} speed={spd} acc={acc}'
            )

    def destroy_node(self):
        if self._serial_ok:
            for sid in self._servo_ids:
                self._sms.enable_torque(sid, 0)
                self.get_logger().info(f'Torque ID={sid} OFF (shutdown)')
            self._sms.end()
        super().destroy_node()


def main():
    rclpy.init()
    node = ServoDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
