#!/usr/bin/env python3
"""
键盘控制节点
- 读取键盘输入，发布指令到 /servo_command (std_msgs/String)
- 订阅 /servo_status (std_msgs/String, JSON)，显示舵机状态
"""
import json
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

HELP = """
══════════════════════════════════════════════════════════════════════
  舵机键盘控制 — 命令发送到 /servo_command 话题，由 servo_driver 执行
══════════════════════════════════════════════════════════════════════

  控制命令:
  ──────────────────────────────────────────────────────────────────
  <pos> [speed] [acc]    移动到目标位置
                          pos   : 0~4095 (0°~360°)
                          speed : 0~4095, 默认 500
                          acc   : 0~255,   默认 10
                          例: 2048
                          例: 2048 1000 20

  t <0|1>                扭矩开关
                          1 = 扭矩使能（上电），0 = 释放（可自由转动）
                          例: t 1

  查询:
  ──────────────────────────────────────────────────────────────────
  s                      显示所有舵机状态
                          (位置/速度/负载/电压/温度/运动/电流)

  设置:
  ──────────────────────────────────────────────────────────────────
  r <id>                 切换当前控制的舵机 ID
                          例: r 1

  其他:
  ──────────────────────────────────────────────────────────────────
  h                      显示帮助
  q                      退出

  底层协议:
  ──────────────────────────────────────────────────────────────────
  keyboard_control 发布文本指令到 /servo_command 话题:
    m <id> <pos> <speed> <acc>     → 伺服驱动执行
    t <id> <0|1>                   → 扭矩控制
  servo_driver 返回 JSON 状态到 /servo_status 话题"""


class KeyboardControl(Node):

    def __init__(self):
        super().__init__('keyboard_control')

        self._cmd_pub = self.create_publisher(String, 'servo_command', 10)
        self._status_sub = self.create_subscription(
            String, 'servo_status', self._status_callback, 10
        )

        self._servo_id = 2  # 默认控制舵机2
        self._default_speed = 500
        self._default_acc = 10
        self._latest = {}  # id -> status dict
        self._running = True

        self.get_logger().info('Keyboard control ready')
        print(f'当前控制舵机ID: {self._servo_id}')
        print(HELP)

        self._thread = threading.Thread(target=self._input_loop, daemon=True)
        self._thread.start()

    def _status_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self._latest[data['id']] = data
        except Exception:
            pass

    def _input_loop(self):
        while self._running and rclpy.ok():
            try:
                line = input('> ').strip()
                if not line:
                    continue
                self._dispatch(line)
            except (EOFError, KeyboardInterrupt):
                break

    def _dispatch(self, line):
        parts = line.split()

        if parts[0] == 'q':
            self._running = False
            print('退出...')
            return

        if parts[0] == 'h':
            print(HELP)
            return

        if parts[0] == 's':
            self._show_status()
            return

        if parts[0] == 'r' and len(parts) >= 2:
            self._servo_id = int(parts[1])
            print(f'切换到舵机ID: {self._servo_id}')
            return

        if parts[0] == 't' and len(parts) >= 2:
            enable = int(parts[1])
            self._pub(f't {self._servo_id} {enable}')
            return

        # 默认: 移动指令  <pos> [speed] [acc]
        try:
            pos = int(parts[0])
            spd = int(parts[1]) if len(parts) > 1 else self._default_speed
            acc = int(parts[2]) if len(parts) > 2 else self._default_acc
            self._pub(f'm {self._servo_id} {pos} {spd} {acc}')
        except ValueError:
            print(f'未知命令: {line}，输入 h 查看帮助')

    def _pub(self, text):
        msg = String()
        msg.data = text
        self._cmd_pub.publish(msg)
        print(f'已发送: {text}')

    def _show_status(self):
        if not self._latest:
            print('暂无舵机状态数据')
            return
        print(f'{"ID":>3} {"Pos":>6} {"Spd":>6} {"Load":>6} '
              f'{"V":>4} {"T":>4} {"Mov":>3} {"mA":>5}')
        print('-' * 48)
        for sid in sorted(self._latest.keys()):
            s = self._latest[sid]
            print(f'{sid:>3} {s["position"]:>6} {s["speed"]:>6} '
                  f'{s["load"]:>6} {s["voltage"]:>4} '
                  f'{s["temperature"]:>4} {s["moving"]:>3} '
                  f'{s["current"]:>5}')

    def destroy_node(self):
        self._running = False
        super().destroy_node()


def main():
    rclpy.init()
    node = KeyboardControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
