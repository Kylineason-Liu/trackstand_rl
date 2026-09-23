#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial
from sensor_msgs.msg import Joy   # 新增导入

class SBUSRawReader(Node):
    def __init__(self):
        super().__init__('sbus_raw_reader')

        # 声明参数
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        # 可调整帧长度（25 或 35），默认25
        self.declare_parameter('frame_len', 35)

        port = self.get_parameter('port').value
        baudrate = self.get_parameter('baudrate').value
        self.frame_len = self.get_parameter('frame_len').value

        # 打开串口
        self.ser = serial.Serial(port, baudrate, timeout=0.1)
        self.get_logger().info(f'Opened {port} at {baudrate} baud, frame_len={self.frame_len}')

        self.buffer = bytearray()          # 接收缓冲区
        self.latest_frame = None           # 存储最新一帧

        # 创建 Joy 发布者（新增）
        self.joy_pub = self.create_publisher(Joy, 'joy', 10)

        # 高频读取（20ms），确保不丢数据
        self.read_timer = self.create_timer(0.02, self._read_data)
        # 每秒打印一次最新帧
        self.print_timer = self.create_timer(1.0, self._print_latest)

    def _read_data(self):
        """读取所有待处理数据，并提取最新完整帧"""
        if self.ser.in_waiting > 0:
            raw = self.ser.read(self.ser.in_waiting)
            self.buffer.extend(raw)

            # 从缓冲区末尾向前搜索帧头 0x0F
            pos = self.buffer.rfind(b'\x0F')
            if pos != -1:
                # 检查从帧头到末尾是否足够一帧长度
                if len(self.buffer) - pos >= self.frame_len:
                    # 提取最新一帧
                    frame = self.buffer[pos:pos + self.frame_len]
                    self.latest_frame = frame

                    # ===== 新增：解码并发布 Joy =====
                    self._decode_and_publish(frame)
                    # ================================

    def _decode_and_publish(self, frame):
        """根据帧长度解码并发布 Joy 消息"""
        # 仅当帧长度为 25 或 35 时才处理
        if self.frame_len == 25:
            # 标准 SBUS 25 字节解码
            if frame[0] != 0x0F or frame[24] != 0x00:
                return
            channels = self._decode_25(frame)
        elif self.frame_len == 35:
            # 35 字节解码（高字节在前，无位压缩）
            if frame[0] != 0x0F:
                return
            channels = self._decode_35(frame)
        else:
            return  # 不支持的帧长度

        # 构建 Joy 消息
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'rc'
        msg.axes = [int(ch) for ch in channels]   # 16个原始值
        msg.buttons = []         
        self.joy_pub.publish(msg)

    def _decode_25(self, f):
        """标准 25 字节 SBUS 解码，返回 16 通道值列表"""
        ch = []
        for i in range(16):
            b = 1 + i * 11 // 8
            bit = i * 11 % 8
            v = (f[b] >> bit) | (f[b + 1] << (8 - bit))
            if bit > 5:
                v |= f[b + 2] << (16 - bit)
            ch.append(v & 0x07FF)
        return ch

    def _decode_35(self, f):
        """35 字节解码（每个通道2字节，高字节在前）"""
        ch = []
        for i in range(16):
            high = f[1 + i*2]
            low  = f[2 + i*2]
            ch.append((high << 8) | low)
        return ch

    def _print_latest(self):
        """每秒打印最新一帧的原始十六进制数据（不变）"""
        if self.latest_frame is not None:
            hex_str = ' '.join(f'{b:02X}' for b in self.latest_frame)
            self.get_logger().info(f'Latest frame ({len(self.latest_frame)} bytes): {hex_str}')
        else:
            self.get_logger().info('No frame received yet')

def main(args=None):
    rclpy.init(args=args)
    node = SBUSRawReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()