import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist


class JoyToCmd(Node):
    def __init__(self):
        super().__init__('joy_to_cmd')
        self.sub = self.create_subscription(Joy, 'joy', self._cb, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)

    def _cb(self, msg: Joy):
        twist = Twist()
        twist.linear.x = (msg.axes[3]-992)/8.0
        twist.linear.y = (msg.axes[0]-992)/8.0
        self.pub.publish(twist)


def main():
    rclpy.init()
    rclpy.spin(JoyToCmd())
    rclpy.shutdown()
