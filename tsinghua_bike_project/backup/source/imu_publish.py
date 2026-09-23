

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField
from std_msgs.msg import Float32MultiArray
from YbImuLib import YbImuSerial

SERIAL_PORT = "/dev/ttyUSB0"


class ImuPublisher(Node):
    def __init__(self):
        super().__init__("imu_publisher")

        # --- open serial ---
        self.imu = None
        
        self.imu = YbImuSerial(SERIAL_PORT)
            
        if self.imu is None:
            self.get_logger().error("Failed to open IMU serial port")
            return
        self.imu.create_receive_threading()
        self.imu.set_algo_type(6)  # 9-axis fusion (gyro + accel + mag) for stable yaw

        # --- publishers ---
        self.pub_imu = self.create_publisher(Imu, "imu/data_raw", 10)
        self.pub_mag = self.create_publisher(MagneticField, "imu/mag", 10)
        self.pub_euler = self.create_publisher(Float32MultiArray, "imu/euler", 10)

        # --- timer (10 Hz) ---
        self.timer = self.create_timer(0.1, self.publish_data)
        self.get_logger().info("Publishing IMU data at 10 Hz...")

    def destroy_node(self):
        if self.imu is not None:
            self.imu._dev.close()
        super().destroy_node()

    def publish_data(self):
        if self.imu is None:
            return

        t = self.get_clock().now().to_msg()

        ax, ay, az = self.imu.get_accelerometer_data()
        gx, gy, gz = self.imu.get_gyroscope_data()
        mx, my, mz = self.imu.get_magnetometer_data()
        qw, qx, qy, qz = self.imu.get_imu_quaternion_data()
        roll, pitch, yaw = self.imu.get_imu_attitude_data(ToAngle=True)

        # IMU message
        msg_imu = Imu()
        msg_imu.header.stamp = t
        msg_imu.header.frame_id = "imu_link"
        msg_imu.linear_acceleration.x = ax
        msg_imu.linear_acceleration.y = ay
        msg_imu.linear_acceleration.z = az
        msg_imu.angular_velocity.x = gx
        msg_imu.angular_velocity.y = gy
        msg_imu.angular_velocity.z = gz
        msg_imu.orientation.w = qw
        msg_imu.orientation.x = qx
        msg_imu.orientation.y = qy
        msg_imu.orientation.z = qz
        self.pub_imu.publish(msg_imu)

        # Mag message
        msg_mag = MagneticField()
        msg_mag.header.stamp = t
        msg_mag.header.frame_id = "imu_link"
        msg_mag.magnetic_field.x = mx
        msg_mag.magnetic_field.y = my
        msg_mag.magnetic_field.z = mz
        self.pub_mag.publish(msg_mag)

        # Euler message
        msg_euler = Float32MultiArray()
        msg_euler.data = [roll, pitch, yaw]
        self.pub_euler.publish(msg_euler)


def main(args=None):
    rclpy.init(args=args)
    node = ImuPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
