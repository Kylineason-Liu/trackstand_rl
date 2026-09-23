#!/usr/bin/env python3

"""
请先确保YbImuLib已正确安装 / Please make sure YbImuLib is properly installed
"""

import time
from YbImuLib import YbImuSerial


# Serial device path; replace with your actual port / 串口设备路径，请替换为实际端口
SERIAL_PORT = "/dev/ttyUSB0"

# Interval between data prints in seconds / 数据打印间隔（秒）
READ_INTERVAL = 0.1


def create_serial_device() -> YbImuSerial:
    """Create a serial IMU instance.

        创建串口 IMU 实例
    """

    imu = YbImuSerial(SERIAL_PORT, debug=False)
    # Serial mode requires a background thread to parse frames continuously / 串口模式需要开启后台线程持续解析数据帧
    imu.create_receive_threading()
    return imu


def print_sensor_snapshot(imu: YbImuSerial) -> None:
    """Print a snapshot of sensor data.

    Retrieve all available sensor data and print a snapshot.
    读取所有可用传感器数据并打印快照。
    """

    ax, ay, az = imu.get_accelerometer_data()
    gx, gy, gz = imu.get_gyroscope_data()
    mx, my, mz = imu.get_magnetometer_data()
    qw, qx, qy, qz = imu.get_imu_quaternion_data()
    roll, pitch, yaw = imu.get_imu_attitude_data(ToAngle=True)
    height, temperature, pressure, pressure_contrast = imu.get_baro_data()

    print(
        "------ Sensor Data ------\n"
        f"Acceleration [g]:      x={ax: .3f}, y={ay: .3f}, z={az: .3f}\n"
        f"Gyroscope [rad/s]:     x={gx: .3f}, y={gy: .3f}, z={gz: .3f}\n"
        f"Magnetometer [uT]:     x={mx: .3f}, y={my: .3f}, z={mz: .3f}\n"
        f"Quaternion:            w={qw: .5f}, x={qx: .5f}, y={qy: .5f}, z={qz: .5f}\n"
        f"Euler Angle [deg]:     roll={roll: .2f}, pitch={pitch: .2f}, yaw={yaw: .2f}\n"
        f"Barometer:             height={height: .2f} m, temperature={temperature: .2f} °C\n"
        f"                       pressure={pressure: .5f} Pa, pressure_diff={pressure_contrast: .5f} Pa\n"
        "-------------------------"
    )


def main() -> None:
    imu = create_serial_device()

    try:
        version = imu.get_version()
        print(f"Firmware version: {version}")
    except Exception:
        print("Firmware version: unavailable in serial mode")

    print("Press Ctrl+C to exit the program.")
    try:
        while True:
            print_sensor_snapshot(imu)
            time.sleep(READ_INTERVAL)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
