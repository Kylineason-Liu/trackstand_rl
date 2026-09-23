#!/usr/bin/env python3

"""
请先确保YbImuLib已正确安装 / Please make sure YbImuLib is properly installed
"""

import time
from YbImuLib import YbImuI2c


# I2C bus number, corresponding to /dev/i2c-<bus> / I2C 总线号，对应 /dev/i2c-<bus>
I2C_BUS = 7

# Interval between data prints in seconds / 数据打印间隔（秒）
READ_INTERVAL = 0.1


def create_i2c_device() -> YbImuI2c:
    """Create an I2C IMU instance.

        创建 I2C IMU 实例
    """

    return YbImuI2c(port=I2C_BUS, debug=False)


def print_sensor_snapshot(imu: YbImuI2c) -> None:
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
    imu = create_i2c_device()

    try:
        version = imu.get_version()
        print(f"Firmware version: {version}")
    except Exception:
        print("Firmware version: unavailable in i2c mode")

    print("Press Ctrl+C to exit the program.")
    try:
        while True:
            print_sensor_snapshot(imu)
            time.sleep(READ_INTERVAL)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
