#!/usr/bin/env python3

import time
from typing import Optional

from YbImuLib import YbImuSerial


# Serial port path / 串口设备路径
SERIAL_PORT = "/dev/ttyUSB0"

# Extra wait time after calibration in seconds / 校准后额外等待时间（秒）
POST_CALIBRATION_WAIT = 2.0


def create_serial_device() -> YbImuSerial:
    """Create a serial IMU instance and start the receive thread.

    创建串口 IMU 实例并启动接收线程。
    """

    imu = YbImuSerial(SERIAL_PORT, debug=False)
    imu.create_receive_threading()
    time.sleep(0.1)
    return imu


def format_state(state: Optional[int]) -> str:
    """Convert calibration state to readable text.

    将校准结果状态转换为可读文本。
    """

    if state == 1:
        return "success"
    if state == 0 or state is None:
        return "incomplete"
    return f"code {state}"


def main() -> None:
    imu = create_serial_device()

    try:
        version = imu.get_version()
        print(f"Firmware version: {version}")
    except Exception:
        print("Firmware version: unavailable.")

    while True:
        try:
            user_input = input(
                "Enter current ambient temperature in Celsius (e.g., 25.0): "
            ).strip()
            calibration_temperature = float(user_input)
            if not (-50.0 <= calibration_temperature <= 50.0):
                print(
                    "Temperature must be within -50.0 to 50.0 °C. Please re-enter."
                )
                continue
            break
        except ValueError:
            print("Invalid temperature value. Please input a numeric value.")

    print(
        "Starting temperature calibration (stabilize the device at the specified temperature)..."
    )
    state = imu.calibration_temperature(calibration_temperature)

    if state is None and getattr(imu, "_rx_func", None) == getattr(imu, "FUNC_CALIB_TEMP", None):
        state = getattr(imu, "_rx_state", None)

    print(f"Temperature calibration result: {format_state(state)}")

    time.sleep(POST_CALIBRATION_WAIT)
    print("Calibration script finished.")


if __name__ == "__main__":
    main()
