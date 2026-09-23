from machine import I2C, Pin
import struct
import time
import math


READ_INTERVAL = 0.1  # 传感器读取间隔（秒） Sensor read interval in seconds


class YbImuI2c(object):

    def __init__(self, debug=False):

        # 创建I2C通信对象   Create I2C communication object
        self.dev = I2C(0, scl=Pin(5), sda=Pin(4), freq=400000)  # 使用I2C0，SCL=GP5，SDA=GP4    
                                                                # Use I2C0, SCL=GP5, SDA=GP4

        self._delay_time = 0.001
        self._debug = debug
        self._addr = 0x23

        self.FUNC_VERSION = 0x01
        self.FUNC_RAW_ACCEL = 0x04
        self.FUNC_RAW_GYRO = 0x0A
        self.FUNC_RAW_MAG = 0x10
        self.FUNC_QUAT = 0x16
        self.FUNC_EULER = 0x26
        self.FUNC_BARO = 0x32

        self.FUNC_ALGO_TYPE = 0x61

        self.FUNC_CALIB_IMU = 0x70
        self.FUNC_CALIB_MAG = 0x71
        self.FUNC_CALIB_TEMP = 0x73

        self.FUNC_RESET_FLASH = 0xA0


    def __del__(self):
        pass
        

    def _print_log(self, cmd_data, log="Send"):
        if not self._debug:
            return
        # print ("Send: [0x" + ', 0x'.join('{:02X}'.format(x) for x in cmd_data) + "]")
        print (str(log) + ": [" + ''.join('{:02X}'.format(x) for x in cmd_data) + "]")


    # 写数据到I2C寄存器 Write data to I2C register
    def _send_data(self, reg, cmd_data):
        self.dev.writeto_mem(self._addr, reg, bytes([cmd_data]))

    def _send_data_array(self, reg, cmd_array):
        self.dev.writeto_mem(self._addr, reg, bytes(cmd_array))

    # 从I2C寄存器读取数据   Reading data from I2C registers
    def i2c_read(self, addr, reg, length):
        return self.dev.readfrom_mem(addr, reg, length)
    
    def _read_data(self, reg, num):
        if num > 32:
            num = 32
        return self.i2c_read(self._addr, reg, num)
    

    def wait_calibration(self, func, name, timeout_ms=None):
        count = 0
        while True:
            time.sleep(0.1)
            values = self._read_data(func, 1)
            if values[0]:
                if self._debug:
                    print(name, values[0])
                return values[0]
            if timeout_ms is not None:
                count = count + 1
                if count > timeout_ms:
                    if self._debug:
                        print(name, "timeout")
                    return None


    # 校准IMU(陀螺仪和加速度计) / Calibrate IMU (gyroscope and accelerometer)
    def calibration_imu(self):
        cmd = [self.FUNC_CALIB_IMU, 0x01]
        self._send_data(cmd[0], cmd[1])
        self._print_log(cmd, "cali imu")
        if self._delay_time > 0:
            time.sleep(self._delay_time)
        self.wait_calibration(self.FUNC_CALIB_IMU, "cali imu", 7000)


    # 校准磁力计    / Calibrate magnetometer
    def calibration_mag(self):
        cmd = [self.FUNC_CALIB_MAG, 0x01]
        self._send_data(cmd[0], cmd[1])
        self._print_log(cmd, "cali mag")
        if self._delay_time > 0:
            time.sleep(self._delay_time)
        self.wait_calibration(self.FUNC_CALIB_MAG, "cali mag")


    
    # 校准温度   / Calibrate temperature
    def calibration_temperature(self, now_temperature):
        if now_temperature > 50 or now_temperature < -50:
            return
        value = bytearray(struct.pack('h', int(now_temperature*100)))
        cmd = [self.FUNC_CALIB_TEMP, value[0], value[1]]
        self._send_data_array(cmd[0], cmd[1:])
        self._print_log(cmd, "cali temp")
        if self._delay_time > 0:
            time.sleep(self._delay_time)
        self.wait_calibration(self.FUNC_CALIB_TEMP, "cali temp", 2000)


    # 重置用户数据。请谨慎操作，会将所有校准值清零。 / Reset user data. Please operate with caution, all calibration values will be cleared.
    def reset_user_data(self):
        cmd = [self.FUNC_RESET_FLASH, 0x01]
        self._send_data(cmd[0], cmd[1])
        self._print_log(cmd, "reset user data")
        if self._delay_time > 0:
            time.sleep(self._delay_time)
        time.sleep(1)



    # 获取加速度计三轴数据，返回accel=[a_x, a_y, a_z]
    # Get accelerometer triaxial data, return accel=[a_x, a_y, a_z]
    def get_accelerometer_data(self):
        values = self._read_data(self.FUNC_RAW_ACCEL, 6)
        # 转化单位为g
        accel_ratio = 16 / 32767.0
        a_x = struct.unpack('h', bytearray(values[0:2]))[0]*accel_ratio
        a_y = struct.unpack('h', bytearray(values[2:4]))[0]*accel_ratio
        a_z = struct.unpack('h', bytearray(values[4:6]))[0]*accel_ratio
        accel = [a_x, a_y, a_z]
        if self._delay_time > 0:
            time.sleep(self._delay_time)
        return accel

    # 获取陀螺仪三轴数据，返回gyro=[g_x, g_y, g_z]
    # Get the gyro triaxial data, return gyro=[g_x, g_y, g_z]
    def get_gyroscope_data(self):
        values = self._read_data(self.FUNC_RAW_GYRO, 6)
        # 转化单位为rad/s
        AtoR = math.pi / 180.0
        gyro_ratio = (2000 / 32767.0) * AtoR
        g_x = struct.unpack('h', bytearray(values[0:2]))[0]*gyro_ratio
        g_y = struct.unpack('h', bytearray(values[2:4]))[0]*gyro_ratio
        g_z = struct.unpack('h', bytearray(values[4:6]))[0]*gyro_ratio
        gyro = [g_x, g_y, g_z]
        if self._delay_time > 0:
            time.sleep(self._delay_time)
        return gyro

    # 获取磁力计三轴数据，返回mag=[m_x, m_y, m_z]
    def get_magnetometer_data(self):
        values = self._read_data(self.FUNC_RAW_MAG, 6)
        # 转化单位为uT
        mag_ratio = 800.0 / 32767.0
        m_x = struct.unpack('h', bytearray(values[0:2]))[0]*mag_ratio
        m_y = struct.unpack('h', bytearray(values[2:4]))[0]*mag_ratio
        m_z = struct.unpack('h', bytearray(values[4:6]))[0]*mag_ratio
        mag = [m_x, m_y, m_z]
        if self._delay_time > 0:
            time.sleep(self._delay_time)
        return mag

    # 获取IMU的四元数，返回quat=[w, x, y, z]
    def get_imu_quaternion_data(self):
        values = self._read_data(self.FUNC_QUAT, 16)
        q0 = struct.unpack('f', bytearray(values[0:4]))[0]
        q1 = struct.unpack('f', bytearray(values[4:8]))[0]
        q2 = struct.unpack('f', bytearray(values[8:12]))[0]
        q3 = struct.unpack('f', bytearray(values[12:16]))[0]
        quat = [q0, q1, q2, q3]
        if self._delay_time > 0:
            time.sleep(self._delay_time)
        return quat

    # 获取板子姿态角，返回euler=[roll, pitch, yaw]
    # ToAngle=True返回角度，ToAngle=False返回弧度。
    def get_imu_attitude_data(self, ToAngle=True):
        values = self._read_data(self.FUNC_EULER, 12)
        roll = struct.unpack('f', bytearray(values[0:4]))[0]
        pitch = struct.unpack('f', bytearray(values[4:8]))[0]
        yaw = struct.unpack('f', bytearray(values[8:12]))[0]
        if ToAngle:
            RtoA = 180.0 / math.pi
            roll = roll * RtoA
            pitch = pitch * RtoA
            yaw = yaw * RtoA
        euler = [roll, pitch, yaw]
        if self._delay_time > 0:
            time.sleep(self._delay_time)
        return euler

    # 获取气压计的数据，返回baro=[height, temperature, pressure, pressure_contrast]
    def get_baro_data(self):
        values = self._read_data(self.FUNC_BARO, 16)
        height = round(struct.unpack('f', bytearray(values[0:4]))[0], 2)
        temperature = round(struct.unpack('f', bytearray(values[4:8]))[0], 2)
        pressure = round(struct.unpack('f', bytearray(values[8:12]))[0], 5)
        pressure_contrast = round(struct.unpack('f', bytearray(values[12:16]))[0], 5)
        baro = [height, temperature, pressure, pressure_contrast]
        if self._delay_time > 0:
            time.sleep(self._delay_time)
        return baro

    # 获取底层单片机版本号，如1.1
    # Get the underlying microcontroller version number, such as 1.1
    def get_version(self):
        values = self._read_data(self.FUNC_VERSION, 3)
        version = "%d.%d.%d" % (values[0], values[1], values[2])
        if self._delay_time > 0:
            time.sleep(self._delay_time)
        return version

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
    imu = YbImuI2c(debug=False)

    try:
        version = imu.get_version()
        print(f"Firmware version: {version}")
    except Exception:
        print("Firmware version: unavailable in serial mode")

    try:
        interval_ms = max(int(READ_INTERVAL * 1000), 20)
        poll_sleep_ms = 5
        next_print = time.ticks_ms()
        while True:
            now = time.ticks_ms()
            if time.ticks_diff(now, next_print) >= 0:
                print_sensor_snapshot(imu)
                next_print = time.ticks_add(now, interval_ms)
            time.sleep_ms(poll_sleep_ms)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
