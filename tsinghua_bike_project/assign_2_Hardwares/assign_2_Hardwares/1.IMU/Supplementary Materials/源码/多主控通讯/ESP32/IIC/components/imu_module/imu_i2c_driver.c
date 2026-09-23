#include "imu_i2c_driver.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define IMU_I2C_ADDRESS 0x23
#define delay_ms(ms) vTaskDelay(pdMS_TO_TICKS(ms))// 毫秒延时宏 | Millisecond delay macro
/**
 * @brief 将寄存器中的两个字节转换为有符号 16 位值
 *        Convert two bytes from register to signed 16-bit integer.
 */
static int16_t to_int16(const uint8_t *bytes)
{
    return (int16_t)((bytes[1] << 8) + bytes[0]);
}

/**
 * @brief 将寄存器中的四个字节转换为浮点数
 *        Convert four bytes from register to IEEE754 float.
 */
static float to_float(const uint8_t *bytes)
{
    float value;
    memcpy(&value, bytes, sizeof(float));
    return value;
}

int IMU_ReadBytes(uint8_t dev_addr, uint8_t reg_addr, uint8_t *buf, uint16_t len)
{
    return i2cRead(dev_addr, reg_addr, (uint8_t)len, buf);
}
int IMU_WriteBytes(uint8_t dev_addr, uint8_t reg_addr, const uint8_t *buf, uint16_t len)
{
    return i2cWrite(dev_addr, reg_addr, (uint8_t)len, (uint8_t *)buf);
}

/**
 * @brief 读取指定寄存器内容
 *        Read raw bytes from the given register.
 */
static int read_register(uint8_t reg, uint8_t *register_data, uint16_t length)
{
    return IMU_ReadBytes(IMU_I2C_ADDRESS, reg, register_data, length);
}

/**
 * @brief 写入指定寄存器
 *        Write raw bytes into the given register.
 */
static int write_register(uint8_t reg, const uint8_t *register_data, uint16_t length)
{
    return IMU_WriteBytes(IMU_I2C_ADDRESS, reg, register_data, length);
}

static int i2c_wait_calibration(uint8_t function, uint32_t timeout_ms)
{
    uint32_t elapsed = 0;
    while (1) {
        uint8_t state = 0;
        if (read_register(function, &state, 1) != 0) {
            return -2; 
        }
        if (state != 0) {
            return state;
        }
        if (timeout_ms != 0 && elapsed >= timeout_ms) {
            return -1;
        }
        delay_ms(100);
        elapsed += 100;
    }
}

int IMU_I2C_SendCommand(uint8_t function, uint16_t value)
{
    uint8_t register_data[2];
    register_data[0] = (uint8_t)(value & 0xFF);
    register_data[1] = (uint8_t)((value >> 8) & 0xFF);
    return write_register(function, register_data, 2);
}

/* ---------- 校准指令 / Calibration helpers ---------- */
int IMU_I2C_CalibrationImu(void)
{
    uint8_t data = 0x01;
    if (write_register(IMU_FUNC_CALIB_IMU, &data, 1) != 0) {
        printf("[IMU-I2C] Calibration imu send failed\r\n");
        return -2;
    }
    int state = i2c_wait_calibration(IMU_FUNC_CALIB_IMU, 7000);
    if (state == 1) { printf("[IMU-I2C] Calibration imu success\r\n"); return 0; }
    if (state == -1){ printf("[IMU-I2C] Calibration imu timeout\r\n"); return -1; }
    if (state == -2){ printf("[IMU-I2C] Calibration imu read failed\r\n"); return -2; }
    printf("[IMU-I2C] Calibration imu failed (code=%d)\r\n", state);
    return -state;
}

int IMU_I2C_CalibrationMag(void)
{
    uint8_t data = 0x01;
    if (write_register(IMU_FUNC_CALIB_MAG, &data, 1) != 0) {
        printf("[IMU-I2C] Calibration mag send failed\r\n");
        return -2;
    }
    int state = i2c_wait_calibration(IMU_FUNC_CALIB_MAG, 0);
    if (state == 1) { printf("[IMU-I2C] Calibration mag success\r\n"); return 0; }
    if (state == -1){ printf("[IMU-I2C] Calibration mag timeout\r\n"); return -1; }
    if (state == -2){ printf("[IMU-I2C] Calibration mag read failed\r\n"); return -2; }
    printf("[IMU-I2C] Calibration mag failed (code=%d)\r\n", state);
    return -state;
}

int IMU_I2C_CalibrationTemp(float now_temperature)
{
    if (now_temperature > 50.0f || now_temperature < -50.0f) {
        return -1;
    }
    int16_t raw = (int16_t)(now_temperature * 100.0f);
    uint8_t data[2] = {
        (uint8_t)(raw & 0xFF),
        (uint8_t)((raw >> 8) & 0xFF)
    };
    if (write_register(IMU_FUNC_CALIB_TEMP, data, 2) != 0) {
        printf("[IMU-I2C] Calibration temp send failed\r\n");
        return -2;
    }
    int state = i2c_wait_calibration(IMU_FUNC_CALIB_TEMP, 2000);
    if (state == 1) { printf("[IMU-I2C] Calibration temp success\r\n"); return 0; }
    if (state == -1){ printf("[IMU-I2C] Calibration temp timeout\r\n"); return -1; }
    if (state == -2){ printf("[IMU-I2C] Calibration temp read failed\r\n"); return -2; }
    printf("[IMU-I2C] Calibration temp failed (code=%d)\r\n", state);
    return -state;
}

/**
 * @brief 读取加速度数据（单位 g）
 *        Read acceleration in g.
 */
int IMU_I2C_ReadAccelerometer(float out[3])
{
    uint8_t register_data[6];
    if (read_register(IMU_FUNC_RAW_ACCEL, register_data, 6) != 0) {
        return -1;
    }
    if (out != NULL) {
        float ratio = 16.0f / 32767.0f;
        out[0] = to_int16(&register_data[0]) * ratio;
        out[1] = to_int16(&register_data[2]) * ratio;
        out[2] = to_int16(&register_data[4]) * ratio;
    }
    return 0;
}

/**
 * @brief 读取角速度（单位 rad/s）
 *        Read angular velocity in rad/s.
 */
int IMU_I2C_ReadGyroscope(float out[3])
{
    uint8_t register_data[6];
    if (read_register(IMU_FUNC_RAW_GYRO, register_data, 6) != 0) {
        return -1;
    }
    if (out != NULL) {
        float ratio = (2000.0f / 32767.0f) * (3.1415926f / 180.0f);
        out[0] = to_int16(&register_data[0]) * ratio;
        out[1] = to_int16(&register_data[2]) * ratio;
        out[2] = to_int16(&register_data[4]) * ratio;
    }
    return 0;
}

/**
 * @brief 读取磁场强度（单位 uT）
 *        Read magnetic field strength in micro tesla.
 */
int IMU_I2C_ReadMagnetometer(float out[3])
{
    uint8_t register_data[6];
    if (read_register(IMU_FUNC_RAW_MAG, register_data, 6) != 0) {
        return -1;
    }
    if (out != NULL) {
        float ratio = 800.0f / 32767.0f;
        out[0] = to_int16(&register_data[0]) * ratio;
        out[1] = to_int16(&register_data[2]) * ratio;
        out[2] = to_int16(&register_data[4]) * ratio;
    }
    return 0;
}

/**
 * @brief 读取四元数
 *        Read quaternion (w, x, y, z).
 */
int IMU_I2C_ReadQuaternion(float out[4])
{
    uint8_t register_data[16];
    if (read_register(IMU_FUNC_QUAT, register_data, 16) != 0) {
        return -1;
    }
    if (out != NULL) {
        out[0] = to_float(&register_data[0]);
        out[1] = to_float(&register_data[4]);
        out[2] = to_float(&register_data[8]);
        out[3] = to_float(&register_data[12]);
    }
    return 0;
}

/**
 * @brief 读取欧拉角（弧度）
 *        Read Euler angles (rad).
 */
int IMU_I2C_ReadEuler(float out[3])
{
    uint8_t register_data[12];
    if (read_register(IMU_FUNC_EULER, register_data, 12) != 0) {
        return -1;
    }
    if (out != NULL) {
        const float RAD2DEG = 57.2957795f;
        out[0] = to_float(&register_data[0]) * RAD2DEG;
        out[1] = to_float(&register_data[4]) * RAD2DEG;
        out[2] = to_float(&register_data[8]) * RAD2DEG;
    }
    return 0;
}

/**
 * @brief 读取气压相关数据：高度、温度、气压、气压差
 *        Read barometric data: height, temperature, pressure, delta.
 */
int IMU_I2C_ReadBarometer(float out[4])
{
    uint8_t register_data[16];
    if (read_register(IMU_FUNC_BARO, register_data, 16) != 0) {
        return -1;
    }
    if (out != NULL) {
        out[0] = to_float(&register_data[0]);
        out[1] = to_float(&register_data[4]);
        out[2] = to_float(&register_data[8]);
        out[3] = to_float(&register_data[12]);
    }
    return 0;
}

/**
 * @brief 读取固件版本字符串
 *        Read firmware version as string.
 */
int IMU_I2C_ReadVersion()
{
    uint8_t register_data[3] = {-1};
    if (read_register(IMU_FUNC_VERSION, register_data, 3) != 0) {
        return -1;
    }
    if (register_data[0] != -1) {
        printf("Version:%u.%u.%u\r\n", register_data[0], register_data[1], register_data[2]);
    }
    else {
        printf("Version:-1");
    }
    return 0;
}

/**
 * @brief 一次性读取所有常用数据
 *        Read all common sensor values in a single call.
 */
int IMU_I2C_ReadAll(imu_measurement_t *out)
{
    if (out == NULL) {
        return -1;
    }
    if (IMU_I2C_ReadAccelerometer(out->accel) != 0) {
        return -1;
    }
    if (IMU_I2C_ReadGyroscope(out->gyro) != 0) {
        return -1;
    }
    if (IMU_I2C_ReadMagnetometer(out->mag) != 0) {
        return -1;
    }
    if (IMU_I2C_ReadQuaternion(out->quat) != 0) {
        return -1;
    }
    if (IMU_I2C_ReadEuler(out->euler) != 0) {
        return -1;
    }
    if (IMU_I2C_ReadBarometer(out->baro) != 0) {
        return -1;
    }
    return 0;
}
