#ifndef IMU_I2C_DRIVER_H
#define IMU_I2C_DRIVER_H

#include "Wire.h"
#include "bsp_iic.hpp"
#include <Arduino.h>

#define IMU_FUNC_VERSION        0x01
#define IMU_FUNC_RAW_ACCEL      0x04
#define IMU_FUNC_RAW_GYRO       0x0A
#define IMU_FUNC_RAW_MAG        0x10
#define IMU_FUNC_QUAT           0x16
#define IMU_FUNC_EULER          0x26
#define IMU_FUNC_BARO           0x32
#define IMU_FUNC_CALIB_IMU      0x70
#define IMU_FUNC_CALIB_MAG      0x71
#define IMU_FUNC_CALIB_BARO     0x72
#define IMU_FUNC_CALIB_TEMP     0x73
#define IMU_FUNC_REQUEST_DATA   0x80
#define IMU_FUNC_RETURN_STATE   0x81
#define IMU_FUNC_RESET_FLASH    0xA0
#define IMU_FUNC_REBOOT_DEVICE  0xA1

/* IMU 测量数据结构 / IMU measurement data structure */
typedef struct {
    float accel[3];
    float gyro[3];
    float mag[3];
    float quat[4];
    float euler[3];
    float baro[4];
    char  version[8];
} imu_measurement_t;

int IMU_ReadBytes(uint8_t dev_addr, uint8_t reg_addr, uint8_t *buf, uint16_t len);
int IMU_WriteBytes(uint8_t dev_addr, uint8_t reg_addr, const uint8_t *buf, uint16_t len);

int  IMU_I2C_SendCommand(uint8_t function, uint16_t value);

/** 读取加速度 / Read acceleration (g) */
int  IMU_I2C_ReadAccelerometer(float out[3]);
/** 读取角速度 / Read angular velocity (rad/s) */
int  IMU_I2C_ReadGyroscope(float out[3]);
/** 读取磁场 / Read magnetic field (uT) */
int  IMU_I2C_ReadMagnetometer(float out[3]);
/** 读取四元数 / Read quaternion */
int  IMU_I2C_ReadQuaternion(float out[4]);
/** 读取欧拉角 / Read Euler angles (rad) */
int  IMU_I2C_ReadEuler(float out[3]);
/** 读取气压相关数据 / Read barometer-related data */
int  IMU_I2C_ReadBarometer(float out[4]);
/** 读取固件版本 / Read firmware version string */
int  IMU_I2C_ReadVersion();
/** 一次性读取全部数据 / Read all available data in one call */
int  IMU_I2C_ReadAll(imu_measurement_t *out);

/* 校准命令 / Calibration helpers */
int  IMU_I2C_CalibrationImu(void);
int  IMU_I2C_CalibrationMag(void);
int  IMU_I2C_CalibrationTemp(float now_temperature);

void print_sensor_data(const imu_measurement_t &data);


#endif
