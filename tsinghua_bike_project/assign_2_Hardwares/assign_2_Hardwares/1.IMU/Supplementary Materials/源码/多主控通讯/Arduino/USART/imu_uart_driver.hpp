#ifndef IMU_UART_DRIVER_H
#define IMU_UART_DRIVER_H

#include <stdint.h>
#include <Arduino.h>
#include "usart.hpp"


/* 配置项 / Config */
#ifndef IMU_UART_RX_BUF_SIZE
#define IMU_UART_RX_BUF_SIZE 256  /* 环形接收缓冲区大小 / RX ring buffer size */
#endif

#define FRAME_HEAD1 0x7E
#define FRAME_HEAD2 0x23


/* 功能码 / Function codes */
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

/* 结构体：一次性获取所有传感器数据 / Struct: get all sensor data at once */
typedef struct {
    float accel[3];
    float gyro[3];
    float mag[3];
    float quat[4];
    float euler[3];
    float baro[4];
    char  version[8];
} imu_measurement_t;

int IMU_UART_Send(const uint8_t *data, uint16_t len);

void IMU_UART_Init(void);

void IMU_UART_RxBytes(volatile uint8_t *data, uint16_t len);

void IMU_UART_Process(void);

int  IMU_UART_SendCommand(uint8_t function, const uint8_t *params, uint8_t param_len);

void IMU_UART_ClearAutoReportData(void);

int  IMU_UART_GetAccelerometer(float out[3]);
int  IMU_UART_GetGyroscope(float out[3]);
int  IMU_UART_GetMagnetometer(float out[3]);
int  IMU_UART_GetQuaternion(float out[4]);
int  IMU_UART_GetEuler(float out[3]);
int  IMU_UART_GetBarometer(float out[4]);
void  IMU_UART_GetVersion(void);
int  IMU_UART_GetAll(imu_measurement_t *out);


void Send_IMU_Data(uint8_t Data);
void Send_IMU_Array(uint8_t *pData, uint8_t Length);

int  IMU_UART_CalibrationImu(void);
int  IMU_UART_CalibrationMag(void);
int  IMU_UART_CalibrationTemp(float now_temperature);
int  IMU_UART_ResetUserData(void);
int  IMU_UART_WaitCalibration(uint8_t function, uint32_t timeout_ms);
void print_sensor_data(const imu_measurement_t &data);

#endif /* IMU_UART_DRIVER_H */
