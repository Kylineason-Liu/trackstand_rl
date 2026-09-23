#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "sdkconfig.h"
#include "i2c_module.h"
#include "imu_i2c_driver.h"

#define delay_ms(ms) vTaskDelay(pdMS_TO_TICKS(ms))// 毫秒延时宏 | Millisecond delay macro

imu_measurement_t imu_data;

void app_main(void)
{
	
	//iic通信初始化	iic communication initialization
	if (i2c_module_init() != ESP_OK) {
        ESP_LOGE("MAIN", "I2C initialization failed!");
        return;
    }
	
	IMU_I2C_ReadVersion();

	while(1)
	{
        IMU_I2C_ReadAll(&imu_data);
        printf("------------------------Sensor Data--------------------------\n");
        printf("Acceleration [g]:    x = %.3f y = %.3f z = %.3f\n", imu_data.accel[0], imu_data.accel[1], imu_data.accel[2]);
        printf("Gyroscope [rad/s]:   x = %.3f y = %.3f z = %.3f\n", imu_data.gyro[0],  imu_data.gyro[1],  imu_data.gyro[2]);
        printf("Magnetometer [uT]:   x = %.3f y = %.3f z = %.3f\n", imu_data.mag[0],   imu_data.mag[1],   imu_data.mag[2]);
        printf("Euler Angle [deg]:   roll = %.3f pitch = %.3f yaw = %.3f\n", imu_data.euler[0], imu_data.euler[1], imu_data.euler[2]);
        printf("Quaternion:          w = %.3f x = %.3f y = %.3f z = %.3f\n", imu_data.quat[0], imu_data.quat[1], imu_data.quat[2], imu_data.quat[3]);
        printf("Barometer:           height = %.3f(m) temperature = %.3f(C)\n", imu_data.baro[0], imu_data.baro[1]);
        printf("                     pressure = %.3f(Pa) pressure-diff = %.3f(Pc)\n", imu_data.baro[2], imu_data.baro[3]);
        printf("-------------------------------------------------------------\n");

        delay_ms(100);

	}
}
