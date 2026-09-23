#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "sdkconfig.h"
#include "uart_module.h"
#include "imu_uart_driver.h"

#define delay_ms(ms) vTaskDelay(pdMS_TO_TICKS(ms))// 毫秒延时宏 | Millisecond delay macro

imu_measurement_t imu_data;

void IMUReadData_Task(void *arg) {
	while (1) {
		IMU_UART_Process();
		IMU_UART_GetAll(&imu_data);
		printf("------------------------Sensor Data--------------------------\n");
		printf("Acceleration [g]:    x = %.3f y = %.3f z = %.3f\n", imu_data.accel[0], imu_data.accel[1], imu_data.accel[2]);
		printf("Gyroscope [rad/s]:   x = %.3f y = %.3f z = %.3f\n", imu_data.gyro[0],  imu_data.gyro[1],  imu_data.gyro[2]);
		printf("Magnetometer [uT]:   x = %.3f y = %.3f z = %.3f\n", imu_data.mag[0],   imu_data.mag[1],   imu_data.mag[2]);
		printf("Euler Angle [deg]:   roll = %.3f pitch = %.3f yaw = %.3f\n", imu_data.euler[0], imu_data.euler[1], imu_data.euler[2]);
		printf("Quaternion:          w = %.3f x = %.3f y = %.3f z = %.3f\n", imu_data.quat[0], imu_data.quat[1], imu_data.quat[2], imu_data.quat[3]);
		printf("Barometer:           height = %.3f(m) temperature = %.3f(C)\n", imu_data.baro[0], imu_data.baro[1]);
		printf("                     pressure = %.3f(Pa) pressure-diff = %.3f(Pc)\n", imu_data.baro[2], imu_data.baro[3]);
		printf("-------------------------------------------------------------\n");

		delay_ms(100);// 防止任务卡死 | Preventing tasks from getting stuck

	}
}

void app_main(void)
{
	//初始化UART1通信 | Initialize UART1 communication
	uart1_init();	
	
    xTaskCreate(
        UART_Process_Task, 		// 任务函数	Task Function
        "UART_Process",         // 任务名称	Task Name
        4096,                   // 堆栈大小（字节）	Stack size (bytes)
        NULL, 
        2,                      // 优先级（数值越大优先级越高）	Priority (the larger the value, the higher the priority)
        NULL
    );
	IMU_UART_GetVersion();// 获取IMU版本信息 | Get IMU version info
	xTaskCreate(IMUReadData_Task, "IMURadData", 4096, NULL, 1, NULL);

}
