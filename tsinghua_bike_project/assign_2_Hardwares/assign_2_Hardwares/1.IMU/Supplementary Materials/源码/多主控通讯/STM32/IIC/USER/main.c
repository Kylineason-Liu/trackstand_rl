//IMU -> STM32
//SCL -> PB10
//SDA -> PB11
#include "AllHeader.h"

imu_measurement_t imu_data;

int main(void)
{	
	bsp_init();
	
	TIM3_Init();
	
	//电机模块iic通信初始化	Motor module iic communication initialization
	IIC_IMU_Init();
	
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

        delay_ms(70);

	}
	
}

