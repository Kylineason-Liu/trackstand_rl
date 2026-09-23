#include "imu_i2c_driver.hpp"

imu_measurement_t imu_data;
static unsigned long lastPrintAt = 0;
const unsigned long printIntervalMs = 150;  // 只用作控制打印的频率，数字越大打印越慢 / Used only to control print frequency, larger number means slower printing

void setup() {
  Serial.begin(115200);
  IIC_Init();
  IMU_I2C_ReadVersion();
}

void loop() {
  if (millis() - lastPrintAt >= printIntervalMs) {
      lastPrintAt = millis();
      IMU_I2C_ReadAll(&imu_data);
      print_sensor_data(imu_data);
  }
}

