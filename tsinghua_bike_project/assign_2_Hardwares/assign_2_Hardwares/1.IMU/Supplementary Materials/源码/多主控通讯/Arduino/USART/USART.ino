//IMU -> Arduino UNO    uart module -> Arduino UNO
//TX -> RX(0)               RX      ->      3
//RX -> TX(1)
#include <Arduino.h>
#include "usart.hpp"
#include "imu_uart_driver.hpp"

imu_measurement_t imu_data;

static unsigned long lastPrintAt = 0;
const unsigned long printIntervalMs = 150;  // 只用作控制打印的频率，数字越大打印越慢 / Used only to control print frequency, larger number means slower printing

void setup(){
  
  Usart_init();
  IMU_UART_GetVersion();
} 

void loop(){
  // Receive a bunch of bytes to fill the buffer / 接收一堆字节以填充缓冲区
  for (int i = 0; i < 128; i++) {
    USART_Recieve();
  }
  
  // Process all received data / 处理所有接收到的数据
  IMU_UART_Process();


  if (millis() - lastPrintAt >= printIntervalMs) {
      lastPrintAt = millis();
      IMU_UART_GetAll(&imu_data);
      print_sensor_data(imu_data);
  }
}
