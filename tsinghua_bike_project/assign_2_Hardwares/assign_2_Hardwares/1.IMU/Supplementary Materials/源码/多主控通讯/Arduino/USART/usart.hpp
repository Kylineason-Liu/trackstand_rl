#ifndef __BSP_MOTOR_USART_H_
#define __BSP_MOTOR_USART_H_

#include <Arduino.h>
#include "imu_uart_driver.hpp"
#include <SoftwareSerial.h>

extern SoftwareSerial printSerial;
//#define  printSerial Serial

void Usart_init (void);
void USART_Send_U8(uint8_t Data);
void USART_Send_ArrayU8(uint8_t *pData, uint16_t Length);
void USART_Recieve(void);



#endif

