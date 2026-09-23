#ifndef __UART_MODULE_H__
#define __UART_MODULE_H__

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "imu_uart_driver.h"

// 引脚配置 Pin Configuration
#define UART1_TX_PIN    36
#define UART1_RX_PIN    35
#define UART_BAUD  115200

extern QueueHandle_t uart_queue;

void uart1_init(void);
int USART_SendArray(uint8_t* data, uint16_t len);
int USART_SendData(uint8_t data);
void UART_Process_Task(void *pvParameters);


#endif