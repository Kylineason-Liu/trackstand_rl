/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    usart.h
  * @brief   This file contains all the function prototypes for
  *          the usart.c file
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __USART_H__
#define __USART_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* USER CODE BEGIN Includes */

#include "stdbool.h"
/* USER CODE END Includes */

extern UART_HandleTypeDef huart1;

extern UART_HandleTypeDef huart3;

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

void MX_USART1_UART_Init(void);
void MX_USART3_UART_Init(void);

/* USER CODE BEGIN Prototypes */
void UART_IDLEIT_Enable(UART_HandleTypeDef *huart);
void UART_Receive_DMA(UART_HandleTypeDef *huart, uint8_t *pData, uint16_t Size);
HAL_StatusTypeDef UART_Transmit_DMA(UART_HandleTypeDef *huart, uint8_t *pData, uint16_t Size);
bool UART_GET_FLAG(UART_HandleTypeDef *huart, uint32_t FLAG);
void UART_DMAStop(UART_HandleTypeDef *huart);
void UART_CLEAR_IDLEFLAG(UART_HandleTypeDef *huart);
uint32_t UART_DMA_GetReceiveCounter(UART_HandleTypeDef *huart);
void UART_CLEAR_OREFLAG(UART_HandleTypeDef *huart);
void UART_EN_RS485_RX(UART_HandleTypeDef *huart);
void UART_EN_RS485_TX(UART_HandleTypeDef *huart);
bool UART_GET_EN_STATE(UART_HandleTypeDef *huart);

/* USER CODE END Prototypes */

#ifdef __cplusplus
}
#endif

#endif /* __USART_H__ */

