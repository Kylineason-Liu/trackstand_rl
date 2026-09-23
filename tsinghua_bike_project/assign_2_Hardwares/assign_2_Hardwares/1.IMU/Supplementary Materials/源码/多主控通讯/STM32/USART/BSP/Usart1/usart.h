#ifndef __USART_H
#define __USART_H
#include "AllHeader.h"



void uart_init(u32 bound);
void USART1_Send_U8(uint8_t ch);
void USART1_Send_ArrayU8(uint8_t *BufferPtr, uint16_t Length);
#endif


