#ifndef __BSP_TIMER_H__
#define __BSP_TIMER_H__

#include "AllHeader.h"


void TIM6_Init(void);
void TIM7_Init(void);
void TIM3_Init(void);

void power_decect(void);
void cotrol_led(void);

void delay_time(u16 time);
void my_delay(u16 s);


#endif
