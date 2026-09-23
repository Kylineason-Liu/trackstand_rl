#ifndef __BSP_MOTOR_IIC_H_
#define __BSP_MOTOR_IIC_H_

#include "Wire.h"

void IIC_Init(void);
int i2cWrite(uint8_t devAddr, uint8_t regAddr, uint8_t length, uint8_t *data);
int i2cRead(uint8_t devAddr, uint8_t regAddr, uint8_t length, uint8_t *data);

#endif
