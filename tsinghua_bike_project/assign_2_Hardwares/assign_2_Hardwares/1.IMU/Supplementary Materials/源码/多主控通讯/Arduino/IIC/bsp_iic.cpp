#include "bsp_iic.hpp"

int32_t Encoder_Offset[4];
int32_t Encoder_Now[4];

// I2C 写函数	I2C Write Function
int i2cWrite(uint8_t devAddr, uint8_t regAddr, uint8_t length, uint8_t *data) {
  Wire.beginTransmission(devAddr);
  Wire.write(regAddr);
  for (uint8_t i = 0; i < length; i++) {
    Wire.write(data[i]);
  }
  Wire.endTransmission();
  return 0;
}

// I2C 读函数	I2C Read Function
int i2cRead(uint8_t devAddr, uint8_t regAddr, uint8_t length, uint8_t *data) {
  Wire.beginTransmission(devAddr);
  Wire.write(regAddr);
  Wire.endTransmission(false);
  Wire.requestFrom(devAddr, length);
  for (uint8_t i = 0; i < length; i++) {
    data[i] = Wire.read();
  }
  return 0;
}
// IIC 初始化	IIC Initialization
void IIC_Init(void)
{			
	Wire.begin();
}

