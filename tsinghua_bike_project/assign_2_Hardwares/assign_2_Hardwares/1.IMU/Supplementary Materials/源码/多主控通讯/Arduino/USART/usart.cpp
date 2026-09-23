#include "usart.hpp"

#define  mySerial Serial
SoftwareSerial printSerial(2, 3);//2:RX 3:TX

//电机串口初始化	serial port initialization
void Usart_init (void)
{
  mySerial.begin(115200);
  printSerial.begin(115200);
  printSerial.println("please wait...");
}

void USART_Send_U8(uint8_t Data)
{
  mySerial.write(Data);
}
void USART_Send_ArrayU8(uint8_t *pData, uint16_t Length)
{
  while (Length--)
  {
    USART_Send_U8(*pData);
    pData++;
  }
}

void USART_Recieve(void)
{
  char Rx_Temp = 0;

  if (mySerial.available())
  {
    Rx_Temp = char(mySerial.read());
    IMU_UART_RxBytes(&Rx_Temp, 1);
  }
}


