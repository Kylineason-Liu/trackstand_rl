#include "usart.h"

#include <errno.h>
#include <stdint.h>

#define RE_0_BUFF_LEN_MAX	128

volatile uint8_t  recv0_buff[RE_0_BUFF_LEN_MAX] = {0};
volatile uint16_t recv0_length = 0;
volatile uint8_t  recv0_flag = 0;

static void UART_SendByteBlocking(uint8_t data)
{
	while (DL_UART_isBusy(UART_0_INST) == true) {
	}
	DL_UART_Main_transmitData(UART_0_INST, data);
}

static void UART_SendBufferBlocking(const uint8_t *buffer, uint32_t length)
{
	if (buffer == NULL || length == 0U) {
		return;
	}
	for (uint32_t i = 0; i < length; ++i) {
		UART_SendByteBlocking(buffer[i]);
	}
}

void USART_Init(void)
{
	//清除串口中断标志
	//Clear the serial port interrupt flag
	NVIC_ClearPendingIRQ(UART_0_INST_INT_IRQN);
	//使能串口中断
	//Enable serial port interrupt
	NVIC_EnableIRQ(UART_0_INST_INT_IRQN);
}

//串口发送一个字节
//The serial port sends a byte
void USART_SendData(unsigned char data)
{
	UART_SendByteBlocking(data);
}


#if !defined(__MICROLIB)
//不使用微库的话就需要添加下面的函数
//If you don't use the micro library, you need to add the following function
#if (__ARMCLIB_VERSION <= 6000000)
//如果编译器是AC5  就定义下面这个结构体
//If the compiler is AC5, define the following structure
struct __FILE
{
	int handle;
};
#endif

FILE __stdout;

//定义_sys_exit()以避免使用半主机模式
//Define _sys_exit() to avoid using semihosting mode
void _sys_exit(int x)
{
	x = x;
}
#endif


//printf函数重定义
//printf function redefinition
int fputc(int ch, FILE *stream)
{
	//当串口0忙的时候等待，不忙的时候再发送传进来的字符
	//Wait when serial port 0 is busy, and send the incoming characters when it is not busy
	while( DL_UART_isBusy(UART_0_INST) == true );
	
	DL_UART_Main_transmitData(UART_0_INST, ch);
	
	return ch;
}
int fputs(const char* restrict s, FILE* restrict stream) {

    uint16_t char_len=0;
    while(*s!=0)
    {
        while( DL_UART_isBusy(UART_0_INST) == true );
        DL_UART_Main_transmitData(UART_0_INST, *s++);
        char_len++;
    }
    return char_len;
}
int puts(const char* _ptr)
{
 return 0;
}
//串口的中断服务函数
//Serial port interrupt service function
void UART_0_INST_IRQHandler(void)
{
	uint8_t receivedData = 0;
	
	//如果产生了串口中断
	//If a serial port interrupt occurs
	switch( DL_UART_getPendingInterrupt(UART_0_INST) )
	{
		case DL_UART_IIDX_RX://如果是接收中断	If it is a receive interrupt
			
			// 接收发送过来的数据保存	Receive and save the data sent
			receivedData = DL_UART_Main_receiveData(UART_0_INST);
		
			// 检查缓冲区是否已满	Check if the buffer is full
			if (recv0_length < RE_0_BUFF_LEN_MAX - 1)
			{
				recv0_buff[recv0_length++] = receivedData;
			}
			else
			{
				recv0_length = 0;
			}

			// 标记接收标志	Mark receiving flag
			recv0_flag = 1;
		
			break;
		
		default://其他的串口中断	Other serial port interrupts
			break;
	}
}
