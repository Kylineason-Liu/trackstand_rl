#include "delay.h"


static uint8_t fac_us = 0;  //us延时倍乘数	us delay multiplier
static uint16_t fac_ms = 0; //ms延时倍乘数	ms delay multiplier

void delay_init(void)
{
	uint8_t SYSCLK = SystemCoreClock / 1000000;
	SysTick_CLKSourceConfig(SysTick_CLKSource_HCLK_Div8); // 选择外部时钟  HCLK/8	Select external clock HCLK/8
	fac_us = SYSCLK / 8;
	fac_ms = (uint16_t)fac_us * 1000;
}


/**********************************************************
** 函数名: delay_ms		Function name: delay_ms
** 功能描述: 延时nms		Function description: Delay nms
** 输入参数: nms			Input parameter: nms
** 输出参数: 无			Output parameter: None
** 说明：SysTick->LOAD为24位寄存器,所以,最大延时为:	Note: SysTick->LOAD is a 24-bit register, so the maximum delay is:
		nms<=0xffffff*8*1000/SYSCLK					nms<=0xffffff*8*1000/SYSCLK
		SYSCLK单位为Hz,nms单位为ms					SYSCLK is in Hz, nms is in ms.
		对72M条件下,nms<=1864 						Under 72M conditions, nms<=1864
***********************************************************/
void delay_ms(uint16_t nms)
{
	uint32_t temp;
	SysTick->LOAD = (uint32_t)nms * fac_ms; //时间加载(SysTick->LOAD为24bit)	Time loading (SysTick->LOAD is 24bit)
	SysTick->VAL = 0x00;			   //清空计数器	Clear counter
	SysTick->CTRL = 0x01;			   //开始倒数	Start countdown
	do
	{
		temp = SysTick->CTRL;
	} while (temp & 0x01 && !(temp & (1 << 16))); //等待时间到达	Waiting time to arrive
	SysTick->CTRL = 0x00;						  //关闭计数器	Close Counter
	SysTick->VAL = 0X00;						  //清空计数器	Clear counter
}

/**********************************************************
** 函数名: delay_us						Function name: delay_us
** 功能描述: 延时nus，nus为要延时的us数.	Function description: Delay nus, nus is the number of us to delay.
** 输入参数: nus							Input parameter: nus
** 输出参数: 无							Output parameter: None
***********************************************************/
void delay_us(uint32_t nus)
{
	uint32_t temp;
	SysTick->LOAD = nus * fac_us; //时间加载		Time loading
	SysTick->VAL = 0x00;		  //清空计数器	Clear counter
	SysTick->CTRL = 0x01;		  //开始倒数		Start countdown
	do
	{
		temp = SysTick->CTRL;
	} while (temp & 0x01 && !(temp & (1 << 16))); //等待时间到达	Waiting time to arrive
	SysTick->CTRL = 0x00;						  //关闭计数器	Close Counter
	SysTick->VAL = 0X00;						  //清空计数器	Clear counter
}




