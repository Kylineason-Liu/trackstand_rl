#include "bsp_timer.h"


static u16 stop_time = 0;//延迟时间  delay time



//定时器6做延迟 10ms的延迟 此方法比delay准确
//Timer 6 has a delay of 10ms. This method is more accurate than delay
void delay_time(u16 time)
{
	stop_time = time;
	while(stop_time);//死等 Wait
}

//延迟1s  Unit second
void my_delay(u16 s)//s
{
	for(int i = 0;i<s;i++)
	{
		delay_time(100);
	}
}


/**************************************************************************
Function function: TIM3 initialization, timed for 10 milliseconds
Entrance parameters: None
Return value: None
函数功能：TIM3初始化，定时10毫秒
入口参数：无
返回  值：无
**************************************************************************/
void TIM3_Init(void)
{
	TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure;
	NVIC_InitTypeDef NVIC_InitStructure;
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM3, ENABLE); //使能定时器的时钟  Enable the clock of the timer
	TIM_TimeBaseStructure.TIM_Prescaler = 7199;			 // 预分频器  Prescaler
	TIM_TimeBaseStructure.TIM_Period = 99;				 //设定计数器自动重装值  Set the automatic reset value of the counter
	TIM_TimeBaseInit(TIM3, &TIM_TimeBaseStructure);
	TIM_ClearFlag(TIM3, TIM_FLAG_Update);                //清除TIM的更新标志位 Clear the update flag of TIM
	TIM_ITConfig(TIM3, TIM_IT_Update, ENABLE);

	//中断优先级NVIC设置
	NVIC_InitStructure.NVIC_IRQChannel = TIM3_IRQn;			  //TIM6中断	TIM6 interrupt
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 4; //先占优先级4级	Preempts priority level 4
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 2;		  //从优先级2级	From priority level 2
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;			  //IRQ通道被使能	IRQ channel is enabled
	NVIC_Init(&NVIC_InitStructure);							  //初始化NVIC寄存器	Initializes NVIC registers

	TIM_Cmd(TIM3, ENABLE);
}


// TIM3中断 //TIM3 Interrupt service
void TIM3_IRQHandler(void)
{
	if (TIM_GetITStatus(TIM3, TIM_IT_Update) != RESET) //检查TIM更新中断发生与否	Check whether TIM update interrupt occurs
	{
		TIM_ClearITPendingBit(TIM3, TIM_IT_Update);    //清除TIMx更新中断标志	Clear TIMx update interrupt flag
		
		if(stop_time>0)
		{
			stop_time --;
		}
	}
}

