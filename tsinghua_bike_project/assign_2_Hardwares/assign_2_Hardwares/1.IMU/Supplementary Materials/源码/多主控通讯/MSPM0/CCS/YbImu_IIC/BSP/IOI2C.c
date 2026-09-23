#include "ioi2c.h"

/**************************************************************************
Function: Simulate IIC start signal
Input   : none
Output  : 1
函数功能：模拟IIC起始信号
入口参数：无
返回  值：1
**************************************************************************/
int IIC_Start(void)
{
	SDA_OUT();
	SDA(1);
	SCL(1);
	delay_us(1);
	SDA(0);
	delay_us(1);	   
	SCL(0);
}

/**************************************************************************
Function: Analog IIC end signal
Input   : none
Output  : none
函数功能：模拟IIC结束信号
入口参数：无
返回  值：无
**************************************************************************/	  
void IIC_Stop(void)
{
	SDA_OUT();
	SCL(0);
	SDA(0);
	
	SCL(1);
	delay_us(1);
	SDA(1);
	delay_us(1);							   	
}

/**************************************************************************
Function: IIC wait the response signal
Input   : none
Output  : 0：No response received；1：Response received
函数功能：IIC等待应答信号
入口参数：无
返回  值：0：没有收到应答；1：收到应答
**************************************************************************/
int IIC_Wait_Ack(void)
{
	char ack = 0;
	unsigned char ack_flag = 10;
	SCL(0);
	SDA(1);
	SDA_IN();
	
	SCL(1);
	while( (SDA_GET()==1) && ( ack_flag ) )
	{
			ack_flag--;
			delay_us(1);
	}
	
	if( ack_flag <= 0 )
	{
			IIC_Stop();
			return 1;
	}
	else
	{
			SCL(0);
			SDA_OUT();
	}
	return ack;
} 

/******************************************************************
 * 函 数 名 称：IIC_Send_Ack
 * 函 数 说 明：主机发送应答或者非应答信号
 * 函 数 形 参：0发送应答  1发送非应答
 * 函 数 返 回：无
 * 备       注：无
 * Function: IIC_Send_Ack
 * Description: The host sends a response or non-response signal
 * Pparameter: 0 sends a response 1 sends a non-response
 * Return: None
 * Notes: None
******************************************************************/
void IIC_Send_Ack(unsigned char ack)
{
        SDA_OUT();
        SCL(0);
        SDA(0);
        delay_us(5);
        if(!ack) SDA(0);
        else         SDA(1);
        SCL(1);
        delay_us(5);
        SCL(0);
        SDA(1);
}

/**************************************************************************
Function: IIC sends a byte
Input   : txd：Byte data sent
Output  : none
函数功能：IIC发送一个字节
入口参数：txd：发送的字节数据
返回  值：无
**************************************************************************/	  
void IIC_Send_Byte(u8 txd)
{                        
	int i = 0;
	SDA_OUT();
	SCL(0);//拉低时钟开始数据传输 Pull the clock low to start data transmission
	
	for( i = 0; i < 8; i++ )
	{
			SDA( (txd & 0x80) >> 7 );
			delay_us(1);
			SCL(1);
			delay_us(5);
			SCL(0);
			delay_us(5);
			txd<<=1;
	}     
} 	 
  
/**************************************************************************
Function: IIC write data to register
Input   : addr：Device address；reg：Register address；len;Number of bytes；data：Data
Output  : 0：Write successfully；1：Failed to write
函数功能：IIC写数据到寄存器
入口参数：addr：设备地址；reg：寄存器地址；len;字节数；data：数据
返回  值：0：成功写入；1：没有成功写入
**************************************************************************/
int i2cWrite(uint8_t addr, uint8_t reg, uint8_t len, uint8_t *data)
{
    uint16_t i = 0;
        IIC_Start();
        IIC_Send_Byte((addr<<1)|0);
        if( IIC_Wait_Ack() == 1 ) {IIC_Stop();return 1;}
        IIC_Send_Byte(reg);
        if( IIC_Wait_Ack() == 1 ) {IIC_Stop();return 2;}
    
        for(i=0;i<len;i++)
    {
        IIC_Send_Byte(data[i]);
        if( IIC_Wait_Ack() == 1 ) {IIC_Stop();return (3+i);}
    }        
        IIC_Stop();
    return 0;
}
/**************************************************************************
Function: IIC read register data
Input   : addr：Device address；reg：Register address；len;Number of bytes；*buf：Data read out
Output  : 0：Read successfully；1：Failed to read
函数功能：IIC读寄存器的数据
入口参数：addr：设备地址；reg：寄存器地址；len;字节数；*buf：读出数据缓存
返回  值：0：成功读出；1：没有成功读出
**************************************************************************/

int i2cRead(uint8_t addr, uint8_t reg, uint8_t len, uint8_t *buf)
{
	uint8_t i;
	IIC_Start();
	IIC_Send_Byte((addr<<1)|0);
	if( IIC_Wait_Ack() == 1 ) {IIC_Stop();return 1;}
	IIC_Send_Byte(reg);
	if( IIC_Wait_Ack() == 1 ) {IIC_Stop();return 2;}
	
	IIC_Start();
	IIC_Send_Byte((addr<<1)|1);
	if( IIC_Wait_Ack() == 1 ) {IIC_Stop();return 3;}
	
	for(i=0;i<(len-1);i++){
			buf[i]=IIC_Read_Byte();
			IIC_Send_Ack(0);
	}
	buf[i]=IIC_Read_Byte();
	IIC_Send_Ack(1);         
	IIC_Stop();
	return 0;
}

/**************************************************************************
Function: IIC reads a byte
Input   : ack：Send response signal or not；1：Send；0：Do not send
Output  : receive：Data read
函数功能：IIC读取一个位
入口参数：ack：是否发送应答信号；1：发送；0：不发送
返回  值：receive：读取的数据
**************************************************************************/ 
u8 IIC_Read_Byte(void)
{
	unsigned char i,receive=0;
	SDA_IN();//SDA设置为输入 SDA is set as input
    for(i=0;i<8;i++ )
        {
        SCL(0);
        delay_us(5);
        SCL(1);
        delay_us(5);
        receive<<=1;
        if( SDA_GET() )
        {        
            receive|=1;   
        }
        delay_us(5); 
    }                                         
	SCL(0); 
	return receive;
}

