#include "comm_rs485.h"
#include "usart.h"
#include "crc16_modbus.h"
#include "stdio.h"
#include "string.h"

uint8_t comm_uart_rx_buffer[RX_LEN] = {0};
uint8_t comm_uart_tx_buffer[RX_LEN] = {0};

_RS485_Para_t ZE300_RS485BACK_DATA;

void CommRs485_Init(void)
{
    UART_IDLEIT_Enable(&huart3);
    UART_Receive_DMA(&huart3, comm_uart_rx_buffer, RX_LEN);
    UART_EN_RS485_RX(&huart3);
}

/* 数据发送完成中断 */
void HAL_UART_TxCpltCallback(UART_HandleTypeDef  *huart)
{
    UART_EN_RS485_RX(huart);
}

static void _TransmitProtocolData(UartProtocolData_t* pData)
{
    uint16_t tmp_crc = CalcCRC16_Modbus(comm_uart_tx_buffer, (pData->data_len + 5));
    pData->buf[pData->data_len] = (uint8_t)tmp_crc;
    pData->buf[pData->data_len + 1] = (uint8_t)(tmp_crc>>8);
    
    UART_EN_RS485_TX(&huart3);
    if (HAL_OK != UART_Transmit_DMA(&huart3, comm_uart_tx_buffer, pData->data_len +7))
    {
        UART_EN_RS485_RX(&huart3);
    }
}


/* 发送不带参数的命令 */
static void _SendNoParaCmd(uint8_t dev_addr, uint8_t cmd)
{
	UartProtocolData_t *pData = (UartProtocolData_t*)comm_uart_tx_buffer;
    pData->head = UART_TX_PROTOCOL_HEADER_V3;
    pData->pack_num = 0;
    pData->addr = dev_addr;
    pData->cmd = cmd;
    pData->data_len = 0;
    
    _TransmitProtocolData(pData);
}


/* 系统复位 */
void CommRs485_Restart(uint8_t dev_addr)
{
    _SendNoParaCmd(dev_addr, eRestart);
}

/* 编码器校准*/
void CommRs485_EncoderCalibration(uint8_t dev_addr)
{
	_SendNoParaCmd(dev_addr, eEncoderCalibration);
}

/* 获取版本信息 */
void CommRs485_GetVerInfo(uint8_t dev_addr)
{
    _SendNoParaCmd(dev_addr, eGetVerInfo);
}

/* 获取实时信息 */
void CommRs485_GetRealtimeInfo(uint8_t dev_addr)
{
    _SendNoParaCmd(dev_addr, eGetRealtimeInfo);
}

/* 清除故障 */
void CommRs485_ResetFault(uint8_t dev_addr)
{
    _SendNoParaCmd(dev_addr, eResetFault);
}

/* 用户参数读取*/
void CommRs485_ReadeUserPara(uint8_t dev_addr)
{
    _SendNoParaCmd(dev_addr, eReadUserPara);
}

/* 用户参数写入*/
void CommRs485_WriteUserPara(uint8_t dev_addr, UserParameter_t user_para)
{
	UartProtocolData_t *pData = (UartProtocolData_t*)comm_uart_tx_buffer;
    uint32_t tmp_val;
	
    pData->head = UART_TX_PROTOCOL_HEADER_V3;
    pData->pack_num = 0;
    pData->addr = dev_addr;
    pData->cmd = eWriteUserPara;
    pData->data_len =0x10;
	
	memcpy(&pData->buf[0], &user_para.EncoderType, sizeof(user_para.EncoderType));
	memcpy(&pData->buf[1], &user_para.EncoderDirection, sizeof(user_para.EncoderDirection));
	memcpy(&pData->buf[2], &user_para.SeCEncoderEnable, sizeof(user_para.SeCEncoderEnable));
	memcpy(&pData->buf[3], &user_para.VelocityFiltFactor, sizeof(user_para.VelocityFiltFactor));
	memcpy(&pData->buf[4], &user_para.DevAddr, sizeof(user_para.DevAddr));
	memcpy(&pData->buf[5], &user_para.RS485Baud, sizeof(user_para.RS485Baud));
	memcpy(&pData->buf[6], &user_para.CANBaud, sizeof(user_para.CANBaud));
	memcpy(&pData->buf[7], &user_para.CANopenEnable, sizeof(user_para.CANopenEnable));
	
	tmp_val  = user_para.BusVoltageMax *100;
	memcpy(&pData->buf[8], &tmp_val, sizeof(tmp_val));
	
	memcpy(&pData->buf[10], &user_para.VoltageFaultTime, sizeof(user_para.VoltageFaultTime));
	
	tmp_val  = user_para.BusCurrentMax *100;
	memcpy(&pData->buf[11], &tmp_val, sizeof(tmp_val));
	
	memcpy(&pData->buf[13], &user_para.CurrentFaultTime, sizeof(user_para.CurrentFaultTime));
	memcpy(&pData->buf[14], &user_para.Temperature, sizeof(user_para.Temperature));
	memcpy(&pData->buf[15], &user_para.TemperatureFaultTime, sizeof(user_para.TemperatureFaultTime));
	
	_TransmitProtocolData(pData);
}
/* 电机硬件参数读取*/
void CommRs485_ReadeMotorPara(uint8_t dev_addr)
{
    _SendNoParaCmd(dev_addr, eReadMotorPara);
}

/* 电机硬件参数写入*/
void CommRs485_WriteMotorPara(uint8_t dev_addr, MotorParameter_t motor_para)
{
	UartProtocolData_t *pData = (UartProtocolData_t*)comm_uart_tx_buffer;
	
    pData->head = UART_TX_PROTOCOL_HEADER_V3;
    pData->pack_num = 0;
    pData->addr = dev_addr;
    pData->cmd = eWriteMotorPara;
    pData->data_len =0x1E;

	memcpy(&pData->buf[0], &motor_para.MotorName, sizeof(motor_para.MotorName));
	memcpy(&pData->buf[16], &motor_para.MotorPole, sizeof(motor_para.MotorPole));
	memcpy(&pData->buf[17], &motor_para.PhaseRes, sizeof(motor_para.PhaseRes));
	memcpy(&pData->buf[21], &motor_para.PhaseIcl, sizeof(motor_para.PhaseIcl));
	memcpy(&pData->buf[25], &motor_para.TorqueConstant, sizeof(motor_para.TorqueConstant));
	memcpy(&pData->buf[29], &motor_para.ReductionRatio, sizeof(motor_para.ReductionRatio));
	
	_TransmitProtocolData(pData);
}

/* 读取运动控制参数 */
void CommRs485_GetMotionCtrlPara(uint8_t dev_addr)
{
    _SendNoParaCmd(dev_addr, eGetMotionCtrlPara);
}

/* 写运动控制参数 */
void CommRs485_WriteMotionCtrlPara(uint8_t dev_addr, MotionCtrlPara_t pid , bool is_save)
{
	UartProtocolData_t *pData = (UartProtocolData_t*)comm_uart_tx_buffer;
    uint32_t tmp_val;
    
    pData->head = UART_TX_PROTOCOL_HEADER_V3;
    pData->pack_num = 0;
    pData->addr = dev_addr;
    pData->cmd = (is_save == true) ? eWriteAndSaveMotionCtrlPara : eWriteMotionCtrlPara ;
    pData->data_len =0x18;
    
    memcpy(&pData->buf[0], &pid.pos_kp, sizeof(pid.pos_kp));
    memcpy(&pData->buf[4], &pid.pos_ki, sizeof(pid.pos_ki));
    
    tmp_val = pid.pos_out_limit*100;
    memcpy(&pData->buf[8], &tmp_val, sizeof(tmp_val));
    
    memcpy(&pData->buf[12], &pid.vel_kp, sizeof(pid.vel_kp));
    memcpy(&pData->buf[16], &pid.vel_ki, sizeof(pid.vel_ki));
    
    tmp_val = pid.vel_out_limit*1000;
    memcpy(&pData->buf[20], &tmp_val, sizeof(tmp_val));
    
    _TransmitProtocolData(pData);
}


/* 设置当前位置为原点 */
void CommRs485_SetZeroByCurrentRawAngle(uint8_t dev_addr)
{
    _SendNoParaCmd(dev_addr, eSetZeroByCurrentRawAngle);
	
}

/* 参数恢复默认值 */
void CommRs485_RestSysPara(uint8_t dev_addr)
{
    _SendNoParaCmd(dev_addr, eRestSysPara);
}

/* Q轴电流控制*/
void CommRs485_Torque(uint8_t dev_addr, float target_Iq, float target_Slope)
{
    UartProtocolData_t *pData = (UartProtocolData_t*)comm_uart_tx_buffer;
    
    int32_t tmp_iq = (int32_t)(target_Iq*1000);
    uint32_t tmp_slope = (uint32_t)(target_Slope*1000);
    
    pData->head = UART_TX_PROTOCOL_HEADER_V3;
    pData->pack_num = 0;
    pData->addr = dev_addr;
    pData->cmd = eTorqueCtrl;
    pData->data_len = 8;
    
    memcpy(&pData->buf[0], &tmp_iq, sizeof(tmp_iq));
    memcpy(&pData->buf[4], &tmp_slope, sizeof(tmp_slope));
    
    _TransmitProtocolData(pData);
}

/* 速度控制，输入的target_vel单位为Rpm，target_acc单位为Rpm/s*/
void CommRs485_VelocityCtrl(uint8_t dev_addr, float target_vel, float target_acc)
{
    UartProtocolData_t *pData = (UartProtocolData_t*)comm_uart_tx_buffer;
    
    int32_t tmp_vel = (int32_t)(target_vel*100);
    uint32_t tmp_acc = (uint32_t)(target_acc*100);
    
    pData->head = UART_TX_PROTOCOL_HEADER_V3;
    pData->pack_num = 0;
    pData->addr = dev_addr;
    pData->cmd = eVelocityCtrl;
    pData->data_len = 8;
    
    memcpy(&pData->buf[0], &tmp_vel, sizeof(tmp_vel));
    memcpy(&pData->buf[4], &tmp_acc, sizeof(tmp_acc));
    
    _TransmitProtocolData(pData);
}

/* 绝对位置控制，单位为count，一圈16384 */
void CommRs485_AbsolutePositionCtrl(uint8_t dev_addr, int32_t target_count)
{
    UartProtocolData_t *pData = (UartProtocolData_t*)comm_uart_tx_buffer;
        
    pData->head = UART_TX_PROTOCOL_HEADER_V3;
    pData->pack_num = 0;
    pData->addr = dev_addr;
    pData->cmd = eAbsolutePositionCtrl;
    pData->data_len = 4;
    
    memcpy(&pData->buf[0], &target_count, sizeof(target_count));
    
    _TransmitProtocolData(pData);
}

/* 相对位置控制，单位为count，一圈16384 */
void CommRs485_RelativePositionCtrl(uint8_t dev_addr, int32_t target_count)
{
    UartProtocolData_t *pData = (UartProtocolData_t*)comm_uart_tx_buffer;
    
    pData->head = UART_TX_PROTOCOL_HEADER_V3;
    pData->pack_num = 0;
    pData->addr = dev_addr;
    pData->cmd = eRelativePositionCtrl;
    pData->data_len = 4;
    
    memcpy(&pData->buf[0], &target_count, sizeof(target_count));
    
    _TransmitProtocolData(pData);
}

/* 设置最短距离返回原点 */
void CommRs485_ShortestHomePosition(uint8_t dev_addr)
{
    UartProtocolData_t *pData = (UartProtocolData_t*)comm_uart_tx_buffer;
    
    pData->head = UART_TX_PROTOCOL_HEADER_V3;
    pData->pack_num = 0;
    pData->addr = dev_addr;
    pData->cmd = eShortestHomePosition;
    pData->data_len = 0;
    
    _TransmitProtocolData(pData);
}


/* 抱阀控制开关输出控制 */
void CommRs485_eBreakCtrl(uint8_t dev_addr, uint16_t target_state)
{
    UartProtocolData_t *pData = (UartProtocolData_t*)comm_uart_tx_buffer;
    
    int32_t tmp_state = (int32_t)(target_state);
    
    pData->head = UART_TX_PROTOCOL_HEADER_V3;
    pData->pack_num = 0;
    pData->addr = dev_addr;
    pData->cmd = eBreakCtrl;
    pData->data_len = 1;
    
    memcpy(&pData->buf[0], &tmp_state, sizeof(tmp_state));
    
    _TransmitProtocolData(pData);
}


/* 参数恢复默认值 */
void CommRs485_DisableCtrl(uint8_t dev_addr)
{
    _SendNoParaCmd(dev_addr, eDisableCtrl);
}

/* 串口空闲中断 */
void UART_IDLE_IT_Callback(UART_HandleTypeDef *huart)
{
    uint8_t size = 0;
    
    if(UART_GET_FLAG(huart, UART_FLAG_IDLE) != RESET)
    {
        UART_DMAStop(huart);
        UART_CLEAR_IDLEFLAG(huart);
        
        size = (RX_LEN - UART_DMA_GetReceiveCounter(huart));
        if(size != 0)
        {
            
            if((size >= 7) && (CalcCRC16_Modbus(comm_uart_rx_buffer, size) == 0))
            {
                UartProtocolData_t *pData = (UartProtocolData_t*)comm_uart_rx_buffer;
                if(pData->head == UART_RX_PROTOCOL_HEADER_V3)
                {
                    switch(pData->cmd)
                    {
                        case eGetVerInfo:
						{
                            memcpy(&ZE300_RS485BACK_DATA.BoardVerInfo.boot_ver, &(pData->buf[0]), sizeof(ZE300_RS485BACK_DATA.BoardVerInfo.software_ver));
                            memcpy(&ZE300_RS485BACK_DATA.BoardVerInfo.software_ver, &(pData->buf[2]), sizeof(ZE300_RS485BACK_DATA.BoardVerInfo.software_ver));
                            memcpy(&ZE300_RS485BACK_DATA.BoardVerInfo.hardware_ver, &(pData->buf[4]), sizeof(ZE300_RS485BACK_DATA.BoardVerInfo.hardware_ver));
                            memcpy(&ZE300_RS485BACK_DATA.BoardVerInfo.aux_rs485_ver, &(pData->buf[6]), sizeof(ZE300_RS485BACK_DATA.BoardVerInfo.aux_rs485_ver));
                            memcpy(&ZE300_RS485BACK_DATA.BoardVerInfo.modbus_rs485_ver, &(pData->buf[7]), sizeof(ZE300_RS485BACK_DATA.BoardVerInfo.modbus_rs485_ver));
                            memcpy(&ZE300_RS485BACK_DATA.BoardVerInfo.aux_can_ver, &(pData->buf[8]), sizeof(ZE300_RS485BACK_DATA.BoardVerInfo.aux_can_ver));
                            memcpy(&ZE300_RS485BACK_DATA.BoardVerInfo.canopen_ver, &(pData->buf[9]), sizeof(ZE300_RS485BACK_DATA.BoardVerInfo.canopen_ver));
                            memcpy(&ZE300_RS485BACK_DATA.BoardVerInfo.UID, &(pData->buf[10]), sizeof(ZE300_RS485BACK_DATA.BoardVerInfo.UID));
                            
                            printf("\r\n");
							printf("Boot软件版本 :%d\r\n", ZE300_RS485BACK_DATA.BoardVerInfo.boot_ver);
							printf("应用软件版本 :%d\r\n", ZE300_RS485BACK_DATA.BoardVerInfo.software_ver);
							printf("硬件版本     :%02X\r\n", ZE300_RS485BACK_DATA.BoardVerInfo.hardware_ver);
							printf("RS48-自定义  :%d\r\n", ZE300_RS485BACK_DATA.BoardVerInfo.aux_rs485_ver);
							printf("RS485-Modbus :%d\r\n", ZE300_RS485BACK_DATA.BoardVerInfo.modbus_rs485_ver);
							printf("CAN-自定义   :%d\r\n", ZE300_RS485BACK_DATA.BoardVerInfo.aux_can_ver);
							printf("Can-Canopen  :%d\r\n", ZE300_RS485BACK_DATA.BoardVerInfo.canopen_ver);
							printf("UID唯一序列号:");
							for(int i = 0; i < 12; i++)
							{
								printf("%02X",ZE300_RS485BACK_DATA.BoardVerInfo.UID[i]);
							}
							printf("\r\n");
                            printf("\r\n");
						}
						break;
                        
						case eReadUserPara:
						case eWriteUserPara:
						{
                            memcpy(&ZE300_RS485BACK_DATA.UserParameter.EncoderType, &(pData->buf[10]), sizeof(ZE300_RS485BACK_DATA.UserParameter.EncoderType));
							memcpy(&ZE300_RS485BACK_DATA.UserParameter.EncoderDirection, &(pData->buf[11]), sizeof(ZE300_RS485BACK_DATA.UserParameter.EncoderDirection));
							memcpy(&ZE300_RS485BACK_DATA.UserParameter.SeCEncoderEnable, &(pData->buf[12]), sizeof(ZE300_RS485BACK_DATA.UserParameter.SeCEncoderEnable));
							memcpy(&ZE300_RS485BACK_DATA.UserParameter.VelocityFiltFactor, &(pData->buf[13]), sizeof(ZE300_RS485BACK_DATA.UserParameter.VelocityFiltFactor));
							memcpy(&ZE300_RS485BACK_DATA.UserParameter.DevAddr, &(pData->buf[14]), sizeof(ZE300_RS485BACK_DATA.UserParameter.DevAddr));
							memcpy(&ZE300_RS485BACK_DATA.UserParameter.RS485Baud, &(pData->buf[15]), sizeof(ZE300_RS485BACK_DATA.UserParameter.RS485Baud));
							memcpy(&ZE300_RS485BACK_DATA.UserParameter.CANBaud, &(pData->buf[16]), sizeof(ZE300_RS485BACK_DATA.UserParameter.CANBaud));
							memcpy(&ZE300_RS485BACK_DATA.UserParameter.CANopenEnable, &(pData->buf[17]), sizeof(ZE300_RS485BACK_DATA.UserParameter.CANopenEnable));
							memcpy(&ZE300_RS485BACK_DATA.UserParameter.BusVoltageMax, &(pData->buf[18]), sizeof(ZE300_RS485BACK_DATA.UserParameter.BusVoltageMax));
							memcpy(&ZE300_RS485BACK_DATA.UserParameter.VoltageFaultTime, &(pData->buf[20]), sizeof(ZE300_RS485BACK_DATA.UserParameter.VoltageFaultTime));
							memcpy(&ZE300_RS485BACK_DATA.UserParameter.BusCurrentMax, &(pData->buf[21]), sizeof(ZE300_RS485BACK_DATA.UserParameter.BusCurrentMax));
							memcpy(&ZE300_RS485BACK_DATA.UserParameter.CurrentFaultTime, &(pData->buf[23]), sizeof(ZE300_RS485BACK_DATA.UserParameter.CurrentFaultTime));
							memcpy(&ZE300_RS485BACK_DATA.UserParameter.Temperature, &(pData->buf[24]), sizeof(ZE300_RS485BACK_DATA.UserParameter.Temperature));
							memcpy(&ZE300_RS485BACK_DATA.UserParameter.TemperatureFaultTime, &(pData->buf[25]), sizeof(ZE300_RS485BACK_DATA.UserParameter.TemperatureFaultTime));
							
							printf("编码器型号：%d\r\n",ZE300_RS485BACK_DATA.UserParameter.EncoderType);
							printf("编码器方向：%d\r\n",ZE300_RS485BACK_DATA.UserParameter.EncoderDirection);
							printf("第二编码器：%d\r\n",ZE300_RS485BACK_DATA.UserParameter.SeCEncoderEnable);
							printf("速度滤波系数：%d\r\n",ZE300_RS485BACK_DATA.UserParameter.VelocityFiltFactor);
							printf("设备地址：%d\r\n",ZE300_RS485BACK_DATA.UserParameter.DevAddr);
							printf("RS485波特率：%d\r\n",ZE300_RS485BACK_DATA.UserParameter.RS485Baud);
							printf("CAN波特率：%d\r\n",ZE300_RS485BACK_DATA.UserParameter.CANBaud);
							printf("CANopen使能：%d\r\n",ZE300_RS485BACK_DATA.UserParameter.CANopenEnable);
							printf("最大母线电压：%fV\r\n",ZE300_RS485BACK_DATA.UserParameter.BusVoltageMax*0.01);
							printf("报电压故障时长：%ds\r\n",ZE300_RS485BACK_DATA.UserParameter.VoltageFaultTime);
							printf("最大母线电流：%fA\r\n",ZE300_RS485BACK_DATA.UserParameter.BusCurrentMax*0.01);
							printf("报电流故障时长：%ds\r\n",ZE300_RS485BACK_DATA.UserParameter.CurrentFaultTime);
							printf("最大温度：%d℃\r\n",ZE300_RS485BACK_DATA.UserParameter.Temperature);
							printf("报温度故障时长：%ds\r\n",ZE300_RS485BACK_DATA.UserParameter.TemperatureFaultTime);
						}
						break;
						
						case eReadMotorPara:
						case eWriteMotorPara:
						{
							memcpy(&ZE300_RS485BACK_DATA.MotorParameter.MotorName, &(pData->buf[0]), sizeof(ZE300_RS485BACK_DATA.MotorParameter.MotorName));
							memcpy(&ZE300_RS485BACK_DATA.MotorParameter.MotorPole, &(pData->buf[16]), sizeof(ZE300_RS485BACK_DATA.MotorParameter.MotorPole));
							memcpy(&ZE300_RS485BACK_DATA.MotorParameter.PhaseRes, &(pData->buf[17]), sizeof(ZE300_RS485BACK_DATA.MotorParameter.PhaseRes));
							memcpy(&ZE300_RS485BACK_DATA.MotorParameter.PhaseIcl, &(pData->buf[21]), sizeof(ZE300_RS485BACK_DATA.MotorParameter.PhaseIcl));
							memcpy(&ZE300_RS485BACK_DATA.MotorParameter.TorqueConstant, &(pData->buf[25]), sizeof(ZE300_RS485BACK_DATA.MotorParameter.TorqueConstant));
							memcpy(&ZE300_RS485BACK_DATA.MotorParameter.ReductionRatio, &(pData->buf[29]), sizeof(ZE300_RS485BACK_DATA.MotorParameter.ReductionRatio));
	
							printf("电机名称：%s\r\n",ZE300_RS485BACK_DATA.MotorParameter.MotorName);
							printf("极对数：%d\r\n",ZE300_RS485BACK_DATA.MotorParameter.MotorPole);
							printf("相电阻：%fΩ\r\n",ZE300_RS485BACK_DATA.MotorParameter.PhaseRes);
							printf("相电感：%fmH\r\n",ZE300_RS485BACK_DATA.MotorParameter.PhaseIcl);
							printf("力矩常数：%fN/m\r\n",ZE300_RS485BACK_DATA.MotorParameter.TorqueConstant);
							printf("减速比：%d\r\n",ZE300_RS485BACK_DATA.MotorParameter.ReductionRatio);
						}
						break;
						
                        case eGetRealtimeInfo:
						case eEncoderCalibration:
                        case eTorqueCtrl:
                        case eVelocityCtrl:
                        case eAbsolutePositionCtrl:
                        case eRelativePositionCtrl:
                        case eShortestHomePosition:
                        case eDisableCtrl:
                        {
                            memcpy(&ZE300_RS485BACK_DATA.RealTimeInfo.MecRawAngle, &(pData->buf[0]), sizeof(ZE300_RS485BACK_DATA.RealTimeInfo.MecRawAngle));                    /* 单圈绝对值角度[0]-[1] */
                            memcpy(&ZE300_RS485BACK_DATA.RealTimeInfo.MecPosition, &(pData->buf[2]), sizeof(ZE300_RS485BACK_DATA.RealTimeInfo.MecPosition));                    /* 多圈绝对值角度[2]-[5] */
                            memcpy(&ZE300_RS485BACK_DATA.RealTimeInfo.MecVel_Mulit100, &(pData->buf[6]), sizeof(ZE300_RS485BACK_DATA.RealTimeInfo.MecVel_Mulit100));            /* 机械速度[6]-[9] */
                            memcpy(&ZE300_RS485BACK_DATA.RealTimeInfo.Iq_Mulit1000, &(pData->buf[10]), sizeof(ZE300_RS485BACK_DATA.RealTimeInfo.Iq_Mulit1000));                 /* Q轴电流[10]-[13] */
                            memcpy(&ZE300_RS485BACK_DATA.RealTimeInfo.BusVoltage_multi100, &(pData->buf[14]), sizeof(ZE300_RS485BACK_DATA.RealTimeInfo.BusVoltage_multi100));   /* 母线电压[14]-[15] */
                            memcpy(&ZE300_RS485BACK_DATA.RealTimeInfo.BusCurrent_multi100, &(pData->buf[16]), sizeof(ZE300_RS485BACK_DATA.RealTimeInfo.BusCurrent_multi100));   /* 母线电流[16]-[17] */
                            ZE300_RS485BACK_DATA.RealTimeInfo.Temperature = pData->buf[18];     /* 工作温度[18] */
                            ZE300_RS485BACK_DATA.RealTimeInfo.RunMode = pData->buf[19];         /* 运行模式[19] */
                            ZE300_RS485BACK_DATA.RealTimeInfo.CtrlState = pData->buf[20];       /* 电机状态[20] */
                            ZE300_RS485BACK_DATA.RealTimeInfo.SysFault = pData->buf[21];        /* 故障码[21] */
                            
                            printf("\r\n");
                            printf("Addr:%d\r\n", pData->addr);
                            printf("MecRawAngle:%f°\r\n", (ZE300_RS485BACK_DATA.RealTimeInfo.MecRawAngle/16384.0*360));
                            printf("MecPosition:%f°\r\n", (ZE300_RS485BACK_DATA.RealTimeInfo.MecPosition/16384.0*360));
                            printf("MecVelocity:%fRpm\r\n", (ZE300_RS485BACK_DATA.RealTimeInfo.MecVel_Mulit100/100.0));
                            printf("Iq:%fA\r\n", (ZE300_RS485BACK_DATA.RealTimeInfo.Iq_Mulit1000/1000.0));
                            printf("BusVoltage:%fV\r\n", (ZE300_RS485BACK_DATA.RealTimeInfo.BusVoltage_multi100/100.0));
                            printf("BusCurrent:%fA\r\n", (ZE300_RS485BACK_DATA.RealTimeInfo.BusCurrent_multi100/100.0));
                            printf("Temperature:%d℃\r\n", ZE300_RS485BACK_DATA.RealTimeInfo.Temperature);
                            printf("RunMode:%d\r\n", ZE300_RS485BACK_DATA.RealTimeInfo.RunMode);
                            printf("CtrlState:%d\r\n", ZE300_RS485BACK_DATA.RealTimeInfo.CtrlState);
                            printf("SysFault:%2x\r\n", ZE300_RS485BACK_DATA.RealTimeInfo.SysFault);
                        }
						break;
						
						case eResetFault: 
						{
							ZE300_RS485BACK_DATA.Fault_data = pData->buf[0];
                            
                            printf("\r\n");
                            printf("Fault_code:%02X\r\n",ZE300_RS485BACK_DATA.Fault_data);
                            printf("\r\n");

						}
						break;
						
						
						case eGetMotionCtrlPara:
                        case eWriteMotionCtrlPara:
                        case eWriteAndSaveMotionCtrlPara:
						{
                            uint32_t tmp_val;
							
                            memcpy(&ZE300_RS485BACK_DATA.MotionCtrlPara.pos_kp, &(pData->buf[0]), sizeof(ZE300_RS485BACK_DATA.MotionCtrlPara.pos_kp));
                            memcpy(&ZE300_RS485BACK_DATA.MotionCtrlPara.pos_ki, &(pData->buf[4]), sizeof(ZE300_RS485BACK_DATA.MotionCtrlPara.pos_ki));
                            
                            memcpy(&tmp_val, &(pData->buf[8]), sizeof(tmp_val));
                            ZE300_RS485BACK_DATA.MotionCtrlPara.pos_out_limit = tmp_val*0.01f;
                            
                            memcpy(&ZE300_RS485BACK_DATA.MotionCtrlPara.vel_kp, &(pData->buf[12]), sizeof(ZE300_RS485BACK_DATA.MotionCtrlPara.vel_kp));
                            memcpy(&ZE300_RS485BACK_DATA.MotionCtrlPara.vel_ki, &(pData->buf[16]), sizeof(ZE300_RS485BACK_DATA.MotionCtrlPara.vel_ki));
                            
                            memcpy(&tmp_val, &(pData->buf[20]), sizeof(tmp_val));
                            ZE300_RS485BACK_DATA.MotionCtrlPara.vel_out_limit = tmp_val*0.001f;
							
                            printf("\r\n");
                            printf("pos_kp:%f\r\n", ZE300_RS485BACK_DATA.MotionCtrlPara.pos_kp);
                            printf("pos_ki:%f\r\n", ZE300_RS485BACK_DATA.MotionCtrlPara.pos_ki);
                            printf("pos_out_limit:%f\r\n", ZE300_RS485BACK_DATA.MotionCtrlPara.pos_out_limit);
                            printf("velocity_kp:%f\r\n", ZE300_RS485BACK_DATA.MotionCtrlPara.vel_kp);
                            printf("velocity_ki:%f\r\n", ZE300_RS485BACK_DATA.MotionCtrlPara.vel_ki);
                            printf("velocity_outline:%f\r\n", ZE300_RS485BACK_DATA.MotionCtrlPara.vel_out_limit);
                            printf("\r\n");
						}
						break;
                        
                        default:
                            break;
                    }
                }
            }
        }
        
        UART_CLEAR_OREFLAG(huart);
        UART_Receive_DMA(huart, comm_uart_rx_buffer, RX_LEN);
    }
}



