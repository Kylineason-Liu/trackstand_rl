#ifndef __COMM_RS485_H
#define __COMM_RS485_H

#include "main.h"
#include "stdbool.h"

#define RX_LEN  256
/* 3.0协议头 */
#define UART_TX_PROTOCOL_HEADER_V3  0xAE
#define UART_RX_PROTOCOL_HEADER_V3  0xAC

typedef struct
{
    uint8_t head;               /**!< 协议头 */
    uint8_t pack_num;           /**!< 包序号 */
    uint8_t addr;               /**!< RS485地址 */
    uint8_t cmd;                /**!< 命令码 */
    uint8_t data_len;           /**!< 数据字段长度 */
    uint8_t buf[RX_LEN - 5];    /**!< 数据段 */
}UartProtocolData_t;


/* 命令枚举 */
enum{
    eRestart                    = 0x00,     /**< 软件复位 */
    eGetVerInfo                 = 0x0A,     /**< 获取Boot软件版本、应用软件版本、硬件版本、RS485协议版本、CAN协议版本、UID */
    eGetRealtimeInfo            = 0x0B,     /**< 读取实时数据（单圈绝对值角度、多圈绝对值角度、速度、Q轴电流、母线电压、母线电流、工作温度、运行状态、电机状态、故障码） */
    eResetFault                 = 0x0F,     /**< 清除故障 */
	eReadUserPara				= 0x10,		/**< 读取用户参数 */
	eWriteUserPara	            = 0x11,     /**< 写入并保存用户参数*/
	eReadMotorPara				= 0x12,     /**< 读取电机硬件参数*/
	eWriteMotorPara 			= 0x13,		/**< 写入并保存电机硬件参数*/
    eGetMotionCtrlPara          = 0x14,     /**< 读取运动控制参数 */
    eWriteMotionCtrlPara        = 0x15,     /**< 写入运动控制参数 */
    eWriteAndSaveMotionCtrlPara = 0x16,     /**< 写入并保存运动控制参数 */
    eSetZeroByCurrentRawAngle   = 0x1D,     /**< 设置当前位置为原点 */
	eEncoderCalibration			= 0x1E,		/**< 编码器校准 */
    eRestSysPara                = 0x1F,     /**< 参数恢复默认值 */
    eTorqueCtrl                 = 0x20,     /**< 力矩控制 */
    eVelocityCtrl               = 0x21,     /**< 速度控制 */
    eAbsolutePositionCtrl       = 0x22,     /**< 绝对值位置控制 */
    eRelativePositionCtrl       = 0x23,     /**< 相对位置控制 */
    eShortestHomePosition       = 0x24,     /**< 最短距离回原点 */
	eBreakCtrl                  = 0x2E,     /**< 抱阀控制开关输出控制 */
    eDisableCtrl                = 0x2F,     /**< 失能电机输出 */
};

typedef struct
{
    uint8_t boot_ver;              
    uint8_t software_ver;              
    uint8_t hardware_ver;              
    uint8_t aux_rs485_ver;              
    uint8_t modbus_rs485_ver;              
    uint8_t aux_can_ver;   
	uint8_t canopen_ver;
	uint8_t UID[12];
}BoardVerInfo_t;

typedef struct
{
	uint8_t EncoderType;
	uint8_t EncoderDirection;
	uint8_t SeCEncoderEnable;
	uint8_t VelocityFiltFactor;
	uint8_t DevAddr;
	uint8_t RS485Baud;
	uint8_t CANBaud;
	uint8_t CANopenEnable;
	uint16_t BusVoltageMax;
	uint8_t VoltageFaultTime;
	uint16_t BusCurrentMax;
	uint8_t CurrentFaultTime;
	uint8_t	Temperature;
	uint8_t TemperatureFaultTime;
}UserParameter_t;

typedef struct 
{
	char MotorName[16];
	uint8_t MotorPole;
	float PhaseRes;
	float PhaseIcl;
	float TorqueConstant;
	uint8_t ReductionRatio;
}MotorParameter_t;

typedef struct
{
    uint16_t MecRawAngle;
    int32_t MecPosition;
    int32_t MecVel_Mulit100;
    int32_t Iq_Mulit1000;
    uint16_t BusVoltage_multi100;
    uint16_t BusCurrent_multi100;
    uint8_t Temperature;
    uint8_t RunMode;
    uint8_t CtrlState;
    uint8_t SysFault;
}RealTimeInfo_t;

typedef struct
{
    float pos_kp;              
    float pos_ki;           
    float pos_out_limit;     
	float vel_kp;
	float vel_ki;
    float vel_out_limit;     
}MotionCtrlPara_t;

typedef struct
{
	BoardVerInfo_t BoardVerInfo;
	UserParameter_t UserParameter;
	MotorParameter_t MotorParameter;
	RealTimeInfo_t RealTimeInfo;
	MotionCtrlPara_t MotionCtrlPara;
	uint8_t Fault_data;
}_RS485_Para_t;

void CommRs485_Init(void);
void CommRs485_Restart(uint8_t dev_addr);
void CommRs485_EncoderCalibration(uint8_t dev_addr);
void CommRs485_GetVerInfo(uint8_t dev_addr);
void CommRs485_GetRealtimeInfo(uint8_t dev_addr);
void CommRs485_ResetFault(uint8_t dev_addr);
void CommRs485_ReadeUserPara(uint8_t dev_addr);
void CommRs485_WriteUserPara(uint8_t dev_addr, UserParameter_t user_para);
void CommRs485_ReadeMotorPara(uint8_t dev_addr);
void CommRs485_WriteMotorPara(uint8_t dev_addr, MotorParameter_t motor_para);
void CommRs485_GetMotionCtrlPara(uint8_t dev_addr);
void CommRs485_WriteMotionCtrlPara(uint8_t dev_addr, MotionCtrlPara_t pid , bool is_save);
void CommRs485_SetZeroByCurrentRawAngle(uint8_t dev_addr);
void CommRs485_RestSysPara(uint8_t dev_addr);
void CommRs485_Torque(uint8_t dev_addr, float target_Iq, float target_Slope);
void CommRs485_VelocityCtrl(uint8_t dev_addr, float target_vel, float target_acc);
void CommRs485_AbsolutePositionCtrl(uint8_t dev_addr, int32_t target_count);
void CommRs485_RelativePositionCtrl(uint8_t dev_addr, int32_t target_count);
void CommRs485_ShortestHomePosition(uint8_t dev_addr);
void CommRs485_eBreakCtrl(uint8_t dev_addr, uint16_t target_state);
void CommRs485_DisableCtrl(uint8_t dev_addr);

void UART_IDLE_IT_Callback(UART_HandleTypeDef *huart);

#endif

