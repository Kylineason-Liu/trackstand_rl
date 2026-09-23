#include "imu_uart_driver.h"

#include <stdio.h>
#include <string.h>
#include <math.h>


/* ---------- 环形缓冲 / RX ring buffer ---------- */
static volatile uint8_t  s_rx_buffer[IMU_UART_RX_BUF_SIZE];
static volatile uint16_t s_rx_write_index = 0;  /* 写入位置 / write index */
static volatile uint16_t s_rx_read_index  = 0;  /* 读取位置 / read index */

static inline uint16_t _rxbuf_next(uint16_t index)
{
    return (uint16_t)((index + 1u) % IMU_UART_RX_BUF_SIZE);
}

static inline int _rxbuf_is_empty(void)
{
    return s_rx_write_index == s_rx_read_index;
}

static inline void _rxbuf_push(uint8_t byte_value)
{
    uint16_t next_index = _rxbuf_next(s_rx_write_index);
    if (next_index == s_rx_read_index) {
        /* 缓冲区满时丢弃最旧数据 / drop oldest byte when buffer is full */
        s_rx_read_index = _rxbuf_next(s_rx_read_index);
    }
    s_rx_buffer[s_rx_write_index] = byte_value;
    s_rx_write_index = next_index;
}

static inline int _rxbuf_pop(uint8_t *out_byte)
{
    if (_rxbuf_is_empty()) {
        return -1;
    }
    *out_byte = s_rx_buffer[s_rx_read_index];
    s_rx_read_index = _rxbuf_next(s_rx_read_index);
    return 0;
}

/** 将两个字节转换为 int16 / Convert two bytes to int16 */
static int16_t to_int16(const uint8_t *bytes)
{
    return (int16_t)((bytes[1] << 8) + bytes[0]);
}

/** 将四个字节转换为 float / Convert four bytes to float */
static float to_float(const uint8_t *bytes)
{
    float v;
    memcpy(&v, bytes, sizeof(float));
    return v;
}


/* ---------- 内部状态/ Internal cached state ---------- */
static volatile float s_ax = 0.0f, s_ay = 0.0f, s_az = 0.0f;
static volatile float s_gx = 0.0f, s_gy = 0.0f, s_gz = 0.0f;
static volatile float s_mx = 0.0f, s_my = 0.0f, s_mz = 0.0f;
static volatile float s_roll = 0.0f, s_pitch = 0.0f, s_yaw = 0.0f;
static volatile float s_q0 = 0.0f, s_q1 = 0.0f, s_q2 = 0.0f, s_q3 = 0.0f;
static volatile float s_height = 0.0f, s_temperature = 0.0f, s_pressure = 0.0f, s_pressure_contrast = 0.0f;
static volatile int   s_version_high = -1, s_version_mid = 0, s_version_low = 0;
static volatile uint8_t s_last_rx_function = 0;
static volatile int16_t s_last_rx_state = 0;

/* ---------- 解析数据帧 / Parse one complete frame ---------- */
static void _parse_frame_data(uint8_t frame_function, const uint8_t *frame_data)
{
    if (frame_function == IMU_FUNC_RAW_ACCEL) {
        float accel_ratio = 16.0f / 32767.0f;
        s_ax = to_int16(&frame_data[0])  * accel_ratio;
        s_ay = to_int16(&frame_data[2])  * accel_ratio;
        s_az = to_int16(&frame_data[4])  * accel_ratio;

        float deg_to_rad = 3.14159265358979323846f / 180.0f;
        float gyro_ratio  = (2000.0f / 32767.0f) * deg_to_rad;
        s_gx = to_int16(&frame_data[6])  * gyro_ratio;
        s_gy = to_int16(&frame_data[8])  * gyro_ratio;
        s_gz = to_int16(&frame_data[10]) * gyro_ratio;

        float mag_ratio = 800.0f / 32767.0f;
        s_mx = to_int16(&frame_data[12]) * mag_ratio;
        s_my = to_int16(&frame_data[14]) * mag_ratio;
        s_mz = to_int16(&frame_data[16]) * mag_ratio;
    } else if (frame_function == IMU_FUNC_EULER) {
        s_roll  = to_float(&frame_data[0]);
        s_pitch = to_float(&frame_data[4]);
        s_yaw   = to_float(&frame_data[8]);
    } else if (frame_function == IMU_FUNC_QUAT) {
        s_q0 = to_float(&frame_data[0]);
        s_q1 = to_float(&frame_data[4]);
        s_q2 = to_float(&frame_data[8]);
        s_q3 = to_float(&frame_data[12]);
    } else if (frame_function == IMU_FUNC_BARO) {
        s_height            = to_float(&frame_data[0]);
        s_temperature       = to_float(&frame_data[4]);
        s_pressure          = to_float(&frame_data[8]);
        s_pressure_contrast = to_float(&frame_data[12]);
    } else if (frame_function == IMU_FUNC_VERSION) {
        s_version_high = frame_data[0];
        s_version_mid  = frame_data[1];
        s_version_low  = frame_data[2];
    } else if (frame_function == IMU_FUNC_RETURN_STATE) {
        s_last_rx_function = frame_data[0];
        s_last_rx_state    = (int16_t)frame_data[1];
    }
}

/* ---------- 帧发送接口 / Command sender ---------- */
int IMU_UART_SendCommand(uint8_t function, const uint8_t *params, uint8_t param_len)
{
    if (param_len > 3 || (param_len > 0 && params == NULL)) {
        return -1;
    }

    uint8_t frame[8] = {FRAME_HEAD1, FRAME_HEAD2, 0, function, 0, 0, 0, 0};

    for (uint8_t i = 0; i < param_len; ++i) {
        frame[4 + i] = params[i];
    }

    uint8_t frame_len = (uint8_t)(4 + param_len + 1);
    frame[2] = frame_len;

    uint8_t checksum = 0;
    for (uint8_t i = 0; i < frame_len - 1; ++i) {
        checksum = (uint8_t)(checksum + frame[i]);
    }
    frame[frame_len - 1] = checksum;

    Send_IMU_Array(frame, frame_len);
    return 0;

}

/** 初始化接口，当前为占位符 / Init hook (placeholder). */
void IMU_UART_Init(void)
{
	GPIO_InitTypeDef GPIO_InitStructure;
	USART_InitTypeDef USART_InitStructure;
	NVIC_InitTypeDef NVIC_InitStructure;
	// 打开串口GPIO的时钟	Turn on the serial GPIO clock
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);

	// 打开串口外设的时钟	Enable the clock of the serial port peripheral
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART2, ENABLE);

	// 将USART Tx的GPIO配置为推挽复用模式		Configure the GPIO of USART Tx to push-pull multiplexing mode
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_2;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &GPIO_InitStructure);

	// 将USART Rx的GPIO配置为浮空输入模式		Configure the GPIO of USART Rx to floating input mode
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_3;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
	GPIO_Init(GPIOA, &GPIO_InitStructure);

	//Usart2 NVIC 配置	Usart2 NVIC Configuration
	NVIC_InitStructure.NVIC_IRQChannel = USART2_IRQn;
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1; //抢占优先级	Preemption priority
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 2;		  //子优先级		Subpriority
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;			  //IRQ通道使能	IRQ channel enable
	NVIC_Init(&NVIC_InitStructure);							  //根据指定的参数初始化NVIC寄存器	Initializes the NVIC registers according to the specified parameters

	
	// 配置波特率	Configuring the baud rate
	USART_InitStructure.USART_BaudRate = 115200;
	// 配置 针数据字长	Configuration Pin Data Word Length
	USART_InitStructure.USART_WordLength = USART_WordLength_8b;
	// 配置停止位	Configuring stop bits
	USART_InitStructure.USART_StopBits = USART_StopBits_1;
	// 配置校验位	Configuring the check digit
	USART_InitStructure.USART_Parity = USART_Parity_No;
	// 配置硬件流控制	Configuring Hardware Flow Control
	USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
	// 配置工作模式，收发一起		Configure the working mode, send and receive together
	USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;

	// 完成串口的初始化配置	Complete the initial configuration of the serial port
	USART_Init(USART2, &USART_InitStructure);

	//开启串口接收中断	Enable serial port receive interrupt
	USART_ITConfig(USART2, USART_IT_RXNE, ENABLE);
	// 使能串口	Enable the serial port
	USART_Cmd(USART2, ENABLE);

}

/**
 * @brief 中断接收入口，将新数据写入环形缓冲
 *        ISR entry to push received bytes into ring buffer
 */
void IMU_UART_RxBytes(volatile uint8_t *data, uint16_t len)
{
    if (!data || len == 0) return;
    for (uint16_t i = 0; i < len; ++i) {
        _rxbuf_push(data[i]);
    }
}

/**
 * @brief 解析环形缓冲中的数据，提取完整帧并更新缓存
 *        Process RX ring buffer, parse frames and update internal cache
 */
void IMU_UART_Process(void)
{
    enum {
        RX_STATE_EXPECT_HEAD1 = 0,
        RX_STATE_EXPECT_HEAD2,
        RX_STATE_EXPECT_LENGTH,
        RX_STATE_EXPECT_FUNCTION,
        RX_STATE_COLLECT_DATA
    };

    static uint8_t  rx_state = RX_STATE_EXPECT_HEAD1;
    static uint8_t  frame_length = 0;
    static uint8_t  frame_function = 0;
    static uint8_t  frame_buffer[64]; /* 数据区 + 校验 / data section + checksum */
    static uint16_t frame_index = 0;

    uint8_t current_byte = 0;

    while (_rxbuf_pop(&current_byte) == 0) {
        switch (rx_state) {
        case RX_STATE_EXPECT_HEAD1:
            rx_state = (current_byte == FRAME_HEAD1) ? RX_STATE_EXPECT_HEAD2 : RX_STATE_EXPECT_HEAD1;
            break;

        case RX_STATE_EXPECT_HEAD2:
            rx_state = (current_byte == FRAME_HEAD2) ? RX_STATE_EXPECT_LENGTH : RX_STATE_EXPECT_HEAD1;
            break;

        case RX_STATE_EXPECT_LENGTH:
            frame_length = current_byte;
            rx_state = RX_STATE_EXPECT_FUNCTION;
            break;

        case RX_STATE_EXPECT_FUNCTION:
            frame_function = current_byte;
            frame_index = 0;
            rx_state = RX_STATE_COLLECT_DATA;
            break;

        case RX_STATE_COLLECT_DATA: {
            uint16_t data_length = (frame_length >= 4) ? (uint16_t)(frame_length - 4) : 0;
            if (data_length == 0 || data_length > sizeof(frame_buffer)) {
                rx_state = RX_STATE_EXPECT_HEAD1;
                break;
            }

            frame_buffer[frame_index++] = current_byte;
            if (frame_index >= data_length) {
                uint8_t calculated_checksum = (uint8_t)(FRAME_HEAD1 + FRAME_HEAD2 + frame_length + frame_function);
                for (uint16_t i = 0; i < data_length - 1; ++i) {
                    calculated_checksum = (uint8_t)(calculated_checksum + frame_buffer[i]);
                }

                uint8_t received_checksum = frame_buffer[data_length - 1];
                if (calculated_checksum == received_checksum) {
                    _parse_frame_data(frame_function, frame_buffer);
                }
                rx_state = RX_STATE_EXPECT_HEAD1;
            }
        } break;

        default:
            rx_state = RX_STATE_EXPECT_HEAD1;
            break;
        }
    }
}

/* ---------------- 读取数据 / Read Data ---------------- */
int IMU_UART_GetAccelerometer(float out[3])
{
    if (!out) return -1;
    out[0] = s_ax; out[1] = s_ay; out[2] = s_az;
    return 0;
}

/**
 * @brief 读取角速度数据（rad/s）
 *        Read angular velocity in rad/s.
 */
int IMU_UART_GetGyroscope(float out[3])
{
    if (!out) return -1;
    out[0] = s_gx; out[1] = s_gy; out[2] = s_gz;
    return 0;
}

/**
 * @brief 读取磁场数据（uT）
 *        Read magnetic field in micro tesla.
 */
int IMU_UART_GetMagnetometer(float out[3])
{
    if (!out) return -1;
    out[0] = s_mx; out[1] = s_my; out[2] = s_mz;
    return 0;
}

/**
 * @brief 读取四元数
 *        Read quaternion (w, x, y, z).
 */
int IMU_UART_GetQuaternion(float out[4])
{
    if (!out) return -1;
    out[0] = s_q0; out[1] = s_q1; out[2] = s_q2; out[3] = s_q3;
    return 0;
}

/**
 * @brief 读取欧拉角（角度）
 *        Read Euler angles in radians.
 */
int IMU_UART_GetEuler(float out[3])
{
    if (!out) return -1;
    const float RAD2DEG = 57.2957795f;
    out[0] = s_roll  * RAD2DEG;
    out[1] = s_pitch * RAD2DEG;
    out[2] = s_yaw   * RAD2DEG;
    return 0;
}

/**
 * @brief 读取气压相关数据：高度、温度、气压、气压差
 *        Read barometric data: height, temperature, pressure, delta.
 */
int IMU_UART_GetBarometer(float out[4])
{
    if (!out) return -1;
    out[0] = s_height; out[1] = s_temperature; out[2] = s_pressure; out[3] = s_pressure_contrast;
    return 0;
}

/**
 * @brief 读取固件版本字符串
 *        Read firmware version string.
 */
void IMU_UART_GetVersion(void)
{
   if (s_version_high < 0) {
    uint8_t payload[2] = {IMU_FUNC_VERSION, 0x00};
        IMU_UART_SendCommand(IMU_FUNC_REQUEST_DATA, payload, (uint8_t)sizeof(payload));

        for (int i = 0; i < 20; ++i) {
            IMU_UART_Process();
            if (s_version_high >= 0) {
                printf("Version:%d.%d.%d\r\n", s_version_high, s_version_mid, s_version_low);
                return;
            }
            delay_ms(5);
        }
        printf("Version:-1\r\n");
        return;
    }
}

/**
 * @brief 一次性读取全部常用数据
 *        Read all common sensor values at once.
 */
int IMU_UART_GetAll(imu_measurement_t *out)
{
    if (!out) return -1;
    IMU_UART_GetAccelerometer(out->accel);
    IMU_UART_GetGyroscope(out->gyro);
    IMU_UART_GetMagnetometer(out->mag);
    IMU_UART_GetQuaternion(out->quat);
    IMU_UART_GetEuler(out->euler);
    IMU_UART_GetBarometer(out->baro);
    return 0;
}

/* ---------- 清理缓存 / Clear cached auto-reported data ---------- */
void IMU_UART_ClearAutoReportData(void)
{
    s_ax = s_ay = s_az = 0.0f;
    s_gx = s_gy = s_gz = 0.0f;
    s_mx = s_my = s_mz = 0.0f;
    s_roll = s_pitch = s_yaw = 0.0f;
    s_q0 = s_q1 = s_q2 = s_q3 = 0.0f;
    s_height = s_temperature = s_pressure = s_pressure_contrast = 0.0f;
}

static int _calibration_with_wait(uint8_t function, const uint8_t *payload, uint8_t payload_len,
                                  const char *label, uint32_t timeout_ms)
{
    s_last_rx_function = 0;
    s_last_rx_state = -1;

    int rc = IMU_UART_SendCommand(function, payload, payload_len);
    if (rc != 0) {
        return rc;
    }

    int result = IMU_UART_WaitCalibration(function, timeout_ms);
    if (!label) {
        label = "unknown";
    }

    if (result == -1) {
        printf("[IMU] Calibration %s timeout\r\n", label);
    } else if (result == 1) {
        printf("[IMU] Calibration %s success\r\n", label);
    } else {
        printf("[IMU] Calibration %s failed (code=%d)\r\n", label, result);
    }

    return result;
}

/*校准API*/
int IMU_UART_CalibrationImu(void)
{
    uint8_t payload[2] = {0x01, 0x5F};
    return _calibration_with_wait(IMU_FUNC_CALIB_IMU, payload, (uint8_t)sizeof(payload), "imu", 7000);
}

int IMU_UART_CalibrationMag(void)
{
    uint8_t payload[2] = {0x01, 0x5F};
    return _calibration_with_wait(IMU_FUNC_CALIB_MAG, payload, (uint8_t)sizeof(payload), "mag", 0);
}

int IMU_UART_CalibrationTemp(float now_temperature)
{
    if (now_temperature > 50.0f || now_temperature < -50.0f) {
        return -1;
    }
    int16_t temperature_raw = (int16_t)(now_temperature * 100.0f);
    uint8_t param_low  = (uint8_t)(temperature_raw & 0xFF);
    uint8_t param_high = (uint8_t)((temperature_raw >> 8) & 0xFF);
    uint8_t payload[3] = {param_low, param_high, 0x5F};
    return _calibration_with_wait(IMU_FUNC_CALIB_TEMP, payload, (uint8_t)sizeof(payload), "temp", 2000);
}

int IMU_UART_ResetUserData(void)
{
    uint8_t payload[2] = {0x01, 0x5F};
    return IMU_UART_SendCommand(IMU_FUNC_RESET_FLASH, payload, (uint8_t)sizeof(payload));
}

int IMU_UART_WaitCalibration(uint8_t function, uint32_t timeout_ms)
{
    uint32_t elapsed_ms = 0;
    while (1) {
        IMU_UART_Process();

        if (s_last_rx_function == function) {
            return s_last_rx_state;
        }

        if (timeout_ms != 0 && elapsed_ms >= timeout_ms) {
            return -1;
        }

        delay_ms(1);
        if (timeout_ms != 0) {
            ++elapsed_ms;
        }
    }
}



void Send_IMU_Data(uint8_t Data)
{
	while (USART_GetFlagStatus(USART2, USART_FLAG_TXE) == RESET)
		;
	USART_SendData(USART2, Data);
}

void Send_IMU_Array(uint8_t *pData, uint8_t Length)
{
	while (Length--)
	{
		Send_IMU_Data(*pData);
		pData++;
	}
}


/*  串口中断接收处理 */
/* Serial port interrupt reception processing */
void USART2_IRQHandler(void)
{
	static volatile uint8_t Rx2_Temp = 0;
    static volatile uint16_t recv_length = 0;
    static volatile uint8_t  Rx_Array[128] = {0};

	
	while (USART_GetFlagStatus(USART2, USART_FLAG_RXNE) == RESET);
			
    // 接收发送过来的数据保存	Receive and save the data sent
    Rx2_Temp = USART_ReceiveData(USART2);
    //printf("Received Byte in IRQ: %02X\n", Rx2_Temp); // 打印接收到的字节

    // 检查缓冲区是否已满	Check if the buffer is full
    if (recv_length < 128 - 1)
    {
        Rx_Array[recv_length++] = Rx2_Temp;
    }
    else
    {
        IMU_UART_RxBytes(Rx_Array, recv_length);
        recv_length = 0;
    }			
    
	
}
