#include "uart_module.h"

#define BUF_SIZE (1024)         // 发送缓冲区大小 | Tx buffer size
#define RD_BUF_SIZE (BUF_SIZE)  // 接收缓冲区大小 | Rx buffer size

QueueHandle_t uart_queue;// UART事件队列句柄 | UART event queue handle

// 通过串口发送一串数据 
int USART_SendArray(uint8_t* data, uint16_t len)
{
    const int txBytes = uart_write_bytes(UART_NUM_1, data, len);
    return txBytes;
}

// 通过串口发送一个字节 
int USART_SendData(uint8_t data)
{
    uint8_t data1 = data;// 创建临时变量避免内存对齐问题 | Avoid memory alignment issues
    const int txBytes = uart_write_bytes(UART_NUM_1, &data1, 1);
    return txBytes;
}

void uart1_init() {
    // UART参数配置 | UART parameter configuration
    uart_config_t uart_config = {
        .baud_rate = UART_BAUD,                 // 波特率：115200 | Common baud rate: 115200
        .data_bits = UART_DATA_8_BITS,          // 8位数据位 | 8 data bits
        .parity    = UART_PARITY_DISABLE,       // 无校验位 | No parity
        .stop_bits = UART_STOP_BITS_1,          // 1位停止位 | 1 stop bit
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,  // 禁用硬件流控 | Disable hardware flow control
        .source_clk = UART_SCLK_APB,            // 使用APB时钟源 | Use APB clock source
    };

    uart_param_config(UART_NUM_1, &uart_config);// 应用配置参数 | Apply configuration
    uart_set_pin(UART_NUM_1, UART1_TX_PIN, UART1_RX_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);// 设置GPIO引脚 | Set GPIO pins
    uart_driver_install(UART_NUM_1, 2048, 2048, 50, &uart_queue, 0);// 安装UART驱动 | Install UART driver
}

//UART数据处理任务 | Task: UART Data Processing
void UART_Process_Task(void *pvParameters) {
    uart_event_t event;// UART事件结构体 | UART event structure
    uint8_t* dtmp = (uint8_t*) malloc(RD_BUF_SIZE);// 分配临时缓冲区 | Allocate temp buffer
    while(1)
    {// 等待队列事件（阻塞式）| Wait for queue event (blocking)
        if (xQueueReceive(uart_queue, (void *)&event, (TickType_t)portMAX_DELAY)) {
            switch (event.type) 
            {
                case UART_DATA:// 数据到达事件 | Data received event
                    int len = uart_read_bytes(UART_NUM_1, dtmp, event.size, portMAX_DELAY);// 读取UART数据 | Read UART data
                    for (int i = 0; i < len; i++) {// 逐字节处理数据 | Process data byte by byte
                        IMU_UART_RxBytes(&dtmp[i],1);
                    }
                    break;
                case UART_BUFFER_FULL:// 缓冲区满处理 | Buffer full handling
                    uart_flush_input(UART_NUM_1);// 清空输入缓冲区 | Clear input buffer
                    xQueueReset(uart_queue);    // 重置事件队列 | Reset event queue
                    break;
                case UART_FIFO_OVF:// FIFO溢出处理 | FIFO overflow handling
                    uart_flush_input(UART_NUM_1);// 清空硬件FIFO | Clear hardware FIFO
                    xQueueReset(uart_queue);    // 重置事件队列 | Reset event queue
                    break;
                default:
                    break;
            }
        }
    }
    free(dtmp);// 释放内存 | Free memory
    dtmp = NULL;// 避免野指针 | Prevent dangling pointer
    vTaskDelete(NULL);// 删除当前任务 | Delete current task
}
