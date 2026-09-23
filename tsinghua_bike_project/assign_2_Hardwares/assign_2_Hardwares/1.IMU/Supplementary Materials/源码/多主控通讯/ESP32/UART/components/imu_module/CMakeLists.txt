idf_component_register(
    SRCS "imu_uart_driver.c"
    INCLUDE_DIRS "."
    REQUIRES uart_module 
)