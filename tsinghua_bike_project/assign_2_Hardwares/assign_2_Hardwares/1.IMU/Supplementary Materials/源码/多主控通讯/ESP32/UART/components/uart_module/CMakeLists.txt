idf_component_register(
    SRCS "uart_module.c"
    INCLUDE_DIRS "."
    REQUIRES driver
    REQUIRES imu_module
)