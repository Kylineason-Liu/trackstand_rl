################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

# Each subdirectory must supply rules for building sources it contributes
BSP/%.o: ../BSP/%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: Arm Compiler'
	"E:/SoftWare/ti/CCS/ccs/tools/compiler/ti-cgt-armllvm_4.0.2.LTS/bin/tiarmclang.exe" -c @"syscfg/device.opt"  -march=thumbv6m -mcpu=cortex-m0plus -mfloat-abi=soft -mlittle-endian -mthumb -O0 -I"C:/Users/leo/workspace_ccstheia/YbImu_IIC" -I"C:/Users/leo/workspace_ccstheia/YbImu_IIC/Debug" -I"E:/SoftWare/ti/mspm0_sdk_2_02_00_05/source/third_party/CMSIS/Core/Include" -I"E:/SoftWare/ti/mspm0_sdk_2_02_00_05/source" -I"C:/Users/leo/workspace_ccstheia/YbImu_IIC/BSP" -gdwarf-3 -MMD -MP -MF"BSP/$(basename $(<F)).d_raw" -MT"$(@)" -I"C:/Users/leo/workspace_ccstheia/YbImu_IIC/Debug/syscfg"  $(GEN_OPTS__FLAG) -o"$@" "$<"
	@echo 'Finished building: "$<"'
	@echo ' '


