################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: Arm Compiler'
	"E:/SoftWare/ti/CCS/ccs/tools/compiler/ti-cgt-armllvm_4.0.2.LTS/bin/tiarmclang.exe" -c @"syscfg/device.opt"  -march=thumbv6m -mcpu=cortex-m0plus -mfloat-abi=soft -mlittle-endian -mthumb -O0 -I"C:/Users/leo/workspace_ccstheia/YbImu_Uart" -I"C:/Users/leo/workspace_ccstheia/YbImu_Uart/Debug" -I"E:/SoftWare/ti/mspm0_sdk_2_02_00_05/source/third_party/CMSIS/Core/Include" -I"E:/SoftWare/ti/mspm0_sdk_2_02_00_05/source" -I"C:/Users/leo/workspace_ccstheia/YbImu_Uart/BSP" -gdwarf-3 -MMD -MP -MF"$(basename $(<F)).d_raw" -MT"$(@)" -I"C:/Users/leo/workspace_ccstheia/YbImu_Uart/Debug/syscfg"  $(GEN_OPTS__FLAG) -o"$@" "$<"
	@echo 'Finished building: "$<"'
	@echo ' '

build-601910367: ../empty.syscfg
	@echo 'Building file: "$<"'
	@echo 'Invoking: SysConfig'
	"E:/SoftWare/ti/CCS/ccs/utils/sysconfig_1.23.0/sysconfig_cli.bat" --script "C:/Users/leo/workspace_ccstheia/YbImu_Uart/empty.syscfg" -o "syscfg" -s "E:/SoftWare/ti/mspm0_sdk_2_02_00_05/.metadata/product.json" --compiler ticlang
	@echo 'Finished building: "$<"'
	@echo ' '

syscfg/device_linker.cmd: build-601910367 ../empty.syscfg
syscfg/device.opt: build-601910367
syscfg/device.cmd.genlibs: build-601910367
syscfg/ti_msp_dl_config.c: build-601910367
syscfg/ti_msp_dl_config.h: build-601910367
syscfg: build-601910367

syscfg/%.o: ./syscfg/%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: Arm Compiler'
	"E:/SoftWare/ti/CCS/ccs/tools/compiler/ti-cgt-armllvm_4.0.2.LTS/bin/tiarmclang.exe" -c @"syscfg/device.opt"  -march=thumbv6m -mcpu=cortex-m0plus -mfloat-abi=soft -mlittle-endian -mthumb -O0 -I"C:/Users/leo/workspace_ccstheia/YbImu_Uart" -I"C:/Users/leo/workspace_ccstheia/YbImu_Uart/Debug" -I"E:/SoftWare/ti/mspm0_sdk_2_02_00_05/source/third_party/CMSIS/Core/Include" -I"E:/SoftWare/ti/mspm0_sdk_2_02_00_05/source" -I"C:/Users/leo/workspace_ccstheia/YbImu_Uart/BSP" -gdwarf-3 -MMD -MP -MF"syscfg/$(basename $(<F)).d_raw" -MT"$(@)" -I"C:/Users/leo/workspace_ccstheia/YbImu_Uart/Debug/syscfg"  $(GEN_OPTS__FLAG) -o"$@" "$<"
	@echo 'Finished building: "$<"'
	@echo ' '

startup_mspm0g350x_ticlang.o: E:/SoftWare/ti/mspm0_sdk_2_02_00_05/source/ti/devices/msp/m0p/startup_system_files/ticlang/startup_mspm0g350x_ticlang.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: Arm Compiler'
	"E:/SoftWare/ti/CCS/ccs/tools/compiler/ti-cgt-armllvm_4.0.2.LTS/bin/tiarmclang.exe" -c @"syscfg/device.opt"  -march=thumbv6m -mcpu=cortex-m0plus -mfloat-abi=soft -mlittle-endian -mthumb -O0 -I"C:/Users/leo/workspace_ccstheia/YbImu_Uart" -I"C:/Users/leo/workspace_ccstheia/YbImu_Uart/Debug" -I"E:/SoftWare/ti/mspm0_sdk_2_02_00_05/source/third_party/CMSIS/Core/Include" -I"E:/SoftWare/ti/mspm0_sdk_2_02_00_05/source" -I"C:/Users/leo/workspace_ccstheia/YbImu_Uart/BSP" -gdwarf-3 -MMD -MP -MF"$(basename $(<F)).d_raw" -MT"$(@)" -I"C:/Users/leo/workspace_ccstheia/YbImu_Uart/Debug/syscfg"  $(GEN_OPTS__FLAG) -o"$@" "$<"
	@echo 'Finished building: "$<"'
	@echo ' '


