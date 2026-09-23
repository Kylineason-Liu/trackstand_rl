#include "user_adc.h"
#include "adc.h"

#define ADC_REF_VOLTAGE		3.3f
#define ADC_VALUE_MAX		4096U
#define VBUS_FACTOR			((ADC_REF_VOLTAGE*((33.0f + 3.3f)/3.3f))/ADC_VALUE_MAX)

#define ADC_BUF_SIZE		10
#define ADC2_CHANNEL		2

uint16_t adc1_value[ADC_BUF_SIZE];
uint16_t adc2_value[ADC_BUF_SIZE][ADC2_CHANNEL];		/**< 两个通道 */

void UserAdc_Init(void)
{
	__HAL_ADC_ENABLE(&hadc1);
	__HAL_ADC_ENABLE(&hadc2);
	
    HAL_ADCEx_Calibration_Start(&hadc1, ADC_SINGLE_ENDED);
	HAL_ADCEx_Calibration_Start(&hadc2, ADC_SINGLE_ENDED);
	
	HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc1_value, ADC_BUF_SIZE);
	HAL_ADC_Start_DMA(&hadc2, (uint32_t*)adc2_value, (ADC_BUF_SIZE*ADC2_CHANNEL));
}

/* 获取电压 */
float UserADC_GetVoltage(void)
{
	uint32_t sum = 0;
	
	for(uint8_t i = 0; i < ADC_BUF_SIZE; i++)
	{
		sum += adc1_value[i];
	}
	
	return (float)((sum/ADC_BUF_SIZE)*VBUS_FACTOR);
}

/* 模拟输入通道0的值 */
float UsrADC_GetAdcInput0(void)
{
	uint32_t sum = 0;
	
	for(uint8_t i = 0; i < ADC_BUF_SIZE; i++)
	{
		sum += adc2_value[i][0];
	}
	
	return (float)((sum/ADC_BUF_SIZE)*ADC_REF_VOLTAGE/ADC_VALUE_MAX);
}

/* 模拟输入通道1的值 */
float UsrADC_GetAdcInput1(void)
{
	uint32_t sum = 0;
	
	for(uint8_t i = 0; i < ADC_BUF_SIZE; i++)
	{
		sum += adc2_value[i][1];
	}
	
	return (float)((sum/ADC_BUF_SIZE)*ADC_REF_VOLTAGE/ADC_VALUE_MAX);
}
