#include "user_button.h"

UserBnt_TypeDef bnt_key1;
UserBnt_TypeDef bnt_key2;
UserBnt_TypeDef bnt_key3;

static uint8_t _GetButton_GPIO(uint8_t button_id)
{
	uint8_t ret_val = 0;
	
	switch(button_id)
	{
		case 1:
			ret_val =  HAL_GPIO_ReadPin(KEY1_GPIO_Port, KEY1_Pin);
			break;
		
		case 2:
			ret_val =  HAL_GPIO_ReadPin(KEY2_GPIO_Port, KEY2_Pin);
			break;
		
		case 3:
			ret_val =  HAL_GPIO_ReadPin(KEY3_GPIO_Port, KEY3_Pin);
			break;

		default:
			break;
	}
	
	return ret_val;
}


static void _KEY1_SINGLE_CLICK_Handler(void* pVal)
{
    bnt_key1.trigger = true;
}

static void _KEY2_SINGLE_CLICK_Handler(void* pVal)
{
    bnt_key2.trigger = true;
}

static void _KEY3_SINGLE_CLICK_Handler(void* pVal)
{
    bnt_key3.trigger = true;
}



/* ³õÊ¼»¯ */
void UserButton_Init(void)
{
	button_init(&(bnt_key1._Super), _GetButton_GPIO, 0, 1);
	button_init(&(bnt_key2._Super), _GetButton_GPIO, 0, 2);
	button_init(&(bnt_key3._Super), _GetButton_GPIO, 0, 3);
    
    button_attach(&(bnt_key1._Super), SINGLE_CLICK, _KEY1_SINGLE_CLICK_Handler);
    button_attach(&(bnt_key2._Super), SINGLE_CLICK, _KEY2_SINGLE_CLICK_Handler);
    button_attach(&(bnt_key3._Super), SINGLE_CLICK, _KEY3_SINGLE_CLICK_Handler);
	
	button_start(&(bnt_key1._Super));
	button_start(&(bnt_key2._Super));
	button_start(&(bnt_key3._Super));
}


bool UserButton_GetTriggerState(UserBnt_TypeDef *userbnt)
{
    if(userbnt->trigger)
    {
        userbnt->trigger = false;
        
        return true;
    }
    
    return false;
}








