#ifndef __USER_BUTTON_H
#define __USER_BUTTON_H

#include "main.h"
#include "multi_button.h"
#include "stdbool.h"

typedef struct 
{
    struct Button _Super;
    bool trigger;
}UserBnt_TypeDef;

extern UserBnt_TypeDef bnt_key1;
extern UserBnt_TypeDef bnt_key2;
extern UserBnt_TypeDef bnt_key3;


void UserButton_Init(void);
bool UserButton_GetTriggerState(UserBnt_TypeDef *userbnt);

#endif
