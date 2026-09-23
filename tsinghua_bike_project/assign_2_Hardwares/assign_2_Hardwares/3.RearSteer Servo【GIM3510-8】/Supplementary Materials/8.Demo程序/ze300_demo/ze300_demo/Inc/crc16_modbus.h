#ifndef __CRC16_MODBUS_H
#define __CRC16_MODBUS_H
#ifdef __cplusplus
 extern "C" {
#endif

#include "main.h"

#define usMBCRC16   CalcCRC16_Modbus

uint16_t CalcCRC16_Modbus( uint8_t *Frame, uint16_t Len );

#ifdef __cplusplus
}
#endif

#endif /*_CRC16_MODBUS_H */
