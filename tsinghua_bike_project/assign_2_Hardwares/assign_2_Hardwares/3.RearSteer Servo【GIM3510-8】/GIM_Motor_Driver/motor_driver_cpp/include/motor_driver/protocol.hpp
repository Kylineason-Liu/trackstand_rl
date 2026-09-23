#pragma once

/**
 * @file protocol.hpp
 * @brief CAN communication protocol definitions for motor control.
 *
 * Protocol version: V3.09b0
 * - Standard CAN frame (11-bit ID), data frame format
 * - Little-endian byte order
 * - Default baud rate: 1MHz
 * - Default device address: 0x01
 */

#include <cstdint>
#include <cstring>
#include <string>
#include <array>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace motor_driver {

// ============================================================================
// CAN Communication Constants
// ============================================================================

constexpr uint8_t  BROADCAST_ADDR    = 0x00;   ///< All slaves execute, no response
constexpr uint8_t  PUBLIC_ADDR       = 0xFF;   ///< All slaves respond
constexpr uint8_t  DEFAULT_DEV_ADDR  = 0x01;   ///< Default device address
constexpr uint16_t HOST_PREFIX       = 0x100;  ///< Host sends on (0x100 | dev_addr)
constexpr uint16_t MIT_MODE_PREFIX   = 0x400;  ///< MIT mode: StdID Bit[10] = 1

constexpr double COUNTS_PER_REV      = 16384.0;
constexpr double DEGREES_PER_COUNT   = 360.0 / COUNTS_PER_REV;
constexpr double RADIANS_PER_COUNT   = (2.0 * M_PI) / COUNTS_PER_REV;

// ============================================================================
// Command Codes
// ============================================================================

enum class Command : uint8_t {
    // System
    REBOOT            = 0x00,
    READ_VERSION      = 0xA0,
    READ_Q_CURRENT    = 0xA1,
    READ_SPEED        = 0xA2,
    READ_ANGLE        = 0xA3,
    READ_STATUS_ALL   = 0xA4,
    READ_SYS_STATUS   = 0xAE,
    CLEAR_FAULT       = 0xAF,

    // Parameters
    READ_MOTOR_PARAMS       = 0xB0,
    SET_ORIGIN              = 0xB1,
    POS_MODE_MAX_SPEED      = 0xB2,
    POS_SPEED_MAX_CURRENT   = 0xB3,
    Q_CURRENT_SLOPE         = 0xB4,
    SPEED_ACCELERATION      = 0xB5,
    POS_LOOP_KP             = 0xB6,
    POS_LOOP_KI             = 0xB7,
    SPEED_LOOP_KP           = 0xB8,
    SPEED_LOOP_KI           = 0xB9,
    DEVICE_ADDR             = 0xBA,

    // Control
    Q_CURRENT_CTRL    = 0xC0,
    SPEED_CTRL        = 0xC1,
    ABS_POS_CTRL      = 0xC2,
    REL_POS_CTRL      = 0xC3,
    SHORTEST_ORIGIN   = 0xC4,
    CAN_TIMEOUT_CFG   = 0xCD,
    BRAKE_CTRL        = 0xCE,
    MOTOR_DISABLE     = 0xCF,

    // Advanced
    TRAPEZOID_ACC     = 0xD0,
    TRAPEZOID_DEC     = 0xD1,
    POS_FILTER_BW     = 0xD5,
    POS_FILTER_INERTIA= 0xD6,
    POS_FILTER_FF_LIM = 0xD7,
    TRAPEZOID_POS     = 0xDA,
    POS_FILTER_CTRL   = 0xDC,

    // MIT mode
    MIT_CONFIG        = 0xF0,
    MIT_STATUS        = 0xF1,
};

// ============================================================================
// Enums
// ============================================================================

enum class RunMode : uint8_t {
    OFF               = 0,
    VOLTAGE_CONTROL   = 1,
    Q_CURRENT_CONTROL = 2,
    SPEED_CONTROL     = 3,
    POSITION_CONTROL  = 4,
};

enum class PositionType : uint8_t {
    ABSOLUTE = 0x00,
    RELATIVE = 0x01,
};

enum class ParamAccess : uint8_t {
    READ       = 0x01,
    WRITE_2BYTE = 0x03,
    WRITE_4BYTE = 0x05,
    WRITE_6BYTE = 0x07,
};

enum class BrakeAction : uint8_t {
    DISCONNECT  = 0x00,
    CONNECT     = 0x01,
    READ_STATUS = 0xFF,
};

// ============================================================================
// Data Transfer Objects
// ============================================================================

struct VersionInfo {
    uint16_t boot_version = 0;
    uint16_t app_version = 0;
    uint16_t hardware_version = 0;
    uint8_t  can_protocol_version = 0;
};

struct SystemStatus {
    double   bus_voltage = 0.0;    ///< Volts
    double   bus_current = 0.0;    ///< Amperes
    uint8_t  temperature = 0;     ///< Celsius
    RunMode  run_mode = RunMode::OFF;
    uint8_t  fault_code = 0;
};

struct MotorParams {
    uint8_t pole_pairs = 0;
    float   torque_constant = 0.0f;  ///< Nm/A
};

struct StatusAll {
    uint8_t temperature = 0;      ///< Celsius
    double  q_current = 0.0;      ///< Amperes
    double  speed = 0.0;          ///< Rpm
    double  angle = 0.0;          ///< Degrees (single-turn)
};

struct AngleInfo {
    double single_turn_angle = 0.0;  ///< Degrees (0-360)
    double multi_turn_angle = 0.0;   ///< Degrees (cumulative)
};

struct MITStatus {
    double position = 0.0;    ///< rad
    double velocity = 0.0;    ///< rad/s
    double torque = 0.0;      ///< Nm
    bool   is_mit_mode = false;
    bool   has_fault = false;
};

struct MITConfig {
    double pos_max = 0.0;     ///< rad
    double vel_max = 0.0;     ///< rad/s
    double t_max = 0.0;       ///< Nm
};

struct CANTimeoutConfig {
    bool    enabled = false;
    uint16_t timeout_ms = 5000;
    bool    no_software_clear = false;
    bool    brake_disconnect = false;
};

struct FaultInfo {
    bool voltage_fault = false;
    bool current_fault = false;
    bool temperature_fault = false;
    bool encoder_fault = false;
    bool communication_fault = false;
    bool hardware_fault = false;
    bool software_fault = false;
};

// ============================================================================
// Data Packing / Unpacking (Little-Endian)
// ============================================================================

namespace detail {

// Pack — append to byte vector
inline void pack_1u(std::vector<uint8_t>& buf, uint8_t v) { buf.push_back(v); }
inline void pack_2u(std::vector<uint8_t>& buf, uint16_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}
inline void pack_4u(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}
inline void pack_2s(std::vector<uint8_t>& buf, int16_t v)  { pack_2u(buf, static_cast<uint16_t>(v)); }
inline void pack_4s(std::vector<uint8_t>& buf, int32_t v)  { pack_4u(buf, static_cast<uint32_t>(v)); }
inline void pack_4f(std::vector<uint8_t>& buf, float v) {
    uint32_t raw;
    std::memcpy(&raw, &v, sizeof(raw));
    pack_4u(buf, raw);
}

// Unpack — read from byte array at offset
inline uint8_t  unpack_1u(const uint8_t* data, size_t offset = 0) { return data[offset]; }
inline uint16_t unpack_2u(const uint8_t* data, size_t offset = 0) {
    return static_cast<uint16_t>(data[offset]) |
           (static_cast<uint16_t>(data[offset + 1]) << 8);
}
inline uint32_t unpack_4u(const uint8_t* data, size_t offset = 0) {
    return static_cast<uint32_t>(data[offset]) |
           (static_cast<uint32_t>(data[offset + 1]) << 8) |
           (static_cast<uint32_t>(data[offset + 2]) << 16) |
           (static_cast<uint32_t>(data[offset + 3]) << 24);
}
inline int16_t unpack_2s(const uint8_t* data, size_t offset = 0) {
    return static_cast<int16_t>(unpack_2u(data, offset));
}
inline int32_t unpack_4s(const uint8_t* data, size_t offset = 0) {
    return static_cast<int32_t>(unpack_4u(data, offset));
}
inline float unpack_4f(const uint8_t* data, size_t offset = 0) {
    uint32_t raw = unpack_4u(data, offset);
    float result;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
}

} // namespace detail

// ============================================================================
// CAN ID Helpers
// ============================================================================

inline uint16_t build_can_id(uint8_t dev_addr, bool use_host_prefix = true) {
    return use_host_prefix ? (HOST_PREFIX | dev_addr) : dev_addr;
}

inline uint16_t build_mit_can_id(uint8_t dev_addr, bool use_host_prefix = true) {
    uint16_t base = use_host_prefix ? (HOST_PREFIX | dev_addr) : dev_addr;
    return MIT_MODE_PREFIX | base;
}

// ============================================================================
// Conversion Helpers
// ============================================================================

inline double counts_to_degrees(int32_t counts) { return counts * DEGREES_PER_COUNT; }
inline double counts_to_radians(int32_t counts) { return counts * RADIANS_PER_COUNT; }
inline int32_t degrees_to_counts(double degrees) {
    return static_cast<int32_t>(std::round(degrees / DEGREES_PER_COUNT));
}
inline int32_t radians_to_counts(double radians) {
    return static_cast<int32_t>(std::round(radians / RADIANS_PER_COUNT));
}

inline FaultInfo parse_fault_code(uint8_t code) {
    FaultInfo f;
    f.voltage_fault        = (code & (1 << 0)) != 0;
    f.current_fault        = (code & (1 << 1)) != 0;
    f.temperature_fault    = (code & (1 << 2)) != 0;
    f.encoder_fault        = (code & (1 << 3)) != 0;
    f.communication_fault  = (code & (1 << 5)) != 0;
    f.hardware_fault       = (code & (1 << 6)) != 0;
    f.software_fault       = (code & (1 << 7)) != 0;
    return f;
}

} // namespace motor_driver
