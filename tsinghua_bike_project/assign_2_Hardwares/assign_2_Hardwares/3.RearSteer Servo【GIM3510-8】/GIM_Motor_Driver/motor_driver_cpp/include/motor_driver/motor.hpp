#pragma once

/**
 * @file motor.hpp
 * @brief Single motor driver class for CAN bus control.
 *
 * Provides a high-level interface for all commands defined in the
 * custom CAN protocol V3.09b0.
 *
 * Usage:
 * @code
 *   CANInterface can("can0");
 *   can.open();
 *   Motor motor(can, 0x01);
 *   auto status = motor.read_system_status();
 *   motor.set_speed(100.0);
 *   can.close();
 * @endcode
 */

#include <vector>
#include <cmath>
#include <optional>
#include <thread>
#include <chrono>

#include "protocol.hpp"
#include "can_interface.hpp"

namespace motor_driver {

class Motor {
public:
    /**
     * @param can Reference to an OPEN CANInterface
     * @param dev_addr Motor device address (1-254)
     * @param timeout_ms Default command timeout in ms
     */
    Motor(CANInterface& can, uint8_t dev_addr = DEFAULT_DEV_ADDR, int timeout_ms = 500)
        : can_(can), dev_addr_(dev_addr), timeout_ms_(timeout_ms)
    {
        if (dev_addr < 1 || dev_addr > 254) {
            throw std::invalid_argument("Device address must be 1-254, got " + std::to_string(dev_addr));
        }
    }

    // ------------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------------

    uint8_t dev_addr() const { return dev_addr_; }
    int timeout_ms() const { return timeout_ms_; }
    void set_timeout_ms(int ms) { timeout_ms_ = ms; }
    CANInterface& can_interface() { return can_; }
    const CANInterface& can_interface() const { return can_; }

    /** Get cached torque constant, reads from motor if not cached. */
    float torque_constant() {
        if (!torque_constant_.has_value()) {
            auto params = read_motor_params();
            torque_constant_ = params.torque_constant;
        }
        return torque_constant_.value_or(0.0f);
    }

    // =========================================================================
    // System Commands
    // =========================================================================

    /** Reboot the motor. No response expected. */
    void reboot() {
        std::vector<uint8_t> data;
        data.push_back(static_cast<uint8_t>(Command::REBOOT));
        // 7 bytes padding
        for (int i = 0; i < 7; i++) {
            data.push_back((i % 2 == 0) ? 0xFF : 0x00);
        }
        can_.send_command(dev_addr_, data);
    }

    /** Read firmware versions. */
    VersionInfo read_version() {
        auto resp = cmd(Command::READ_VERSION);
        VersionInfo v;
        v.boot_version         = detail::unpack_2u(resp.data(), 1);
        v.app_version          = detail::unpack_2u(resp.data(), 3);
        v.hardware_version     = detail::unpack_2u(resp.data(), 5);
        v.can_protocol_version = detail::unpack_1u(resp.data(), 7);
        return v;
    }

    /** Read real-time Q-axis current (Amperes). */
    double read_q_current() {
        auto resp = cmd(Command::READ_Q_CURRENT);
        return detail::unpack_4s(resp.data(), 1) * 0.001;
    }

    /** Read real-time rotation speed (Rpm). */
    double read_speed() {
        auto resp = cmd(Command::READ_SPEED);
        return detail::unpack_4s(resp.data(), 1) * 0.01;
    }

    /** Read absolute angle (single-turn and multi-turn) in degrees. */
    AngleInfo read_angle() {
        auto resp = cmd(Command::READ_ANGLE);
        AngleInfo a;
        a.single_turn_angle = counts_to_degrees(detail::unpack_2u(resp.data(), 1));
        a.multi_turn_angle  = counts_to_degrees(detail::unpack_4s(resp.data(), 3));
        return a;
    }

    /** Read temperature, Q-current, speed, angle together. */
    StatusAll read_status_all() {
        auto resp = cmd(Command::READ_STATUS_ALL);
        StatusAll s;
        s.temperature = detail::unpack_1u(resp.data(), 1);
        s.q_current   = detail::unpack_2s(resp.data(), 2) * 0.001;
        s.speed       = detail::unpack_2s(resp.data(), 4) * 0.01;
        s.angle       = counts_to_degrees(detail::unpack_2u(resp.data(), 6));
        return s;
    }

    /** Read full system status (voltage, current, temp, mode, fault). */
    SystemStatus read_system_status() {
        auto resp = cmd(Command::READ_SYS_STATUS);
        SystemStatus s;
        s.bus_voltage = detail::unpack_2u(resp.data(), 1) * 0.01;
        s.bus_current = detail::unpack_2u(resp.data(), 3) * 0.01;
        s.temperature = detail::unpack_1u(resp.data(), 5);
        s.run_mode    = static_cast<RunMode>(detail::unpack_1u(resp.data(), 6));
        s.fault_code  = detail::unpack_1u(resp.data(), 7);
        return s;
    }

    /** Clear faults. Returns fault code after clear (0 = no fault). */
    uint8_t clear_fault() {
        auto resp = cmd(Command::CLEAR_FAULT);
        return detail::unpack_1u(resp.data(), 1);
    }

    /** Get parsed fault information. */
    FaultInfo get_fault_info() {
        return parse_fault_code(read_system_status().fault_code);
    }

    // =========================================================================
    // Parameter Commands
    // =========================================================================

    /** Read motor parameters (torque constant, pole pairs). */
    MotorParams read_motor_params() {
        auto resp = cmd(Command::READ_MOTOR_PARAMS);
        MotorParams p;
        p.pole_pairs      = (resp.size() > 1) ? detail::unpack_1u(resp.data(), 1) : 0;
        p.torque_constant = detail::unpack_4f(resp.data(), 2);
        torque_constant_  = p.torque_constant;
        return p;
    }

    /** Set current position as origin (persists across power cycles). */
    uint16_t set_origin() {
        auto resp = cmd(Command::SET_ORIGIN);
        return detail::unpack_2u(resp.data(), 1);
    }

    // ---- Position mode max speed (0xB2) ----
    uint32_t get_position_max_speed_raw() { return read_param_4u(Command::POS_MODE_MAX_SPEED); }
    void set_position_max_speed_raw(uint32_t v) { write_param_4u(Command::POS_MODE_MAX_SPEED, v); }
    void set_position_max_speed_rpm(double rpm) {
        set_position_max_speed_raw(static_cast<uint32_t>(rpm / 0.01));
    }

    // ---- Position/Speed mode max Q-current (0xB3) ----
    uint32_t get_max_q_current_raw() { return read_param_4u(Command::POS_SPEED_MAX_CURRENT); }
    void set_max_q_current_raw(uint32_t v) { write_param_4u(Command::POS_SPEED_MAX_CURRENT, v); }
    void set_max_q_current_amps(double amps) {
        set_max_q_current_raw(static_cast<uint32_t>(amps / 0.001));
    }

    // ---- Q-axis current slope (0xB4) ----
    uint32_t get_q_current_slope() { return read_param_4u(Command::Q_CURRENT_SLOPE); }
    void set_q_current_slope(uint32_t v) { write_param_4u(Command::Q_CURRENT_SLOPE, v); }

    // ---- Speed acceleration (0xB5) ----
    uint32_t get_speed_acceleration() { return read_param_4u(Command::SPEED_ACCELERATION); }
    void set_speed_acceleration(uint32_t v) { write_param_4u(Command::SPEED_ACCELERATION, v); }
    void set_speed_acceleration_rpm_s(double rpm_s) {
        set_speed_acceleration(static_cast<uint32_t>(rpm_s / 0.01));
    }

    // ---- Position loop Kp (0xB6) ----
    float get_pos_kp() { return read_param_4f(Command::POS_LOOP_KP); }
    void set_pos_kp(float v) { write_param_4f(Command::POS_LOOP_KP, v); }

    // ---- Position loop Ki (0xB7) ----
    float get_pos_ki() { return read_param_4f(Command::POS_LOOP_KI); }
    void set_pos_ki(float v) { write_param_4f(Command::POS_LOOP_KI, v); }

    // ---- Speed loop Kp (0xB8) ----
    float get_speed_kp() { return read_param_4f(Command::SPEED_LOOP_KP); }
    void set_speed_kp(float v) { write_param_4f(Command::SPEED_LOOP_KP, v); }

    // ---- Speed loop Ki (0xB9) ----
    float get_speed_ki() { return read_param_4f(Command::SPEED_LOOP_KI); }
    void set_speed_ki(float v) { write_param_4f(Command::SPEED_LOOP_KI, v); }

    // ---- Device address (0xBA) ----
    uint8_t read_device_addr() {
        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(Command::DEVICE_ADDR));
        detail::pack_1u(data, static_cast<uint8_t>(ParamAccess::READ));
        auto resp = can_.send_and_receive(dev_addr_, data, timeout_ms_);
        return detail::unpack_1u(resp.data(), 1);
    }

    void set_device_addr(uint8_t new_addr) {
        if (new_addr < 1 || new_addr > 254) {
            throw std::invalid_argument("Device address must be 1-254");
        }
        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(Command::DEVICE_ADDR));
        detail::pack_1u(data, static_cast<uint8_t>(ParamAccess::WRITE_4BYTE));
        detail::pack_2u(data, new_addr);
        can_.send_and_receive(dev_addr_, data, timeout_ms_);
    }

    // =========================================================================
    // Control Commands
    // =========================================================================

    /**
     * @brief Q-axis current (torque) control.
     * @param current_a Target current in Amperes
     * @return Actual Q-axis current (A)
     */
    double set_q_current(double current_a) {
        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(Command::Q_CURRENT_CTRL));
        detail::pack_4s(data, static_cast<int32_t>(std::round(current_a / 0.001)));
        auto resp = can_.send_and_receive(dev_addr_, data, timeout_ms_);
        return detail::unpack_4s(resp.data(), 1) * 0.001;
    }

    /**
     * @brief Speed control.
     * @param speed_rpm Target speed in Rpm
     * @return Actual speed (Rpm)
     */
    double set_speed(double speed_rpm) {
        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(Command::SPEED_CTRL));
        detail::pack_4s(data, static_cast<int32_t>(std::round(speed_rpm / 0.01)));
        auto resp = can_.send_and_receive(dev_addr_, data, timeout_ms_);
        return detail::unpack_4s(resp.data(), 1) * 0.01;
    }

    /**
     * @brief Absolute position control.
     * @param counts Target position in encoder counts (16384/rev)
     * @return Current angle info
     */
    AngleInfo set_absolute_position(int32_t counts) {
        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(Command::ABS_POS_CTRL));
        detail::pack_4s(data, counts);
        auto resp = can_.send_and_receive(dev_addr_, data, timeout_ms_);
        AngleInfo a;
        a.single_turn_angle = counts_to_degrees(detail::unpack_2u(resp.data(), 1));
        a.multi_turn_angle  = counts_to_degrees(detail::unpack_4s(resp.data(), 3));
        return a;
    }

    AngleInfo set_absolute_position_degrees(double degrees) {
        return set_absolute_position(degrees_to_counts(degrees));
    }

    /**
     * @brief Relative position control.
     * @param delta_counts Position delta in encoder counts
     */
    AngleInfo set_relative_position(int32_t delta_counts) {
        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(Command::REL_POS_CTRL));
        detail::pack_4s(data, delta_counts);
        auto resp = can_.send_and_receive(dev_addr_, data, timeout_ms_);
        AngleInfo a;
        a.single_turn_angle = counts_to_degrees(detail::unpack_2u(resp.data(), 1));
        a.multi_turn_angle  = counts_to_degrees(detail::unpack_4s(resp.data(), 3));
        return a;
    }

    AngleInfo set_relative_position_degrees(double delta_degrees) {
        return set_relative_position(degrees_to_counts(delta_degrees));
    }

    /** Shortest path to origin (rotates ≤180°). */
    AngleInfo shortest_path_to_origin() {
        auto resp = cmd(Command::SHORTEST_ORIGIN);
        AngleInfo a;
        a.single_turn_angle = counts_to_degrees(detail::unpack_2u(resp.data(), 1));
        a.multi_turn_angle  = counts_to_degrees(detail::unpack_4s(resp.data(), 3));
        return a;
    }

    /** Disable motor (free state). Returns system status. */
    SystemStatus disable() {
        auto resp = cmd(Command::MOTOR_DISABLE);
        SystemStatus s;
        s.bus_voltage = detail::unpack_2u(resp.data(), 1) * 0.01;
        s.bus_current = detail::unpack_2u(resp.data(), 3) * 0.01;
        s.temperature = detail::unpack_1u(resp.data(), 5);
        s.run_mode    = static_cast<RunMode>(detail::unpack_1u(resp.data(), 6));
        s.fault_code  = detail::unpack_1u(resp.data(), 7);
        return s;
    }

    /** Clear faults to enable motor for control. */
    void enable() {
        clear_fault();
    }

    /** Convenience — same as disable(). */
    SystemStatus emergency_stop() { return disable(); }

    // ---- CAN Timeout Configuration (0xCD) ----

    void configure_can_timeout(bool enabled, uint16_t timeout_ms = 5000,
                                bool no_software_clear = false, bool brake_disconnect = false) {
        uint8_t action = 0;
        if (no_software_clear) action |= 0x01;
        if (brake_disconnect)  action |= 0x02;

        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(Command::CAN_TIMEOUT_CFG));
        detail::pack_1u(data, static_cast<uint8_t>(ParamAccess::WRITE_4BYTE));
        detail::pack_1u(data, enabled ? 0x01 : 0x00);
        detail::pack_2u(data, timeout_ms);
        detail::pack_1u(data, action);
        can_.send_and_receive(dev_addr_, data, timeout_ms_);
    }

    CANTimeoutConfig read_can_timeout_config() {
        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(Command::CAN_TIMEOUT_CFG));
        detail::pack_1u(data, static_cast<uint8_t>(ParamAccess::READ));
        auto resp = can_.send_and_receive(dev_addr_, data, timeout_ms_);
        CANTimeoutConfig c;
        c.enabled            = (detail::unpack_1u(resp.data(), 1) == 0x01);
        c.timeout_ms         = detail::unpack_2u(resp.data(), 2);
        c.no_software_clear  = (detail::unpack_1u(resp.data(), 4) & 0x01) != 0;
        c.brake_disconnect   = (detail::unpack_1u(resp.data(), 4) & 0x02) != 0;
        return c;
    }

    // ---- Brake Control (0xCE) ----

    bool brake_connect() {
        auto resp = cmd(Command::BRAKE_CTRL, {static_cast<uint8_t>(BrakeAction::CONNECT)});
        return detail::unpack_1u(resp.data(), 1) == 0x01;
    }
    bool brake_disconnect() {
        auto resp = cmd(Command::BRAKE_CTRL, {static_cast<uint8_t>(BrakeAction::DISCONNECT)});
        return detail::unpack_1u(resp.data(), 1) == 0x00;
    }
    bool brake_read_status() {
        auto resp = cmd(Command::BRAKE_CTRL, {static_cast<uint8_t>(BrakeAction::READ_STATUS)});
        return detail::unpack_1u(resp.data(), 1) == 0x01;
    }

    // =========================================================================
    // Advanced Control
    // =========================================================================

    // ---- Trapezoid Acceleration (0xD0) ----
    uint32_t get_trapezoid_acc() { return read_param_4u(Command::TRAPEZOID_ACC); }
    void set_trapezoid_acc(uint32_t v) { write_param_4u(Command::TRAPEZOID_ACC, v); }

    // ---- Trapezoid Deceleration (0xD1) ----
    uint32_t get_trapezoid_dec() { return read_param_4u(Command::TRAPEZOID_DEC); }
    void set_trapezoid_dec(uint32_t v) { write_param_4u(Command::TRAPEZOID_DEC, v); }

    // ---- Position Filter Bandwidth (0xD5) ----
    uint16_t get_position_filter_bandwidth() {
        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(Command::POS_FILTER_BW));
        detail::pack_1u(data, static_cast<uint8_t>(ParamAccess::READ));
        auto resp = can_.send_and_receive(dev_addr_, data, timeout_ms_);
        return detail::unpack_2u(resp.data(), 1);
    }
    void set_position_filter_bandwidth(uint16_t hz) {
        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(Command::POS_FILTER_BW));
        detail::pack_1u(data, static_cast<uint8_t>(ParamAccess::WRITE_2BYTE));
        detail::pack_2u(data, hz);
        can_.send_and_receive(dev_addr_, data, timeout_ms_);
    }

    // ---- Position Filter Inertia (0xD6) ----
    float get_position_filter_inertia() { return read_param_4f(Command::POS_FILTER_INERTIA); }
    void set_position_filter_inertia(float v) { write_param_4f(Command::POS_FILTER_INERTIA, v); }

    // ---- Position Filter Feedforward Limit (0xD7) ----
    uint32_t get_position_filter_ff_limit() { return read_param_4u(Command::POS_FILTER_FF_LIM); }
    void set_position_filter_ff_limit(uint32_t v) { write_param_4u(Command::POS_FILTER_FF_LIM, v); }

    // ---- Trapezoid Position Control (0xDA) ----
    AngleInfo trapezoid_position(int32_t counts, PositionType ptype = PositionType::ABSOLUTE) {
        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(Command::TRAPEZOID_POS));
        detail::pack_1u(data, static_cast<uint8_t>(ptype));
        detail::pack_4s(data, counts);
        auto resp = can_.send_and_receive(dev_addr_, data, timeout_ms_);
        AngleInfo a;
        a.single_turn_angle = counts_to_degrees(detail::unpack_2u(resp.data(), 1));
        a.multi_turn_angle  = counts_to_degrees(detail::unpack_4s(resp.data(), 3));
        return a;
    }

    AngleInfo trapezoid_position_degrees(double degrees, bool absolute = true) {
        return trapezoid_position(
            degrees_to_counts(degrees),
            absolute ? PositionType::ABSOLUTE : PositionType::RELATIVE);
    }

    // ---- Position Filter Control (0xDC) ----
    AngleInfo position_filter_control(int32_t counts, PositionType ptype = PositionType::ABSOLUTE) {
        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(Command::POS_FILTER_CTRL));
        detail::pack_1u(data, static_cast<uint8_t>(ptype));
        detail::pack_4s(data, counts);
        auto resp = can_.send_and_receive(dev_addr_, data, timeout_ms_);
        AngleInfo a;
        a.single_turn_angle = counts_to_degrees(detail::unpack_2u(resp.data(), 1));
        a.multi_turn_angle  = counts_to_degrees(detail::unpack_4s(resp.data(), 3));
        return a;
    }

    AngleInfo position_filter_control_degrees(double degrees, bool absolute = true) {
        return position_filter_control(
            degrees_to_counts(degrees),
            absolute ? PositionType::ABSOLUTE : PositionType::RELATIVE);
    }

    // =========================================================================
    // MIT Mode
    // =========================================================================

    /** Configure MIT mode limits (saved to non-volatile memory). */
    MITConfig mit_configure(double pos_max_rad, double vel_max_rad_s, double t_max_nm) {
        uint16_t raw_pos = static_cast<uint16_t>(std::round(pos_max_rad / 0.1));
        uint16_t raw_vel = static_cast<uint16_t>(std::round(vel_max_rad_s / 0.01));
        uint16_t raw_t   = static_cast<uint16_t>(std::round(t_max_nm / 0.01));

        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(Command::MIT_CONFIG));
        detail::pack_1u(data, static_cast<uint8_t>(ParamAccess::WRITE_6BYTE));
        detail::pack_2u(data, raw_pos);
        detail::pack_2u(data, raw_vel);
        detail::pack_2u(data, raw_t);
        auto resp = can_.send_and_receive(dev_addr_, data, timeout_ms_);

        MITConfig c;
        c.pos_max = detail::unpack_2u(resp.data(), 1) * 0.1;
        c.vel_max = detail::unpack_2u(resp.data(), 3) * 0.01;
        c.t_max   = detail::unpack_2u(resp.data(), 5) * 0.01;
        return c;
    }

    /** Read MIT mode configuration. */
    MITConfig mit_read_config() {
        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(Command::MIT_CONFIG));
        detail::pack_1u(data, static_cast<uint8_t>(ParamAccess::READ));
        auto resp = can_.send_and_receive(dev_addr_, data, timeout_ms_);
        MITConfig c;
        c.pos_max = detail::unpack_2u(resp.data(), 1) * 0.1;
        c.vel_max = detail::unpack_2u(resp.data(), 3) * 0.01;
        c.t_max   = detail::unpack_2u(resp.data(), 5) * 0.01;
        return c;
    }

    /** Read MIT mode real-time status. */
    MITStatus mit_read_status() {
        auto resp = cmd(Command::MIT_STATUS);
        auto cfg = mit_read_config();

        uint16_t pos_raw = (static_cast<uint16_t>(resp[1]) << 8) | resp[2];
        uint16_t vel_raw = (static_cast<uint16_t>(resp[3]) << 4) | (resp[4] >> 4);
        uint16_t tor_raw = (static_cast<uint16_t>(resp[4] & 0x0F) << 8) | resp[5];

        MITStatus s;
        s.position    = mit_decode(pos_raw, 65535, cfg.pos_max);
        s.velocity    = mit_decode(vel_raw, 4095, cfg.vel_max);
        s.torque      = mit_decode(tor_raw, 4095, cfg.t_max);
        s.is_mit_mode = (resp[6] & 0x01) != 0;
        s.has_fault   = (resp[6] & 0x02) != 0;
        return s;
    }

    /**
     * @brief Send MIT mode control command.
     * @param position_rad Target position in radians
     * @param velocity_rad_s Feed-forward velocity in rad/s
     * @param kp Position gain (0-500)
     * @param kd Velocity gain (0-5)
     * @param torque_nm Feed-forward torque in Nm
     * @return MIT status response
     */
    MITStatus mit_control(double position_rad, double velocity_rad_s = 0.0,
                          double kp = 0.0, double kd = 0.0, double torque_nm = 0.0) {
        auto cfg = mit_read_config();

        uint16_t pos_enc = mit_encode(position_rad, cfg.pos_max, 65535);
        uint16_t vel_enc = mit_encode(velocity_rad_s, cfg.vel_max, 4095);
        uint16_t tor_enc = mit_encode(torque_nm, cfg.t_max, 4095);
        uint16_t kp_enc  = mit_encode(kp, 500.0, 4095);
        uint16_t kd_enc  = mit_encode(kd, 5.0, 4095);

        std::vector<uint8_t> data(8);
        data[0] = (pos_enc >> 8) & 0xFF;
        data[1] = pos_enc & 0xFF;
        data[2] = (vel_enc >> 4) & 0xFF;
        data[3] = static_cast<uint8_t>(((vel_enc & 0x0F) << 4) | ((kp_enc >> 8) & 0x0F));
        data[4] = kp_enc & 0xFF;
        data[5] = (kd_enc >> 4) & 0xFF;
        data[6] = static_cast<uint8_t>(((kd_enc & 0x0F) << 4) | ((tor_enc >> 8) & 0x0F));
        data[7] = tor_enc & 0xFF;

        can_.send_mit_command(dev_addr_, data);
        return mit_read_status();
    }

    // =========================================================================
    // Convenience
    // =========================================================================

    /** Configure multiple PID gains at once. Pass -1 to skip. */
    void configure_pid(float pos_kp = -1, float pos_ki = -1,
                       float speed_kp = -1, float speed_ki = -1) {
        if (pos_kp >= 0) set_pos_kp(pos_kp);
        if (pos_ki >= 0) set_pos_ki(pos_ki);
        if (speed_kp >= 0) set_speed_kp(speed_kp);
        if (speed_ki >= 0) set_speed_ki(speed_ki);
    }

    /** Wait until all faults are cleared. */
    bool wait_for_fault_clear(int timeout_ms = 5000) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            auto status = read_system_status();
            if (status.fault_code == 0) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return false;
    }

private:
    // ------------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------------

    /** Send a command (1 byte) and receive the response. */
    std::vector<uint8_t> cmd(Command command, const std::vector<uint8_t>& extra = {}) {
        std::vector<uint8_t> data;
        data.push_back(static_cast<uint8_t>(command));
        data.insert(data.end(), extra.begin(), extra.end());
        return can_.send_and_receive(dev_addr_, data, timeout_ms_);
    }

    uint32_t read_param_4u(Command cmd) {
        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(cmd));
        detail::pack_1u(data, static_cast<uint8_t>(ParamAccess::READ));
        auto resp = can_.send_and_receive(dev_addr_, data, timeout_ms_);
        return detail::unpack_4u(resp.data(), 1);
    }

    void write_param_4u(Command cmd, uint32_t value) {
        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(cmd));
        detail::pack_1u(data, static_cast<uint8_t>(ParamAccess::WRITE_4BYTE));
        detail::pack_4u(data, value);
        can_.send_and_receive(dev_addr_, data, timeout_ms_);
    }

    float read_param_4f(Command cmd) {
        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(cmd));
        detail::pack_1u(data, static_cast<uint8_t>(ParamAccess::READ));
        auto resp = can_.send_and_receive(dev_addr_, data, timeout_ms_);
        return detail::unpack_4f(resp.data(), 1);
    }

    void write_param_4f(Command cmd, float value) {
        std::vector<uint8_t> data;
        detail::pack_1u(data, static_cast<uint8_t>(cmd));
        detail::pack_1u(data, static_cast<uint8_t>(ParamAccess::WRITE_4BYTE));
        detail::pack_4f(data, value);
        can_.send_and_receive(dev_addr_, data, timeout_ms_);
    }

    static double mit_encode(double value, double max_val, uint16_t max_enc) {
        double ratio = value / max_val;
        if (ratio > 1.0) ratio = 1.0;
        if (ratio < -1.0) ratio = -1.0;
        double encoded = (ratio + 1.0) / 2.0 * max_enc;
        return encoded;
    }

    static double mit_decode(uint16_t encoded, uint16_t max_enc, double max_val) {
        double ratio = (static_cast<double>(encoded) / max_enc) * 2.0 - 1.0;
        return ratio * max_val;
    }

    // ------------------------------------------------------------------------
    // Members
    // ------------------------------------------------------------------------

    CANInterface& can_;
    uint8_t dev_addr_;
    int timeout_ms_;
    std::optional<float> torque_constant_;
};

} // namespace motor_driver
