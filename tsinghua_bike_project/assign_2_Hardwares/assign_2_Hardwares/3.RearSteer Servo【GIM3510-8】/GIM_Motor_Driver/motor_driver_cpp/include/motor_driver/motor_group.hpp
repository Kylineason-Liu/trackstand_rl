#pragma once

/**
 * @file motor_group.hpp
 * @brief Multi-motor group for synchronized control.
 *
 * Usage:
 * @code
 *   CANInterface can("can0");
 *   can.open();
 *
 *   Motor left(can, 0x01);
 *   Motor right(can, 0x02);
 *   MotorGroup motors;
 *   motors.add("left", left);
 *   motors.add("right", right);
 *
 *   motors.set_speed({{"left", 100.0}, {"right", -100.0}});
 *   motors.disable_all();
 * @endcode
 */

#include <unordered_map>
#include <string>
#include <vector>
#include <functional>
#include <iostream>

#include "motor.hpp"

namespace motor_driver {

class MotorGroup {
public:
    MotorGroup() = default;

    /** Add a motor to the group. */
    void add(const std::string& name, Motor& motor) {
        motors_[name] = &motor;
    }

    /** Get a motor by name. */
    Motor& get(const std::string& name) {
        auto it = motors_.find(name);
        if (it == motors_.end()) {
            throw std::out_of_range("No motor named '" + name + "' in group");
        }
        return *it->second;
    }

    /** Number of motors in the group. */
    size_t size() const { return motors_.size(); }

    /** Check if a motor exists in the group. */
    bool contains(const std::string& name) const {
        return motors_.find(name) != motors_.end();
    }

    /** Iterator over all motors. */
    template <typename Func>
    void for_each(Func&& func) {
        for (auto& [name, motor] : motors_) {
            func(name, *motor);
        }
    }

    template <typename Func>
    void for_each(Func&& func) const {
        for (const auto& [name, motor] : motors_) {
            func(name, *motor);
        }
    }

    // =========================================================================
    // System Commands
    // =========================================================================

    /** Reboot all motors. */
    void reboot_all() {
        for_each([](const std::string& name, Motor& m) {
            try { m.reboot(); } catch (const std::exception& e) {
                std::cerr << "Failed to reboot '" << name << "': " << e.what() << std::endl;
            }
        });
    }

    /** Read versions from all motors. */
    std::unordered_map<std::string, VersionInfo> read_all_versions() {
        std::unordered_map<std::string, VersionInfo> results;
        for_each([&](const std::string& name, Motor& m) {
            try { results[name] = m.read_version(); }
            catch (const std::exception& e) {
                std::cerr << "Failed to read version from '" << name << "': " << e.what() << std::endl;
            }
        });
        return results;
    }

    /** Clear faults on all motors. */
    std::unordered_map<std::string, uint8_t> clear_all_faults() {
        std::unordered_map<std::string, uint8_t> results;
        for_each([&](const std::string& name, Motor& m) {
            try { results[name] = m.clear_fault(); }
            catch (const std::exception& e) {
                std::cerr << "Failed to clear fault on '" << name << "': " << e.what() << std::endl;
            }
        });
        return results;
    }

    /** Disable all motors. */
    void disable_all() {
        for_each([](const std::string& name, Motor& m) {
            try { m.disable(); } catch (const std::exception& e) {
                std::cerr << "Failed to disable '" << name << "': " << e.what() << std::endl;
            }
        });
    }

    /** Emergency stop all motors. */
    void emergency_stop_all() {
        for_each([](const std::string& name, Motor& m) {
            try { m.emergency_stop(); } catch (const std::exception& e) {
                std::cerr << "Failed to stop '" << name << "': " << e.what() << std::endl;
            }
        });
    }

    /** Enable all motors (clear faults). */
    void enable_all() {
        for_each([](const std::string& name, Motor& m) {
            try { m.enable(); } catch (const std::exception& e) {
                std::cerr << "Failed to enable '" << name << "': " << e.what() << std::endl;
            }
        });
    }

    // =========================================================================
    // Control Commands
    // =========================================================================

    /**
     * @brief Set speed for multiple motors.
     * @param speeds Map of motor name -> speed in Rpm
     * @return Map of motor name -> actual speed
     */
    std::unordered_map<std::string, double> set_speed(
        const std::unordered_map<std::string, double>& speeds)
    {
        std::unordered_map<std::string, double> results;
        for (const auto& [name, rpm] : speeds) {
            auto it = motors_.find(name);
            if (it == motors_.end()) {
                std::cerr << "Unknown motor '" << name << "' in set_speed" << std::endl;
                continue;
            }
            try {
                results[name] = it->second->set_speed(rpm);
            } catch (const std::exception& e) {
                std::cerr << "set_speed failed for '" << name << "': " << e.what() << std::endl;
            }
        }
        return results;
    }

    /**
     * @brief Set Q-axis current for multiple motors.
     * @param currents Map of motor name -> current in Amperes
     */
    std::unordered_map<std::string, double> set_q_current(
        const std::unordered_map<std::string, double>& currents)
    {
        std::unordered_map<std::string, double> results;
        for (const auto& [name, amps] : currents) {
            auto it = motors_.find(name);
            if (it == motors_.end()) continue;
            try {
                results[name] = it->second->set_q_current(amps);
            } catch (const std::exception& e) {
                std::cerr << "set_q_current failed for '" << name << "': " << e.what() << std::endl;
            }
        }
        return results;
    }

    /**
     * @brief Synchronized speed control — sends all commands first, then
     *        collects all responses for minimal inter-motor delay.
     */
    std::unordered_map<std::string, double> synced_speed(
        const std::unordered_map<std::string, double>& speeds)
    {
        // Phase 1: Send all commands
        for (const auto& [name, rpm] : speeds) {
            auto it = motors_.find(name);
            if (it == motors_.end()) continue;
            Motor& m = *it->second;
            std::vector<uint8_t> data;
            detail::pack_1u(data, static_cast<uint8_t>(Command::SPEED_CTRL));
            detail::pack_4s(data, static_cast<int32_t>(std::round(rpm / 0.01)));
            try {
                m.can_interface().send_command(m.dev_addr(), data);
            } catch (...) {}
        }

        // Phase 2: Collect all responses
        std::unordered_map<std::string, double> results;
        for (const auto& [name, rpm] : speeds) {
            auto it = motors_.find(name);
            if (it == motors_.end()) continue;
            Motor& m = *it->second;
            try {
                uint8_t buf[8];
                m.can_interface().receive(m.dev_addr(), m.timeout_ms(), buf);
                results[name] = detail::unpack_4s(buf, 1) * 0.01;
            } catch (const std::exception& e) {
                std::cerr << "synced_speed failed for '" << name << "': " << e.what() << std::endl;
            }
        }
        return results;
    }

    // =========================================================================
    // Broadcast Commands (address 0x00 — all execute, none respond)
    // =========================================================================

    /** Broadcast disable to ALL motors on the bus. */
    void broadcast_disable() {
        if (motors_.empty()) return;
        auto* first = motors_.begin()->second;
        std::vector<uint8_t> data;
        data.push_back(static_cast<uint8_t>(Command::MOTOR_DISABLE));
        first->can_interface().send_command(BROADCAST_ADDR, data, false);
    }

    /** Broadcast clear fault to ALL motors on the bus. */
    void broadcast_clear_fault() {
        if (motors_.empty()) return;
        auto* first = motors_.begin()->second;
        std::vector<uint8_t> data;
        data.push_back(static_cast<uint8_t>(Command::CLEAR_FAULT));
        first->can_interface().send_command(BROADCAST_ADDR, data, false);
    }

    // =========================================================================
    // Status
    // =========================================================================

    /** Check if any motor has an active fault. */
    bool any_fault() {
        bool has_fault = false;
        for_each([&](const std::string&, Motor& m) {
            if (has_fault) return;
            try {
                if (m.read_system_status().fault_code != 0) has_fault = true;
            } catch (...) {}
        });
        return has_fault;
    }

    /** Check if all motors are healthy. */
    bool all_healthy() { return !any_fault(); }

    /** Wait for all motor faults to clear. */
    bool wait_all_faults_clear(int timeout_ms = 5000) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if (!any_fault()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return false;
    }

    /** Configure PID on all motors with the same values. */
    void configure_all_pid(float pos_kp = -1, float pos_ki = -1,
                           float speed_kp = -1, float speed_ki = -1) {
        for_each([&](const std::string&, Motor& m) {
            try { m.configure_pid(pos_kp, pos_ki, speed_kp, speed_ki); }
            catch (const std::exception& e) {
                std::cerr << "PID config failed: " << e.what() << std::endl;
            }
        });
    }

private:
    std::unordered_map<std::string, Motor*> motors_;
};

} // namespace motor_driver
