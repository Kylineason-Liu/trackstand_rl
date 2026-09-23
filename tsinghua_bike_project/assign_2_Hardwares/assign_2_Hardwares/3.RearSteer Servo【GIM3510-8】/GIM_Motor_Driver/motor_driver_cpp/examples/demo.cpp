/**
 * @file demo.cpp
 * @brief Comprehensive demo: all control modes, MIT mode, fault handling.
 */

#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>

#include <motor_driver/can_interface.hpp>
#include <motor_driver/motor.hpp>
#include <motor_driver/motor_group.hpp>

using namespace motor_driver;
using namespace std::chrono_literals;

void print_separator(const std::string& title) {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << "============================================================" << std::endl;
}

void demo_basic_info(Motor& motor, const std::string& name) {
    print_separator("Basic Info - " + name);
    try {
        auto ver = motor.read_version();
        std::cout << "  Boot version:    " << ver.boot_version << std::endl;
        std::cout << "  App version:     " << ver.app_version << std::endl;
        std::cout << "  Hardware version:" << ver.hardware_version << std::endl;
        std::cout << "  CAN proto ver:   " << static_cast<int>(ver.can_protocol_version) << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  Failed: " << e.what() << std::endl;
    }

    try {
        auto params = motor.read_motor_params();
        std::cout << "  Torque constant: " << params.torque_constant << " Nm/A" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  Failed to read params: " << e.what() << std::endl;
    }
}

void demo_system_status(Motor& motor, const std::string& name) {
    print_separator("System Status - " + name);
    try {
        auto s = motor.read_system_status();
        std::cout << "  Bus voltage:  " << s.bus_voltage << " V" << std::endl;
        std::cout << "  Bus current:  " << s.bus_current << " A" << std::endl;
        std::cout << "  Temperature:  " << static_cast<int>(s.temperature) << " °C" << std::endl;
        std::cout << "  Run mode:     " << static_cast<int>(s.run_mode) << std::endl;
        std::cout << "  Fault code:   0x" << std::hex << static_cast<int>(s.fault_code) << std::dec << std::endl;

        if (s.fault_code != 0) {
            auto faults = parse_fault_code(s.fault_code);
            std::cout << "  ⚠ Active faults: ";
            if (faults.voltage_fault)        std::cout << "voltage ";
            if (faults.current_fault)        std::cout << "current ";
            if (faults.temperature_fault)    std::cout << "temperature ";
            if (faults.encoder_fault)        std::cout << "encoder ";
            if (faults.communication_fault)  std::cout << "communication ";
            if (faults.hardware_fault)       std::cout << "hardware ";
            if (faults.software_fault)       std::cout << "software ";
            std::cout << std::endl;
        } else {
            std::cout << "  ✓ No faults" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "  Failed: " << e.what() << std::endl;
    }
}

void demo_realtime_data(Motor& motor, const std::string& name) {
    print_separator("Real-time Data - " + name);
    try {
        auto s = motor.read_status_all();
        std::cout << "  Temperature:  " << static_cast<int>(s.temperature) << " °C" << std::endl;
        std::cout << "  Q-axis curr:  " << s.q_current << " A" << std::endl;
        std::cout << "  Speed:        " << s.speed << " Rpm" << std::endl;
        std::cout << "  Angle:        " << s.angle << "°" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  Failed: " << e.what() << std::endl;
    }
}

void demo_speed_control(MotorGroup& motors) {
    print_separator("Speed Control");
    try {
        std::cout << "  Starting: left=50Rpm, right=-50Rpm" << std::endl;
        motors.set_speed({{"left", 50.0}, {"right", -50.0}});
        std::this_thread::sleep_for(1s);

        std::cout << "  Stopping..." << std::endl;
        motors.set_speed({{"left", 0.0}, {"right", 0.0}});
    } catch (const std::exception& e) {
        std::cout << "  Failed: " << e.what() << std::endl;
    }
}

void demo_mit_mode(Motor& motor, const std::string& name) {
    print_separator("MIT Mode - " + name);
    try {
        auto cfg = motor.mit_read_config();
        std::cout << "  Pos_Max: " << cfg.pos_max << " rad" << std::endl;
        std::cout << "  Vel_Max: " << cfg.vel_max << " rad/s" << std::endl;
        std::cout << "  T_Max:   " << cfg.t_max << " Nm" << std::endl;

        auto status = motor.mit_read_status();
        std::cout << "  Position:  " << status.position << " rad" << std::endl;
        std::cout << "  Velocity:  " << status.velocity << " rad/s" << std::endl;
        std::cout << "  Torque:    " << status.torque << " Nm" << std::endl;
        std::cout << "  MIT mode:  " << (status.is_mit_mode ? "yes" : "no") << std::endl;
        std::cout << "  Has fault: " << (status.has_fault ? "yes" : "no") << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  Failed: " << e.what() << std::endl;
    }
}

int main() {
    try {
        CANInterface can("can0");
        can.open();
        std::cout << "CAN bus opened on can0" << std::endl;

        Motor left(can, 0x01);
        Motor right(can, 0x02);
        MotorGroup motors;
        motors.add("left", left);
        motors.add("right", right);

        // Basic info
        demo_basic_info(left, "Left Motor");
        demo_basic_info(right, "Right Motor");

        // System status
        demo_system_status(left, "Left Motor");
        demo_system_status(right, "Right Motor");

        // Real-time data
        demo_realtime_data(left, "Left Motor");

        // Fault handling
        print_separator("Fault Handling");
        if (motors.any_fault()) {
            std::cout << "  Clearing faults..." << std::endl;
            motors.clear_all_faults();
        } else {
            std::cout << "  All motors healthy" << std::endl;
        }

        // PID configuration
        print_separator("PID Configuration");
        try {
            std::cout << "  Position Kp: " << left.get_pos_kp() << std::endl;
            std::cout << "  Position Ki: " << left.get_pos_ki() << std::endl;
            std::cout << "  Speed Kp:    " << left.get_speed_kp() << std::endl;
            std::cout << "  Speed Ki:    " << left.get_speed_ki() << std::endl;
        } catch (...) {
            std::cout << "  Failed to read PID values" << std::endl;
        }

        // Control demos
        motors.enable_all();

        demo_speed_control(motors);
        demo_mit_mode(left, "Left Motor");

        // Cleanup
        print_separator("Cleanup");
        motors.emergency_stop_all();
        std::cout << "  All motors disabled." << std::endl;

        can.close();
        std::cout << "\n✓ Demo completed." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
