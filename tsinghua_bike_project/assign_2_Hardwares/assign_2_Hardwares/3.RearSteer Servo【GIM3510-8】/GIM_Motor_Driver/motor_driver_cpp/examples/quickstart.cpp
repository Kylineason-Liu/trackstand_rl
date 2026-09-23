/**
 * @file quickstart.cpp
 * @brief Quick start: operate two motors with minimal code.
 *
 * Build:
 *   mkdir build && cd build && cmake .. && make
 *
 * Before running:
 *   sudo ip link set can0 up type can bitrate 1000000
 */

#include <iostream>
#include <thread>
#include <chrono>

#include <motor_driver/can_interface.hpp>
#include <motor_driver/motor.hpp>
#include <motor_driver/motor_group.hpp>

using namespace motor_driver;

int main() {
    try {
        // 1. Open CAN bus
        CANInterface can("can0");
        can.open();
        std::cout << "CAN bus opened" << std::endl;

        // 2. Create two motors (addresses 0x01 and 0x02)
        Motor steer(can, 0x02);
        Motor rear(can, 0x01);
        MotorGroup motors;
        motors.add("rear", rear);
        motors.add("steer", steer);

        // 3. Check status
        try {
            auto steer_status = steer.read_status_all();
            std::cout << "Steer motor temp: " << static_cast<int>(steer_status.temperature) << "°C" << std::endl;
            std::cout << "Steer motor fault: 0x" << std::hex
                      << static_cast<int>(steer.read_system_status().fault_code) << std::dec << std::endl;

            auto rear_status = rear.read_status_all();
            std::cout << "Rear motor temp: " << static_cast<int>(rear_status.temperature) << "°C" << std::endl;
            std::cout << "Rear motor fault: 0x" << std::hex
                      << static_cast<int>(rear.read_system_status().fault_code) << std::dec << std::endl;
        } catch (const CANTimeoutError& e) {
            std::cerr << "Warning: Motor not responding — " << e.what() << std::endl;
        }

        // 4. Clear faults and enable
        motors.clear_all_faults();

        // 5. Control - both motors same direction
        std::cout << "\nRunning both motors at 10 Rpm..." << std::endl;
        motors.set_speed({{"steer", 10.0}, {"rear", 10.0}});

        std::this_thread::sleep_for(std::chrono::seconds(2));

        // 6. Stop and disable
        std::cout << "Stopping motors..." << std::endl;
        motors.set_speed({{"steer", 0.0}, {"rear", 0.0}});
        motors.disable_all();

        can.close();
        std::cout << "Done." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
