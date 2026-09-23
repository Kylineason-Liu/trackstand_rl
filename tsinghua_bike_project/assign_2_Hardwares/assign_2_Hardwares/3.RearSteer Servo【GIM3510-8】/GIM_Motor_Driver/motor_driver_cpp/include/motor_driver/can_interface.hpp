#pragma once

/**
 * @file can_interface.hpp
 * @brief Linux SocketCAN wrapper for motor communication.
 *
 * Provides RAII-style CAN bus management with send/receive,
 * timeout handling, and MIT mode frame support.
 */

#include <string>
#include <stdexcept>
#include <chrono>
#include <memory>

#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <cstring>

#include "protocol.hpp"

namespace motor_driver {

// ============================================================================
// Exceptions
// ============================================================================

class CANError : public std::runtime_error {
public:
    explicit CANError(const std::string& msg) : std::runtime_error(msg) {}
};

class CANTimeoutError : public CANError {
public:
    explicit CANTimeoutError(const std::string& msg) : CANError(msg) {}
};

class MotorResponseError : public CANError {
public:
    MotorResponseError(const std::string& msg, uint8_t dev = 0, uint8_t cmd = 0, uint8_t fault = 0)
        : CANError(msg), dev_addr(dev), command(cmd), fault_code(fault) {}
    uint8_t dev_addr;
    uint8_t command;
    uint8_t fault_code;
};

// ============================================================================
// CANInterface
// ============================================================================

class CANInterface {
public:
    /**
     * @brief Construct a CAN interface.
     * @param channel CAN interface name (e.g., "can0", "vcan0")
     * @param timeout_ms Default receive timeout in milliseconds
     */
    explicit CANInterface(const std::string& channel = "can0", int timeout_ms = 500)
        : channel_(channel), timeout_ms_(timeout_ms), sockfd_(-1) {}

    ~CANInterface() { close(); }

    // Non-copyable, movable
    CANInterface(const CANInterface&) = delete;
    CANInterface& operator=(const CANInterface&) = delete;
    CANInterface(CANInterface&& other) noexcept
        : channel_(std::move(other.channel_))
        , timeout_ms_(other.timeout_ms_)
        , sockfd_(other.sockfd_) {
        other.sockfd_ = -1;
    }
    CANInterface& operator=(CANInterface&& other) noexcept {
        if (this != &other) {
            close();
            channel_ = std::move(other.channel_);
            timeout_ms_ = other.timeout_ms_;
            sockfd_ = other.sockfd_;
            other.sockfd_ = -1;
        }
        return *this;
    }

    // ------------------------------------------------------------------------
    // Open / Close
    // ------------------------------------------------------------------------

    /** Open the CAN socket and bind to the interface. */
    void open() {
        if (is_open()) return;

        // Create socket
        sockfd_ = ::socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (sockfd_ < 0) {
            throw CANError("Failed to create CAN socket: " + std::string(strerror(errno)));
        }

        // Get interface index
        struct ifreq ifr;
        std::strncpy(ifr.ifr_name, channel_.c_str(), IFNAMSIZ - 1);
        if (::ioctl(sockfd_, SIOCGIFINDEX, &ifr) < 0) {
            int saved_errno = errno;
            ::close(sockfd_);
            sockfd_ = -1;
            throw CANError("Failed to get index for " + channel_ + ": " + strerror(saved_errno));
        }

        // Bind
        struct sockaddr_can addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.can_family = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;
        if (::bind(sockfd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
            int saved_errno = errno;
            ::close(sockfd_);
            sockfd_ = -1;
            throw CANError("Failed to bind to " + channel_ + ": " + strerror(saved_errno));
        }

        // Set receive timeout
        struct timeval tv;
        tv.tv_sec = timeout_ms_ / 1000;
        tv.tv_usec = (timeout_ms_ % 1000) * 1000;
        ::setsockopt(sockfd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    }

    /** Close the CAN socket. */
    void close() {
        if (sockfd_ >= 0) {
            ::close(sockfd_);
            sockfd_ = -1;
        }
    }

    /** Check if socket is open. */
    bool is_open() const { return sockfd_ >= 0; }

    /** Get raw socket fd for advanced use. */
    int sockfd() const { return sockfd_; }

    // ------------------------------------------------------------------------
    // Send / Receive
    // ------------------------------------------------------------------------

    /**
     * @brief Send a raw CAN frame.
     * @param can_id 11-bit CAN ID
     * @param data Data bytes (0-8)
     * @param len Data length
     */
    void send(uint16_t can_id, const uint8_t* data, size_t len) {
        if (!is_open()) throw CANError("CAN bus not open");

        struct can_frame frame;
        std::memset(&frame, 0, sizeof(frame));
        frame.can_id = can_id & CAN_SFF_MASK;
        frame.can_dlc = static_cast<uint8_t>(len > 8 ? 8 : len);
        if (len > 0) std::memcpy(frame.data, data, frame.can_dlc);

        if (::write(sockfd_, &frame, sizeof(frame)) != sizeof(frame)) {
            throw CANError("Failed to send CAN frame: " + std::string(strerror(errno)));
        }
    }

    /** Send a CAN frame from a vector. */
    void send(uint16_t can_id, const std::vector<uint8_t>& data) {
        send(can_id, data.data(), data.size());
    }

    /**
     * @brief Receive a CAN frame with optional timeout.
     * @param can_id Expected arbitration ID (ignored if 0xFFFF)
     * @param timeout_ms Timeout in ms, 0 = use default, -1 = block
     * @param[out] out_data Buffer to hold received data (must be at least 8 bytes)
     * @return Number of data bytes received
     * @throws CANTimeoutError on timeout
     */
    size_t receive(uint16_t can_id, int timeout_ms, uint8_t* out_data) {
        if (!is_open()) throw CANError("CAN bus not open");

        if (timeout_ms != 0) {
            struct timeval tv;
            int ms = (timeout_ms > 0) ? timeout_ms : timeout_ms_;
            tv.tv_sec = ms / 1000;
            tv.tv_usec = (ms % 1000) * 1000;
            ::setsockopt(sockfd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        }

        struct can_frame frame;
        ssize_t n = ::read(sockfd_, &frame, sizeof(frame));

        if (timeout_ms != 0) {
            // Restore default timeout
            struct timeval tv;
            tv.tv_sec = timeout_ms_ / 1000;
            tv.tv_usec = (timeout_ms_ % 1000) * 1000;
            ::setsockopt(sockfd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        }

        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                throw CANTimeoutError("CAN receive timeout");
            }
            throw CANError("CAN receive error: " + std::string(strerror(errno)));
        }
        if (n != sizeof(struct can_frame)) {
            throw CANError("Incomplete CAN frame received");
        }

        if (can_id != 0xFFFF && (frame.can_id & CAN_SFF_MASK) != (can_id & CAN_SFF_MASK)) {
            // Not our frame — in a real implementation, you might want to
            // buffer this or retry. For simplicity, we throw.
            throw CANError("Unexpected CAN ID: got 0x" +
                           std::to_string(frame.can_id & CAN_SFF_MASK) +
                           " expected 0x" + std::to_string(can_id & CAN_SFF_MASK));
        }

        size_t len = frame.can_dlc;
        if (len > 0) std::memcpy(out_data, frame.data, len);
        return len;
    }

    // ------------------------------------------------------------------------
    // Motor-specific helpers
    // ------------------------------------------------------------------------

    /** Send a command to a specific motor address. */
    void send_command(uint8_t dev_addr, const std::vector<uint8_t>& data, bool use_host_prefix = true) {
        uint16_t cid = build_can_id(dev_addr, use_host_prefix);
        send(cid, data);
    }

    /** Send an MIT mode control command. */
    void send_mit_command(uint8_t dev_addr, const std::vector<uint8_t>& data, bool use_host_prefix = true) {
        uint16_t cid = build_mit_can_id(dev_addr, use_host_prefix);
        send(cid, data);
    }

    /**
     * @brief Send a command and wait for the motor's response.
     * @param dev_addr Motor device address
     * @param data Command bytes (including command code)
     * @param timeout_ms Receive timeout (-1 = default)
     * @param use_host_prefix Send on (0x100 | dev_addr)
     * @return Received data as vector
     */
    std::vector<uint8_t> send_and_receive(
        uint8_t dev_addr,
        const std::vector<uint8_t>& data,
        int timeout_ms = -1,
        bool use_host_prefix = true)
    {
        send_command(dev_addr, data, use_host_prefix);

        // Motor responds on its own dev_addr
        uint8_t buf[8] = {};
        size_t len = receive(dev_addr, timeout_ms, buf);
        return std::vector<uint8_t>(buf, buf + len);
    }

    /** Flush receive buffer. */
    void flush() {
        if (!is_open()) return;
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 1000; // 1ms poll
        ::setsockopt(sockfd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        struct can_frame frame;
        while (::read(sockfd_, &frame, sizeof(frame)) > 0) {}

        // Restore default timeout
        tv.tv_sec = timeout_ms_ / 1000;
        tv.tv_usec = (timeout_ms_ % 1000) * 1000;
        ::setsockopt(sockfd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    }

private:
    std::string channel_;
    int timeout_ms_;
    int sockfd_;
};

} // namespace motor_driver
