import struct
import serial


# Memory register addresses (SMS_STS)
SMS_STS_TORQUE_ENABLE = 40
SMS_STS_ACC = 41
SMS_STS_GOAL_POSITION_L = 42
SMS_STS_GOAL_POSITION_H = 43
SMS_STS_GOAL_TIME_L = 44
SMS_STS_GOAL_TIME_H = 45
SMS_STS_GOAL_SPEED_L = 46
SMS_STS_GOAL_SPEED_H = 47
SMS_STS_LOCK = 55

SMS_STS_PRESENT_POSITION_L = 56
SMS_STS_PRESENT_POSITION_H = 57
SMS_STS_PRESENT_SPEED_L = 58
SMS_STS_PRESENT_SPEED_H = 59
SMS_STS_PRESENT_LOAD_L = 60
SMS_STS_PRESENT_LOAD_H = 61
SMS_STS_PRESENT_VOLTAGE = 62
SMS_STS_PRESENT_TEMPERATURE = 63
SMS_STS_MOVING = 66
SMS_STS_PRESENT_CURRENT_L = 69
SMS_STS_PRESENT_CURRENT_H = 70

# Instructions
INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03


class SMS_STS:
    """Pure Python implementation of FEETECH SMS_STS servo serial protocol."""

    def __init__(self, end=0):
        self._end = end  # 0 = little-endian (default for ARM/x86)
        self._serial = None
        self._mem = bytearray(15)  # feedback buffer (registers 56~70 = 15 bytes)

    def begin(self, baud_rate, port):
        self._serial = serial.Serial(
            port=port,
            baudrate=baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
        )
        return self._serial.is_open

    def end(self):
        if self._serial and self._serial.is_open:
            self._serial.close()

    @property
    def is_connected(self):
        return self._serial is not None and self._serial.is_open

    # --- low-level protocol ---

    @staticmethod
    def _checksum(data):
        s = sum(data) & 0xFF
        return (~s) & 0xFF

    def _pack16(self, value):
        fmt = '<h' if self._end == 0 else '>h'
        return struct.pack(fmt, value)

    def _send_packet(self, packet):
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()
        self._serial.write(packet)
        self._serial.flush()

    def _read_exactly(self, n):
        data = self._serial.read(n)
        if len(data) != n:
            return None
        return data

    def _check_head(self):
        h = bytearray([0, 0])
        for _ in range(11):
            b = self._read_exactly(1)
            if b is None:
                return False
            h[1] = h[0]
            h[0] = b[0]
            if h[0] == 0xFF and h[1] == 0xFF:
                return True
        return False

    def _ack(self, sid):
        if sid == 0xFE:
            return True
        if not self._check_head():
            return False
        resp = self._read_exactly(4)
        if resp is None or resp[0] != sid or resp[1] != 2:
            return False
        return resp[2] == 0 and self._checksum(resp[0:3]) == resp[3]

    def _make_packet(self, sid, inst, params=b''):
        length = 2 + len(params)
        body = bytes([sid, length, inst]) + params
        return bytes([0xFF, 0xFF]) + body + bytes([self._checksum(body)])

    # --- register read / write ---

    def read(self, sid, addr, length):
        packet = self._make_packet(sid, INST_READ, bytes([addr, length]))
        self._send_packet(packet)

        if not self._check_head():
            return None
        hdr = self._read_exactly(3)
        if hdr is None or hdr[0] != sid:
            return None
        if hdr[1] != length + 2:
            return None
        error = hdr[2]

        data = self._read_exactly(length)
        if data is None:
            return None
        csum_byte = self._read_exactly(1)
        if csum_byte is None:
            return None

        calc = self._checksum(bytes([sid, hdr[1], error]) + data)
        if error or calc != csum_byte[0]:
            return None
        return bytes(data)

    def write(self, sid, addr, data):
        packet = self._make_packet(sid, INST_WRITE, bytes([addr]) + bytes(data))
        self._send_packet(packet)
        return self._ack(sid)

    def write_byte(self, sid, addr, value):
        return self.write(sid, addr, bytes([value]))

    # --- high-level servo commands ---

    def write_pos_ex(self, sid, position, speed, acc=0):
        if position < 0:
            position = abs(position) | (1 << 15)
        data = (
            bytes([acc])
            + self._pack16(position)
            + bytes([0, 0])
            + self._pack16(speed)
        )
        return self.write(sid, SMS_STS_ACC, data)

    def enable_torque(self, sid, enable):
        return self.write_byte(sid, SMS_STS_TORQUE_ENABLE, enable)

    def feedback(self, sid):
        data = self.read(
            sid,
            SMS_STS_PRESENT_POSITION_L,
            SMS_STS_PRESENT_CURRENT_H - SMS_STS_PRESENT_POSITION_L + 1,
        )
        if data is None:
            return -1
        self._mem = data
        return len(data)

    # --- status readers (id=-1 reads from buffered _mem) ---

    def _unpack16(self, data, sign_bit):
        raw = struct.unpack('<H', data)[0]
        if raw & (1 << sign_bit):
            raw = -(raw & ~(1 << sign_bit))
        return raw

    def read_pos(self, sid=-1):
        if sid == -1:
            return self._unpack16(self._mem[0:2], 15)
        data = self.read(sid, SMS_STS_PRESENT_POSITION_L, 2)
        return None if data is None else self._unpack16(data, 15)

    def read_speed(self, sid=-1):
        if sid == -1:
            return self._unpack16(self._mem[2:4], 15)
        data = self.read(sid, SMS_STS_PRESENT_SPEED_L, 2)
        return None if data is None else self._unpack16(data, 15)

    def read_load(self, sid=-1):
        if sid == -1:
            return self._unpack16(self._mem[4:6], 10)
        data = self.read(sid, SMS_STS_PRESENT_LOAD_L, 2)
        return None if data is None else self._unpack16(data, 10)

    def read_voltage(self, sid=-1):
        if sid == -1:
            return self._mem[6]
        data = self.read(sid, SMS_STS_PRESENT_VOLTAGE, 1)
        return None if data is None else data[0]

    def read_temper(self, sid=-1):
        if sid == -1:
            return self._mem[7]
        data = self.read(sid, SMS_STS_PRESENT_TEMPERATURE, 1)
        return None if data is None else data[0]

    def read_move(self, sid=-1):
        if sid == -1:
            return self._mem[10]
        data = self.read(sid, SMS_STS_MOVING, 1)
        return None if data is None else data[0]

    def read_current(self, sid=-1):
        if sid == -1:
            return self._unpack16(self._mem[13:15], 15)
        data = self.read(sid, SMS_STS_PRESENT_CURRENT_L, 2)
        return None if data is None else self._unpack16(data, 15)
