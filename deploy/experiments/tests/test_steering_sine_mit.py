"""MIT protocol and failure-path checks; all CAN transport is simulated."""

import contextlib
import io
import math
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import steering_sine_mit as mit


def unpack_fields(payload):
    """Decode using the protocol's explicit bit locations, independently."""
    return (
        int.from_bytes(payload[0:2], "big"),
        (payload[2] << 4) | (payload[3] >> 4),
        ((payload[3] & 15) << 8) | payload[4],
        (payload[5] << 4) | (payload[6] >> 4),
        ((payload[6] & 15) << 8) | payload[7],
    )


class FakeBus:
    def __init__(self, fault=None):
        self.fault = fault
        self.now = 0.0
        self.frames = []
        self.replies = []
        self.filters = []
        self.timeouts = []
        self.closed = False
        self.zero_code = 40000
        self.config = (120, 800, 300)  # Actual F0: +/-12 rad, +/-8 rad/s, +/-3 Nm.
        self.timeout = 1.0
        self.f0_reads = 0
        self.single_count = self.multi_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def bind(self, address):
        self.address = address

    def setsockopt(self, level, option, payload):
        self.filters.append(payload)

    def settimeout(self, value):
        self.timeouts.append(value)
        self.timeout = value

    def send(self, frame):
        can_id, length, padded = struct.unpack("=IB3x8s", frame)
        data = padded[:length]
        self.frames.append((self.now, can_id, length, data))
        if data == b"\xcf":
            if self.fault == "startup_cf_failed" or (
                self.fault == "cf_failed" and any(cid == 0x502 for _, cid, _, _ in self.frames)
            ):
                raise OSError("CF send failed")
            if self.fault != "missing_cf":
                mode = 4 if self.fault == "cf_not_disabled" else 0
                fault_code = 4 if self.fault == "cf_fault" else 0
                reply = b"\xcf" + bytes(5) + bytes([mode, fault_code])
                dlc = 7 if self.fault == "short_cf" else 8
                self.replies.append(struct.pack("=IB3x8s", 2, dlc, reply))
        elif data == b"\xb1":
            if self.fault != "missing_b1":
                if self.fault != "bad_f1_zero":
                    self.zero_code = 32767  # Same near-zero F1 code observed after B1 on the bike.
                dlc = 2 if self.fault == "short_b1" else 3
                self.replies.append(struct.pack("=IB3x8s", 2, dlc, b"\xb1\xbe\xa9"))
        elif data == b"\xf0":
            self.f0_reads += 1
            if self.f0_reads == 2 and self.fault == "f0_readback_mismatch":
                self.config = (120, self.config[1], self.config[2])
            elif self.f0_reads == 2 and self.fault == "missing_f0_readback":
                return len(frame)
            if self.fault != "missing_f0":
                dlc = 6 if self.fault == "short_f0" else 7
                reply = b"\xf0" + struct.pack("<HHH", *self.config)
                self.replies.append(struct.pack("=IB3x8s", 2, dlc, reply))
        elif len(data) == 7 and data[0] == 0xF0:
            if self.fault != "f0_write_not_applied":
                self.config = struct.unpack("<HHH", data[1:])
            if self.fault != "missing_f0_write":
                dlc = 6 if self.fault == "short_f0_write" else 7
                reply = b"\xf0" + struct.pack("<HHH", *self.config)
                self.replies.append(struct.pack("=IB3x8s", 2, dlc, reply))
        elif data == b"\xa3":
            if self.fault != "missing_a3":
                dlc = 6 if self.fault == "short_a3" else 7
                reply = b"\xa3" + struct.pack("<Hi", self.single_count, self.multi_count)
                self.replies.append(struct.pack("=IB3x8s", 2, dlc, reply))
        elif data == b"\xf1":
            if self.fault != "missing_f1":
                status = 2 if self.fault == "motor_fault" else 1 if self.fault == "mit_active" else 0
                velocity = 3000 if self.fault == "moving_f1" else 2048
                fields = bytes([velocity >> 4, ((velocity & 15) << 4) | 8, 0])
                reply = b"\xf1" + self.zero_code.to_bytes(2, "big") + fields + bytes([status])
                dlc = 6 if self.fault == "short_f1" else 7
                self.replies.append(struct.pack("=IB3x8s", 2, dlc, reply))
        elif can_id == 0x502 and self.fault == "send_failed":
            raise OSError("MIT send failed")
        return len(frame)

    def recv(self, size):
        while self.replies:
            if self.timeout:
                self.now += 0.1
            frame = self.replies.pop(0)
            if len(frame) != 16 or not self.filters:
                return frame
            can_id = struct.unpack_from("=I", frame)[0]
            # Model Linux's special error-filter path as well as ordinary ID matching.
            # A CAN_ERR_FLAG mask must never pass a normal motor data frame.
            for filter_id, mask in struct.iter_unpack("=II", self.filters[-1]):
                if mask & 0x20000000:
                    if can_id & 0x20000000:
                        return frame
                elif can_id & mask == filter_id & mask:
                    return frame
        if self.timeout == 0.0:
            raise BlockingIOError("simulated empty nonblocking socket")
        self.now += 0.1
        raise TimeoutError("simulated missing reply")

    def sleep(self, seconds):
        if self.fault == "interrupt":
            raise KeyboardInterrupt
        self.now += seconds


class MITSteeringTests(unittest.TestCase):
    def setUp(self):
        self.args = mit.parse_args(["--execute", "--duration", "0.016"])
        self.defaults = mit.MITRanges(95.5, 45.0, 18.0)

    def run_fake(self, bus):
        output = io.StringIO()
        with patch.object(mit.socket, "socket", return_value=bus), \
                patch.object(mit.time, "monotonic", side_effect=lambda: bus.now), \
                patch.object(mit.time, "sleep", side_effect=bus.sleep), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            mit.execute(self.args)
        return output.getvalue()

    def test_defaults_match_c2_sender_and_gains_are_fixed(self):
        self.assertEqual((self.args.motor_id, self.args.sign, self.args.gear_ratio), (2, -1, 1))
        self.assertEqual((self.args.amplitude, self.args.period), (1, 10))
        self.assertEqual((mit.RATE_HZ, mit.KP_NM_PER_RAD, mit.KD_NM_S_PER_RAD), (200, 10, 0.2))

    def test_known_frame_uses_correct_gain_scaling_and_nibble_order(self):
        # Independently computed nearest codes for p=1, v=0, kp=10, kd=.2, t=0.
        payload = mit.mit_payload(1.0, self.defaults)
        self.assertEqual(payload.hex(), "81578000520a4800")
        self.assertEqual(unpack_fields(payload), (33111, 2048, 82, 164, 2048))

    def test_zero_gain_encodes_zero_not_vendor_helpers_half_scale(self):
        with patch.object(mit, "KP_NM_PER_RAD", 0.0), patch.object(mit, "KD_NM_S_PER_RAD", 0.0):
            fields = unpack_fields(mit.mit_payload(0, self.defaults))
        self.assertEqual(fields[2:4], (0, 0))
        self.assertEqual(mit.encode_unsigned(500, 0, 500, 12), 4095)
        self.assertEqual(mit.encode_unsigned(5, 0, 5, 12), 4095)

    def test_only_position_changes_during_sine(self):
        targets = [mit.sine_position(self.args, t, 5.0) for t in (0, 2.5, 5, 7.5, 10)]
        self.assertEqual(targets, [5, 4, 5, 6, 5])
        for target in targets:
            fields = unpack_fields(mit.mit_payload(target, self.defaults))
            self.assertEqual(fields[1:], (2048, 82, 164, 2048))

    def test_native_origin_is_never_multiplied_by_sign_or_gear(self):
        self.args.gear_ratio = 2
        self.assertEqual(mit.sine_position(self.args, 0, 3), 3)
        self.assertEqual(mit.sine_position(self.args, 2.5, 3), 1)
        self.args.sign = 1
        self.assertEqual(mit.sine_position(self.args, 2.5, 3), 5)

    def test_zero_fields_quantization_is_bounded_to_half_a_code(self):
        for maximum in (0.01, 3, 18, 45, 655.35):
            code = mit.encode_unsigned(0, -maximum, maximum, 12)
            self.assertEqual(code, 2048)
            self.assertAlmostEqual(mit.decode_symmetric(code, maximum, 12), maximum / 4095, places=12)

    def test_f1_origin_can_be_reencoded_without_changing_its_code(self):
        for maximum in (0.1, 12, 95.5, 6553.5):
            ranges = mit.MITRanges(maximum, 8, 3)
            for code in (0, 1, 12345, 32767, 32768, 40000, 65534, 65535):
                origin = mit.decode_symmetric(code, maximum, 16)
                self.assertEqual(unpack_fields(mit.mit_payload(origin, ranges))[0], code)

    def test_actual_f0_ranges_startup_reference_timing_and_shutdown(self):
        bus = FakeBus()
        self.run_fake(bus)
        self.assertEqual([(cid, n, p) for _, cid, n, p in bus.frames[:7]],
                         [(0x102, 1, b"\xcf"), (0x102, 1, b"\xb1"),
                          (0x102, 1, b"\xf0"), (0x102, 7, bytes.fromhex("F01E0020032C01")),
                          (0x102, 1, b"\xf0"), (0x102, 1, b"\xa3"), (0x102, 1, b"\xf1")])
        self.assertEqual(bus.config, (30, 800, 300))
        motion = bus.frames[7:-1]
        self.assertEqual(len(motion), 4)
        zero = (32767 / 65535 * 2 - 1) * 3
        for index, (timestamp, can_id, dlc, payload) in enumerate(motion):
            self.assertAlmostEqual(timestamp, 0.7 + index * 0.005)
            self.assertEqual((can_id, dlc), (0x502, 8))
            code, velocity, kp, kd, torque = unpack_fields(payload)
            target = zero - math.sin(2 * math.pi * index * 0.005 / 10)
            decoded = (code / 65535 * 2 - 1) * 3
            self.assertLessEqual(abs(target - decoded), 3 / 65535 + 1e-12)
            self.assertEqual((velocity, kp, kd, torque), (2048, 82, 164, 2048))
        self.assertEqual(unpack_fields(motion[0][3])[0], bus.zero_code)
        self.assertEqual(bus.frames[-1][1:], (0x102, 1, b"\xcf"))
        self.assertEqual(bus.address, ("can0",))
        self.assertEqual(struct.unpack("=II", bus.filters[0]), (2, 0xC00007FF))
        self.assertEqual(bus.filters[-1], b"")
        self.assertTrue(bus.closed)

    def test_invalid_or_missing_startup_information_prevents_motion(self):
        for fault in ("missing_cf", "short_cf", "cf_not_disabled", "cf_fault", "missing_b1", "short_b1",
                      "missing_f0", "short_f0", "missing_f0_write", "short_f0_write", "f0_write_not_applied",
                      "missing_f0_readback", "f0_readback_mismatch", "missing_a3", "short_a3", "missing_f1",
                      "short_f1", "motor_fault", "mit_active", "bad_f1_zero", "moving_f1"):
            with self.subTest(fault=fault):
                bus = FakeBus(fault)
                with self.assertRaises((TimeoutError, RuntimeError)):
                    self.run_fake(bus)
                self.assertTrue(all(cid == 0x102 and p[0] in (0xCF, 0xB1, 0xF0, 0xA3, 0xF1)
                                    for _, cid, _, p in bus.frames))
                self.assertTrue(bus.closed)

    def test_missing_b1_ack_stops_before_f0_and_motion(self):
        bus = FakeBus("missing_b1")
        with self.assertRaisesRegex(TimeoutError, "no valid B1 startup reply"):
            self.run_fake(bus)
        self.assertEqual([data for _, _, _, data in bus.frames], [b"\xcf", b"\xb1"])

    def test_origin_is_reset_on_every_execution(self):
        for previous_code in (65535, 1000):
            bus = FakeBus()
            bus.zero_code = previous_code
            self.run_fake(bus)
            self.assertEqual([data for _, _, _, data in bus.frames].count(b"\xb1"), 1)
            self.assertEqual(unpack_fields(bus.frames[7][3])[0], 32767)

    def test_default_ranges_produce_the_documented_three_rad_write(self):
        bus = FakeBus()
        bus.config = (955, 4500, 1800)
        output = self.run_fake(bus)
        self.assertEqual(bus.frames[3][1:], (0x102, 7, bytes.fromhex("F01E0094110807")))
        self.assertEqual(bus.config, (30, 4500, 1800))
        self.assertIn("F0 verified: +/-3 rad", output)
        self.assertIn("Startup verified: A3 single=0, multi=0", output)
        self.assertIn("status=0x00", output)

    def test_nondefault_velocity_and_torque_ranges_are_preserved_byte_for_byte(self):
        for velocity, torque in ((1, 1), (1234, 5678), (65535, 65535)):
            with self.subTest(velocity=velocity, torque=torque):
                bus = FakeBus()
                bus.config = (955, velocity, torque)
                self.run_fake(bus)
                self.assertEqual(bus.frames[3][3], b"\xf0\x1e\x00" + struct.pack("<HH", velocity, torque))
                self.assertEqual(bus.config, (30, velocity, torque))

    def test_a3_single_and_multiturn_zero_are_both_verified(self):
        for single, multi in ((3, 0), (16381, 0), (0, 3), (0, -3), (0, 31 * 16384)):
            with self.subTest(single=single, multi=multi):
                bus = FakeBus()
                bus.single_count, bus.multi_count = single, multi
                with self.assertRaisesRegex(RuntimeError, "A3 not near B1 zero"):
                    self.run_fake(bus)
                self.assertFalse(any(cid == 0x502 for _, cid, _, _ in bus.frames))

    def test_a3_zero_tolerance_includes_wraparound_and_signed_multiturn(self):
        for single, multi in ((0, 0), (2, 2), (16382, -2)):
            with self.subTest(single=single, multi=multi):
                bus = FakeBus()
                bus.single_count, bus.multi_count = single, multi
                self.run_fake(bus)

    def test_f0_readback_verifies_preserved_fields_as_well_as_position(self):
        class ChangedTorqueBus(FakeBus):
            def send(self, frame):
                if frame[8:9] == b"\xf0" and frame[4] == 1 and self.f0_reads == 1:
                    self.config = (30, 800, 301)
                return super().send(frame)

        bus = ChangedTorqueBus()
        with self.assertRaisesRegex(RuntimeError, "F0 read-back does not match"):
            self.run_fake(bus)
        self.assertFalse(any(cid == 0x502 for _, cid, _, _ in bus.frames))

    def test_queued_old_f1_position_is_discarded_before_startup_query(self):
        class StaleBus(FakeBus):
            def send(self, frame):
                result = super().send(frame)
                if frame[8] == 0xA3:
                    old = b"\xf1" + (40000).to_bytes(2, "big") + b"\x80\x08\x00\x00"
                    self.replies.append(struct.pack("=IB3x8s", 2, 7, old))
                return result

        bus = StaleBus()
        self.run_fake(bus)
        self.assertEqual(unpack_fields(bus.frames[7][3])[0], 32767)

    def test_f0_write_ignores_old_range_reply_before_matching_ack(self):
        class OldConfigBus(FakeBus):
            def send(self, frame):
                result = super().send(frame)
                if frame[8] == 0xF0 and frame[4] == 7:
                    old = bytes.fromhex("F0BB0394110807")
                    self.replies.insert(0, struct.pack("=IB3x8s", 2, 7, old))
                return result

        self.run_fake(OldConfigBus())

    def test_continuous_reply_flood_prevents_request_instead_of_hanging(self):
        class FloodBus(FakeBus):
            def recv(self, size):
                return struct.pack("=IB3x8s", 2, 7, bytes.fromhex("F17FFF7FF7FF00"))

        bus = FloodBus()
        with self.assertRaisesRegex(RuntimeError, "CAN reply queue did not clear"):
            self.run_fake(bus)
        self.assertEqual(bus.frames, [])
        self.assertEqual(bus.timeout, mit.REPLY_TIMEOUT_S)
        self.assertTrue(bus.closed)

    def test_startup_disable_failure_stops_before_origin_reset(self):
        bus = FakeBus("startup_cf_failed")
        with self.assertRaises(OSError):
            self.run_fake(bus)
        self.assertEqual([data for _, _, _, data in bus.frames], [b"\xcf"])

    def test_error_only_filter_reproduces_missing_motor_reply(self):
        bus = FakeBus()
        bus.setsockopt(mit.socket.SOL_CAN_RAW, mit.socket.CAN_RAW_FILTER,
                       struct.pack("=II", 2, 0xE00007FF))
        with self.assertRaisesRegex(TimeoutError, "no valid F0 startup reply"):
            mit.read_reply(bus, 2, 0xF0)

    def test_invalid_f0_ranges_do_not_fall_back_to_defaults(self):
        for config in ((0, 800, 300), (120, 0, 300), (120, 800, 0)):
            with self.subTest(config=config):
                bus = FakeBus()
                bus.config = config
                with self.assertRaises(ValueError):
                    self.run_fake(bus)
                self.assertEqual(len(bus.frames), 3)

    def test_unrelated_and_malformed_replies_are_skipped(self):
        class NoisyBus(FakeBus):
            def send(self, frame):
                result = super().send(frame)
                if frame[8] == 0xB1:
                    self.replies[:0] = [
                        struct.pack("=IB3x8s", 1, 3, b"\xb1\xbe\xa9"),
                        struct.pack("=IB3x8s", 2, 3, b"\xf0\xbe\xa9"),
                        struct.pack("=IB3x8s", 2, 7, b"\xb1\xbe\xa9"),
                    ]
                elif frame[8] == 0xF0:
                    valid = b"\xf0" + struct.pack("<HHH", *self.config)
                    self.replies[:0] = [
                        b"too short",
                        struct.pack("=IB3x8s", 1, 7, valid),
                        struct.pack("=IB3x8s", 0x80000002, 7, valid),
                        struct.pack("=IB3x8s", 2, 7, b"\xf1" + valid[1:]),
                        struct.pack("=IB3x8s", 2, 6, valid),
                    ]
                return result

        bus = NoisyBus()
        self.run_fake(bus)
        self.assertEqual(unpack_fields(bus.frames[7][3])[0], bus.zero_code)
        self.assertEqual(bus.frames[-1][1:], (0x102, 1, b"\xcf"))

    def test_full_sine_range_checked_before_first_motion(self):
        bus = FakeBus()
        self.args.amplitude = 3.1  # Exceeds configured +/-3 rad even after zeroing.
        with self.assertRaises(ValueError):
            self.run_fake(bus)
        self.assertEqual(len(bus.frames), 7)
        self.assertTrue(bus.closed)

    def test_send_failure_and_interrupt_attempt_cf_and_close_socket(self):
        for fault, exception in (("send_failed", OSError), ("interrupt", KeyboardInterrupt)):
            with self.subTest(fault=fault):
                bus = FakeBus(fault)
                with self.assertRaises(exception):
                    self.run_fake(bus)
                self.assertEqual(bus.frames[-1][1:], (0x102, 1, b"\xcf"))
                self.assertTrue(bus.closed)

    def test_failed_disable_is_reported_as_an_error(self):
        bus = FakeBus("cf_failed")
        with self.assertRaises(OSError):
            self.run_fake(bus)
        self.assertTrue(bus.closed)

    def test_preview_never_opens_can(self):
        output = io.StringIO()
        with patch.object(mit.socket, "socket", side_effect=AssertionError("CAN must stay closed")), \
                contextlib.redirect_stdout(output):
            self.assertEqual(mit.main(["--preview", "--duration", "0.01"]), 0)
        text = output.getvalue()
        self.assertIn("Pos_Max=3 rad, assumed velocity/torque defaults", text)
        self.assertIn("502#80008000520A4800", text)
        self.assertEqual(text.count("t="), 2)

    def test_bad_arguments_rejected_before_can_open(self):
        invalid = (["--motor-id", "0"], ["--motor-id", "255"], ["--sign", "0"],
                   ["--amplitude", "nan"], ["--amplitude", "inf"], ["--amplitude", "0"],
                   ["--period", "0"], ["--period", "0.01"], ["--period", "inf"],
                   ["--gear-ratio", "0"], ["--duration", "-1"], ["--preview", "--execute"])
        for args in invalid:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    mit.parse_args(args)

    def test_out_of_range_positions_are_rejected_instead_of_clipped(self):
        for position in (-95.501, 95.501, math.inf, math.nan):
            with self.subTest(position=position), self.assertRaises(ValueError):
                mit.mit_payload(position, self.defaults)


if __name__ == "__main__":
    unittest.main()
