"""MIT collector protocol, timing and read-only transport checks; no real CAN."""

from collections import deque
import contextlib
import csv
import io
import math
from pathlib import Path
import queue
import struct
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import motor_feedback_mit as recorder
import steering_sine_mit as sender
from test_steering_sine_mit import FakeBus as SineBus


FRAME = struct.Struct("=IB3x8s")
CONFIG = bytes.fromhex("F0BB0394110807")  # +/-95.5 rad, +/-45 rad/s, +/-18 Nm.
TARGET = bytes.fromhex("81578000520A4800")  # p code=33111; v/t=2048, kp=82, kd=164.
ZERO = bytes.fromhex("F1800080080000")
FEEDBACK = bytes.fromhex("F181577D080203")  # p=33111, v=2000, t=2050; MIT active + fault.
MIT_CONFIG = bytes.fromhex("F01E0094110807")  # Current sender: +/-3 rad; other ranges unchanged.
MIT_ZERO = bytes.fromhex("F17FFF80080000")
MIT_TARGET = sender.mit_payload(-.1, sender.MITRanges(3, 45, 18))
MIT_FEEDBACK = b"\xf1" + MIT_TARGET[:2] + bytes.fromhex("80080001")


def accepts_frame(frame, filters):
    if not filters or len(frame) != FRAME.size:
        return True
    can_id = FRAME.unpack(frame)[0]
    for filter_id, mask in struct.iter_unpack("=II", filters):
        # Linux routes ERR-masked filters separately, not by ordinary ID matching.
        if mask & 0x20000000:
            if can_id & 0x20000000 and can_id & (mask & 0x1FFFFFFF):
                return True
        elif not can_id & 0x20000000 and can_id & mask == filter_id & mask:
            return True
    return False


class State:
    def __init__(self):
        self.now = 0.0
        self.sent = []
        self.query_pending = deque()
        self.capture_pending = queue.Queue()
        self.capture_ready = threading.Event()
        self.capture_enabled = False
        self.query_closed = self.capture_closed = False
        self.query_filters = self.capture_filters = b""
        self.query_options = []
        self.capture_options = []
        self.timeout = None
        self.suppress = set()
        self.counts = {1: 20000, 2: -20000}
        self.configs = {1: CONFIG, 2: CONFIG}
        self.injected = False

    def clock(self):
        return self.now

    def wall_clock(self):
        return 1700000000 + self.now

    def sleep(self, delay):
        self.now += delay


class QueryBus:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.state.query_closed = True

    def bind(self, address):
        assert address == ("can0",)

    def setsockopt(self, level, option, payload):
        self.state.query_options.append((level, option, payload))
        if level == recorder.socket.SOL_CAN_RAW and option == recorder.socket.CAN_RAW_FILTER:
            self.state.query_filters = payload

    def setblocking(self, blocking):
        self.state.timeout = None if blocking else 0

    def settimeout(self, timeout):
        self.state.timeout = timeout

    def send(self, frame):
        host, length, data = FRAME.unpack(frame)
        device, command = host & 0xFF, data[0]
        assert host == 0x100 | device
        assert length == 1 and command in (0xA1, 0xA2, 0xA3, 0xB0, 0xF0, 0xF1)
        self.state.sent.append((self.state.now, device, command))
        if command in self.state.suppress:
            return len(frame)
        if command == 0xF0:
            payload = self.state.configs[device]
        elif command == 0xF1:
            payload = ZERO
        elif command == 0xB0:
            payload = b"\xb0\x00" + struct.pack("<f", 1.25) + b"\x00"
        elif command == 0xA3:
            payload = b"\xa3\x00\x01" + struct.pack("<i", self.state.counts[device])
        else:
            raw = -12345 if command == 0xA2 else -1234
            payload = bytes([command]) + struct.pack("<i", raw)
        reply = FRAME.pack(device, len(payload), payload)
        self.state.query_pending.append(reply)
        if self.state.capture_enabled:
            self.state.capture_ready.clear()
            self.state.capture_pending.put((frame, [], recorder.socket.MSG_DONTROUTE))
            self.state.capture_pending.put(reply)
            if device == 2 and command == 0xA3 and not self.state.injected:
                self.state.capture_pending.put(FRAME.pack(0x502, 8, TARGET))
                self.state.capture_pending.put(FRAME.pack(2, 7, FEEDBACK))
                self.state.capture_pending.put(FRAME.pack(0x102, 1, b"\xcf"))
                self.state.injected = True
            assert self.state.capture_ready.wait(1.0), "observer did not process copies of replies"
        return len(frame)

    def recv(self, size):
        assert size == FRAME.size
        while self.state.query_pending:
            if self.state.timeout != 0:
                self.state.now += 0.0001
            item = self.state.query_pending.popleft()
            frame = item[0] if isinstance(item, tuple) else item
            if accepts_frame(frame, self.state.query_filters):
                return item
        if self.state.timeout == 0:
            raise BlockingIOError
        self.state.now += self.state.timeout
        raise TimeoutError

    def recvmsg(self, size, ancillary_size):
        item = self.recv(size)
        if isinstance(item, tuple):
            frame, ancillary, flags = item
        else:
            frame, ancillary, flags = item, [], 0
        return frame, ancillary, flags, ("can0",)


class ObserverBus:
    """Deliberately has no send method."""

    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.state.capture_closed = True

    def bind(self, address):
        assert address == ("can0",)

    def setsockopt(self, level, option, payload):
        self.state.capture_options.append((level, option, payload))
        if level == recorder.socket.SOL_CAN_RAW and option == recorder.socket.CAN_RAW_FILTER:
            self.state.capture_filters = payload

    def settimeout(self, timeout):
        self.timeout = timeout

    def setblocking(self, blocking):
        self.timeout = None if blocking else 0

    def recv(self, size):
        assert size == FRAME.size
        while True:
            if self.state.capture_pending.empty():
                self.state.capture_ready.set()
            try:
                item = self.state.capture_pending.get(timeout=self.timeout)
            except queue.Empty:
                if self.timeout == 0:
                    raise BlockingIOError
                raise TimeoutError
            frame = item[0] if isinstance(item, tuple) else item
            if accepts_frame(frame, self.state.capture_filters):
                return item

    def recvmsg(self, size, ancillary_size):
        item = self.recv(size)
        if isinstance(item, tuple):
            frame, ancillary, flags = item
        else:
            frame, ancillary, flags = item, [], 0
        return frame, ancillary, flags, ("can0",)


class MITFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.motor = recorder.Motor("steering", 2, 1.0, -1, -1)
        self.by_id = {value: self.motor for value in (2, 0x102, 0x402, 0x502)}

    def process(self, can_id, payload, stamp=10.0, dlc=None):
        return recorder.process_frame(FRAME.pack(can_id, len(payload) if dlc is None else dlc, payload),
                                      self.by_id, 5.0, stamp, 1700000000 + stamp)

    def initialise(self):
        self.process(2, CONFIG, 6)
        return self.process(2, ZERO, 7)

    @contextlib.contextmanager
    def fake_clock(self, state):
        with patch.object(recorder.time, "monotonic", side_effect=state.clock), \
                patch.object(recorder.time, "time", side_effect=state.wall_clock), \
                patch.object(recorder.time, "sleep", side_effect=state.sleep), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield

    def test_defaults_match_original_collector(self):
        args = recorder.parse_args([])
        self.assertEqual((args.rate, args.timeout, args.duration), (200, .002, 0))
        self.assertEqual(args.motors, ["steering", "drive"])
        self.assertEqual((args.steering_id, args.drive_id), (2, 1))
        self.assertEqual((args.steering_encoder_sign, args.drive_encoder_sign), (-1, -1))
        self.assertTrue(args.output.startswith("motor_feedback_mit_"))

    def test_f0_little_endian_units_and_nondefault_ranges(self):
        row = self.process(2, CONFIG)
        self.assertEqual((row["mit_position_max_rad"], row["mit_velocity_max_rad_s"],
                          row["mit_torque_max_nm"]), (95.5, 45, 18))
        custom = b"\xf0" + struct.pack("<HHH", 123, 2345, 678)
        row = self.process(2, custom)
        self.assertAlmostEqual(row["mit_position_max_rad"], 12.3)
        self.assertAlmostEqual(row["mit_velocity_max_rad_s"], 23.45)
        self.assertAlmostEqual(row["mit_torque_max_nm"], 6.78)

    def test_target_decoding_all_fields_signs_origin_and_timestamp(self):
        self.initialise()
        self.motor.gear_ratio = 1.7
        for can_id in (0x402, 0x502):
            row = self.process(can_id, TARGET)
            self.assertEqual(row["record_type"], "mit_target")
            self.assertEqual([row[f"mit_{name}_code"] for name in ("position", "velocity", "kp", "kd", "torque")],
                             [33111, 2048, 82, 164, 2048])
            p = (33111 / 65535 * 2 - 1) * 95.5
            zero = (32768 / 65535 * 2 - 1) * 95.5
            self.assertAlmostEqual(row["mit_target_position_motor_rad"], p)
            self.assertAlmostEqual(row["mit_target_position_rad"], -(p - zero) / 1.7)
            self.assertAlmostEqual(row["mit_target_velocity_rad_s"], -45 / 4095 / 1.7)
            self.assertAlmostEqual(row["mit_feedforward_torque_nm"], -18 / 4095)
            self.assertAlmostEqual(row["mit_kp_nm_per_motor_rad"], 82 * 500 / 4095)
            self.assertAlmostEqual(row["mit_kd_nm_s_per_motor_rad"], 164 * 5 / 4095)
            self.assertEqual((row["elapsed_s"], row["time_unix_s"]), (5, 1700000010))
            self.assertEqual(row["mit_payload_hex"], TARGET.hex().upper())
            self.assertTrue(math.isnan(row["torque_nm"]))  # Target is not measured torque.

    def test_f1_feedback_status_decoding_and_first_zero_is_retained(self):
        first = self.initialise()
        self.assertEqual(first["mit_feedback_position_rad"], 0)
        self.assertFalse(first["mit_is_active"])
        self.assertFalse(first["mit_has_fault"])
        row = self.process(2, FEEDBACK)
        self.assertEqual(row["record_type"], "mit_feedback")
        self.assertAlmostEqual(row["mit_feedback_velocity_motor_rad_s"], (2000 / 4095 * 2 - 1) * 45)
        self.assertAlmostEqual(row["mit_feedback_torque_motor_nm"], (2050 / 4095 * 2 - 1) * 18)
        self.assertEqual(row["mit_feedback_torque_nm"], -row["mit_feedback_torque_motor_nm"])
        self.assertEqual(row["mit_zero_position_motor_rad"], first["mit_zero_position_motor_rad"])
        self.assertEqual(row["mit_feedback_elapsed_s"], row["elapsed_s"])
        self.assertTrue(row["mit_is_active"] and row["mit_has_fault"])

    def test_sine_sender_and_collector_agree_for_actual_nondefault_ranges(self):
        self.process(2, b"\xf0" + struct.pack("<HHH", 120, 800, 300))
        first = self.process(2, ZERO)
        zero = first["mit_zero_position_motor_rad"]
        args = sender.parse_args(["--amplitude", "0.1", "--period", "0.5"])
        ranges = sender.MITRanges(12, 8, 3)
        for elapsed, relative in ((0, 0), (.125, .1), (.25, 0), (.375, -.1)):
            target = sender.sine_position(args, elapsed, zero)
            payload = sender.mit_payload(target, ranges)
            row = self.process(0x502, payload)
            self.assertAlmostEqual(row["mit_target_position_rad"], relative, delta=12 / 65535)
            self.assertAlmostEqual(row["mit_target_velocity_motor_rad_s"], 8 / 4095)
            self.assertAlmostEqual(row["mit_feedforward_torque_motor_nm"], 3 / 4095)

    def test_unknown_ranges_preserve_raw_codes_without_fabricating_units(self):
        for can_id, payload in ((0x502, TARGET), (2, FEEDBACK)):
            row = self.process(can_id, payload)
            self.assertFalse(row["mit_decode_ready"])
            self.assertEqual(row["mit_position_code"], 33111)
            self.assertEqual(row["mit_payload_hex"], payload.hex().upper())
            self.assertTrue(math.isnan(row["mit_target_position_motor_rad"]))
            self.assertTrue(math.isnan(row["mit_feedback_position_motor_rad"]))
        self.assertTrue(math.isnan(self.motor.mit_zero_position_motor_rad))

    def test_no_zero_yet_keeps_absolute_target_but_relative_is_unknown(self):
        self.process(2, CONFIG)
        row = self.process(0x502, TARGET)
        self.assertTrue(math.isfinite(row["mit_target_position_motor_rad"]))
        self.assertTrue(math.isnan(row["mit_target_position_rad"]))

    def test_config_write_invalidates_until_confirmation_and_preserves_old_rows(self):
        self.initialise()
        earlier = self.process(0x502, TARGET, 8)
        requested = b"\xf0" + struct.pack("<HHH", 120, 800, 300)
        row = self.process(0x102, requested, 9)
        self.assertFalse(row["mit_decode_ready"])
        self.assertTrue(row["mit_config_write_pending"])
        self.assertTrue(math.isnan(row["mit_zero_position_motor_rad"]))
        unknown = self.process(0x502, TARGET, 10)
        self.assertFalse(unknown["mit_decode_ready"])
        stale = self.process(2, CONFIG, 10.5)
        self.assertFalse(stale["mit_decode_ready"])
        self.assertIn("differs", stale["can_note"])
        confirmed = self.process(2, requested, 11)
        self.assertFalse(confirmed["mit_config_write_pending"])
        later = self.process(0x502, TARGET, 12)
        self.assertAlmostEqual(later["mit_target_position_motor_rad"], (33111 / 65535 * 2 - 1) * 12)
        self.assertEqual(earlier["mit_position_max_rad"], 95.5)
        self.assertFalse(unknown["mit_decode_ready"])  # Never retroactively decoded with later ranges.

    def test_invalid_ranges_never_use_defaults(self):
        self.initialise()
        row = self.process(2, b"\xf0" + struct.pack("<HHH", 0, 4500, 1800))
        self.assertFalse(row["mit_decode_ready"])
        self.assertFalse(self.process(0x502, TARGET)["mit_decode_ready"])

    def test_only_selected_standard_mit_frames_are_targets(self):
        frames = [(0x503, TARGET),
                  (0x80000502, TARGET), (0x40000502, TARGET), (0x20000502, TARGET),
                  (0x80000002, b"\xf1" + bytes(6))]
        for can_id, payload in frames:
            with self.subTest(can_id=can_id):
                self.assertIsNone(self.process(can_id, payload))
        self.assertIsNone(recorder.process_frame(b"short", self.by_id, 5, 10, 123))
        row = self.process(0x102, b"\xcf")
        self.assertEqual(row["record_type"], "mit_disable")
        self.assertEqual(self.process(2, b"\xcf" + bytes(7))["record_type"], "motor_status")
        for can_id, data in ((0x102, b"\xc2" + bytes(4)), (2, TARGET),
                             (0x502, TARGET[:7]), (2, b"\xf1" + bytes(5))):
            self.assertEqual(self.process(can_id, data)["record_type"], "can_frame")

    def test_queries_only_allow_f0_f1_and_time_out_without_state_guess(self):
        state = State()
        bus = QueryBus(state)
        with self.fake_clock(state):
            recorder.query_mit_read(bus, 2, 0xF0, .002)
            state.suppress.add(0xF1)
            recorder.query_mit_read(bus, 2, 0xF1, .002)
            for opcode in (0xC0, 0xC2, 0xCF, 0xB1):
                with self.assertRaises(ValueError):
                    recorder.query_mit_read(bus, 2, opcode, .002)
        self.assertEqual([command for _, _, command in state.sent], [0xF0, 0xF1])
        self.assertIsNone(self.motor.mit_ranges)

    def test_legacy_measurements_and_missing_samples_are_unchanged(self):
        state = State()
        bus = QueryBus(state)
        with self.fake_clock(state):
            first = recorder.collect_motor(bus, self.motor, 0, .002)
            self.assertEqual(first["position_unwrapped_rad"], 0)
            state.counts[2] += 40000
            row = recorder.collect_motor(bus, self.motor, 0, .002)
            self.assertAlmostEqual(row["position_unwrapped_rad"], -40000 * 2 * math.pi / 16384)
            self.assertAlmostEqual(row["velocity_rad_s"], 123.45 * 2 * math.pi / 60)
            self.assertAlmostEqual(row["torque_nm"], 1.234 * 1.25)
            self.assertLess(row["position_elapsed_s"], row["velocity_elapsed_s"])
            self.assertLess(row["velocity_elapsed_s"], row["current_elapsed_s"])
            state.suppress.update((0xA1, 0xA2, 0xA3))
            missing = recorder.collect_motor(bus, self.motor, 0, .002)
        for field in ("position_unwrapped_rad", "velocity_rad_s", "torque_nm", "current_a"):
            self.assertTrue(math.isnan(missing[field]))
        self.assertIsNone(missing["encoder_count"])

    def test_full_csv_capture_read_only_shutdown_and_no_overwrite(self):
        state = State()
        state.capture_enabled = True
        bus, observer = QueryBus(state), ObserverBus(state)
        with tempfile.TemporaryDirectory() as directory, self.fake_clock(state):
            path = Path(directory) / "mit.csv"
            args = recorder.parse_args(["--duration", ".015", "--output", str(path)])
            with patch.object(recorder.socket, "socket", side_effect=[bus, observer]):
                recorder.run(args)
            with path.open(newline="") as log:
                rows = list(csv.DictReader(log))
            self.assertEqual(list(rows[0]), list(recorder.FIELDS))
            measured = [row for row in rows if row["record_type"] == "feedback"]
            self.assertEqual([row["motor"] for row in measured], ["steering", "drive"] * 3)
            targets = [row for row in rows if row["record_type"] == "mit_target"]
            self.assertEqual(len(targets), 1)
            self.assertTrue(math.isfinite(float(targets[0]["mit_target_position_rad"])))
            self.assertEqual(sum(row["record_type"] == "mit_disable" for row in rows), 1)
            self.assertEqual(sum(row["record_type"] == "mit_feedback" for row in rows), 3)
            self.assertEqual(sum(row["record_type"] == "mit_config" for row in rows), 2)
            self.assertTrue(state.query_closed and state.capture_closed)
            self.assertTrue(all(opcode in (0xA1, 0xA2, 0xA3, 0xB0, 0xF0, 0xF1)
                                for _, _, opcode in state.sent))
            query_ids = [entry[0] for entry in struct.iter_unpack("=II", state.query_filters)]
            capture_ids = [entry[0] for entry in struct.iter_unpack("=II", state.capture_filters)]
            self.assertEqual(query_ids, [2, 1])
            self.assertEqual(capture_ids, [2, 0x102, 0x402, 0x502, 1, 0x101, 0x401, 0x501,
                                          0, 0xFF, 0x400, 0x4FF])
            for options, filters in ((state.query_options, state.query_filters),
                                     (state.capture_options, state.capture_filters)):
                self.assertIn((recorder.socket.SOL_SOCKET, recorder.SO_RXQ_OVFL, 1), options)
                for can_id, mask in struct.iter_unpack("=II", filters):
                    self.assertEqual(mask, 0xC00007FF)
                    self.assertTrue(accepts_frame(FRAME.pack(can_id, 1, b"\xa1"), filters))
                    for flag in (0x80000000, 0x40000000, 0x20000000):
                        self.assertFalse(accepts_frame(FRAME.pack(can_id | flag, 1, b"\xa1"), filters))
            calls = len(state.sent)
            with patch.object(recorder.socket, "socket", side_effect=AssertionError("must not open CAN")):
                with self.assertRaises(FileExistsError):
                    recorder.run(args)
            self.assertEqual(len(state.sent), calls)

    def test_capture_failure_reaches_writer(self):
        class BrokenBus:
            def recvmsg(self, size, ancillary_size):
                raise OSError("observer disconnected")

        pending = queue.Queue()
        recorder.receive_mit(BrokenBus(), [self.motor], 0, threading.Event(), pending)
        writer = csv.DictWriter(io.StringIO(), fieldnames=recorder.FIELDS)
        with self.assertRaisesRegex(OSError, "observer disconnected"):
            recorder.write_pending(writer, pending)

    def test_all_six_queries_use_exact_request_and_reply_formats(self):
        state, bus = State(), None
        bus = QueryBus(state)
        with self.fake_clock(state):
            for opcode, expected_length in recorder.READ_LENGTHS.items():
                reply = recorder.read_query(bus, 2, opcode, .002)
                self.assertEqual(len(reply.data), expected_length)
                self.assertEqual(reply.data[0], opcode)
                self.assertEqual(reply.received_s, state.now)
                self.assertEqual(reply.wall_time_s, 1700000000 + state.now)
        self.assertEqual([opcode for _, _, opcode in state.sent], list(recorder.READ_LENGTHS))

    def test_query_skips_short_wrong_id_opcode_dlc_flags_and_local_frames(self):
        state = State()

        class NoisyBus(QueryBus):
            def send(self, frame):
                result = super().send(frame)
                good = b"\xa1" + struct.pack("<i", 9999)
                noise = [b"short", FRAME.pack(1, 5, good), FRAME.pack(2, 4, good),
                         FRAME.pack(2, 5, b"\xa2" + good[1:]), FRAME.pack(0x80000002, 5, good),
                         FRAME.pack(0x40000002, 5, good), FRAME.pack(0x20000002, 5, good),
                         (FRAME.pack(2, 5, good), [], recorder.socket.MSG_DONTROUTE)]
                self.state.query_pending.extendleft(reversed(noise))
                return result

        with self.fake_clock(state):
            reply = recorder.read_query(NoisyBus(state), 2, 0xA1, .002)
        self.assertEqual(struct.unpack("<i", reply.data[1:])[0], -1234)

    def test_stale_drain_is_bounded_and_never_sends_with_uncleared_backlog(self):
        class FloodBus(QueryBus):
            def recvmsg(self, size, ancillary_size):
                return FRAME.pack(2, 5, b"\xa1" + bytes(4)), [], 0, ("can0",)

        state = State()
        with self.fake_clock(state):
            self.assertIsNone(recorder.read_query(FloodBus(state), 2, 0xA1, .002))
        self.assertEqual(state.sent, [])

    def test_send_timeout_and_short_send_are_errors_not_missing_measurements(self):
        for failure in ("timeout", "partial"):
            class FailedSend(QueryBus):
                def send(self, frame):
                    if failure == "timeout":
                        raise TimeoutError("transmission blocked")
                    return 15

            state = State()
            with self.subTest(failure=failure), self.fake_clock(state), self.assertRaises(OSError):
                recorder.read_query(FailedSend(state), 2, 0xA1, .002)

    def test_b0_wrong_lengths_and_invalid_constants_never_produce_guessed_torque(self):
        for dlc, constant in ((6, 1.25), (8, 1.25), (7, 0), (7, math.nan), (7, math.inf), (7, 100)):
            class BadConstantBus(QueryBus):
                def send(self, frame):
                    result = super().send(frame)
                    if frame[8] == 0xB0:
                        self.state.query_pending[-1] = FRAME.pack(2, dlc,
                            b"\xb0\x00" + struct.pack("<f", constant) + b"\x00\x00")
                    return result

            state = State()
            motor = recorder.Motor("steering", 2, 1, -1, -1)
            with self.subTest(dlc=dlc, constant=constant), self.fake_clock(state):
                row = recorder.collect_motor(BadConstantBus(state), motor, 0, .002)
                self.assertTrue(math.isnan(row["torque_nm"]))
                self.assertAlmostEqual(row["current_a"], -1.234)
                self.assertEqual(row["b0_reply_received"], dlc == 7)

    def test_native_id_local_f0_write_is_not_mistaken_for_motor_reply(self):
        self.initialise()
        frame = FRAME.pack(2, 7, CONFIG)
        row = recorder.process_frame(frame, self.by_id, 5, 10, 20, local_tx=True)
        self.assertEqual(row["mit_config_source"], "write_request")
        self.assertEqual(row["can_direction"], "local_tx")
        self.assertFalse(row["mit_decode_ready"])
        self.assertEqual(self.process(2, CONFIG)["mit_config_source"], "reply")

    def test_broadcast_and_public_mit_frames_are_recorded_for_each_selected_motor(self):
        steering = recorder.Motor("steering", 2, 1, -1, -1)
        drive = recorder.Motor("drive", 1, 1, 1, -1)
        state = State()
        state.capture_pending.put(FRAME.pack(2, 7, CONFIG))
        state.capture_pending.put(FRAME.pack(1, 7, CONFIG))
        state.capture_pending.put(FRAME.pack(0x400, 8, TARGET))
        state.capture_pending.put(FRAME.pack(0x4FF, 8, TARGET))
        stopped, pending = threading.Event(), queue.Queue()
        stopped.set()
        with self.fake_clock(state):
            recorder.receive_mit(ObserverBus(state), [steering, drive], 0, stopped, pending)
        rows = list(pending.queue)
        targets = [row for row in rows if row["record_type"] == "mit_target"]
        self.assertEqual([(row["mit_can_id"], row["motor"]) for row in targets],
                         [(0x400, "steering"), (0x400, "drive"), (0x4FF, "steering"), (0x4FF, "drive")])

    def test_shutdown_drains_all_queued_frames_including_final_cf(self):
        state = State()
        for can_id, data in ((2, CONFIG), (2, ZERO), (0x502, TARGET), (2, FEEDBACK), (0x102, b"\xcf")):
            state.capture_pending.put(FRAME.pack(can_id, len(data), data))
        stopped, pending = threading.Event(), queue.Queue()
        stopped.set()
        with self.fake_clock(state):
            recorder.receive_mit(ObserverBus(state), [self.motor], 0, stopped, pending)
        self.assertEqual([row["record_type"] for row in list(pending.queue)],
                         ["mit_config", "mit_feedback", "mit_target", "mit_feedback", "mit_disable"])

    def test_live_writer_batch_returns_even_when_producer_keeps_queue_nonempty(self):
        pending = queue.Queue()
        pending.put({"n": 0})
        calls = []

        class Writer:
            def writerow(self, row):
                calls.append(row)
                pending.put({"n": row["n"] + 1})

        recorder.write_pending(Writer(), pending)
        self.assertEqual(calls, [{"n": 0}])
        self.assertEqual(pending.qsize(), 1)

    def test_shutdown_flood_and_writer_backlog_raise_explicit_errors(self):
        for failure in ("shutdown_flood", "writer_backlog"):
            state = State()

            class FloodBus(ObserverBus):
                def recvmsg(self, size, ancillary_size):
                    state.now += .1 if failure == "shutdown_flood" else 0
                    return FRAME.pack(0x502, 8, TARGET), [], 0, ("can0",)

            stopped, pending = threading.Event(), queue.Queue()
            stopped.set()
            if failure == "writer_backlog":
                pending.put({"existing": 1})
            with self.subTest(failure=failure), self.fake_clock(state), \
                    patch.object(recorder, "MAX_PENDING_ROWS", 1 if failure == "writer_backlog" else 20000):
                recorder.receive_mit(FloodBus(state), [self.motor], 0, stopped, pending)
            error = list(pending.queue)[-1]
            self.assertIsInstance(error, RuntimeError)
            self.assertIn("drain timed out" if failure == "shutdown_flood" else "backlog exceeded", str(error))

    def test_receive_overflow_is_visible_and_stops_capture_after_preserving_frame(self):
        state = State()
        ancillary = [(recorder.socket.SOL_SOCKET, recorder.SO_RXQ_OVFL, struct.pack("=I", 5))]
        state.capture_pending.put((FRAME.pack(0x502, 8, TARGET), ancillary, 0))
        stopped, pending = threading.Event(), queue.Queue()
        stopped.set()
        with self.fake_clock(state):
            recorder.receive_mit(ObserverBus(state), [self.motor], 0, stopped, pending)
        row, error = pending.get_nowait(), pending.get_nowait()
        self.assertEqual(row["socket_dropped_frames"], 5)
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("dropped CAN frames", str(error))

    def test_query_overflow_and_truncated_metadata_are_not_silently_ignored(self):
        for flags, ancillary in (
            (recorder.socket.MSG_TRUNC, []), (recorder.socket.MSG_CTRUNC, []),
            (0, [(recorder.socket.SOL_SOCKET, recorder.SO_RXQ_OVFL, struct.pack("=I", 1))]),
        ):
            state = State()
            state.query_pending.append((FRAME.pack(2, 5, b"\xa1" + bytes(4)), ancillary, flags))
            with self.subTest(flags=flags), self.fake_clock(state), self.assertRaises(RuntimeError):
                recorder.read_query(QueryBus(state), 2, 0xA1, .002)

    def test_query_deadline_includes_receipt_and_invalid_overflow_metadata_fails(self):
        state = State()

        class LateBus(QueryBus):
            def recvmsg(bus, size, ancillary_size):
                packet = super().recvmsg(size, ancillary_size)
                state.now += .003
                return packet

        with self.fake_clock(state):
            self.assertIsNone(recorder.read_query(LateBus(state), 2, 0xA1, .002))
        self.assertEqual(len(state.sent), 1)
        state = State()
        state.query_pending.append((FRAME.pack(2, 5, b"\xa1" + bytes(4)),
                                    [(recorder.socket.SOL_SOCKET, recorder.SO_RXQ_OVFL, bytes(3))], 0))
        with self.fake_clock(state), self.assertRaisesRegex(RuntimeError, "invalid SO_RXQ_OVFL"):
            recorder.read_query(QueryBus(state), 2, 0xA1, .002)

    def test_signed_int32_measurements_and_multi_turn_difference_do_not_overflow(self):
        state = State()
        self.motor.gear_ratio = 1.7
        state.counts[2] = -(1 << 31)

        class EdgeBus(QueryBus):
            def send(bus, frame):
                result = super().send(frame)
                if frame[8] in (0xA1, 0xA2):
                    raw = -(1 << 31) if frame[8] == 0xA1 else (1 << 31) - 1
                    state.query_pending[-1] = FRAME.pack(2, 5, bytes([frame[8]]) + struct.pack("<i", raw))
                return result

        with self.fake_clock(state):
            first = recorder.collect_motor(EdgeBus(state), self.motor, 0, .002)
            state.counts[2] = (1 << 31) - 1
            second = recorder.collect_motor(EdgeBus(state), self.motor, 0, .002)
        self.assertEqual(first["position_unwrapped_rad"], 0)
        self.assertAlmostEqual(second["position_unwrapped_rad"], -((1 << 32) - 1) * 2 * math.pi / 16384 / 1.7)
        self.assertAlmostEqual(second["velocity_rad_s"], -((1 << 31) - 1) * .01 * 2 * math.pi / 60 / 1.7)
        self.assertAlmostEqual(second["current_a"], -(1 << 31) * .001)
        self.assertAlmostEqual(second["torque_nm"], (1 << 31) * .001 * 1.25)

    def test_cf_ack_and_unsolicited_ae_status_are_decoded(self):
        for opcode in (0xAE, 0xCF):
            data = bytes([opcode]) + struct.pack("<HHBBB", 2428, 12, 38, 0, 0x29)
            row = self.process(2, data)
            self.assertEqual(row["record_type"], "motor_status")
            self.assertEqual(row["can_opcode"], opcode)
            self.assertAlmostEqual(row["bus_voltage_v"], 24.28)
            self.assertAlmostEqual(row["bus_current_a"], .12)
            self.assertEqual(row["bus_current_raw"], 12)
            self.assertEqual((row["motor_temperature_c"], row["motor_mode_raw"], row["motor_fault_code"]),
                             (38, 0, 0x29))

    def test_origin_request_pauses_without_transmitting_and_reboot_stops_capture(self):
        for data in (b"\xb1", b"\x00\xff\x00\xff\x00\xff\x00\xff"):
            motor = recorder.Motor("steering", 2, 1, -1, -1, zero_count=123)
            motor.mit_zero_position_motor_rad = .5
            state = State()
            state.capture_pending.put((FRAME.pack(0x102, len(data), data), [], recorder.socket.MSG_DONTROUTE))
            stopped, pending = threading.Event(), queue.Queue()
            stopped.set()
            with self.subTest(data=data), self.fake_clock(state):
                recorder.receive_mit(ObserverBus(state), [motor], 0, stopped, pending)
            self.assertEqual(pending.get_nowait()["record_type"], "origin_reset")
            if data == b"\xb1":
                self.assertTrue(pending.empty())
                with self.fake_clock(state):
                    paused = recorder.collect_motor(QueryBus(state), motor, 0, .002)
                self.assertTrue(paused["origin_pending"])
                self.assertTrue(math.isnan(paused["position_unwrapped_rad"]))
                self.assertEqual(state.sent, [])
            else:
                self.assertIsInstance(pending.get_nowait(), RuntimeError)
            self.assertTrue(motor.reference_invalid)
            self.assertEqual(motor.reference_epoch, 1)
            self.assertIsNone(motor.zero_count)
            self.assertTrue(math.isnan(motor.mit_zero_position_motor_rad))
            self.assertIsNone(motor.mit_ranges)

    def test_b1_acknowledgement_starts_new_epoch_without_mixing_references(self):
        self.initialise()
        self.motor.zero_count = 123
        self.process(0x102, b"\xb1", 8)
        self.process(2, CONFIG, 8.1)  # Still pending; cannot restore decoding yet.
        self.process(2, ZERO, 8.2)
        self.assertIsNone(self.motor.mit_ranges)
        self.assertTrue(math.isnan(self.motor.mit_zero_position_motor_rad))
        ack = self.process(2, b"\xb1\x7b\x00", 9)
        self.assertEqual((ack["reference_epoch"], ack["origin_pending"]), (1, False))
        self.process(2, b"\xa3" + struct.pack("<Hi", 123, 123), 9.05)
        self.assertIsNone(self.motor.zero_count)  # Pre-F0 traffic cannot become the new preflight zero.
        self.process(2, CONFIG, 9.1)
        self.assertEqual(self.process(2, FEEDBACK, 9.2)["mit_feedback_position_rad"], 0)
        state = State()
        with self.fake_clock(state):
            paused = recorder.collect_motor(QueryBus(state), self.motor, 0, .002)
        self.assertTrue(paused["startup_pending"])
        self.assertEqual(state.sent, [])
        self.process(0x502, TARGET, 9.3)
        with self.fake_clock(state):
            row = recorder.collect_motor(QueryBus(state), self.motor, 0, .002)
        self.assertEqual(row["zero_count"], -20000)
        self.assertEqual(row["position_unwrapped_rad"], 0)
        self.assertEqual(row["reference_epoch"], 1)
        self.assertFalse(row["origin_pending"])
        # An ack without a captured request still invalidates the old references.
        unsolicited = self.process(2, b"\xb1\x7b\x00", 10)
        self.assertEqual(unsolicited["reference_epoch"], 2)
        self.assertIsNone(self.motor.zero_count)

    def test_origin_change_during_a3_discards_entire_mixed_cycle(self):
        self.initialise()
        state = State()

        class ResetBus(QueryBus):
            def send(bus, frame):
                result = super().send(frame)
                if frame[8] == 0xA3:
                    self.process(0x102, b"\xb1")
                    self.process(2, b"\xb1\x00\x00")
                return result

        with self.fake_clock(state):
            row = recorder.collect_motor(ResetBus(state), self.motor, 0, .002)
        self.assertEqual(row["reference_epoch"], 1)
        self.assertIn("discarded", row["can_note"])
        self.assertIsNone(self.motor.zero_count)
        for field in ("position_unwrapped_rad", "velocity_rad_s", "current_a", "torque_nm"):
            self.assertTrue(math.isnan(row[field]))

    def test_csv_records_whole_sender_startup_b1_mit_and_exit_sequence(self):
        state = State()
        state.capture_enabled = state.injected = True

        class StartupBus(QueryBus):
            reset_done = False

            def send(bus, frame):
                result = super().send(frame)
                if frame[8] == 0xA3 and not bus.reset_done:
                    bus.reset_done = True
                    state.capture_ready.clear()
                    state.configs[2] = MIT_CONFIG
                    state.counts[2] = 50  # A later poll must not become the pre-motion zero.
                    for can_id, data in ((0x102, b"\xcf"), (2, b"\xcf" + bytes(7)),
                                         (0x102, b"\xb1"), (2, b"\xb1\x00\x00"),
                                         (0x102, b"\xf0"), (2, CONFIG), (0x102, MIT_CONFIG), (2, MIT_CONFIG),
                                         (0x102, b"\xf0"), (2, MIT_CONFIG),
                                         (0x102, b"\xa3"), (2, b"\xa3" + bytes(6)),
                                         (0x102, b"\xf1"), (2, MIT_ZERO),
                                         (0x502, MIT_TARGET), (2, MIT_FEEDBACK),
                                         (0x102, b"\xcf"), (2, b"\xcf" + bytes(7))):
                        flags = recorder.socket.MSG_DONTROUTE if can_id in (0x102, 0x502) else 0
                        state.capture_pending.put((FRAME.pack(can_id, len(data), data), [], flags))
                    self.assertTrue(state.capture_ready.wait(1))
                return result

        with tempfile.TemporaryDirectory() as directory, self.fake_clock(state):
            path = Path(directory) / "startup.csv"
            args = recorder.parse_args(["--motors", "steering", "--duration", ".015", "--output", str(path)])
            with patch.object(recorder.socket, "socket", side_effect=[StartupBus(state), ObserverBus(state)]):
                recorder.run(args)
            with path.open(newline="") as log:
                rows = list(csv.DictReader(log))
        origin_rows = [row for row in rows if row["record_type"] == "origin_reset"]
        self.assertEqual([row["origin_pending"] for row in origin_rows], ["True", "False"])
        self.assertEqual([row["reference_epoch"] for row in origin_rows], ["1", "1"])
        self.assertEqual(sum(row["record_type"] == "mit_target" for row in rows), 1)
        self.assertEqual(sum(row["record_type"] == "mit_disable" for row in rows), 2)
        self.assertEqual(sum(row["record_type"] == "motor_status" for row in rows), 2)
        target = next(row for row in rows if row["record_type"] == "mit_target")
        self.assertEqual(float(target["mit_position_max_rad"]), 3)
        self.assertAlmostEqual(float(target["mit_target_position_rad"]), .1, delta=6 / 65535)
        self.assertEqual(target["startup_pending"], "False")
        measurements = [row for row in rows if row["record_type"] == "feedback"]
        self.assertTrue(math.isnan(float(measurements[0]["position_unwrapped_rad"])))
        self.assertAlmostEqual(float(measurements[-1]["position_unwrapped_rad"]), -50 * 2 * math.pi / 16384)
        self.assertEqual(measurements[-1]["zero_count"], "0")
        self.assertEqual(measurements[-1]["reference_epoch"], "1")

    def observe_current_sender(self, bus, state):
        """Run the real sender functions; only the CAN transport is simulated."""
        observed, samples, startup_query_counts = [], [], []
        original_send, original_recv = bus.send, bus.recv

        def capture(frame, local_tx):
            row = recorder.process_frame(frame, self.by_id, 0, bus.now, 1700000000 + bus.now, local_tx)
            if row is not None:
                observed.append(row)
            before = len(state.sent)
            samples.append(recorder.collect_motor(QueryBus(state), self.motor, 0, .002))
            if not any(can_id == 0x502 for _, can_id, _, _ in bus.frames):
                startup_query_counts.append(len(state.sent) - before)

        def send(frame):
            result = original_send(frame)
            capture(frame, True)
            return result

        def recv(size):
            frame = original_recv(size)
            capture(frame, False)
            return frame

        args = sender.parse_args(["--execute", "--amplitude", ".1", "--period", ".5", "--duration", ".016"])
        with patch.object(bus, "send", side_effect=send), patch.object(bus, "recv", side_effect=recv), \
                patch.object(sender.socket, "socket", return_value=bus), \
                patch.object(sender.time, "monotonic", side_effect=lambda: bus.now), \
                patch.object(sender.time, "time", side_effect=lambda: 1700000000 + bus.now), \
                patch.object(sender.time, "sleep", side_effect=bus.sleep), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                sender.execute(args)
            finally:
                self.assertTrue(startup_query_counts)
                self.assertEqual(set(startup_query_counts), {0})
        return observed, samples

    def test_current_sender_execution_preserves_custom_ranges_and_startup_zeros(self):
        for config in ((955, 4500, 1800), (120, 800, 300), (30, 1234, 456)):
            with self.subTest(config=config):
                self.setUp()
                bus, state = SineBus(), State()
                bus.config = config
                state.counts[2] = 100  # The first active poll is already away from zero.
                observed, samples = self.observe_current_sender(bus, state)
                self.assertEqual(self.motor.mit_ranges, (3, config[1] * .01, config[2] * .01))
                self.assertFalse(self.motor.startup_pending)
                self.assertFalse(self.motor.mit_config_write_pending)
                targets = [row for row in observed if row["record_type"] == "mit_target"]
                self.assertEqual(len(targets), 4)
                self.assertEqual(targets[0]["mit_target_position_rad"], 0)
                self.assertEqual(targets[0]["mit_position_max_rad"], sender.POSITION_MAX_RAD)
                self.assertAlmostEqual(targets[0]["mit_kp_nm_per_motor_rad"], 82 * 500 / 4095)
                self.assertAlmostEqual(targets[0]["mit_kd_nm_s_per_motor_rad"], 164 * 5 / 4095)
                for row in targets:
                    self.assertAlmostEqual(row["mit_target_velocity_motor_rad_s"], config[1] * .01 / 4095)
                    self.assertAlmostEqual(row["mit_feedforward_torque_motor_nm"], config[2] * .01 / 4095)
                active = next(row for row in samples if row["a3_reply_received"])
                self.assertEqual(active["zero_count"], 0)
                self.assertAlmostEqual(active["position_unwrapped_rad"], -100 * 2 * math.pi / 16384)
                self.assertAlmostEqual(active["mit_zero_position_motor_rad"], -3 / 65535)
                self.assertTrue(all(opcode in (0xA1, 0xA2, 0xA3, 0xB0) for _, _, opcode in state.sent))
                # The sender alone issued F0 writes; the recorder never did.
                writes = [row for row in observed if row["mit_config_source"] == "write_request"]
                self.assertEqual(len(writes), 1)
                self.assertEqual(writes[0]["mit_requested_position_max_rad"], 3)
                self.assertTrue(writes[0]["mit_config_write_pending"])

    def test_failed_current_sender_startup_stays_passive(self):
        for fault in ("missing_b1", "f0_write_not_applied", "f0_readback_mismatch", "moving_f1"):
            with self.subTest(fault=fault):
                self.setUp()
                bus, state = SineBus(fault), State()
                with self.assertRaises((TimeoutError, RuntimeError)):
                    self.observe_current_sender(bus, state)
                self.assertEqual(state.sent, [])
                self.assertTrue(self.motor.startup_pending)
                self.assertFalse(any(can_id == 0x502 for _, can_id, _, _ in bus.frames))
                if fault in ("f0_write_not_applied", "f0_readback_mismatch"):
                    self.assertTrue(self.motor.mit_config_write_pending)
                    self.assertIsNone(self.motor.mit_ranges)

    def test_f0_range_change_replaces_old_quantized_f1_zero(self):
        previous = self.initialise()
        requested = self.process(0x102, MIT_CONFIG)
        self.assertTrue(math.isnan(requested["mit_zero_position_motor_rad"]))
        self.assertEqual(requested["mit_requested_position_max_rad"], 3)
        self.process(2, MIT_CONFIG)
        self.process(2, MIT_ZERO)
        args = sender.parse_args(["--amplitude", ".1", "--period", ".5"])
        zero = self.motor.mit_zero_position_motor_rad
        target = sender.mit_payload(sender.sine_position(args, .125, zero), sender.MITRanges(3, 45, 18))
        row = self.process(0x502, target)
        self.assertAlmostEqual(row["mit_target_position_rad"], .1, delta=3 / 65535)
        self.assertAlmostEqual(row["mit_zero_position_motor_rad"], -3 / 65535)
        self.assertAlmostEqual(previous["mit_zero_position_motor_rad"], 95.5 / 65535)
        self.assertFalse(row["startup_pending"])

    def test_preflight_pause_is_per_motor_and_exit_cf_resumes_reads(self):
        self.initialise()
        self.process(0x102, b"\xcf")
        self.process(2, b"\xcf" + bytes(7))
        state = State()
        with self.fake_clock(state):
            paused = recorder.collect_motor(QueryBus(state), self.motor, 0, .002)
            self.assertEqual(state.sent, [])
            drive = recorder.Motor("drive", 1, 1, 1, -1)
            drive_row = recorder.collect_motor(QueryBus(state), drive, 0, .002)
        self.assertTrue(paused["startup_pending"])
        self.assertFalse(drive_row["startup_pending"])
        self.assertEqual({device for _, device, _ in state.sent}, {1})
        self.process(0x502, TARGET)
        exit_row = self.process(0x102, b"\xcf")
        self.assertFalse(exit_row["startup_pending"])
        with self.fake_clock(state):
            after_exit = recorder.collect_motor(QueryBus(state), self.motor, 0, .002)
        self.assertTrue(after_exit["a3_reply_received"])

    def test_configuration_change_during_query_discards_cycle_without_origin_change(self):
        self.initialise()
        self.motor.zero_count = 123
        state = State()

        class ReconfigureBus(QueryBus):
            def send(bus, frame):
                result = super().send(frame)
                if frame[8] == 0xA3:
                    self.process(0x102, MIT_CONFIG)
                    self.process(2, MIT_CONFIG)
                    self.process(2, MIT_ZERO)
                    self.process(0x502, MIT_TARGET)
                return result

        with self.fake_clock(state):
            row = recorder.collect_motor(ReconfigureBus(state), self.motor, 0, .002)
        self.assertEqual(row["reference_epoch"], 0)
        self.assertIn("discarded", row["can_note"])
        self.assertEqual(self.motor.zero_count, 123)
        self.assertTrue(math.isnan(row["position_unwrapped_rad"]))
        self.assertTrue(math.isnan(row["torque_nm"]))
        self.assertFalse(any(opcode in (0xA2, 0xA1) for _, _, opcode in state.sent))

    def test_all_other_selected_motor_commands_remain_available_as_raw_rows(self):
        for opcode in range(256):
            for length in range(1, 9):
                payload = bytes([opcode]) + bytes(range(1, length))
                row = self.process(0x102, payload)
                self.assertIsNotNone(row)
                self.assertEqual(row["mit_payload_hex"], payload.hex().upper())
                self.assertEqual(row["can_opcode"], opcode)
        empty = self.process(0x102, b"")
        self.assertEqual(empty["can_length"], 0)
        self.assertIsNone(empty["can_opcode"])

    def test_mit_nibble_boundaries_and_gain_endpoints(self):
        self.initialise()
        for code in (0, 1, 15, 16, 255, 256, 2047, 2048, 4094, 4095):
            payload = bytes([0xF0, 0x01, code >> 4, ((code & 15) << 4) | (code >> 8),
                             code & 255, code >> 4, ((code & 15) << 4) | (code >> 8), code & 255])
            row = self.process(0x502, payload)
            self.assertEqual(row["mit_position_code"], 0xF001)  # First byte is data, never an F0 opcode.
            for field in ("velocity", "kp", "kd", "torque"):
                self.assertEqual(row[f"mit_{field}_code"], code)
            self.assertAlmostEqual(row["mit_kp_nm_per_motor_rad"], code * 500 / 4095)
            self.assertAlmostEqual(row["mit_kd_nm_s_per_motor_rad"], code * 5 / 4095)

    def test_mit_position_and_feedback_endpoints_and_all_read_whitelist_rejections(self):
        self.initialise()
        for position, code, expected in ((0, 0, -1), (65535, 4095, 1)):
            packed = bytes([position >> 8, position & 255, code >> 4,
                            ((code & 15) << 4) | (code >> 8), code & 255])
            target = self.process(0x502, packed[:2] + bytes.fromhex("800000000800"))
            feedback = self.process(2, b"\xf1" + packed + b"\x01")
            self.assertEqual(target["mit_target_position_motor_rad"], expected * 95.5)
            self.assertEqual(feedback["mit_feedback_position_motor_rad"], expected * 95.5)
            self.assertEqual(feedback["mit_feedback_velocity_motor_rad_s"], expected * 45)
            self.assertEqual(feedback["mit_feedback_torque_motor_nm"], expected * 18)
        state = State()
        for opcode in set(range(256)) - set(recorder.READ_LENGTHS):
            with self.assertRaises(ValueError):
                recorder.read_query(QueryBus(state), 2, opcode, .002)
        self.assertEqual(state.sent, [])

    def test_bad_arguments_are_rejected(self):
        for args in (["--rate", "0"], ["--rate", "nan"], ["--duration", "-1"], ["--timeout", "0"],
                     ["--steering-id", "255"], ["--steering-gear-ratio", "0"],
                     ["--motors", "steering", "steering"], ["--steering-id", "1"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    recorder.parse_args(args)

    def test_main_reports_errors_and_restores_signal_handler_without_hardware(self):
        previous = recorder.signal.getsignal(recorder.signal.SIGTERM)
        for error, result in ((KeyboardInterrupt(), 0), (OSError("CAN disconnected"), 1),
                              (RuntimeError("capture lost"), 1), (ValueError("bad frame"), 1)):
            with self.subTest(error=type(error).__name__), \
                    patch.object(recorder, "run", side_effect=error), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(recorder.main([]), result)
            self.assertEqual(recorder.signal.getsignal(recorder.signal.SIGTERM), previous)


if __name__ == "__main__":
    unittest.main()
