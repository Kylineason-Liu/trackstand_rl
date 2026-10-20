"""Steering sweep tests use simulated CAN/clock only, never a real interface."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import heapq
import io
import math
from pathlib import Path
import socket
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import steering_speed_sweep as sweep


class Clock:
    now = 10.

    def monotonic(self):
        return self.now

    def time(self):
        return 1700000000+self.now


class MemoryLog:
    def __init__(self):
        self.rows, self.file, self.fail = [], io.StringIO(), False

    def emit(self, row):
        if self.fail:
            raise OSError("simulated disk failure")
        self.rows.append(row.copy())


class FakeCAN:
    """Instantaneous speed response is a TEST fixture, not an assumed real motor."""

    def __init__(self, clock):
        self.clock, self.frames, self.rx = clock, [], []
        self.sequence, self.mode, self.speed_code, self.closed = 0, 0, 0, False
        self.count, self.updated = 12345., clock.now
        self.config = {0xB3: b"\xb3"+struct.pack("<I", 3000),
                       0xCD: b"\xcd\x00"+struct.pack("<H", 5000)+b"\x00"}
        self.drop, self.temperature, self.fault = set(), 25, 0
        self.bad_cf = self.reverse_speed = self.ignore_zero = self.frozen_position = self.ignore_writes = False
        self.b3_offset, self.readback_offset, self.b3_writes = 0, 0, 0
        self.current_override = None
        self.query_delay = .0002

    def close(self):
        self.closed = True

    def integrate(self):
        if not self.frozen_position:
            self.count += self.speed_code*.01*16384/60*(self.clock.now-self.updated)
        self.updated = self.clock.now

    def enqueue(self, data, cid=2, delay=0., flags=0, ancillary=None):
        self.sequence += 1
        heapq.heappush(self.rx, (self.clock.now+delay, self.sequence,
                                sweep.CAN_FRAME.pack(cid, len(data), data), ancillary or [], flags))

    def status(self, opcode):
        return bytes([opcode])+struct.pack("<HH", 2400, 65000)+bytes([self.temperature, self.mode, self.fault])

    def send(self, frame):
        cid, length, padded = sweep.CAN_FRAME.unpack(frame)
        assert cid == 0x102, "Only the selected steering motor may be commanded"
        data = padded[:length]
        op = data[0]
        self.integrate()
        self.frames.append((self.clock.now, data))
        if op == 0xCF:
            self.mode = 3 if self.bad_cf else 0
            if not self.bad_cf:
                self.speed_code = 0
            reply = self.status(op)
        elif op == 0xC1:
            self.mode = 3
            code = struct.unpack("<i", data[1:])[0]
            if code != 0 or not self.ignore_zero:
                self.speed_code = code*(-1 if self.reverse_speed else 1)
            reply = b"\xc1"+struct.pack("<i", self.speed_code)
        elif op == 0xA3:
            count = round(self.count)
            signed = (count+2**31) % 2**32-2**31
            reply = b"\xa3"+struct.pack("<Hi", count % 16384, signed)
        elif op == 0xA2:
            reply = b"\xa2"+struct.pack("<i", self.speed_code)
        elif op == 0xA1:
            current = self.current_override
            if current is None:
                current = 0 if self.speed_code == 0 else (100 if self.speed_code > 0 else -100)
            reply = b"\xa1"+struct.pack("<i", current)
        elif op == 0xB0:
            reply = b"\xb0\x00"+struct.pack("<f", .78)+b"\x00"
        elif op == 0xB5:
            reply = b"\xb5"+struct.pack("<I", 10000)
        elif op == 0xAE:
            reply = self.status(op)
        elif op in self.config:
            write = len(data) > 1
            if write and not self.ignore_writes:
                if op == 0xB3:
                    self.b3_writes += 1
                    data = b"\xb3"+struct.pack("<I", struct.unpack("<I", data[1:])[0]+self.b3_offset)
                self.config[op] = data
            reply = self.config[op]
            if op == 0xB3 and not write and self.b3_writes:
                reply = b"\xb3"+struct.pack("<I", struct.unpack("<I", reply[1:])[0]+self.readback_offset)
        else:
            raise AssertionError(f"Unexpected opcode {op:02X}")
        if op not in self.drop:
            self.enqueue(reply, delay=self.query_delay)
        return len(frame)

    def recvmsg(self, size, ancillary_size):
        if not self.rx or self.rx[0][0] > self.clock.now:
            raise BlockingIOError
        _, _, frame, ancillary, flags = heapq.heappop(self.rx)
        return frame, ancillary, flags, ("fake",)

    def wait(self, duration):
        ready = self.rx[0][0] if self.rx else math.inf
        self.clock.now = min(self.clock.now+max(0., duration), ready)


class SteeringSweepTests(unittest.TestCase):
    def setUp(self):
        self.flags = ["--speeds", ".4", ".8", "--cycles", "1", "--sweep-angle", ".12",
                      "--analysis-angle", ".06", "--angle-limit", ".6", "--settle", ".02",
                      "--rest", ".03", "--stop-timeout", ".5", "--speed-limit", "2"]
        self.args = sweep.parse_args(self.flags)
        self.clock, self.log = Clock(), MemoryLog()
        self.bus = FakeCAN(self.clock)
        self.runner = sweep.Sweep(self.args, self.bus, self.log, clock=self.clock)
        self.runner.wait = self.bus.wait
        self.out, self.err = redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())
        self.out.__enter__()
        self.err.__enter__()

    def tearDown(self):
        self.err.__exit__(None, None, None)
        self.out.__exit__(None, None, None)

    def start(self):
        self.runner.startup()

    def position_reply(self, position):
        count = self.runner.zero_count+round(position/self.args.encoder_sign/sweep.RADIANS_PER_COUNT)
        signed = (count+2**31) % 2**32-2**31
        return b"\xa3"+struct.pack("<Hi", count % 16384, signed)

    def activate(self):
        self.start()
        self.runner.motion, self.runner.last_command = True, self.clock.now
        self.runner.start_leg(self.clock.now)

    def test_preview_does_not_touch_can_or_create_csv(self):
        with patch.object(sweep, "open_bus") as bus, patch.object(sweep, "CsvLog") as log:
            self.assertEqual(sweep.main(self.flags), 0)
        bus.assert_not_called()
        log.assert_not_called()

    def test_default_configuration_is_steering_specific(self):
        args = sweep.parse_args([])
        self.assertEqual(args.motor_id, 2)
        self.assertEqual((args.command_sign, args.encoder_sign, args.torque_sign), (-1, -1, -1))
        self.assertEqual(args.current_limit_a, 1.)
        self.assertEqual((args.current_guard_factor, args.current_guard_margin_a), (1.5, .05))
        self.assertEqual(args.speeds, [1., 2., 4., 6., 8., 10., 12.])
        self.assertEqual(args.speed_limit, 16.)
        self.assertEqual(args.query_rate, 200.)
        self.assertEqual((args.query_timeout, args.feedback_timeout), (.01, .02))
        self.assertLess(args.sweep_angle, args.angle_limit)
        self.assertLess(args.angle_limit, math.pi/2)
        self.assertFalse(hasattr(args, "acceleration"))

    def test_invalid_parameters_are_rejected_before_can(self):
        for flags in (["--speeds", "-1"], ["--speeds", "0"], ["--speeds", "nan"],
                      ["--speeds", "17"], ["--cycles", "0"], ["--sweep-angle", "1.3"],
                      ["--speed-limit", "24"],
                      ["--angle-limit", "1.6"], ["--analysis-angle", ".9"],
                      ["--query-timeout", ".2"], ["--current-guard-factor", ".9"],
                      ["--current-guard-factor", "1e308", "--current-limit-a", "2"],
                      ["--current-guard-margin-a", "-.1"], ["--current-limit-a", "1e50"],
                      ["--gear-ratio", "1e-12"],
                      ["--motor-id", "0"], ["--command-rate", "1"], ["--stop-timeout", ".1"]):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                sweep.parse_args(flags)

    def test_high_speed_defaults_preserve_feedback_and_position_margin(self):
        args = sweep.parse_args([])
        budget = max(args.query_timeout+1/args.query_rate, args.feedback_timeout)+args.guard_horizon
        self.assertLess(args.sweep_angle+args.speed_limit*budget, args.angle_limit)
        for flags in (["--feedback-timeout", ".1"], ["--query-rate", "10"],
                      ["--query-timeout", ".019"], ["--guard-horizon", ".1"]):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                sweep.parse_args(flags)

    def test_complete_high_speed_sweep_sends_all_requested_speeds(self):
        args = sweep.parse_args(["--cycles", "1", "--rest", ".03"])
        runner = sweep.Sweep(args, self.bus, self.log, clock=self.clock)
        runner.wait = self.bus.wait
        runner.startup()
        runner.run()
        self.assertTrue(runner.shutdown())
        self.assertTrue(runner.complete)
        sent = {data for _, data in self.bus.frames if data[0] == 0xC1}
        for speed in args.speeds:
            for direction in (-1, 1):
                self.assertIn(sweep.speed_payload(args, direction*speed)[0], sent)
        high_speed_rows = [row for row in self.log.rows if abs(row["plateau_speed_rad_s"]) == 12.]
        self.assertTrue(any(row["analysis_window"] for row in high_speed_rows))
        self.assertLess(max(abs(row["position_rad"]) for row in high_speed_rows), args.angle_limit)
        self.assertEqual(self.bus.mode, 0)

    def test_startup_warns_if_read_b5_cannot_reach_fast_targets_within_stroke(self):
        args = sweep.parse_args([])
        runner = sweep.Sweep(args, self.bus, self.log, clock=self.clock)
        runner.wait = self.bus.wait
        output = io.StringIO()
        with redirect_stdout(output):
            runner.startup()
        self.assertIn("WARNING: if firmware follows", output.getvalue())
        self.assertIn("do not treat these legs as steady-speed", output.getvalue())
        self.assertFalse(any(data[0] == 0xB5 and len(data) != 1 for _, data in self.bus.frames))
        self.assertTrue(runner.shutdown())

    def test_plan_has_both_directions_bounded_endpoints_and_center_return(self):
        plan = sweep.make_plan(self.args)
        self.assertEqual((plan[0].kind, plan[0].destination), ("positioning", -.12))
        self.assertEqual([(leg.speed, leg.destination) for leg in plan if leg.kind == "traverse"],
                         [(.4, .12), (-.4, -.12), (.8, .12), (-.8, -.12)])
        self.assertEqual((plan[-1].kind, plan[-1].destination), ("return", 0.))

    def test_c1_units_and_sign(self):
        data, encoded, rpm = sweep.speed_payload(self.args, 2*math.pi)
        self.assertEqual(data, b"\xc1"+struct.pack("<i", -6000))
        self.assertAlmostEqual(encoded, 2*math.pi)
        self.assertEqual(rpm, -60.)

    def test_a3_zero_unwrapped_angle_and_signed_count_rollover(self):
        self.start()
        self.runner.decode(self.position_reply(4.), self.clock.now)
        self.assertAlmostEqual(self.runner.position, 4., delta=sweep.RADIANS_PER_COUNT)
        self.runner.zero_count = 2**31-2
        count = -2**31+8
        self.runner.decode(b"\xa3"+struct.pack("<Hi", count % 16384, count), self.clock.now)
        self.assertAlmostEqual(self.runner.position, -10*sweep.RADIANS_PER_COUNT)

    def test_raw_torque_uses_a1_not_unsigned_bus_current(self):
        self.start()
        self.runner.decode(b"\xa1"+struct.pack("<i", -200), self.clock.now)
        self.runner.emit("test")
        row = self.log.rows[-1]
        self.assertAlmostEqual(row["torque_estimate_nm"], .156, places=6)
        self.assertEqual(row["bus_current_raw"], 65000)
        self.assertEqual(row["zero_count"], 12345)

    def test_full_sweep_has_no_ramp_queries_all_signals_and_restores(self):
        original = self.bus.config.copy()
        self.start()
        self.runner.run()
        self.assertTrue(self.runner.shutdown())
        self.assertTrue(self.runner.complete)
        self.assertEqual(self.bus.config, original)
        self.assertEqual(self.bus.mode, 0)
        codes = [struct.unpack("<i", data[1:])[0] for _, data in self.bus.frames if data[0] == 0xC1]
        allowed = {sweep.speed_payload(self.args, v)[0][1:] for v in (0, -.4, .4, -.8, .8)}
        self.assertTrue(all(struct.pack("<i", code) in allowed for code in codes))
        self.assertTrue(any(code > 0 for code in codes) and any(code < 0 for code in codes))
        nonzero = next(code for code in codes if code)
        self.assertEqual(nonzero, struct.unpack("<i", sweep.speed_payload(self.args, -.4)[0][1:])[0])
        rows = self.log.rows
        self.assertTrue({"A1", "A2", "A3", "B0", "B5", "AE"}.issubset({row.get("opcode_hex") for row in rows}))
        self.assertTrue(any(row["analysis_window"] for row in rows))
        self.assertLess(max(abs(row["position_rad"]) for row in rows if math.isfinite(row["position_rad"])), .6)
        self.assertLess(abs(self.runner.position), .02)
        self.assertTrue(all(set(row).issubset(sweep.FIELDS) for row in rows))
        for _, data in self.bus.frames:
            self.assertIn(data[0], sweep.LENGTHS)
            if data[0] not in (0xB3, 0xCD, 0xC1):
                self.assertEqual(len(data), 1)  # No origins, B5 or gains written.

    def test_target_crossing_forces_zero_without_waiting_for_periodic_tick(self):
        self.activate()
        self.runner.service()
        old_tick = self.runner.next_command
        self.clock.now += .001
        self.runner.decode(self.position_reply(-.121), self.clock.now)
        self.runner.service()
        self.assertLess(self.clock.now, old_tick)
        self.assertEqual(self.runner.phase, "stopping")
        self.assertEqual(self.bus.frames[-1][1], b"\xc1"+struct.pack("<i", 0))

    def test_hard_boundary_and_outward_prediction_abort(self):
        self.activate()
        with self.assertRaisesRegex(RuntimeError, "hard steering"):
            self.runner.decode(self.position_reply(.61), self.clock.now)
        self.runner.position, self.runner.velocity = .59, 1.
        with self.assertRaisesRegex(RuntimeError, "predictive"):
            self.runner.service()
        self.assertTrue(self.runner.shutdown())

    def test_position_is_required_and_stale_position_stops(self):
        self.activate()
        self.runner.times["position"] = self.clock.now-.2
        with self.assertRaisesRegex(RuntimeError, "stale position"):
            self.runner.service()
        self.assertTrue(self.runner.shutdown())

    def test_stale_speed_current_status_and_command_gap_stop(self):
        self.activate()
        for name in ("velocity", "current", "status"):
            previous = self.runner.times[name]
            self.runner.times[name] = self.clock.now-1.
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "stale"):
                self.runner.service()
            self.runner.times[name] = previous
        self.runner.last_command = self.clock.now-.11
        with self.assertRaisesRegex(RuntimeError, "scheduling gap"):
            self.runner.service()
        self.assertTrue(self.runner.shutdown())

    def test_missing_disable_reply_does_not_restore_higher_current_limit(self):
        self.start()
        self.bus.drop.add(0xCF)
        before = len(self.bus.frames)
        self.assertFalse(self.runner.shutdown())
        self.assertTrue(all(data == b"\xcf" for _, data in self.bus.frames[before:]))
        self.assertEqual(self.bus.config[0xB3], b"\xb3"+struct.pack("<I", 1000))

    def test_failed_socket_setup_closes_without_sending(self):
        bus = Mock()
        bus.setsockopt.side_effect = OSError("unavailable kernel option")
        with patch.object(sweep.socket, "socket", return_value=bus), self.assertRaises(OSError):
            sweep.open_bus(self.args)
        bus.close.assert_called_once()
        bus.send.assert_not_called()

    def test_missing_a3_during_motion_stops_and_disables(self):
        self.start()
        self.bus.drop.add(0xA3)
        with self.assertRaises(TimeoutError):
            self.runner.run()
        self.assertTrue(self.runner.shutdown())
        self.assertEqual(self.bus.mode, 0)

    def test_frozen_position_cannot_run_forever(self):
        self.start()
        self.bus.frozen_position = True
        with self.assertRaisesRegex(RuntimeError, "leg timeout"):
            self.runner.run()
        self.assertTrue(self.runner.shutdown())

    def test_reverse_direction_stops(self):
        self.start()
        self.bus.reverse_speed = True
        with self.assertRaisesRegex(RuntimeError, "direction mismatch"):
            self.runner.run()
        self.assertTrue(self.runner.shutdown())

    def test_failed_zero_stop_cannot_continue_to_next_leg(self):
        self.start()
        self.bus.ignore_zero = True
        with self.assertRaisesRegex(RuntimeError, "stop/rest|position guard|position limit"):
            self.runner.run()
        self.assertEqual(self.runner.leg_index, 0)
        self.assertTrue(self.runner.shutdown())

    def test_current_guard_is_relaxed_but_not_removed(self):
        self.start()
        self.runner.decode(b"\xa1"+struct.pack("<i", 1450), self.clock.now)
        self.assertAlmostEqual(self.runner.current_guard, 1.55)
        for current in (1551, -1551):
            with self.subTest(current=current), self.assertRaisesRegex(RuntimeError, "current"):
                self.runner.decode(b"\xa1"+struct.pack("<i", current), self.clock.now)
        self.assertTrue(self.runner.shutdown())

    def test_lower_existing_current_limit_never_raised(self):
        self.bus.config[0xB3] = b"\xb3"+struct.pack("<I", 400)
        self.start()
        self.assertAlmostEqual(self.runner.current_limit, .4)
        self.assertAlmostEqual(self.runner.current_guard, .65)
        self.assertTrue(self.runner.shutdown())

    def test_current_guard_uses_actual_quantized_b3(self):
        self.bus.b3_offset = -1
        self.start()
        self.assertAlmostEqual(self.runner.current_limit, .999)
        self.assertAlmostEqual(self.runner.current_guard, 1.5485)
        self.assertTrue(self.runner.shutdown())
        self.assertEqual(self.bus.config[0xB3], b"\xb3"+struct.pack("<I", 2999))

    def test_b3_verification_is_still_strict_and_one_sided(self):
        request = b"\xb3"+struct.pack("<I", 1000)
        for current, accepted in ((999, True), (1000, True), (1001, False), (998, False), (0, False)):
            self.assertEqual(sweep.setting_matches(request, b"\xb3"+struct.pack("<I", current)), accepted)
        self.assertFalse(sweep.setting_matches(b"\xb3\x01", b"\xb3\x01"))
        self.assertFalse(sweep.setting_matches(b"\xcd\x01\xf4\x01\x00", b"\xcd\x01\xf3\x01\x00"))

    def test_bad_setting_ack_prevents_all_c1(self):
        self.bus.ignore_writes = True
        with self.assertRaisesRegex(RuntimeError, "setting ack"):
            self.start()
        self.assertFalse(any(data[0] == 0xC1 for _, data in self.bus.frames))
        self.assertTrue(self.runner.shutdown())

    def test_readback_is_independently_verified(self):
        self.bus.readback_offset = 1
        with self.assertRaisesRegex(RuntimeError, "readback"):
            self.start()
        self.assertFalse(any(data[0] == 0xC1 for _, data in self.bus.frames))
        self.bus.readback_offset = 0
        self.assertTrue(self.runner.shutdown())

    def test_does_not_restore_until_cf_and_fresh_ae_confirm_disabled(self):
        self.start()
        self.bus.bad_cf = True
        self.bus.enqueue(self.bus.status(0xCF))  # Stale mode=0 is not sufficient.
        self.assertFalse(self.runner.shutdown())
        self.assertEqual(self.bus.config[0xB3], b"\xb3"+struct.pack("<I", 1000))

    def test_shutdown_sends_cf_before_processing_unsafe_backlog(self):
        self.start()
        self.bus.enqueue(b"\xa2"+struct.pack("<i", 10000000))
        prior = len(self.bus.frames)
        self.assertTrue(self.runner.shutdown())
        self.assertEqual(self.bus.frames[prior][1], b"\xcf")

    def test_logging_failure_cannot_prevent_disable_and_restore(self):
        self.start()
        self.log.fail = True
        with self.assertRaises(OSError):
            self.runner.run()
        self.assertFalse(self.runner.shutdown())
        self.assertEqual(self.bus.mode, 0)
        self.assertEqual(self.bus.config[0xB3], b"\xb3"+struct.pack("<I", 3000))

    def test_stop_request_can_interrupt_with_cleanup(self):
        self.start()
        self.runner.stop_requested = lambda: True
        with self.assertRaises(KeyboardInterrupt):
            self.runner.run()
        self.assertTrue(self.runner.shutdown())

    def test_fault_temperature_overspeed_and_mode_exit_abort(self):
        self.activate()
        for payload in (b"\xa2"+struct.pack("<i", -100000), self.bus.status(0xCF),
                        b"\xae"+struct.pack("<HH", 2400, 0)+bytes([60, 3, 0]),
                        b"\xae"+struct.pack("<HH", 2400, 0)+bytes([25, 3, 2])):
            with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                self.runner.decode(payload, self.clock.now)
        self.assertTrue(self.runner.shutdown())

    def test_invalid_b0_and_changed_torque_constant_abort(self):
        self.activate()
        for value in (float("nan"), -1., 0., .3):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.runner.decode(b"\xb0\x00"+struct.pack("<f", value)+b"\x00", self.clock.now)
        self.assertTrue(self.runner.shutdown())

    def test_conflicting_writer_bad_dlc_and_overflow_abort(self):
        cases = ((b"\xc1\0\0\0\0", 0x102, []), (b"\xa3\0", 2, []),
                 (b"\xa1\0\0\0\0", 2,
                  [(socket.SOL_SOCKET, sweep.SO_RXQ_OVFL, struct.pack("=I", 1))]))
        for data, cid, ancillary in cases:
            with self.subTest(data=data), self.assertRaises(RuntimeError):
                self.bus.enqueue(data, cid=cid, ancillary=ancillary)
                self.runner.receive()

    def test_socket_works_without_pi_python_can_raw_err_filter_name(self):
        bus = Mock()
        with patch.object(sweep.socket, "socket", return_value=bus):
            available = getattr(sweep.socket, "CAN_RAW_ERR_FILTER", None)
            if available is not None:
                delattr(sweep.socket, "CAN_RAW_ERR_FILTER")
            try:
                self.assertIs(sweep.open_bus(self.args), bus)
            finally:
                if available is not None:
                    setattr(sweep.socket, "CAN_RAW_ERR_FILTER", available)
        bus.setsockopt.assert_any_call(socket.SOL_CAN_RAW, 2, struct.pack("=I", 0x1FFFFFFF))
        bus.bind.assert_called_once_with(("can0",))
        bus.send.assert_not_called()

    def test_csv_creation_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"test.csv"
            log = sweep.CsvLog(path)
            log.emit({"record_type": "test"})
            log.close()
            with path.open(newline="") as stream:
                self.assertEqual(next(csv.reader(stream)), list(sweep.FIELDS))
            with self.assertRaises(FileExistsError):
                sweep.CsvLog(path)

    def test_execute_entrypoint_records_csv_with_fake_bus(self):
        def wait(readable, writable, exceptional, timeout):
            self.bus.wait(timeout)
            return readable, [], []

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"steering.csv"
            with patch.object(sweep, "open_bus", return_value=self.bus), \
                    patch.object(sweep.time, "monotonic", side_effect=self.clock.monotonic), \
                    patch.object(sweep.time, "time", side_effect=self.clock.time), \
                    patch.object(sweep.select, "select", side_effect=wait):
                result = sweep.main(self.flags+["--execute", "--output", str(output)])
            self.assertEqual(result, 0)
            self.assertTrue(self.bus.closed)
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(any(row["analysis_window"] == "True" for row in rows))
            self.assertTrue(any(row["record_type"] == "rx" and row["opcode_hex"] == "A3" for row in rows))
            self.assertEqual(rows[-1]["motor_mode"], "0")


if __name__ == "__main__":
    unittest.main()
