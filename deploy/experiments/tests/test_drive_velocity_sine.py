"""Packet, unit and lifecycle checks with virtual CAN/time; no hardware access."""

from contextlib import redirect_stderr, redirect_stdout
import heapq
import io
import json
import math
from pathlib import Path
import socket
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import drive_velocity_sine as sine


class Clock:
    now = 10.

    def monotonic(self):
        return self.now

    def time(self):
        return 1_700_000_000 + self.now


class MemoryLog:
    def __init__(self):
        self.rows, self.fail = [], False

    def emit(self, row):
        self.check()
        self.rows.append(row.copy())

    def check(self):
        if self.fail:
            raise OSError("disk full")

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.check()


class FakeCAN:
    def __init__(self, clock):
        self.clock, self.rx, self.sent, self.sequence = clock, [], [], 0
        self.speed, self.mode, self.bad_cf, self.fault = 0., 0, False, 0
        self.pause_once, self.paused, self.drop = False, False, set()
        self.drop_after = -math.inf
        self.cf_transition_replies = 0
        self.config = {0xB3: b"\xb3" + struct.pack("<I", 30000),
                       0xB5: b"\xb5" + struct.pack("<I", 0),
                       0xCD: b"\xcd\x00" + struct.pack("<H", 5000) + b"\x00",
                       0xB8: b"\xb8" + struct.pack("<f", .2),
                       0xB9: b"\xb9" + struct.pack("<f", .1)}

    def enqueue(self, payload, cid=1, flags=0, delay=.0002):
        if payload[0] in self.drop and self.clock.now >= self.drop_after:
            return
        self.sequence += 1
        heapq.heappush(self.rx, (self.clock.now + delay, self.sequence,
                                sine.CAN_FRAME.pack(cid, len(payload), payload), flags))

    def status(self, opcode):
        return bytes([opcode]) + struct.pack("<HH", 2400, 20) + bytes([25, self.mode, self.fault])

    def send(self, frame):
        cid, length, padded = sine.CAN_FRAME.unpack(frame)
        data = padded[:length]
        assert cid in (0x101, 0x501), "only rear ID 1 may be commanded"
        if self.pause_once and not self.paused and len(self.sent) > 100:
            self.clock.now += .03
            self.paused = True
        self.sent.append((self.clock.now, cid, data))
        op = data[0]
        if cid == 0x501:
            p, v, t = int.from_bytes(data[:2], "big"), (data[2] << 4) | (data[3] >> 4), (
                (data[6] & 15) << 8) | data[7]
            assert ((data[3] & 15) << 8) | data[4] == 0
            self.speed, self.mode = sine.decode(v, 45., 12), 5
            self.enqueue(bytes((0xf1, p >> 8, p & 255, v >> 4, ((v & 15) << 4) | (t >> 8), t & 255, 1)))
        elif op == 0xCF:
            self.mode = 3 if self.bad_cf or self.cf_transition_replies > 0 else 0
            self.cf_transition_replies = max(0, self.cf_transition_replies - 1)
            self.enqueue(self.status(op))
        elif op in self.config:
            if len(data) > 1:
                assert op != 0xB3, "B3 must remain read-only"
                self.config[op] = data
            self.enqueue(self.config[op])
        elif op == 0xB0:
            self.enqueue(b"\xb0\x00" + struct.pack("<f", .3) + b"\x01")
        elif op == 0xF0:
            self.enqueue(b"\xf0" + struct.pack("<HHH", 955, 4500, 1800))
        elif op in (0xA2, 0xC1):
            if op == 0xC1:
                self.speed, self.mode = struct.unpack("<i", data[1:])[0] * .01 * 2 * math.pi / 60, 3
            self.enqueue(bytes([op]) + struct.pack("<i", round(self.speed * 60 / (2 * math.pi) * 100)))
        elif op == 0xA1:
            self.enqueue(b"\xa1" + struct.pack("<i", 0))
        elif op == 0xA3:
            self.enqueue(b"\xa3" + struct.pack("<Hi", 100, 123456))
        elif op == 0xAE:
            self.enqueue(self.status(op))
        else:
            raise AssertionError(f"unexpected opcode {op:02X}")
        return len(frame)

    def recvmsg(self, *unused):
        if not self.rx or self.rx[0][0] > self.clock.now:
            raise BlockingIOError
        _, _, frame, flags = heapq.heappop(self.rx)
        return frame, [], flags, None

    def select(self, readers, writers, errors, timeout):
        deadline = self.clock.now + timeout
        if self.rx:
            deadline = min(deadline, self.rx[0][0])
        self.clock.now = max(self.clock.now, deadline)
        return ([self] if self.rx and self.rx[0][0] <= self.clock.now else [], [], [])


class VelocitySineTests(unittest.TestCase):
    def make_io(self, method="both"):
        args = SimpleNamespace(method=method, amplitude=4., frequency=.5, gain_speed_unit="rad_s",
                               gain_current_unit="a", ki_timebase="second", ki_period_s=None)
        clock, log = Clock(), MemoryLog()
        bus = FakeCAN(clock)
        return sine.VelocityTest(args, bus, log, clock=clock), bus, log, clock

    def test_unknown_units_block_mit_before_any_can(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            sine.parse_args(["--method", "mit", "--execute"])
        self.assertEqual(sine.parse_args(["--method", "c1", "--execute"]).method, "c1")

    def test_preview_never_opens_can(self):
        with patch.object(sine, "open_bus", side_effect=AssertionError), redirect_stdout(io.StringIO()):
            self.assertEqual(sine.main([]), 0)

    def test_per_tick_requires_firmware_timestep(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            sine.parse_args(["--ki-timebase", "tick"])
        with self.assertRaises(ValueError):
            sine.converted_gains(1, 1, .3, "rad_s", "a", "tick")

    def test_gain_conversion_current_to_torque(self):
        kd, ki = sine.converted_gains(2, 3, .3, "rad_s", "a", "second")
        self.assertAlmostEqual(kd, .6)
        self.assertAlmostEqual(ki, .9)

    def test_gain_conversion_rpm_not_wire_counts(self):
        kd, ki = sine.converted_gains(2, 3, .3, "rpm", "a", "second")
        self.assertAlmostEqual(kd, .6 * 60 / (2 * math.pi))
        self.assertAlmostEqual(ki, .9 * 60 / (2 * math.pi))

    def test_milliamp_centirpm_per_tick_conversion(self):
        kd, ki = sine.converted_gains(.001, .0001, .3, "centirpm", "ma", "tick", .001)
        self.assertAlmostEqual(kd, .001 * .001 * .3 * 6000 / (2 * math.pi))
        self.assertAlmostEqual(ki, .0001 * .001 * .3 * 6000 / (2 * math.pi) / .001)

    def test_integral_real_dt_and_reset(self):
        integral = sine.Integral()
        self.assertEqual(integral.update(2., 10.), 0.)
        self.assertAlmostEqual(integral.update(4., 10.01), .03)
        self.assertAlmostEqual(integral.update(-4., 10.03), .03)
        self.assertEqual(sine.Integral().update(4., 20.), 0.)

    def test_c1_units_and_forward_sign(self):
        packet, actual = sine.c1_payload(2 * math.pi)
        self.assertEqual(packet, b"\xc1" + struct.pack("<i", -6000))
        self.assertAlmostEqual(actual, 2 * math.pi)

    def test_mit_packing_native_sign_and_zero_position_gain(self):
        packet, fields = sine.mit_payload(4., .1, 3., .2, (95.5, 45., 18.))
        self.assertEqual(((packet[3] & 15) << 8) | packet[4], 0)
        self.assertLess(sine.decode((packet[2] << 4) | (packet[3] >> 4), 45., 12), 0)
        self.assertLess(sine.decode(((packet[6] & 15) << 8) | packet[7], 18., 12), 0)
        self.assertAlmostEqual(fields["encoded_target_velocity_rad_s"], 4., delta=45 / 4095)
        self.assertAlmostEqual(fields["estimated_mit_total_torque_nm"], .3, delta=.02)

    def test_mit_rejects_bad_gain_or_excess_total_torque_without_clipping(self):
        for kwargs in ((4, .1, 0, 6, (95.5, 45, 18)), (4, 1, 0, 2, (95.5, 45, 18))):
            with self.assertRaises((ValueError, RuntimeError)):
                sine.mit_payload(*kwargs)

    def test_f1_big_endian_packed_12bit_fields_and_status(self):
        test, bus, log, _ = self.make_io()
        test.ranges = (95.5, 45., 18.)
        p, v, t = 40001, 1800, 1900
        bus.enqueue(bytes((0xf1, p >> 8, p & 255, v >> 4, ((v & 15) << 4) | (t >> 8), t & 255, 1)), delay=0)
        test.receive()
        row = log.rows[-1]
        self.assertAlmostEqual(row["mit_position_native_rad"], sine.decode(p, 95.5, 16))
        self.assertAlmostEqual(row["mit_velocity_rad_s"], -sine.decode(v, 45., 12))
        self.assertAlmostEqual(row["mit_torque_nm"], -sine.decode(t, 18., 12))
        self.assertEqual(test.f1_status, 1)

    def test_full_both_modes_200hz_10seconds_and_restore(self):
        test, bus, log, _ = self.make_io()
        original = dict(bus.config)
        with patch.object(sine.select, "select", bus.select), redirect_stdout(io.StringIO()):
            test.startup()
            test.run_phase("c1")
            test.run_phase("mit")
            test.cleanup()
        self.assertEqual(bus.mode, 0)
        self.assertEqual(bus.config, original)
        self.assertEqual([p["method"] for p in test.phases], ["c1", "mit"])
        for phase in test.phases:
            self.assertAlmostEqual(phase["duration_s"], 10., places=7)
            self.assertEqual(phase["commands"], 2000)
            self.assertAlmostEqual(phase["achieved_rate_hz"], 200., places=6)
        commands = [row for row in log.rows if row["record_type"] == "command"]
        self.assertEqual(len(commands), 4000)
        self.assertEqual(next(row for row in commands if row["method"] == "mit")["integral_error_rad"], 0.)
        self.assertFalse(any(data[0] in (0xB1, 0xB3, 0xF0, 0xB5, 0xB8, 0xB9) and len(data) > 1
                             for _, cid, data in bus.sent if cid != 0x501))
        self.assertTrue(all(row.get("mit_kp", 0) == 0 for row in commands))
        self.assertTrue(all(set(row) <= set(sine.FIELDS) for row in log.rows))

    def test_c1_only_does_not_need_mit_configuration(self):
        test, bus, _, _ = self.make_io("c1")
        with patch.object(sine.select, "select", bus.select), redirect_stdout(io.StringIO()):
            test.startup()
            test.run_phase("c1")
            test.cleanup()
        self.assertFalse(any(cid == 0x501 or data[0] == 0xF0 for _, cid, data in bus.sent))

    def test_delayed_host_skips_slots_without_tight_deadline_abort(self):
        test, bus, _, _ = self.make_io("c1")
        with patch.object(sine.select, "select", bus.select), redirect_stdout(io.StringIO()):
            test.startup()
            bus.pause_once = True
            test.run_phase("c1")
            test.cleanup()
        self.assertTrue(bus.paused)
        self.assertGreater(test.phases[0]["skipped_slots"], 0)
        self.assertAlmostEqual(test.phases[0]["duration_s"], 10., places=7)

    def test_lost_current_feedback_stops_and_disables(self):
        test, bus, _, _ = self.make_io("c1")
        with patch.object(sine.select, "select", bus.select), redirect_stdout(io.StringIO()):
            test.startup()
            bus.drop.add(0xA1)
            bus.drop_after = test.clock.now + .02
            with self.assertRaisesRegex(RuntimeError, "stale"):
                test.run_phase("c1")
            test.cleanup()
        self.assertEqual(bus.mode, 0)

    def test_logging_failure_cannot_prevent_disable_restore(self):
        test, bus, log, _ = self.make_io()
        original = dict(bus.config)
        with patch.object(sine.select, "select", bus.select), redirect_stdout(io.StringIO()):
            test.startup()
            log.fail = True
            test.cleanup()
        self.assertEqual(bus.mode, 0)
        self.assertEqual(bus.config, original)

    def test_unconfirmed_disable_never_restores_settings(self):
        test, bus, _, _ = self.make_io()
        with patch.object(sine.select, "select", bus.select), redirect_stdout(io.StringIO()):
            test.startup()
            bus.bad_cf = True
            previous_writes = len([x for x in bus.sent if x[2][0] == 0xCD and len(x[2]) == 5])
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                test.cleanup()
        self.assertEqual(len([x for x in bus.sent if x[2][0] == 0xCD and len(x[2]) == 5]), previous_writes)

    def test_b3_is_read_only_at_startup_motion_and_cleanup(self):
        test, bus, _, _ = self.make_io("c1")
        original = bus.config[0xB3]
        with patch.object(sine.select, "select", bus.select), redirect_stdout(io.StringIO()):
            test.startup()
            test.run_phase("c1")
            test.cleanup()
        self.assertEqual(bus.config[0xB3], original)
        self.assertNotIn("B3", test.applied_settings)
        self.assertEqual(test.settings["B3"], original.hex().upper())
        self.assertTrue(all(len(data) == 1 for _, cid, data in bus.sent if cid == 0x101 and data[0] == 0xB3))

    def test_reported_20934_milliamp_setting_no_longer_triggers_matching_error(self):
        test, bus, log, _ = self.make_io("c1")
        original = b"\xb3" + struct.pack("<I", 20934)
        bus.config[0xB3] = original
        with patch.object(sine.select, "select", bus.select), redirect_stdout(io.StringIO()):
            test.startup()
            test.run_phase("c1")
            test.cleanup()
        self.assertEqual(bus.config[0xB3], original)
        self.assertEqual(test.phases[0]["commands"], 2000)
        self.assertTrue(any(row.get("opcode") == "B3" for row in log.rows))

    def test_b3_value_is_observed_without_imposing_a_software_setting_limit(self):
        test, bus, _, _ = self.make_io("c1")
        bus.config[0xB3] = b"\xb3" + struct.pack("<I", 82498)
        with patch.object(sine.select, "select", bus.select), redirect_stdout(io.StringIO()):
            test.startup()
            test.cleanup()
        self.assertEqual(struct.unpack("<I", bus.config[0xB3][1:])[0], 82498)

    def test_first_cf_active_reply_is_a_transition_not_immediate_failure(self):
        test, bus, _, _ = self.make_io("c1")
        bus.cf_transition_replies = 2
        with patch.object(sine.select, "select", bus.select), redirect_stdout(io.StringIO()):
            test.startup()
            test.cleanup()
        self.assertEqual(bus.mode, 0)

    def test_multi_turn_reference_not_single_turn_and_rollover(self):
        test, bus, _, _ = self.make_io()
        for count in (2**31 - 2, -2**31 + 3):
            bus.enqueue(b"\xa3" + struct.pack("<Hi", 0, count), delay=0)
            test.receive()
        self.assertEqual(test.zero_count, 2**31 - 2)
        self.assertAlmostEqual(test.position_rad, -5 * 2 * math.pi / 16384)

    def test_measured_over_torque_and_speed_are_physical_guards(self):
        test, bus, _, _ = self.make_io()
        test.kt = .3
        for payload in (b"\xa1" + struct.pack("<i", 30000), b"\xa2" + struct.pack("<i", 100000)):
            bus.enqueue(payload, delay=0)
            with self.assertRaises(RuntimeError):
                test.receive()

    def test_other_writer_detected(self):
        test, bus, _, _ = self.make_io()
        bus.enqueue(b"\xc1" + struct.pack("<i", 0), cid=0x101, flags=socket.MSG_DONTROUTE, delay=0)
        with self.assertRaisesRegex(RuntimeError, "another"):
            test.receive()

    def test_csv_automatic_writer_keeps_every_row_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            with sine.CsvLog(path) as log:
                for i in range(20):
                    log.emit(dict(record_type="command", command_index=i, target_velocity_rad_s=i * .1))
            import csv
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 20)
            self.assertEqual([int(row["command_index"]) for row in rows], list(range(20)))
            with self.assertRaises(FileExistsError):
                with sine.CsvLog(path):
                    pass

    def test_failed_watchdog_restoration_is_reported_with_b3_untouched(self):
        test, bus, _, _ = self.make_io()
        original = dict(bus.config)
        with patch.object(sine.select, "select", bus.select), redirect_stdout(io.StringIO()):
            test.startup()
            bus.drop.add(0xCD)
            with self.assertRaisesRegex(RuntimeError, "restoration"):
                test.cleanup()
        self.assertEqual(bus.mode, 0)
        self.assertEqual(bus.config[0xB3], original[0xB3])

    def test_driver_fault_does_not_block_cleanup(self):
        test, bus, _, _ = self.make_io()
        with patch.object(sine.select, "select", bus.select), redirect_stdout(io.StringIO()):
            test.startup()
            bus.fault = 2
            bus.enqueue(bus.status(0xAE), delay=0)
            with self.assertRaisesRegex(RuntimeError, "driver fault"):
                test.receive()
            test.cleanup()
        self.assertEqual(bus.mode, 0)

    def test_unknown_json_measurements_are_null(self):
        self.assertEqual(sine.json_ready({"ki": math.nan, "x": [math.inf, 1.]}), {"ki": None, "x": [None, 1.]})

    def test_main_c1_writes_strict_json_after_confirmed_cleanup(self):
        clock, log = Clock(), MemoryLog()
        bus = FakeCAN(clock)
        original = sine.VelocityTest
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.csv"
            with (patch.object(sine, "open_bus", return_value=bus), patch.object(sine, "CsvLog", return_value=log),
                  patch.object(sine.select, "select", bus.select), patch.object(bus, "close", create=True),
                  patch.object(sine, "VelocityTest", lambda args, bus, log, stop: original(
                      args, bus, log, stop, clock=clock)), redirect_stdout(io.StringIO())):
                self.assertEqual(sine.main(["--method", "c1", "--execute", "--output", str(path)]), 0)
            report = json.loads(path.with_suffix(".json").read_text(), parse_constant=lambda x: self.fail(x))
            self.assertTrue(report["completed"] and report["disabled_and_restored"])
            self.assertEqual(report["phases"][0]["commands"], 2000)
            self.assertIsNone(report["mit_kd_nm_s_per_rad"])

    def test_can_open_failure_leaves_a_failed_summary_without_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.csv"
            with patch.object(sine, "open_bus", side_effect=OSError("no can0")), redirect_stdout(io.StringIO()), (
                redirect_stderr(io.StringIO())
            ):
                self.assertEqual(sine.main(["--method", "c1", "--execute", "--output", str(path)]), 1)
            report = json.loads(path.with_suffix(".json").read_text())
            self.assertFalse(report["completed"] or report["disabled_and_restored"])
            self.assertIn("no can0", report["error"])

    def test_invalid_converted_gain_stops_before_any_motion_and_still_cleans_up(self):
        clock, log = Clock(), MemoryLog()
        bus = FakeCAN(clock)
        bus.config[0xB8] = b"\xb8" + struct.pack("<f", 100.)
        original = sine.VelocityTest
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.csv"
            with (patch.object(sine, "open_bus", return_value=bus), patch.object(sine, "CsvLog", return_value=log),
                  patch.object(sine.select, "select", bus.select), patch.object(bus, "close", create=True),
                  patch.object(sine, "VelocityTest", lambda args, bus, log, stop: original(
                      args, bus, log, stop, clock=clock)),
                  redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                self.assertEqual(sine.main(["--method", "mit", "--execute", "--output", str(path),
                                            "--gain-speed-unit", "rad_s", "--gain-current-unit", "a",
                                            "--ki-timebase", "second"]), 1)
            report = json.loads(path.with_suffix(".json").read_text())
            self.assertFalse(report["completed"])
            self.assertTrue(report["disabled_and_restored"])
            self.assertFalse(any(cid == 0x501 or data[0] == 0xC1 for _, cid, data in bus.sent))


if __name__ == "__main__":
    unittest.main()
