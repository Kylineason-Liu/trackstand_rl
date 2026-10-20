"""Timed steering sender/collector tests: simulated CAN/time only, no hardware."""

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
            raise OSError("simulated full disk")
        self.rows.append(row.copy())


class FakeCAN:
    """Instantaneous response is a fixture, not a claim about real motor physics."""

    def __init__(self, clock):
        self.clock, self.frames, self.rx = clock, [], []
        self.sequence, self.mode, self.speed_code, self.closed = 0, 0, 0, False
        self.count, self.updated = 12345., clock.now
        self.drop, self.delay = set(), .0002
        self.current_override = self.velocity_override = None
        self.temperature, self.fault, self.voltage = 25, 0, 2400
        self.bad_cf, self.frozen_position, self.noisy_zero = False, False, False
        self.velocity_samples = 0
        self.wait_calls = 0

    def close(self):
        self.closed = True

    def enqueue(self, data, cid=2, delay=0., flags=0, ancillary=None):
        self.sequence += 1
        heapq.heappush(self.rx, (self.clock.now+delay, self.sequence,
                                sweep.CAN_FRAME.pack(cid, len(data), data), ancillary or [], flags))

    def status(self, opcode):
        return bytes([opcode])+struct.pack("<HH", self.voltage, 65000)+bytes(
            [self.temperature, self.mode, self.fault]
        )

    def send(self, frame):
        cid, length, padded = sweep.CAN_FRAME.unpack(frame)
        assert cid == 0x102, "Only the selected steering motor may be commanded"
        if not self.frozen_position:
            self.count += self.speed_code*.01*16384/60*(self.clock.now-self.updated)
        self.updated = self.clock.now
        data = padded[:length]
        op = data[0]
        self.frames.append((self.clock.now, data))
        if op == 0xCF:
            self.mode = 3 if self.bad_cf else 0
            if not self.bad_cf:
                self.speed_code = 0
            result = self.status(op)
        elif op == 0xC1:
            self.mode, self.speed_code = 3, struct.unpack("<i", data[1:])[0]
            result = b"\xc1"+struct.pack("<i", self.speed_code)
        elif op == 0xA3:
            count = round(self.count)
            result = b"\xa3"+struct.pack("<Hi", count % 16384, (count+2**31) % 2**32-2**31)
        elif op == 0xA2:
            code = self.speed_code if self.velocity_override is None else self.velocity_override
            if self.noisy_zero and self.speed_code == 0:
                # Similar magnitude to uploaded zero-command jitter; never continuously inside +/-0.05.
                code = (-350 if self.velocity_samples % 2 else 350)
                self.velocity_samples += 1
            result = b"\xa2"+struct.pack("<i", code)
        elif op == 0xA1:
            current = self.current_override
            if current is None:
                current = 0 if self.speed_code == 0 else (100 if self.speed_code > 0 else -100)
            result = b"\xa1"+struct.pack("<i", current)
        elif op == 0xB0:
            result = b"\xb0\x00"+struct.pack("<f", .78)+b"\x00"
        elif op == 0xB5:
            result = b"\xb5"+struct.pack("<I", 0)
        elif op == 0xAE:
            result = self.status(op)
        else:
            raise AssertionError(f"Unexpected write/query {data.hex()}")
        if op not in self.drop:
            self.enqueue(result, delay=self.delay)
        return len(frame)

    def recvmsg(self, size, ancillary_size):
        if not self.rx or self.rx[0][0] > self.clock.now:
            raise BlockingIOError
        _, _, frame, ancillary, flags = heapq.heappop(self.rx)
        return frame, ancillary, flags, ("fake",)

    def wait(self, duration):
        self.wait_calls += 1
        if self.wait_calls > 10000:
            raise AssertionError("test clock made no scheduler progress")
        ready = self.rx[0][0] if self.rx else math.inf
        self.clock.now = min(self.clock.now+max(0., duration), ready)


class SteeringSweepTests(unittest.TestCase):
    def setUp(self):
        self.flags = ["--speeds", "1", "-1", "--settle", ".02", "--hold", ".04", "--rest", ".03"]
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

    def run_sweep(self):
        self.runner.startup()
        self.runner.run()
        self.assertTrue(self.runner.shutdown())

    def test_preview_has_no_can_or_csv_side_effects(self):
        with patch.object(sweep, "open_bus") as bus, patch.object(sweep, "CsvLog") as log:
            self.assertEqual(sweep.main(self.flags), 0)
        bus.assert_not_called()
        log.assert_not_called()

    def test_default_timing_matches_rear_without_ramps(self):
        args = sweep.parse_args([])
        self.assertEqual((args.settle, args.hold, args.rest), (5., 10., 2.))
        self.assertEqual((args.command_rate, args.query_rate, args.status_rate, args.b0_rate), (50., 100., 5., 1.))
        self.assertEqual(args.motor_id, 2)
        self.assertEqual(args.speeds, [1., 2., 4., 6., 8., 10., 12., -1., -2., -4., -6., -8., -10., -12.])
        self.assertEqual((args.command_sign, args.encoder_sign, args.torque_sign), (-1, -1, -1))
        for name in ("current_limit_a", "speed_limit", "angle_limit", "stopped_speed", "stop_timeout",
                     "feedback_timeout", "query_timeout", "guard_horizon", "watchdog_ms", "cycles", "acceleration"):
            self.assertFalse(hasattr(args, name), name)

    def test_every_default_signed_speed_is_sent_without_intermediate_ramp_values(self):
        args = sweep.parse_args(["--settle", ".01", "--hold", ".02", "--rest", ".01"])
        runner = sweep.Sweep(args, self.bus, self.log, clock=self.clock)
        runner.wait = self.bus.wait
        runner.startup()
        runner.run()
        self.assertTrue(runner.shutdown())
        requests = [row["command_speed_rad_s"] for row in self.log.rows if row["record_type"] == "command"]
        transitions = [v for i, v in enumerate(requests) if v != 0 and (i == 0 or requests[i-1] != v)]
        self.assertEqual(transitions, args.speeds)
        self.assertTrue(set(requests).issubset({0., *args.speeds}))

    def test_plan_is_only_timed_settle_hold_and_zero_rest(self):
        plan = sweep.make_plan(self.args)
        self.assertEqual([p.name for p in plan], ["rest", "settle", "hold", "rest", "settle", "hold", "rest"])
        self.assertEqual([p.speed for p in plan], [0., 1., 1., 0., -1., -1., 0.])
        self.assertEqual([p.plateau for p in plan if p.name == "hold"], [1., -1.])
        for left, right in zip(plan, plan[1:]):
            self.assertAlmostEqual(left.start+left.duration, right.start)
        self.assertAlmostEqual(plan[-1].start+plan[-1].duration, .21)

    def test_signed_speeds_use_exact_order_without_automatic_expansion(self):
        args = sweep.parse_args(["--speeds", "12", "-2", "0", "30"])
        self.assertEqual([p.speed for p in sweep.make_plan(args) if p.name == "hold"], [12., -2., 0., 30.])

    def test_no_software_speed_cap_or_rate_ceiling(self):
        args = sweep.parse_args(["--speeds", "1000", "-1000", "--query-rate", "1000"])
        self.assertEqual(args.speeds, [1000., -1000.])
        self.assertEqual(args.query_rate, 1000.)

    def test_only_wire_and_input_validity_are_rejected(self):
        for flags in (["--speeds", "nan"], ["--speeds", "inf"], ["--speeds", "1e50"],
                      ["--hold", "0"], ["--rest", "-1"], ["--settle", "-1"], ["--repeats", "0"],
                      ["--command-rate", "0"], ["--gear-ratio", "0"], ["--motor-id", "0"]):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                sweep.parse_args(flags)
        self.assertEqual(sweep.parse_args(["--rest", "0", "--settle", "0"]).rest, 0.)

    def test_retired_guard_flags_are_not_silently_accepted(self):
        for flag in ("--angle-limit", "--current-limit-a", "--stop-timeout", "--cycles", "--speed-limit"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                sweep.parse_args([flag, "1"])

    def test_c1_wire_units_and_quantization(self):
        data, encoded, rpm = sweep.speed_payload(self.args, 2*math.pi)
        self.assertEqual(data, b"\xc1"+struct.pack("<i", -6000))
        self.assertAlmostEqual(encoded, 2*math.pi)
        self.assertEqual(rpm, -60.)

    def test_full_sweep_sends_only_steps_and_read_queries_no_settings(self):
        self.run_sweep()
        sent = [data for _, data in self.bus.frames]
        for data in sent:
            self.assertIn(data[0], (0xC1, 0xCF, 0xA1, 0xA2, 0xA3, 0xB0, 0xB5, 0xAE))
            if data[0] != 0xC1:
                self.assertEqual(len(data), 1)
        commands = [row for row in self.log.rows if row["record_type"] == "command"]
        self.assertEqual({row["command_speed_rad_s"] for row in commands}, {0., 1., -1.})
        self.assertTrue(all(row["encoded_command_rad_s"] == sweep.speed_payload(
            self.args, row["command_speed_rad_s"])[1] for row in commands))
        self.assertTrue({"A1", "A2", "A3", "B0", "B5", "AE"}.issubset(
            {row.get("opcode_hex") for row in self.log.rows if row["record_type"] == "rx"}))
        self.assertEqual(self.bus.mode, 0)
        self.assertTrue(all(set(row).issubset(sweep.FIELDS) for row in self.log.rows))

    def test_old_zero_speed_jitter_cannot_prevent_next_timed_step(self):
        self.bus.noisy_zero = True
        self.run_sweep()
        holds = [row for row in self.log.rows if row["record_type"] == "event" and row["phase"] == "hold"]
        self.assertEqual([row["plateau_speed_rad_s"] for row in holds], [1., -1.])
        self.assertTrue(any(abs(row["velocity_rad_s"]) > .05 and row["phase"] == "rest"
                            for row in self.log.rows if math.isfinite(row["velocity_rad_s"])))

    def test_missing_measurement_replies_do_not_abort_timed_commands(self):
        self.bus.drop.update((0xA1, 0xA2, 0xA3, 0xB0, 0xB5, 0xAE))
        self.runner.startup()
        self.runner.run()
        commands = [data for _, data in self.bus.frames if data[0] == 0xC1]
        self.assertIn(sweep.speed_payload(self.args, 1.)[0], commands)
        self.assertIn(sweep.speed_payload(self.args, -1.)[0], commands)
        self.assertTrue(math.isnan(self.runner.current))
        self.assertTrue(math.isnan(self.runner.velocity))
        self.bus.drop.clear()
        self.assertTrue(self.runner.shutdown())

    def test_old_cached_velocity_is_logged_with_age_without_stopping(self):
        self.runner.decode(b"\xa2"+struct.pack("<i", -100), self.clock.now)
        self.bus.drop.add(0xA2)
        self.run_sweep()
        self.assertGreater(self.log.rows[-1]["velocity_age_s"], .2)

    def test_can_send_error_still_allows_cf_cleanup(self):
        real_send = self.bus.send

        def send(frame):
            if sweep.CAN_FRAME.unpack(frame)[2][0] == 0xC1:
                raise OSError("simulated CAN send failure")
            return real_send(frame)

        self.bus.send = send
        self.runner.startup()
        with self.assertRaises(OSError):
            self.runner.run()
        self.assertTrue(self.runner.shutdown())
        self.assertEqual(self.bus.mode, 0)

    def test_frozen_position_does_not_gate_progress(self):
        self.bus.frozen_position = True
        self.run_sweep()
        self.assertEqual(self.runner.position, 0.)

    def test_large_feedback_fault_and_temperature_are_logged_not_thresholded(self):
        self.bus.current_override, self.bus.velocity_override = 20000, -60000
        self.bus.temperature, self.bus.fault, self.bus.voltage = 100, 8, 0
        self.run_sweep()
        self.assertEqual(self.runner.current, 20.)
        self.assertAlmostEqual(self.runner.velocity, 20*math.pi)
        self.assertEqual(self.runner.status["fault_code"], 8)
        self.assertEqual(self.runner.status["motor_temperature_c"], 100)

    def test_position_is_unwrapped_and_cannot_abort_sweep(self):
        args = sweep.parse_args(["--speeds", "100", "--hold", ".08", "--settle", "0", "--rest", ".03"])
        runner = sweep.Sweep(args, self.bus, self.log, clock=self.clock)
        runner.wait = self.bus.wait
        runner.startup()
        runner.run()
        self.assertTrue(runner.shutdown())
        self.assertGreater(max(row["position_rad"] for row in self.log.rows
                               if math.isfinite(row["position_rad"])), 2*math.pi)

    def test_raw_decoding_current_torque_counts_and_status(self):
        now = self.clock.now
        self.runner.decode(b"\xb0\0"+struct.pack("<f", .78)+b"\0", now)
        self.runner.decode(b"\xa1"+struct.pack("<i", -200), now)
        self.runner.decode(b"\xa3"+struct.pack("<Hi", 0, 0), now)
        self.runner.decode(b"\xa3"+struct.pack("<Hi", 0, 16384), now)
        self.runner.decode(self.bus.status(0xAE), now)
        self.runner.emit("test")
        row = self.log.rows[-1]
        self.assertAlmostEqual(row["torque_estimate_nm"], .156, places=6)
        self.assertAlmostEqual(row["position_rad"], -2*math.pi)
        self.assertEqual(row["bus_current_raw"], 65000)

    def test_a3_signed_count_rollover_remains_continuous(self):
        self.runner.decode(b"\xa3"+struct.pack("<Hi", 0, 2**31-2), self.clock.now)
        self.runner.decode(b"\xa3"+struct.pack("<Hi", 0, -2**31+8), self.clock.now)
        self.assertAlmostEqual(self.runner.position, -10*sweep.RADIANS_PER_COUNT)

    def test_c1_reply_is_not_misrepresented_as_new_a2_sample(self):
        self.runner.decode(b"\xa2"+struct.pack("<i", -100), self.clock.now)
        sampled = self.runner.times["velocity"]
        extra = self.runner.decode(b"\xc1"+struct.pack("<i", -200), self.clock.now+1)
        self.assertEqual(self.runner.times["velocity"], sampled)
        self.assertAlmostEqual(extra["reply_velocity_rad_s"], 2*self.runner.velocity)

    def test_unknown_and_malformed_reply_retains_raw_without_fake_sample(self):
        self.bus.enqueue(b"\xa2\x01")
        self.runner.receive()
        self.assertTrue(math.isnan(self.runner.velocity))
        self.assertIn("unsupported", self.log.rows[-1]["note"])

    def test_overflow_is_reported_without_feedback_guard_abort(self):
        self.bus.enqueue(b"\xa1\0\0\0\0", ancillary=[
            (socket.SOL_SOCKET, sweep.SO_RXQ_OVFL, struct.pack("=I", 3))])
        self.runner.receive()
        self.assertIn("drops=3", self.log.rows[-1]["note"])

    def test_invalid_b0_is_nan_not_a_guessed_constant_or_motion_abort(self):
        self.runner.decode(b"\xb0\0"+struct.pack("<f", float("nan"))+b"\0", self.clock.now)
        self.assertTrue(math.isnan(self.runner.constant))

    def test_logging_failure_still_sends_disable(self):
        self.runner.startup()
        self.log.fail = True
        with self.assertRaises(OSError):
            self.runner.run()
        self.assertFalse(self.runner.shutdown())
        self.assertEqual(self.bus.mode, 0)

    def test_interrupt_still_allows_disable_cleanup(self):
        self.runner.startup()
        self.runner.stop_requested = lambda: True
        with self.assertRaises(KeyboardInterrupt):
            self.runner.run()
        self.assertTrue(self.runner.shutdown())
        self.assertEqual(self.bus.mode, 0)

    def test_missing_cf_ack_is_reported_without_any_parameter_writes(self):
        self.runner.startup()
        self.runner.run()
        self.bus.drop.add(0xCF)
        self.assertFalse(self.runner.shutdown())
        self.assertFalse(any(data[0] in (0xB3, 0xCD) for _, data in self.bus.frames))

    def test_stale_disabled_cf_does_not_hide_active_fresh_ae(self):
        self.runner.startup()
        self.runner.run()
        self.bus.enqueue(bytes([0xCF])+struct.pack("<HH", 2400, 0)+bytes([25, 0, 0]))
        self.bus.bad_cf = True
        self.assertFalse(self.runner.shutdown())

    def test_socket_setup_handles_pi_missing_constant(self):
        bus = Mock()
        available = getattr(sweep.socket, "CAN_RAW_ERR_FILTER", None)
        if available is not None:
            delattr(sweep.socket, "CAN_RAW_ERR_FILTER")
        try:
            with patch.object(sweep.socket, "socket", return_value=bus):
                self.assertIs(sweep.open_bus(self.args), bus)
        finally:
            if available is not None:
                setattr(sweep.socket, "CAN_RAW_ERR_FILTER", available)
        bus.setsockopt.assert_any_call(socket.SOL_CAN_RAW, 2, struct.pack("=I", 0x1FFFFFFF))
        bus.send.assert_not_called()

    def test_failed_socket_setup_closes_without_motion(self):
        bus = Mock()
        bus.setsockopt.side_effect = OSError("unavailable option")
        with patch.object(sweep.socket, "socket", return_value=bus), self.assertRaises(OSError):
            sweep.open_bus(self.args)
        bus.close.assert_called_once()
        bus.send.assert_not_called()

    def test_csv_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"record.csv"
            log = sweep.CsvLog(path)
            log.emit({"record_type": "test"})
            log.close()
            with path.open(newline="") as stream:
                self.assertEqual(next(csv.reader(stream)), list(sweep.FIELDS))
            with self.assertRaises(FileExistsError):
                sweep.CsvLog(path)

    def test_execute_entrypoint_records_real_csv_with_simulated_can(self):
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
            self.assertTrue(any(row["record_type"] == "rx" and row["opcode_hex"] == "A3" for row in rows))
            self.assertEqual(rows[-1]["motor_mode"], "0")


if __name__ == "__main__":
    unittest.main()
