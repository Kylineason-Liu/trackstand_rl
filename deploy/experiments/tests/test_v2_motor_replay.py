"""Offline validation only: no socket/CAN interface is ever opened."""

import copy
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "v2_motor_replay.py"
SPEC = importlib.util.spec_from_file_location("v2_motor_replay", SCRIPT)
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


class Clock:
    def __init__(self):
        self.now = 0.
    def monotonic(self):
        return self.now
    def time(self):
        return 1_800_000_000. + self.now
    def sleep(self, seconds):
        self.now += seconds


class Log:
    def __init__(self):
        self.rows = []
    def check(self):
        pass
    def emit(self, row):
        self.rows.append(row)


def args(**overrides):
    result = m.parse_args([])
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


def motors():
    result = [m.Motor("steering", 2, 23.9, 5.6), m.Motor("drive", 1, 61.6, 6.28)]
    for motor in result:
        motor.ranges = (3., 13., 18.)
        motor.kt, motor.current, motor.velocity = .3, 0., 0.
        motor.zero_rad = 0.
        motor.velocity_at = motor.current_at = motor.status_at = 0.
    return result


class Bus:
    """Virtual protocol responder, including retained MIT mode after disable."""
    def __init__(self, values, retained=False):
        self.queue, self.sent = [], []
        self.retained, self.counts = retained, {2: 8380, 1: 2**31-3}
        self.settings = {motor.device_id: {
            0xF0: b"\xf0"+struct.pack("<HHH", 30 if motor.name == "steering" else 955, 1300, 1800),
            0xCD: bytes.fromhex("CD00881300"), 0xB3: b"\xb3"+struct.pack("<I", 1000),
        } for motor in values}
    def send(self, frame):
        cid, length, data = m.FRAME.unpack(frame)
        payload, device = data[:length], cid & 255
        self.sent.append((cid, payload))
        if cid & 0x400:
            reply = self.f1(device, 1)
        else:
            op = payload[0]
            if op in (0xF0, 0xCD):
                if len(payload) > 1:
                    self.settings[device][op] = payload
                reply = self.settings[device][op]
            elif op == 0xB3:
                reply = self.settings[device][op]
            elif op == 0xB0:
                reply = b"\xb0\x00"+struct.pack("<f", .78 if device == 2 else .3)+b"\x00"
            elif op == 0xB1:
                self.counts[device] = 0
                reply = b"\xb1\x12\x34"  # Ack offset is deliberately nonzero.
            elif op == 0xA3:
                count = self.counts[device]
                reply = b"\xa3"+struct.pack("<Hi", count % 16384, count)
            elif op in (0xA1, 0xA2):
                reply = bytes([op])+struct.pack("<i", 0)
            elif op == 0xF1:
                reply = self.f1(device, 0)
            elif op in (0xAE, 0xCF):
                reply = bytes([op])+struct.pack("<HHBBB", 2000, 0, 30, 5 if self.retained else 0, 0)
            else:
                raise AssertionError(f"unexpected opcode{op:02X}")
        self.queue.append(m.FRAME.pack(device, len(reply), reply))
        return len(frame)
    def f1(self, device, status):
        pmax, vmax, tmax = (value*scale for value, scale in zip(
            struct.unpack("<HHH", self.settings[device][0xF0][1:]), (.1, .01, .01)))
        p, v, t = m.encode(0., pmax, 16), m.encode(0., vmax, 12), m.encode(0., tmax, 12)
        return bytes((0xF1, p >> 8, p & 255, v >> 4, ((v & 15) << 4) | (t >> 8), t & 255, status))
    def recvmsg(self, *args):
        if not self.queue:
            raise BlockingIOError
        return self.queue.pop(0), [], 0, None
    def close(self):
        pass


class ReferenceTests(unittest.TestCase):
    def test_real_reference_full_and_prefix(self):
        rows, manifest = m.load_reference(args().targets, 10.)
        self.assertEqual(len(rows), 500)
        self.assertEqual(rows[0]["time_s"], 0.)
        self.assertEqual(manifest["task"], "Trackstand-Balance-v2")
        self.assertEqual(len(m.load_reference(args().targets, 30.)[0]), 1500)

    def test_preview_never_opens_can_or_creates_output(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(m.socket, "socket") as opened:
            output = Path(temporary) / "result.csv"
            with patch.object(sys, "stdout", io.StringIO()):
                self.assertEqual(m.main(["--output", str(output)]), 0)
            opened.assert_not_called()
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_bad_hash_and_bounded_but_wrong_held_target_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("policy_targets_50hz.csv", "manifest.json", "reference_metadata.json"):
                (directory/name).write_bytes((args().targets.parent/name).read_bytes())
            target = directory / "policy_targets_50hz.csv"
            with target.open() as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["held_steering_position_target_rad"] = "0"  # Still bounded and within slew.
            with target.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=m.TARGET_FIELDS)
                writer.writeheader(); writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "hash"):
                m.load_reference(target, .02)
            manifest = json.loads((directory/"manifest.json").read_text())
            manifest["files"][target.name]["sha256"] = m.sha(target)
            (directory/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "steering clip"):
                m.load_reference(target, .02)

    def test_no_later_start_and_no_out_of_range_duration(self):
        with self.assertRaises(SystemExit), patch.object(sys, "stderr", io.StringIO()):
            m.parse_args(["--start", "5"])
        for duration in (0., -.02, .021):
            with self.assertRaises(ValueError):
                m.grid_ticks(duration)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            m.load_reference(args().targets, 30.02)


class EncodingAndStateTests(unittest.TestCase):
    def test_steering_absolute_zero_sign_and_gains(self):
        motor = motors()[0]
        motor.zero_rad = .0001
        data, fields = m.steering_packet(motor, -.3866762817)
        decoded = ((data[0] << 8) | data[1], ((data[3] & 15) << 8) | data[4])
        self.assertGreater(m.decode(decoded[0], motor.ranges[0], 16), motor.zero_rad)
        self.assertAlmostEqual(fields["encoded_position_rad"], -.3866762817,
                               delta=2*motor.ranges[0]/65535)
        self.assertEqual(decoded[1], round(10*4095/500))
        self.assertAlmostEqual(fields["encoded_kd"], round(.2*4095/5)*5/4095)

    def test_rear_internal_velocity_p_and_external_i_sign(self):
        motor, integral = motors()[1], m.Integral()
        motor.velocity = -.2
        integral.torque_nm = .12
        data, fields = m.drive_packet(motor, .8, integral)
        self.assertEqual(((data[3] & 15) << 8) | data[4], 0)
        self.assertEqual((data[5] << 4) | (data[6] >> 4), 246)
        vcode = (data[2] << 4) | (data[3] >> 4)
        tcode = ((data[6] & 15) << 8) | data[7]
        native_v, native_ff = m.decode(vcode, motor.ranges[1], 12), m.decode(tcode, motor.ranges[2], 12)
        native_torque = fields["encoded_kd"]*(native_v-motor.sign*motor.velocity)+native_ff
        self.assertAlmostEqual(motor.sign*native_torque, fields["estimated_total_torque_nm"])
        self.assertAlmostEqual(fields["effective_velocity_target_rad_s"], .8, delta=motor.ranges[1]/4095)
        self.assertLess(native_ff, 0.)

    def test_integral_prime_retention_antiwindup_and_error_correction(self):
        integral = m.Integral()
        integral.update(1., 10., .3, 6.28)
        self.assertEqual(integral.torque_nm, 0.)
        integral.update(1., 10.005, .3, 6.28)
        self.assertAlmostEqual(integral.torque_nm, .0165)
        integral.update(-1., 10.010, .3, 6.28)
        self.assertAlmostEqual(integral.torque_nm, .0165)
        integral.update(100., 10.015, .3, 6.28)
        self.assertAlmostEqual(integral.torque_nm, .0165)
        self.assertEqual(integral.previous_error, 100.)
        self.assertAlmostEqual(integral.error_rad, integral.torque_nm/m.KI)

    def test_signed_units_and_multiturn_rollover(self):
        values, clock, log = motors(), Clock(), Log()
        bus = Bus(values)
        owner = m.ReplayIO(args(), bus, log, values, clock=clock)
        drive = values[1]
        for count in (2**31-1, -2**31+2):
            bus.queue.append(m.FRAME.pack(1, 7, b"\xa3"+struct.pack("<Hi", count % 16384, count)))
            owner.receive()
        self.assertAlmostEqual(drive.position, -3*m.RAD_PER_COUNT)
        for op, value in ((0xA2, -6000), (0xA1, -1000)):
            bus.queue.append(m.FRAME.pack(1, 5, bytes([op])+struct.pack("<i", value)))
            owner.receive()
        self.assertAlmostEqual(drive.velocity, 2*math.pi)
        self.assertAlmostEqual(log.rows[-1]["current_torque_nm"], .3)

    def test_mit_send_invalidates_disabled_state(self):
        values, clock, log = motors(), Clock(), Log()
        bus = Bus(values)
        owner = m.ReplayIO(args(), bus, log, values, clock=clock)
        values[0].disabled_confirmed = True
        owner.send(values[0], m.steering_packet(values[0], 0.)[0], True)
        self.assertFalse(values[0].disabled_confirmed)


class SetupCleanupTests(unittest.TestCase):
    def test_setup_B1_once_software_rear_zero_and_journal_before_writes(self):
        values, clock, log = motors(), Clock(), Log()
        bus = Bus(values)
        owner = m.ReplayIO(args(), bus, log, values, clock=clock)
        snapshots = []
        def persist():
            snapshots.append((len(bus.sent), [copy.deepcopy(motor.restore) for motor in values]))
        rows = m.load_reference(args().targets, .04)[0]
        m.setup(owner, rows, persist)
        b1 = [(cid, data) for cid, data in bus.sent if data[0] == 0xB1]
        self.assertEqual(b1, [(0x102, b"\xb1")])
        self.assertEqual(values[0].zero_count, 0)
        self.assertEqual(values[1].zero_count, 2**31-3)
        self.assertTrue(values[0].origin["verified"])
        self.assertTrue(values[1].origin["verified"])
        for i, (cid, data) in enumerate(bus.sent):
            if data[0] in (0xF0, 0xCD) and len(data) > 1:
                index = 0 if (cid & 255) == 2 else 1
                self.assertTrue(any(count <= i and saved[index] for count, saved in snapshots))
        self.assertEqual(struct.unpack("<H", bus.settings[1][0xCD][2:4])[0], 500)
        originals = [dict(motor.settings) for motor in values]
        start = len(bus.sent)
        m.cleanup(owner)
        self.assertEqual([(cid, data) for cid, data in bus.sent[start:start+2]],
                         [(0x102, b"\xcf"), (0x101, b"\xcf")])
        for motor, original in zip(values, originals):
            self.assertTrue(motor.disabled_confirmed)
            self.assertEqual(motor.restore, [])
            for code in (0xF0, 0xCD):
                self.assertEqual(bus.settings[motor.device_id][code].hex().upper(), original[f"{code:02X}"])

    def test_retained_CF_mode_requires_inactive_F1(self):
        values, clock, log = motors(), Clock(), Log()
        bus = Bus(values, retained=True)
        owner = m.ReplayIO(args(), bus, log, values, clock=clock)
        owner.disable(values[0])
        self.assertTrue(values[0].disabled_confirmed)
        self.assertIn((0x102, b"\xf1"), bus.sent)

    def test_measured_overspeed_fault_and_unexpected_B1_are_fatal(self):
        for reply in (b"\xa2"+struct.pack("<i", -100_000_000), b"\xae"+struct.pack("<HHBBB", 2000, 0, 30, 5, 32),
                      b"\xb1\x00\x00"):
            values, clock, log = motors(), Clock(), Log()
            bus = Bus(values)
            owner = m.ReplayIO(args(), bus, log, values, clock=clock)
            owner.phase = "motion"
            bus.queue.append(m.FRAME.pack(1, len(reply), reply))
            with self.assertRaises(RuntimeError):
                owner.receive()

    def test_primary_failure_retained_even_if_cleanup_fails(self):
        rows, manifest = m.load_reference(args().targets, .02)
        with tempfile.TemporaryDirectory() as temporary:
            a = args(output=Path(temporary)/"test.csv", execute=True)
            with patch.object(m, "open_bus", return_value=Bus(motors())), \
                 patch.object(m, "setup", side_effect=RuntimeError("primary")), \
                 patch.object(m, "cleanup", side_effect=RuntimeError("secondary")), \
                 patch.object(sys, "stdout", io.StringIO()), patch.object(sys, "stderr", io.StringIO()):
                self.assertEqual(m.execute(a, rows, manifest), 1)
            metadata = json.loads(a.output.with_suffix(".json").read_text())
            self.assertIn("primary", metadata["execution_error"])
            self.assertIn("secondary", metadata["cleanup_error"])
            self.assertFalse(metadata["disabled_and_restored"])


class SchedulingTests(unittest.TestCase):
    def test_large_epoch_exact_boundaries_do_not_repeat_previous_target(self):
        for start in (100.0066, 1e6+.0066, 1e7+.0066):
            for index in range(1500):
                self.assertEqual(m.slot_index(start, .02, start+index*.02), index)
                if index:
                    self.assertEqual(m.slot_index(start, .02, start+index*.02-1e-6), index-1)
            self.assertEqual(m.advance_slot(start, .005, 0, start+.005), (2, 1))

    def run_motion(self, rate, late=False):
        clock, values, log = Clock(), motors(), Log()
        owner = m.ReplayIO(args(packet_rate=rate), None, log, values, clock=clock)
        owner.send = lambda motor, payload, mit=False, **fields: owner.emit(motor, "command" if mit else "tx", **fields)
        def refresh():
            for motor in values:
                motor.velocity_at = motor.current_at = motor.status_at = motor.f1_at = clock.now
        owner.drain = refresh
        rows = m.load_reference(args().targets, .20)[0]
        jumped = False
        def wait(_r, _w, _e, timeout):
            nonlocal jumped
            clock.now += max(timeout, 1e-8)
            if late and not jumped and clock.now >= .04:
                clock.now += .037
                jumped = True
            return [], [], []
        with patch.object(m.select, "select", side_effect=wait):
            m.motion(owner, rows)
        return owner, log.rows, clock

    def test_integral_200hz_independent_of_50_or_400hz_packets(self):
        for rate in (50, 200, 400):
            owner, rows, clock = self.run_motion(rate)
            updates = [row for row in rows if row["record_type"] == "integral_update"]
            commands = [row for row in rows if row["record_type"] == "command" and row["motor"] == "drive"]
            self.assertEqual(len(updates), 40)
            self.assertEqual(len(commands), round(rate*.2))
            self.assertLess(abs(clock.now-.2), 1e-6)
            self.assertEqual(owner.skipped_packets, 0)
            self.assertEqual(owner.skipped_integrals, 0)

    def test_delays_skip_anchored_slots_without_stretch_or_burst(self):
        owner, rows, clock = self.run_motion(200, late=True)
        commands = [row for row in rows if row["record_type"] == "command" and row["motor"] == "drive"]
        self.assertGreater(owner.skipped_packets, 0)
        self.assertGreater(owner.skipped_integrals, 0)
        self.assertGreater(owner.skipped_targets, 0)
        self.assertLess(abs(clock.now-.2), 1e-6)
        for row in commands:
            expected = min(9, math.floor(row["replay_elapsed_s"]/.02))
            self.assertEqual(row["target_index"], expected)
        self.assertEqual(len({row["host_monotonic_s"] for row in commands}), len(commands))


if __name__ == "__main__":
    unittest.main()
