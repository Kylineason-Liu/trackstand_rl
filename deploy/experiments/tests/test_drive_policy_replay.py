"""Rear recorded-target replay, with virtual CAN/clock only; no real hardware."""

import contextlib
import csv
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_policy_replay as rear
from test_steering_policy_replay import ReplayCAN


class DriveCAN(ReplayCAN):
    def __init__(self, fault=None):
        super().__init__()
        self.fault, self.current_code, self.mode = fault, 0, 0
        self.cd = b"\xcd\x00" + struct.pack("<H", 5000) + b"\x00"
        self.initial_cd = self.cd
        self.cd_writes = 0
        self.b3_ma, self.b4_raw = 1, 100000  # B3 must NOT be treated as a C0 limit.
        self.constant, self.active_speed_code, self.active_current_offset = .3, 0, 0

    def status(self, op):
        active = self.enabled
        return bytes([op]) + struct.pack("<HH", 2400, 20) + bytes([
            70 if self.fault == "hot" and active else 30,
            3 if self.fault == "wrong_mode" and active else self.mode,
            1 if self.fault == "motor_fault" and active else 0])

    def send(self, frame):
        cid, length, padded = rear.mit.CAN_FRAME.unpack(frame)
        assert cid == 0x101, "Only rear ID1 may be commanded/queried"
        data, op = padded[:length], padded[0]
        self.frames.append((self.now, cid, data))
        if op == 0xC0:
            if self.fault == "send_failed":
                raise OSError("simulated C0 send failure")
            self.enabled, self.mode = True, 2
            self.current_code = struct.unpack("<i", data[1:])[0]
            reply = bytes([op]) + struct.pack("<i", self.current_code + self.active_current_offset)
            if self.pause_after_first_packet:
                self.now += self.pause_after_first_packet
                self.pause_after_first_packet = 0.
        elif op == 0xCF:
            if not (self.fault == "cf_failed" and self.enabled):
                self.enabled, self.mode, self.current_code = False, 0, 0
            reply = self.status(op)
        elif op == 0xAE:
            reply = self.status(op)
        elif op == 0xB0:
            reply = bytes([op, 0]) + struct.pack("<f", self.constant) + b"\x00"
        elif op == 0xB3:
            reply = bytes([op]) + struct.pack("<I", self.b3_ma)
        elif op == 0xB4:
            reply = bytes([op]) + struct.pack("<I", self.b4_raw)
        elif op == 0xCD:
            if len(data) == 5:
                self.cd_writes += 1
                restoring = data == self.initial_cd and self.cd_writes > 1
                if not (restoring and self.fault == "restore_failed"):
                    self.cd = data
                if self.fault == "lost_cd_ack" and self.cd_writes == 1:
                    return len(frame)
            reply = self.cd
        elif op == 0xA1:
            reply = bytes([op]) + struct.pack("<i", self.current_code + self.active_current_offset
                                             if self.enabled else 0)
        elif op == 0xA2:
            reply = bytes([op]) + struct.pack("<i", self.active_speed_code if self.enabled else 0)
        elif op == 0xA3:
            reply = bytes([op]) + struct.pack("<Hi", 1000, 1000)
        else:
            raise AssertionError(f"unexpected rear command {data.hex()}")
        if not (self.enabled and op in self.drop_aux):
            self.enqueue(1, reply, self.reply_delay)
        return len(frame)


class DriveReplayTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.output = Path(self.folder.name) / "rear.csv"

    def args(self, *flags):
        return rear.parse_args(["--pre-zero", ".04", "--duration", ".1", "--output", str(self.output), *flags])

    @contextlib.contextmanager
    def clock(self, bus):
        with patch.object(rear.time, "monotonic", side_effect=lambda: bus.now), \
                patch.object(rear.time, "time", side_effect=lambda: 1700000000 + bus.now), \
                patch.object(rear.mit.select, "select", side_effect=bus.select):
            yield

    def run_bus(self, args, bus):
        plan, digest = rear.load_plan(args)
        with self.clock(bus), patch.object(rear.shared.socket, "socket", return_value=bus) as create:
            rear.execute(args, plan, digest)
            create.assert_called_once()
        return plan

    def rows(self):
        with self.output.open() as stream:
            return list(csv.DictReader(stream))

    def test_preview_has_no_can_or_file_side_effect(self):
        with patch.object(rear.shared.socket, "socket", side_effect=AssertionError("real CAN access")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rear.main(["--output", str(self.output)]), 0)
        self.assertEqual(list(Path(self.folder.name).iterdir()), [])

    def test_scaled_and_submitted_default_plans_match_actual_simulator_prefix_and_core(self):
        bank = rear.HERE.parents[1] / "isaaclab_train/reports/20261025_matched_target_replay/inputs.npz"
        with np.load(bank) as capture:
            for tag, column in (("isaac400", 4), ("mujoco400", 6), ("mujoco800", 8)):
                for level in ("scaled", "submitted"):
                    args = rear.parse_args(["--targets", str(rear.HERE / "replay_targets" /
                                                            f"2500_{tag}_env0_episode1.csv"), "--level", level])
                    plan, _ = rear.load_plan(args)
                    self.assertEqual(len(plan), 350)
                    np.testing.assert_array_equal([r["requested_torque_nm"] for r in plan],
                                                  capture["airborne_rear_nm"][:350, column])
                    self.assertTrue(all(r["phase"] == "pre_zero" for r in plan[:100]))
                    self.assertTrue(all(r["phase"] == "core" for r in plan[100:]))
                    args.post_hold = 1.
                    full, _ = rear.load_plan(args)
                    np.testing.assert_array_equal([r["requested_torque_nm"] for r in full],
                                                  capture["airborne_rear_nm"][:, column])

    def test_c0_units_sign_and_quantization(self):
        for torque in (-.381, -.1, 0., .1, .381):
            data, values = rear.current_payload(torque, .3, -1)
            self.assertEqual(data[0], 0xC0)
            self.assertEqual(len(data), 5)
            self.assertEqual(struct.unpack("<i", data[1:])[0], round(-torque / .3 * 1000))
            self.assertLessEqual(abs(values["encoded_command_torque_nm"] - torque), .00015000001)
        for constant in (float("nan"), 0., -1.):
            with self.assertRaises(ValueError):
                rear.current_payload(.1, constant, -1)
        with self.assertRaises(OverflowError):
            rear.current_payload(1e20, .3, -1)

    def test_both_rear_columns_are_selected_without_hidden_transform(self):
        args = self.args("--start", "0", "--duration", ".02", "--pre-zero", "0")
        path = Path(self.folder.name) / "unit_case.csv"
        with args.targets.open() as stream:
            names = next(csv.reader(stream))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            writer.writerow(dict(sample_index=0, source_time_s=0, episode=1, raw_steering=0, raw_drive=.5,
                                 scaled_steering_rad=0, conditioned_steering_rad=0,
                                 scaled_drive_nm=rear.shared.f32(.5 * rear.shared.f32(6.28)), submitted_drive_nm=.2))
        args.targets = path
        scaled, _ = rear.load_plan(args)
        args.level = "submitted"
        submitted, _ = rear.load_plan(args)
        self.assertEqual(scaled[0]["requested_torque_nm"], rear.shared.f32(.5 * rear.shared.f32(6.28)))
        self.assertEqual(submitted[0]["requested_torque_nm"], rear.shared.f32(.2))

    def test_200hz_repeats_same50hz_targets_and_restores_watchdog(self):
        grouped = {}
        for rate in (50, 200):
            self.output = Path(self.folder.name) / f"rear{rate}.csv"
            args, bus = self.args("--packet-rate", str(rate)), DriveCAN()
            plan = self.run_bus(args, bus)
            meta = json.loads(self.output.with_suffix(".json").read_text())
            self.assertTrue(meta["completed"] and meta["disable_confirmed"] and meta["watchdog_restored"])
            self.assertEqual(meta["host_damping_nm_s_per_rad"], 0.)
            self.assertFalse(meta["steering_motor_commanded"])
            self.assertEqual(bus.cd, bus.initial_cd)
            targets = [r for r in self.rows() if r["record_type"] == "drive_target"]
            ratio = rate // 50
            self.assertEqual(len(targets), len(plan) * ratio)
            np.testing.assert_allclose([float(r["replay_time_s"]) for r in targets],
                                       np.arange(len(targets)) / rate, atol=1e-6)
            grouped[rate] = []
            for index, sample in enumerate(plan):
                group = targets[index * ratio:(index + 1) * ratio]
                self.assertEqual(len({r["payload_hex"] for r in group}), 1)
                self.assertTrue(all(float(r["requested_torque_nm"]) == sample["requested_torque_nm"] for r in group))
                grouped[rate].append(group[0]["payload_hex"])
            for _, cid, data in bus.frames:
                self.assertEqual(cid, 0x101)
                if len(data) > 1:
                    self.assertIn(data[0], (0xC0, 0xCD))
            actual = [r for r in self.rows() if r["can_opcode"] == str(0xC0) and r["can_direction"] == "bus_rx"]
            self.assertEqual(len(actual), len(targets))
            for target, feedback in zip(targets, actual):
                self.assertAlmostEqual(float(target["encoded_command_torque_nm"]), float(feedback["torque_nm"]))
        self.assertEqual(grouped[50], grouped[200])

    def test_no_b3_or_b4_writes_and_b3_is_not_used_as_c0_ceiling(self):
        args, bus = self.args(), DriveCAN()
        bus.b3_ma = 1  # 0.001 A position/speed limit, far below valid C0 test commands.
        self.run_bus(args, bus)
        self.assertTrue(any(len(data) == 5 and data[0] == 0xC0 for _, _, data in bus.frames))
        self.assertTrue(all(len(data) == 1 for _, _, data in bus.frames if data[0] in (0xB3, 0xB4)))
        meta = json.loads(self.output.with_suffix(".json").read_text())
        self.assertEqual(meta["B4_current_slew_raw"], 100000)

    def test_nonzero_wheel_velocity_does_not_add_host_damping(self):
        frames = []
        for speed in (0, -3000):
            self.output = Path(self.folder.name) / f"speed{speed}.csv"
            args, bus = self.args(), DriveCAN()
            bus.active_speed_code = speed
            self.run_bus(args, bus)
            frames.append([data for _, _, data in bus.frames if data[0] == 0xC0])
        self.assertEqual(*frames)

    def test_rear_position_is_multiturn_and_not_a_steering_angle_stop(self):
        args, bus = self.args(), DriveCAN()
        with self.clock(bus), rear.CsvLog(self.output) as log:
            motor = rear.mit.Motor("drive", 1, 1., -1, -1, zero_count=1000)
            controller = rear.DriveIO(bus, log, motor, args)
            controller.phase = "motion"
            count = 1000 - 3 * 16384
            row = controller.row(motor, "can_rx")
            self.assertEqual(controller.decode_reply(motor, b"\xa3" + struct.pack("<Hi", 1000, count), 0., row), "")
            self.assertAlmostEqual(row["position_unwrapped_rad"], 6 * np.pi)

    def test_missed_c0_deadline_stops_without_catchup_burst(self):
        args, bus = self.args(), DriveCAN()
        bus.pause_after_first_packet = .035
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            self.run_bus(args, bus)
        self.assertEqual(sum(data[0] == 0xC0 for _, _, data in bus.frames), 1)
        meta = json.loads(self.output.with_suffix(".json").read_text())
        self.assertTrue(meta["disable_confirmed"] and meta["watchdog_restored"])

    def test_excess_command_current_rejects_plan_without_clipping_or_motion(self):
        args, bus = self.args("--command-current-limit-a", ".001"), DriveCAN()
        with self.assertRaisesRegex(ValueError, "no C0 sent"):
            self.run_bus(args, bus)
        self.assertFalse(any(data[0] == 0xC0 for _, _, data in bus.frames))
        self.assertEqual(bus.cd, bus.initial_cd)

    def test_fault_mode_temperature_speed_or_current_stops_and_restores(self):
        for name in ("motor_fault", "wrong_mode", "hot", "speed", "current", "send_failed"):
            with self.subTest(name=name):
                self.output = Path(self.folder.name) / f"{name}.csv"
                args, bus = self.args(), DriveCAN(name)
                if name == "speed":
                    bus.active_speed_code = -10000  # +10.47 rad/s in bike coordinates.
                if name == "current":
                    bus.active_current_offset = 3000
                with self.assertRaises((RuntimeError, OSError)):
                    self.run_bus(args, bus)
                meta = json.loads(self.output.with_suffix(".json").read_text())
                self.assertFalse(meta["completed"])
                self.assertTrue(meta["disable_confirmed"] and meta["watchdog_restored"])

    def test_c0_replies_do_not_hide_missing_a1_and_a1_cannot_hide_missing_c0(self):
        for op in (0xA1, 0xC0):
            with self.subTest(op=op):
                self.output = Path(self.folder.name) / f"missing{op}.csv"
                args, bus = self.args("--duration", ".3"), DriveCAN()
                bus.drop_aux.add(op)
                with self.assertRaisesRegex(RuntimeError, "stale"):
                    self.run_bus(args, bus)
                meta = json.loads(self.output.with_suffix(".json").read_text())
                self.assertTrue(meta["disable_confirmed"] and meta["watchdog_restored"])

    def test_watchdog_is_restored_after_lost_setup_ack(self):
        args, bus = self.args(), DriveCAN("lost_cd_ack")
        with self.assertRaises(TimeoutError):
            self.run_bus(args, bus)
        self.assertFalse(any(data[0] == 0xC0 for _, _, data in bus.frames))
        self.assertEqual(bus.cd, bus.initial_cd)

    def test_no_watchdog_restore_when_disable_not_confirmed(self):
        args, bus = self.args(), DriveCAN("cf_failed")
        with self.assertRaisesRegex(RuntimeError, "NOT confirmed"):
            self.run_bus(args, bus)
        meta = json.loads(self.output.with_suffix(".json").read_text())
        self.assertFalse(meta["disable_confirmed"] or meta["watchdog_restored"])
        self.assertEqual(bus.cd[1], 1)

    def test_failed_restore_is_reported_separately_from_confirmed_disable(self):
        args, bus = self.args(), DriveCAN("restore_failed")
        with self.assertRaisesRegex(RuntimeError, "NOT confirmed"):
            self.run_bus(args, bus)
        meta = json.loads(self.output.with_suffix(".json").read_text())
        self.assertTrue(meta["disable_confirmed"])
        self.assertFalse(meta["watchdog_restored"])
        self.assertIn("mismatch", meta["CD_restore_error"])

    def test_tighter_existing_watchdog_is_preserved(self):
        args, bus = self.args(), DriveCAN()
        bus.cd = bus.initial_cd = b"\xcd\x01" + struct.pack("<H", 200) + b"\x01"
        self.run_bus(args, bus)
        self.assertEqual(bus.cd, bus.initial_cd)
        self.assertFalse(any(len(data) > 1 and data[0] == 0xCD for _, _, data in bus.frames))

    def test_old_replay_outputs_are_not_overwritten(self):
        args = self.args()
        self.output.write_text("keep")
        with self.assertRaises(FileExistsError), patch.object(rear.shared.socket, "socket") as create:
            plan, digest = rear.load_plan(args)
            rear.execute(args, plan, digest)
        create.assert_not_called()
        self.assertEqual(self.output.read_text(), "keep")

    def test_bad_cli_and_episode_window_rejected(self):
        for flags in (["--duration", ".001"], ["--duration", "0"], ["--current-abort-a", "1"],
                      ["--watchdog-ms", "1"], ["--level", "conditioned"], ["--packet-rate", "800"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    rear.parse_args(flags)
        with self.assertRaises(ValueError):
            rear.load_plan(self.args("--start", "29.98", "--duration", "1"))

    def test_logger_failure_still_disables_and_restores_watchdog(self):
        args, bus = self.args(), DriveCAN()
        original_emit = rear.CsvLog.emit
        def fail_on_motion(log, row):
            if row["record_type"] == "drive_target" and not log.stopping:
                log.failed = RuntimeError("simulated full disk")
                raise log.failed
            return original_emit(log, row)
        with patch.object(rear.CsvLog, "emit", fail_on_motion):
            with self.assertRaises(RuntimeError):
                self.run_bus(args, bus)
        meta = json.loads(self.output.with_suffix(".json").read_text())
        self.assertFalse(meta["completed"])
        self.assertTrue(meta["disable_confirmed"] and meta["watchdog_restored"])
        self.assertEqual(bus.cd, bus.initial_cd)

    def test_missing_kernel_timestamp_refuses_c0_motion(self):
        args, bus = self.args(), DriveCAN()
        bus.no_timestamps = True
        with self.assertRaisesRegex(RuntimeError, "NOT confirmed"):
            self.run_bus(args, bus)
        self.assertFalse(any(data[0] == 0xC0 for _, _, data in bus.frames))
        self.assertTrue(any(data[0] == 0xCF for _, _, data in bus.frames))


if __name__ == "__main__":
    unittest.main()
