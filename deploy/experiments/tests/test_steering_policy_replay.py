"""Virtual CAN/clock replay checks; no test opens a real CAN interface."""

import contextlib
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import steering_policy_replay as replay
from test_steering_sine_mit_io import FakeCAN


class ReplayCAN(FakeCAN):
    def __init__(self, fault=None):
        super().__init__(fault)
        self.zero_native = replay.mit.decode_symmetric(self.positions[2], 12., 16)
        self.no_timestamps = False
        self.active_current_ma = 1500
        self.pause_after_first_packet = 0.

    def send(self, frame):
        cid, size, padded = replay.mit.CAN_FRAME.unpack(frame)
        data, mid = padded[:size], cid & 255
        response = None
        if data == b"\xa2":
            response = b"\xa2" + struct.pack("<i", 0)
        elif data == b"\xa1":
            response = b"\xa1" + struct.pack("<i", self.active_current_ma if self.enabled else 1500)
        elif data == b"\xa3":
            native = replay.mit.decode_symmetric(self.positions[mid], self.configs[mid][0] * .1, 16)
            count = round((native - self.zero_native) / replay.mit.RADIANS_PER_COUNT) if self.enabled else 0
            response = b"\xa3" + struct.pack("<Hi", count % 16384, count)
        elif data == b"\xae":
            response = b"\xae" + struct.pack("<HH", 4800, 125) + bytes([35, 4 if self.enabled else 0, 0])
        elif data == b"\xb3":
            response = b"\xb3" + struct.pack("<I", 10000)
        elif data == b"\xcd":
            response = b"\xcd\x00\xf4\x01\x00"
        if response is not None:
            self.frames.append((self.now, cid, data))
            if not self.enabled or data[0] not in self.drop_aux:
                self.enqueue(mid, response, self.reply_delay)
            return len(frame)
        result = super().send(frame)
        if cid & 0x400 and self.pause_after_first_packet:
            self.now += self.pause_after_first_packet
            self.pause_after_first_packet = 0.
        return result

    def recvmsg(self, frame_size, ancillary_size):
        arrival = self.replies[0][0] if self.replies else self.now
        frame, ancillary, flags, addr = super().recvmsg(frame_size, ancillary_size)
        if not self.no_timestamps:
            whole = math.floor(arrival)
            ancillary = ancillary + [(replay.socket.SOL_SOCKET, replay.KERNEL_TIMESTAMP,
                                      replay.TIMESPEC.pack(1700000000 + whole, round((arrival - whole) * 1e9)))]
        return frame, ancillary, flags, addr


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.output = Path(self.folder.name) / "run.csv"

    def args(self, *extra):
        return replay.parse_args(["--duration", ".08", "--ramp", ".04", "--pre-hold", ".02",
                                  "--post-hold", ".02", "--output", str(self.output), *extra])

    @contextlib.contextmanager
    def clock(self, bus):
        with patch.object(replay.time, "monotonic", side_effect=lambda: bus.now), \
                patch.object(replay.time, "time", side_effect=lambda: 1700000000 + bus.now), \
                patch.object(replay.mit.select, "select", side_effect=bus.select):
            yield

    def run_bus(self, args, bus):
        plan, digest = replay.load_plan(args)
        with self.clock(bus), patch.object(replay.socket, "socket", return_value=bus) as create:
            replay.execute(args, plan, digest)
            create.assert_called_once()
        return plan

    def rows(self):
        with self.output.open() as stream:
            return list(csv.DictReader(stream))

    def test_default_preview_opens_no_socket_and_writes_no_files(self):
        with patch.object(replay.socket, "socket", side_effect=AssertionError("hardware access")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(replay.main(["--output", str(self.output)]), 0)
        self.assertEqual(list(Path(self.folder.name).iterdir()), [])

    def test_exported_files_exactly_match_both_capture_levels(self):
        folder = replay.HERE / "replay_targets"
        manifest = json.loads((folder / "manifest.json").read_text())
        root = replay.HERE.parents[1]
        for name, item in manifest["files"].items():
            path = folder / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            source = root / next(p for p in item["source_files"] if p.endswith("motor_samples.npz")
                                 or p.endswith("_evaluation.npz"))
            steps = source if source.name == "motor_samples.npz" else source.parent / "physics_samples.npz"
            with np.load(source) as native, np.load(steps) as physics:
                actions = native["raw_action"][:len(rows), 0]
                np.testing.assert_array_equal([float(row["raw_steering"]) for row in rows], actions[:, 0])
                np.testing.assert_array_equal([float(row["raw_drive"]) for row in rows], actions[:, 1])
                np.testing.assert_array_equal([float(row["scaled_steering_rad"]) for row in rows],
                                              actions[:, 0] * np.float32(1.57))
                np.testing.assert_array_equal([float(row["conditioned_steering_rad"]) for row in rows],
                                              physics["held_steering_target_rad"][:len(rows), 0, 0])
                np.testing.assert_array_equal([float(row["submitted_drive_nm"]) for row in rows],
                                              physics["submitted_motor_torque_nm"][:len(rows), 0, 0, 1])
                self.assertEqual(len(rows), 1500)

    def test_default_plan_matches_existing_simulator_replay_exactly(self):
        bank = replay.HERE.parents[1] / "isaaclab_train/reports/20261025_matched_target_replay/inputs.npz"
        with np.load(bank) as capture:
            for source, column in (("isaac400", 4), ("mujoco400", 6), ("mujoco800", 8)):
                for level in ("scaled", "conditioned"):
                    args = replay.parse_args(["--targets", str(replay.HERE / "replay_targets" /
                                                               f"2500_{source}_env0_episode1.csv"), "--level", level])
                    plan, _ = replay.load_plan(args)
                    np.testing.assert_array_equal(np.array([r["target_rad"] for r in plan], dtype=np.float32),
                                                  capture["airborne_target_rad"][:, column])

    def test_v1_conditioning_reproduces_recorded_startup_once_per_20ms(self):
        args = self.args("--start", "0", "--duration", "1", "--ramp", "0", "--pre-hold", "0",
                         "--post-hold", "0", "--v1-conditioning")
        plan, _ = replay.load_plan(args)
        values = np.array([row["target_rad"] for row in plan])
        with args.targets.open() as stream:
            original = list(csv.DictReader(stream))[:len(plan)]
        np.testing.assert_array_equal(values, [float(r["conditioned_steering_rad"]) for r in original])
        self.assertLessEqual(np.max(abs(np.diff(np.r_[0, values]))), replay.f32(.478) + 1e-7)

    def test_packet_rates_have_identical_50hz_held_targets_and_valid_units(self):
        results = {}
        for rate in (50, 200):
            self.output = Path(self.folder.name) / f"run{rate}.csv"
            args, bus = self.args("--packet-rate", str(rate)), ReplayCAN()
            plan = self.run_bus(args, bus)
            metadata = json.loads(self.output.with_suffix(".json").read_text())
            self.assertTrue(metadata["completed"] and metadata["disable_confirmed"])
            self.assertFalse(metadata["rear_motor_commanded"])
            targets = [row for row in self.rows() if row["record_type"] == "mit_target"]
            self.assertEqual(len(targets), len(plan) * (rate // 50))
            payloads = [r["mit_payload_hex"] for r in targets]
            grouped = [payloads[i:i + rate // 50] for i in range(0, len(payloads), rate // 50)]
            self.assertTrue(all(len(set(group)) == 1 for group in grouped))
            results[rate] = [group[0] for group in grouped]
            np.testing.assert_allclose([float(row["replay_time_s"]) for row in targets],
                                       np.arange(len(targets)) / rate, atol=1e-6)
            np.testing.assert_allclose([float(r["requested_position_rad"]) for r in targets],
                                       np.repeat([r["target_rad"] for r in plan], rate // 50), atol=0)
            self.assertTrue(all(int(r["mit_kp_code"]) == 82 and int(r["mit_kd_code"]) == 164 for r in targets))
            # Nothing is sent to rear ID 1; no origin/range/current/watchdog changes.
            for _, cid, data in bus.frames:
                self.assertIn(cid, (0x102, 0x502))
                if cid == 0x102:
                    self.assertEqual(len(data), 1)
                    self.assertNotIn(data[0], (0xB1, 0xC0, 0xC1, 0xC2, 0xC3))
            currents = [r for r in self.rows() if r["can_opcode"] == str(0xA1)]
            self.assertTrue(currents)
            self.assertAlmostEqual(float(currents[-1]["torque_nm"]), -1.5 * .78, places=6)
        self.assertEqual(results[50], results[200])

    def test_rejects_existing_output_before_can(self):
        self.output.write_text("preserve")
        args = self.args()
        plan, digest = replay.load_plan(args)
        with patch.object(replay.socket, "socket", side_effect=AssertionError("hardware access")):
            with self.assertRaises(FileExistsError):
                replay.execute(args, plan, digest)
        self.assertEqual(self.output.read_text(), "preserve")

    def test_slew_is_not_advanced_four_times_when_resending_at_200hz(self):
        args = self.args("--ramp", "0", "--pre-hold", "0", "--packet-rate", "200",
                         "--v1-conditioning")
        plan = self.run_bus(args, ReplayCAN())
        rows = [r for r in self.rows() if r["record_type"] == "mit_target"]
        expected = np.repeat([r["target_rad"] for r in plan], 4)
        np.testing.assert_array_equal([float(r["requested_position_rad"]) for r in rows], expected)
        self.assertEqual(plan[0]["target_rad"], replay.f32(.478))

    def test_missed_deadline_aborts_without_burst_or_skipped_source_targets(self):
        args, bus = self.args(), ReplayCAN()
        bus.pause_after_first_packet = .035
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            self.run_bus(args, bus)
        self.assertEqual(sum(cid == 0x502 for _, cid, _ in bus.frames), 1)
        self.assertFalse(bus.enabled)

    def test_current_abort_and_send_failure_still_disable(self):
        for name in ("current", "send_failed"):
            with self.subTest(name=name):
                self.output = Path(self.folder.name) / f"{name}.csv"
                args, bus = self.args(), ReplayCAN("send_failed" if name == "send_failed" else None)
                if name == "current":
                    bus.active_current_ma = 3500
                with self.assertRaises((RuntimeError, OSError)):
                    self.run_bus(args, bus)
                metadata = json.loads(self.output.with_suffix(".json").read_text())
                self.assertFalse(metadata["completed"])
                self.assertTrue(metadata["disable_confirmed"])

    def test_malformed_target_file_is_rejected_before_can(self):
        args = self.args()
        content = args.targets.read_text()
        path = Path(self.folder.name) / "bad.csv"
        path.write_text(content.replace("0.0,1,", "nan,1,", 1))
        args.targets = path
        with self.assertRaises(ValueError):
            replay.load_plan(args)

    def test_fault_or_stale_feedback_stops_and_confirms_disable(self):
        for kind in ("runtime_fault", "runtime_inactive", "no_f1", "no_a1"):
            with self.subTest(kind=kind):
                self.output = Path(self.folder.name) / f"{kind}.csv"
                args = self.args("--duration", ".4")
                bus = ReplayCAN(kind if kind.startswith("runtime") else None)
                bus.drop_mit = kind == "no_f1"
                if kind == "no_a1":
                    bus.drop_aux.add(0xA1)
                with self.assertRaises(RuntimeError):
                    self.run_bus(args, bus)
                metadata = json.loads(self.output.with_suffix(".json").read_text())
                self.assertFalse(metadata["completed"])
                self.assertTrue(metadata["disable_confirmed"])
                self.assertFalse(bus.enabled)

    def test_missing_kernel_timestamp_refuses_motion_and_reports_unconfirmed_disable(self):
        args, bus = self.args(), ReplayCAN()
        bus.no_timestamps = True
        with self.assertRaisesRegex(RuntimeError, "disable NOT confirmed"):
            self.run_bus(args, bus)
        self.assertFalse(any(cid == 0x502 for _, cid, _ in bus.frames))
        self.assertTrue(any(data == b"\xcf" for _, _, data in bus.frames))

    def test_bad_range_refuses_motion_without_reconfiguring_motor(self):
        args, bus = self.args(), ReplayCAN()
        bus.configs[2] = (1, 800, 300)
        with self.assertRaises(ValueError):
            self.run_bus(args, bus)
        self.assertFalse(any(cid == 0x502 for _, cid, _ in bus.frames))
        self.assertTrue(all(len(data) == 1 for _, _, data in bus.frames))

    def test_delayed_receive_preserves_kernel_queue_age(self):
        args, bus = self.args(), ReplayCAN()
        plan, _ = replay.load_plan(args)
        with self.clock(bus), replay.CsvLog(self.output) as log:
            log.stopping = False
            motor = replay.mit.Motor("steering", 2, 1., -1, -1)
            controller = replay.ReplayIO(bus, log, motor, args)
            bus.enqueue(2, b"\xb3" + struct.pack("<I", 10000), delay=0)
            bus.now = .05
            controller.receive()
        row = self.rows()[0]
        self.assertAlmostEqual(float(row["receive_queue_delay_s"]), .05, places=6)
        self.assertAlmostEqual(float(row["elapsed_s"]), 0., places=6)

    def test_old_kernel_arrival_cannot_complete_a_new_query(self):
        args, bus = self.args(), ReplayCAN()
        with self.clock(bus), replay.CsvLog(self.output) as log:
            motor = replay.mit.Motor("steering", 2, 1., -1, -1)
            controller = replay.ReplayIO(bus, log, motor, args)
            controller.pending = replay.mit.PendingRead(motor, 0xB3, .04, .06)
            bus.enqueue(2, b"\xb3" + struct.pack("<I", 10000), delay=0)
            bus.now = .05
            controller.drain()
            self.assertIsNotNone(controller.pending)

    def test_transient_overcurrent_cannot_be_hidden_by_later_normal_sample(self):
        args, bus = self.args(), ReplayCAN()
        with self.clock(bus), replay.CsvLog(self.output) as log:
            motor = replay.mit.Motor("steering", 2, 1., -1, -1)
            controller = replay.ReplayIO(bus, log, motor, args)
            controller.phase = "motion"
            bus.enqueue(2, b"\xa1" + struct.pack("<i", 3500))
            bus.enqueue(2, b"\xa1" + struct.pack("<i", 1000))
            with self.assertRaisesRegex(RuntimeError, "current"):
                controller.drain()

    def test_large_measured_angles_are_logged_without_aborting(self):
        args, bus = self.args(), ReplayCAN()
        with self.clock(bus), replay.CsvLog(self.output) as log:
            motor = replay.mit.Motor("steering", 2, 1., -1, -1, zero_count=0)
            controller = replay.ReplayIO(bus, log, motor, args)
            controller.phase = "motion"
            controller.last_f1_at = bus.now
            bus.enqueue(2, b"\xa1" + struct.pack("<i", 1500))
            bus.enqueue(2, b"\xa2" + struct.pack("<i", 0))
            for count in (6000, -6000):
                bus.enqueue(2, b"\xa3" + struct.pack("<Hi", count % 16384, count))
                controller.drain()
                controller.guard()
                self.assertGreater(abs(controller.position), 1.7)
                self.assertAlmostEqual(controller.position, -count * replay.mit.RADIANS_PER_COUNT)
        angles = [float(row["position_unwrapped_rad"]) for row in self.rows()
                  if row["can_opcode"] == str(0xA3)]
        np.testing.assert_allclose(angles, np.array([-6000, 6000]) * replay.mit.RADIANS_PER_COUNT)

    def test_both_input_columns_are_selected_without_implicit_slew(self):
        args = self.args("--start", "0", "--duration", ".02", "--ramp", "0", "--pre-hold", "0",
                         "--post-hold", "0")
        path = Path(self.folder.name) / "unit_case.csv"
        with args.targets.open() as stream:
            names = next(csv.reader(stream))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            writer.writerow(dict(sample_index=0, source_time_s=0, episode=1, raw_steering=.5, raw_drive=0,
                                 scaled_steering_rad=replay.f32(.5 * replay.f32(1.57)),
                                 conditioned_steering_rad=.2, scaled_drive_nm=0, submitted_drive_nm=0))
        args.targets = path
        scaled, _ = replay.load_plan(args)
        args.level = "conditioned"
        conditioned, _ = replay.load_plan(args)
        self.assertEqual(scaled[0]["target_rad"], replay.f32(.5 * replay.f32(1.57)))
        self.assertEqual(conditioned[0]["target_rad"], replay.f32(.2))

    def test_logger_failure_cannot_prevent_disable_attempt(self):
        args, bus = self.args(), ReplayCAN()
        original_enter = replay.CsvLog.__enter__
        def fail_writer(log):
            log.error = OSError("test disk failure")
            log.file.close()
        def enter(log):
            result = original_enter(log)
            log.thread.join()
            return result
        with patch.object(replay.CsvLog, "_write", fail_writer), patch.object(replay.CsvLog, "__enter__", enter):
            with self.assertRaises(RuntimeError):
                self.run_bus(args, bus)
        metadata = json.loads(self.output.with_suffix(".json").read_text())
        self.assertFalse(metadata["completed"])
        self.assertTrue(metadata["disable_confirmed"])
        self.assertTrue(any(data == b"\xcf" for _, _, data in bus.frames))

    def test_late_encoding_work_cannot_send_an_already_expired_packet(self):
        args, bus = self.args(), ReplayCAN()
        with self.clock(bus), replay.CsvLog(self.output) as log:
            motor = replay.mit.Motor("steering", 2, 1., -1, -1)
            controller = replay.ReplayIO(bus, log, motor, args)
            bus.now = .01
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                controller.send(motor, bytes(8), 0x502, command_deadline_s=0.)
            self.assertFalse(bus.frames)

    def test_bad_intervals_and_double_conditioning_rejected(self):
        for args in (["--start", ".001"], ["--level", "conditioned", "--v1-conditioning"],
                     ["--packet-rate", "800"], ["--max-lateness-ms", "6", "--packet-rate", "200"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    replay.parse_args(args)
        args = self.args("--start", "29.98", "--duration", "1")
        with self.assertRaises(ValueError):
            replay.load_plan(args)


if __name__ == "__main__":
    unittest.main()
