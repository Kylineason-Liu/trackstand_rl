"""Reference timing, C1 sign, late packets and shutdown tests; no real CAN."""

from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import drive_velocity_replay as replay
import record_drive_velocity as recorder
from test_drive_velocity_sine import Clock, FakeCAN, MemoryLog


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.reference_path = self.folder / "reference.csv"
        self.rows = [(i * .02, math.sin(i * .4), i * .001) for i in range(31)]
        recorder.write_capture(self.reference_path, self.rows, .02, {"model": "test"})

    def make_io(self):
        args = replay.parse_args(["--targets", str(self.reference_path)])
        clock, log = Clock(), MemoryLog()
        bus = FakeCAN(clock)
        return replay.ReplayIO(args, bus, log, clock=clock), bus, log, clock

    def test_reference_holds_samples_instead_of_interpolating(self):
        reference = replay.Reference(self.reference_path)
        self.assertEqual(reference.at(.019999)[0], self.rows[0][1])
        self.assertEqual(reference.at(.02)[0], self.rows[1][1])
        self.assertEqual(reference.at(.039)[0], self.rows[1][1])
        self.assertEqual(reference.at(.60)[0], self.rows[30][1])

    def test_refuses_nonfinite_speed_irregular_time_and_over_limit_without_clipping(self):
        for rows in ([(0, 0), (.02, float("nan"))], [(0, 0), (.021, 1)], [(0, 0), (.02, 62)]):
            path = self.folder / "bad.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("time_s", "velocity_rad_s"))
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                replay.Reference(path)

    def test_preview_opens_neither_can_nor_log(self):
        with patch.object(replay.shared, "open_bus", side_effect=AssertionError), redirect_stdout(io.StringIO()):
            self.assertEqual(replay.main(["--targets", str(self.reference_path)]), 0)

    def test_identical_held_payloads_four_resends_and_forward_sign(self):
        test, bus, log, clock = self.make_io()
        with patch.object(replay.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            test.startup()
            test.run_segment("core", .60, replay.Reference(self.reference_path))
            test.cleanup()
        commands = [r for r in log.rows if r["record_type"] == "command"]
        self.assertEqual(len(commands), 120)
        for i in range(30):
            payloads = {r["payload_hex"] for r in commands[i * 4:(i + 1) * 4]}
            self.assertEqual(len(payloads), 1)
            self.assertAlmostEqual(commands[i * 4]["target_velocity_rad_s"], self.rows[i][1])
        self.assertLess(int.from_bytes(bytes.fromhex(commands[4]["payload_hex"])[1:], "little", signed=True), 0)
        self.assertEqual(bus.mode, 0)
        self.assertEqual(test.zero_count, 123456)
        self.assertTrue(all(len(data) == 1 for _, _, data in bus.sent if data[0] in (0xB1, 0xB3, 0xB5, 0xB8, 0xB9)))

    def test_30ms_pause_skips_slots_without_abort_burst_or_stretching(self):
        test, bus, log, _ = self.make_io()
        bus.pause_once = True
        with patch.object(replay.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            test.startup()
            test.run_segment("core", .60, replay.Reference(self.reference_path))
            test.cleanup()
        self.assertTrue(bus.paused)
        self.assertGreater(test.phases[0]["skipped_slots"], 0)
        self.assertLess(test.phases[0]["actual_duration_s"], .601)
        self.assertLess(test.phases[0]["commands"], 120)
        # No burst of old reference samples: every packet uses the current held source sample.
        for row in log.rows:
            if row["record_type"] == "command":
                source = replay.Reference(self.reference_path).at(row["phase_elapsed_s"])[0]
                self.assertAlmostEqual(row["target_velocity_rad_s"], source)

    def test_50hz_changes_packet_rate_but_not_source_targets(self):
        test, bus, log, _ = self.make_io()
        test.args.packet_rate = 50
        with patch.object(replay.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            test.startup()
            test.run_segment("core", .60, replay.Reference(self.reference_path))
            test.cleanup()
        commands = [r for r in log.rows if r["record_type"] == "command"]
        self.assertEqual(len(commands), 30)
        self.assertEqual([r["target_velocity_rad_s"] for r in commands], [r[1] for r in self.rows[:30]])

    def test_pause_inside_c1_send_does_not_cause_an_immediate_catchup(self):
        test, bus, _, _ = self.make_io()
        original_send, paused = bus.send, False

        def send(frame):
            nonlocal paused
            _, length, data = replay.shared.CAN_FRAME.unpack(frame)
            if length == 5 and data[0] == 0xC1 and not paused:
                test.clock.now += .03
                paused = True
            return original_send(frame)

        bus.send = send
        with patch.object(replay.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            test.startup()
            test.run_segment("core", .60, replay.Reference(self.reference_path))
            test.cleanup()
        times = [stamp for stamp, _, data in bus.sent if data[0] == 0xC1]
        self.assertGreater(times[1] - times[0], .0049)
        self.assertGreater(test.phases[0]["skipped_slots"], 0)

    def test_wider_feedback_tolerance_without_tight_packet_check(self):
        test, _, _, clock = self.make_io()
        test.velocity_at = test.current_at = clock.now - .8
        test.status_at = clock.now - 1.8
        test.check_feedback(clock.now)
        test.current_at = clock.now - 1.01
        with self.assertRaisesRegex(RuntimeError, "fresh velocity/current"):
            test.check_feedback(clock.now)

    def test_interrupt_and_log_failure_still_disable_and_restore(self):
        for kind in ("interrupt", "log"):
            test, bus, log, _ = self.make_io()
            with patch.object(replay.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
                test.startup()
                if kind == "interrupt":
                    test.stop_requested = lambda: True
                else:
                    log.fail = True
                with self.assertRaises((KeyboardInterrupt, OSError)):
                    try:
                        test.run_segment("core", .6, replay.Reference(self.reference_path))
                    finally:
                        test.cleanup()
            self.assertEqual(bus.mode, 0)
            self.assertEqual(bus.config[0xCD][1], 0)

    def test_main_saves_reference_origin_settings_and_cleanup_result(self):
        clock, log = Clock(), MemoryLog()
        bus = FakeCAN(clock)
        output = self.folder / "run.csv"
        io_class = replay.ReplayIO
        with patch.object(replay.shared, "open_bus", return_value=bus), \
             patch.object(replay.shared, "CsvLog", return_value=log), \
             patch.object(replay.shared.select, "select", bus.select), \
             patch.object(replay, "ReplayIO", side_effect=lambda *a: io_class(*a, clock=clock)), \
             patch.object(FakeCAN, "close", create=True), redirect_stdout(io.StringIO()):
            argv = ["--execute", "--targets", str(self.reference_path), "--output", str(output)]
            self.assertEqual(replay.main(argv), 0)
        metadata = json.loads(output.with_suffix(".json").read_text())
        self.assertTrue(metadata["completed"] and metadata["disabled_and_restored"])
        self.assertEqual(metadata["initial_multi_turn_count"], 123456)
        self.assertEqual([p["name"] for p in metadata["phases"]], ["pre_zero", "core", "post_zero"])
        self.assertEqual(metadata["reference_sha256"], hashlib.sha256(self.reference_path.read_bytes()).hexdigest())

    def test_plot_compares_fresh_feedback_and_keeps_delay(self):
        output = self.folder / "plot.csv"
        output.with_name(output.stem + "_reference.csv").write_bytes(self.reference_path.read_bytes())
        fields = ("method", "record_type", "opcode", "host_monotonic_s", "encoded_target_velocity_rad_s",
                  "velocity_rad_s", "current_torque_nm", "position_rad")
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            # Delay is preserved: the feedback on the first step is still zero.
            writer.writerow(dict(method="core", record_type="command", host_monotonic_s=10.,
                                 encoded_target_velocity_rad_s=1., velocity_rad_s=999.))
            writer.writerow(dict(method="core", record_type="rx", opcode="A2", host_monotonic_s=10.01,
                                 velocity_rad_s=0.))
            writer.writerow(dict(method="core", record_type="rx", opcode="A2", host_monotonic_s=10.03,
                                 velocity_rad_s=1.))
            writer.writerow(dict(method="core", record_type="rx", opcode="A1", host_monotonic_s=10.01,
                                 current_torque_nm=.1))
            writer.writerow(dict(method="core", record_type="rx", opcode="A3", host_monotonic_s=10.01,
                                 position_rad=0.))
        output.with_suffix(".json").write_text(json.dumps(dict(
            completed=True, reference_sha256=replay.Reference(self.reference_path).sha256,
            packet_rate_hz=200, phases=[dict(name="core", start_monotonic_s=10., planned_duration_s=.6,
                                           actual_duration_s=.6, initial_position_rad=0.)])))
        with redirect_stdout(io.StringIO()):
            replay.plot_run(output)
        metrics = json.loads(output.with_name(output.stem + "_comparison.json").read_text())
        self.assertEqual(metrics["compared_samples"], 2)
        self.assertAlmostEqual(metrics["rms_error_rad_s"], math.sqrt(.5))
        self.assertTrue(output.with_name(output.stem + "_comparison.png").is_file())


class RecorderTests(unittest.TestCase):
    def test_native_end_times_and_exact_boundaries_include_initial_rest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.csv"
            samples = [(i * .0025, i * .1, i * .001) for i in range(17)]
            result = recorder.write_capture(path, samples, .02, {})
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([float(r["time_s"]) for r in rows], [0., .02, .04])
            self.assertEqual([float(r["velocity_rad_s"]) for r in rows], [0., .8, 1.6])
            self.assertEqual(result["native_samples"], 17)

    def test_archive_export_stops_before_reset_and_preserves_true_initial_sample(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            checkpoint = folder / "model_2500.pt"
            checkpoint.write_bytes(b"test checkpoint")
            capture = folder / "motor_samples.npz"
            q = np.arange(48).reshape(3, 8, 1, 2) * .01
            v = q * 2
            np.savez(capture, position_before_rad=q, velocity_before_rad_s=v,
                     position_after_rad=q + .01, velocity_after_rad_s=v + .02,
                     done=np.array([[False], [True], [False]]))
            capture.with_name("experiment.json").write_text(json.dumps(dict(
                capture_completed=True, checkpoint_sha256=recorder.sha(checkpoint), motor_order=["steering", "rear"],
                policy_dt_s=.02, physics_dt_s=.0025, num_envs=1, seed=42, dr=True)))
            args = SimpleNamespace(from_capture=capture, checkpoint=checkpoint, env_index=0, duration=.06,
                                   output=folder / "reference.csv")
            with redirect_stdout(io.StringIO()):
                recorder.export_existing(args)
            manifest = json.loads(args.output.with_suffix(".json").read_text())
            self.assertEqual(manifest["duration_s"], .04)
            self.assertTrue(manifest["episode_terminated"])
            with args.output.open() as stream:
                first = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(first["velocity_rad_s"]), v[0, 0, 0, 1])
            self.assertEqual(float(first["position_rad"]), 0.)


if __name__ == "__main__":
    unittest.main()
