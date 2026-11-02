"""Virtual CAN lifecycle and synthetic PI identification; never run real hardware."""

from contextlib import redirect_stdout
from dataclasses import asdict
import csv
import io
import json
import math
from pathlib import Path
import sys
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import drive_c1_kp_identify as experiment
import analyze_drive_c1_kp as analysis
from test_drive_velocity_sine import Clock, FakeCAN, MemoryLog


def synthetic_phase(frequency, repeat, amplitude=1., base=0.):
    """Known velocity PI + inertial/viscous plant, not measured motor behavior."""
    step, duration = .0005, 1.+max(4., 4/frequency)
    t = np.arange(0., duration+step/2, step)
    ct = np.arange(0., duration, .005)
    wanted = base+amplitude*np.sin(2*math.pi*frequency*ct)
    # Include exact C1 speed quantization, not an ideal continuous sine input.
    target = -np.round(-wanted*6000/(2*math.pi))*.01*2*math.pi/60
    current, speed, position = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    integral = 0.
    for k in range(len(t)-1):
        index = min(int((t[k]+1e-10)/.005), len(target)-1)
        error = target[index]-speed[k]
        current[k] = 1.*error+12.*integral+.1  # True SI P=1 A/(rad/s), I=12 A/rad.
        acceleration = (.3*current[k]-.025*speed[k])/.015
        speed[k+1] = speed[k]+step*acceleration
        position[k+1] = position[k]+step*speed[k]
        integral += step*error
    current[-1] = current[-2]
    vt = np.arange(.0003, duration, .0025)
    it = np.arange(.0007, duration, .0025)
    pt = np.arange(.001, duration, .02)
    q = np.interp(pt, t, position)
    q = np.round(q*16384/(2*math.pi))*2*math.pi/16384
    segment = experiment.Segment(f"synthetic_f{frequency}_r{repeat}_b{base}", "as_found_pi", "sine", repeat,
                                 base, amplitude, frequency, duration, 1.)
    return dict(segment=asdict(segment), actual_duration_s=duration, start_monotonic_s=0., streams=dict(
        command=(ct, target), current=(it, np.interp(it, t, current)),
        velocity=(vt, np.interp(vt, t, speed)), position=(pt, q)))


class CollectorTests(unittest.TestCase):
    def make_io(self):
        args = experiment.parse_args([])
        clock, log = Clock(), MemoryLog()
        bus = FakeCAN(clock)
        return experiment.IdentificationIO(args, bus, log, clock=clock), bus, log, clock

    def test_comprehensive_plan_balances_directions_repeats_and_frequency_levels(self):
        args = experiment.parse_args([])
        plan = experiment.make_plan(args)
        self.assertEqual(sum(p.kind == "steady" for p in plan), 54)
        self.assertEqual(sum(p.kind == "step" for p in plan), 120)
        self.assertEqual(sum(p.kind == "sine" for p in plan), 108)
        self.assertAlmostEqual(sum(p.duration_s for p in plan), 1194.)
        for repeat in (1, 2, 3):
            steady = [p.base_rad_s for p in plan if p.repeat == repeat and p.kind == "steady"]
            self.assertEqual(sorted(steady), sorted(sign*x for x in args.levels for sign in (-1, 1)))
            self.assertEqual({p.frequency_hz for p in plan if p.repeat == repeat and p.kind == "sine"},
                             set(args.frequencies))
        self.assertEqual([asdict(p) for p in plan], [asdict(p) for p in experiment.make_plan(args)])

    def test_preview_never_opens_can_or_creates_files(self):
        with patch.object(experiment.shared, "open_bus", side_effect=AssertionError), redirect_stdout(io.StringIO()):
            self.assertEqual(experiment.main([]), 0)

    def test_gain_and_setting_writes_are_rejected_before_can_send(self):
        motor, bus, _, _ = self.make_io()
        for opcode in (0xB8, 0xB9, 0xB3, 0xB5):
            with self.assertRaisesRegex(ValueError, "READ ONLY"):
                motor.send(bytes([opcode])+b"\0"*4)
        with self.assertRaises(ValueError):
            motor.send(b"\xb1")
        self.assertFalse(bus.sent)

    def test_units_schedule_high_rate_current_and_velocity_and_cleanup(self):
        motor, bus, log, _ = self.make_io()
        original = bus.config[0xB9]
        segment = experiment.Segment("test", "as_found_pi", "steady", 1, 1., 0., 0., .2, .05)
        with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            motor.startup()
            motor.run_segment(segment)
            motor.cleanup()
        commands = [r for r in log.rows if r["record_type"] == "command"]
        self.assertEqual(len(commands), 40)
        self.assertTrue(all(r["encoded_target_velocity_rad_s"] > .99 for r in commands))
        self.assertTrue(all(int.from_bytes(bytes.fromhex(r["payload_hex"])[1:], "little", signed=True) < 0
                            for r in commands))
        self.assertEqual(motor.zero_count, 123456)
        self.assertTrue(motor.disable_confirmed)
        self.assertEqual(bus.config[0xB9], original)
        self.assertEqual(bus.config[0xCD][1], 0)
        self.assertTrue(all(len(data) == 1 for _, _, data in bus.sent if data[0] in (0xB8, 0xB9, 0xB3, 0xB5)))
        self.assertFalse(any(data[0] == 0xB1 for _, _, data in bus.sent))

    def test_dynamic_query_rate_is_separate_from_target_rate(self):
        motor, bus, log, _ = self.make_io()
        segment = experiment.Segment("dynamic", "as_found_pi", "sine", 1, 0., 1., 5., .2, .02)
        with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            motor.startup()
            motor.run_segment(segment)
            motor.cleanup()
        for opcode in ("A1", "A2"):
            queries = [r for r in log.rows if r["method"] == "dynamic" and r["record_type"] == "tx"
                       and r["opcode"] == opcode]
            self.assertGreaterEqual(len(queries), 75)
        self.assertEqual(motor.phases[0]["commands"], 40)

    def test_delay_skips_without_abort_or_catchup(self):
        motor, bus, _, _ = self.make_io()
        bus.pause_once = True
        segment = experiment.Segment("late", "as_found_pi", "step", 1, 0., 1., 0., .6, .1, .1, .4)
        with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            motor.startup()
            motor.run_segment(segment)
            motor.cleanup()
        self.assertTrue(bus.paused)
        self.assertGreater(motor.phases[0]["skipped_slots"], 0)
        self.assertLess(motor.phases[0]["actual_duration_s"], .601)
        times = [t for t, _, payload in bus.sent if payload[0] == 0xC1]
        self.assertGreater(min(np.diff(times)), .0049)

    def test_interrupt_log_failure_and_driver_fault_still_disable(self):
        for fault in ("interrupt", "log", "driver"):
            motor, bus, log, _ = self.make_io()
            segment = experiment.Segment("stop", "as_found_pi", "steady", 1, 1., 0., 0., .2, .05)
            with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
                motor.startup()
                if fault == "interrupt":
                    motor.stop_requested = lambda: True
                elif fault == "log":
                    log.fail = True
                else:
                    bus.fault = 32
                with self.assertRaises((KeyboardInterrupt, OSError, RuntimeError)):
                    try:
                        motor.run_segment(segment)
                    finally:
                        motor.cleanup()
            self.assertEqual(bus.mode, 0)
            self.assertTrue(motor.disable_confirmed)
            self.assertFalse(any(len(d)>1 for _, _, d in bus.sent if d[0] in (0xB8, 0xB9)))

    def test_main_records_three_files_and_never_changes_gains(self):
        clock, bus = Clock(), None
        bus = FakeCAN(clock)
        cls = experiment.IdentificationIO
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"virtual_only.csv"
            with patch.object(experiment.shared, "open_bus", return_value=bus), \
                 patch.object(experiment.shared.select, "select", bus.select), \
                 patch.object(experiment, "IdentificationIO", side_effect=lambda *a: cls(*a, clock=clock)), \
                 patch.object(FakeCAN, "close", create=True), redirect_stdout(io.StringIO()):
                result = experiment.main(["--execute", "--suite", "steady", "--repeats", "1", "--levels", ".1",
                                          "--steady-settle", ".02", "--steady-measure", ".04", "--output", str(output)])
            self.assertEqual(result, 0)
            report = json.loads(output.with_suffix(".json").read_text())
            self.assertTrue(report["completed"] and report["disabled_and_restored"] and report["gains_read_only"])
            self.assertTrue(output.is_file() and output.with_name(output.stem+"_plan.csv").is_file())
            self.assertEqual(report["initial_multi_turn_count"], 123456)
            self.assertFalse(any(len(d)>1 for _, _, d in bus.sent if d[0] in (0xB8, 0xB9)))
            metadata, _, _, _ = analysis.read_capture(output)
            self.assertEqual(metadata["schema"], "drive_c1_kp_identify_v1")


class EstimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.phases = [synthetic_phase(f, repeat, base=base) for repeat in (1, 2, 3)
                      for base in (0., 2.) for f in (.5, 2., 5., 10.)]

    def test_time_domain_recovers_physical_p_and_i_separately(self):
        result, _ = analysis.time_fit(self.phases, 40, 42)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["kp_a_per_rad_s"], 1., delta=.1)
        self.assertAlmostEqual(result["ki_a_per_rad"], 12., delta=1.)
        self.assertGreater(result["heldout_score_before_all_data_refit"]["r_squared"], .95)

    def test_frequency_domain_recovers_physical_gain_without_register_units(self):
        samples = analysis.frequency_samples(self.phases)
        result = analysis.frequency_fit(samples, 40, 42)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["kp_a_per_rad_s"], 1., delta=.15)
        self.assertAlmostEqual(result["ki_a_per_rad"], 12., delta=1.5)
        self.assertLess(result["validation_relative_rms"], .15)

    def test_complex_model_handles_known_output_lag_and_delay(self):
        samples = []
        for repeat in (1, 2, 3):
            for frequency in (.5, 1., 2., 5., 10., 20.):
                w = 2*math.pi*frequency
                gain = (1.+12./(1j*w))*np.exp(-1j*w*.003)/(1+1j*w*.005)
                samples.append(dict(repeat=repeat, frequency_hz=frequency, usable=True,
                                    gain_real_a_per_rad_s=float(gain.real), gain_imag_a_per_rad_s=float(gain.imag)))
        result = analysis.frequency_fit(samples, 40, 42)
        self.assertAlmostEqual(result["kp_a_per_rad_s"], 1., places=8)
        self.assertAlmostEqual(result["ki_a_per_rad"], 12., places=8)
        self.assertEqual(result["apparent_delay_ms"], 3)
        self.assertEqual(result["current_or_measurement_lag_ms"], 5.)
        self.assertEqual(result["heldout_trials"], 6)
        self.assertEqual(len(result["test_frequencies_hz"]), 6)
        two_repeats = analysis.frequency_fit([s for s in samples if s["repeat"] < 3], 40, 42)
        self.assertIsNone(two_repeats["untouched_test_relative_rms"])

    def test_exact_held_fourier_does_not_treat_zoh_as_continuous_target(self):
        f, dt, duration = 10., .02, 1.
        t = np.arange(0., duration, dt)
        y = np.sin(2*math.pi*f*t)
        measured = analysis.held_harmonic(t, y, f, 0., duration)
        expected = -1j*np.exp(-1j*math.pi*f*dt)*np.sinc(f*dt)
        self.assertAlmostEqual(abs(measured-expected), 0., places=12)

    def test_unexcited_data_is_not_turned_into_a_gain(self):
        flat = synthetic_phase(2., 1)
        for key in ("current", "velocity", "position", "command"):
            time, values = flat["streams"][key]
            flat["streams"][key] = time, np.zeros_like(values)
        result, _ = analysis.time_fit([flat]*4, 40, 42)
        self.assertIsNone(result)
        self.assertIsNone(analysis.frequency_fit(analysis.frequency_samples([flat]*4), 40, 42))

    def test_end_to_end_raw_csv_report_recovers_torque_units_and_rejects_gain_writes(self):
        # This fixture is simulated mathematical PI data, never a physical motor log.
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder/"SYNTHETIC_ONLY.csv"
            phases, records, position_offset, origin = [], [], 0., 100.
            zero = 123456
            for original in self.phases:
                phase = {k: v for k, v in original.items() if k != "streams"}
                phase["start_monotonic_s"] = origin
                phases.append(phase)
                name = original["segment"]["name"]
                for key, (times, values) in original["streams"].items():
                    for time, value in zip(times, values):
                        row = dict(method=name, host_monotonic_s=origin+time)
                        if key == "command":
                            payload, encoded = experiment.shared.c1_payload(float(value))
                            row.update(record_type="command", encoded_target_velocity_rad_s=encoded)
                        elif key == "velocity":
                            payload = b"\xa2"+struct.pack("<i", round(-value*6000/(2*math.pi)))
                            row.update(record_type="rx")
                        elif key == "current":
                            payload = b"\xa1"+struct.pack("<i", round(-value*1000))
                            row.update(record_type="rx")
                        else:
                            count = zero-round((value+position_offset)*16384/(2*math.pi))
                            payload = b"\xa3"+struct.pack("<Hi", count % 16384, count)
                            row.update(record_type="rx", position_rad=-(count-zero)*2*math.pi/16384)
                        row.update(payload_hex=payload.hex())
                        records.append(row)
                position_offset += original["streams"]["position"][1][-1]
                origin += original["segment"]["duration_s"]+.1
            fields = ("method", "host_monotonic_s", "record_type", "payload_hex",
                      "encoded_target_velocity_rad_s", "position_rad")
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(sorted(records, key=lambda r: r["host_monotonic_s"]))
            metadata = dict(schema="drive_c1_kp_identify_v1", completed=True, phases=phases,
                            initial_multi_turn_count=zero, torque_constant_nm_per_a=.3,
                            original_settings={"B3": (b"\xb3"+struct.pack("<I", 30000)).hex()})
            path.with_suffix(".json").write_text(json.dumps(metadata))
            report = folder/"analysis"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(analysis.main([str(path), "--output-dir", str(report), "--bootstrap", "40"]), 0)
            stats = json.loads((report/"statistics.json").read_text())
            self.assertAlmostEqual(stats["provisional_initial_simulation_velocity_kp_nm_per_rad_s"], .3, delta=.05)
            self.assertTrue((report/"all_figures.pdf").read_bytes().startswith(b"%PDF"))
            self.assertTrue((report/"index.html").is_file())
            self.assertGreater(stats["time_domain"]["heldout_trials"], 0)
            with path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writerow(dict(record_type="tx", payload_hex="b900000000", method="startup",
                                     host_monotonic_s=0.))
            with self.assertRaisesRegex(ValueError, "B8/B9 writes"):
                analysis.read_capture(path)


if __name__ == "__main__":
    unittest.main()
