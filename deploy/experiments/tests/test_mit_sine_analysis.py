"""Offline sparse-MIT decoding and analysis regression tests; no hardware I/O."""

import csv
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_mit_sine import analyze, check_wire, labels, load_recording, trace
from steering_sine_mit_io import FIELDS, Motor, RADIANS_PER_COUNT, encode, target_fields


class MitSineAnalysisTests(unittest.TestCase):
    @staticmethod
    def row(kind, stamp, payload, **values):
        motor = Motor("steering", 2, 1.0, -1, -1, 0, -3/65535, .78, (3, 45, 18), reference_epoch=1)
        row = dict(record_type=kind, motor="steering", motor_id=2, gear_ratio=1,
                   encoder_sign=-1, torque_sign=-1, zero_count=0, reference_epoch=1,
                   mit_zero_position_motor_rad=motor.zero_rad, torque_constant_nm_per_a=.78,
                   mit_position_max_rad=3, mit_velocity_max_rad_s=45, mit_torque_max_nm=18,
                   phase="motion", elapsed_s=stamp, mit_can_id=0x502 if kind == "mit_target" else 2,
                   can_length=len(payload), mit_payload_hex=payload.hex().upper(),
                   socket_dropped_frames=0)
        row.update(values)
        return row

    @classmethod
    def target(cls, stamp=0, requested=0):
        p = encode(-3/65535-requested, -3, 3, 16)
        v, kp, kd, t = 2048, 82, 164, 2048
        payload = bytes((p >> 8, p & 255, v >> 4, (v & 15) << 4 | kp >> 8, kp & 255,
                         kd >> 4, (kd & 15) << 4 | t >> 8, t & 255))
        motor = Motor("steering", 2, 1, -1, -1, 0, -3/65535, .78, (3, 45, 18))
        return cls.row("mit_target", stamp, payload, requested_position_rad=requested,
                       command_interval_s=math.nan if stamp == 0 else .02,
                       command_lateness_s=0, skipped_command_ticks=0, **target_fields(motor, payload))

    def test_filename_gain_decimals_and_frequency(self):
        for token, expected in (("01", .1), ("02", .2), ("0.2", .2)):
            r = labels(Path(f"202610071628_mit_10_{token}_5Hz.csv"))
            self.assertEqual(r["kp"], 10)
            self.assertEqual(r["kd"], expected)
            self.assertEqual(r["frequency_hz"], 5)
            self.assertEqual(r["amplitude_rad"], 1)

    def test_unknown_filename_rejected(self):
        with self.assertRaises(ValueError):
            labels(Path("unknown.csv"))

    def test_target_wire_and_absolute_origin(self):
        row = self.target(requested=.37)
        check_wire(row)
        self.assertAlmostEqual(row["mit_target_position_rad"], .37, delta=3/65535)
        self.assertNotEqual(row["mit_target_velocity_motor_rad_s"], 0)  # Recorded quantization is retained.
        self.assertNotEqual(row["mit_feedforward_torque_motor_nm"], 0)

    def test_corrupt_target_value_rejected(self):
        row = self.target(requested=.37)
        row["mit_target_position_rad"] += .001
        with self.assertRaisesRegex(ValueError, "Wire/CSV mismatch"):
            check_wire(row)

    def test_wrong_address_or_length_rejected(self):
        for field, value in (("mit_can_id", 0x501), ("can_length", 7)):
            row = self.target()
            row[field] = value
            with self.assertRaises(ValueError):
                check_wire(row)

    def test_sparse_a3_feedback_count_sign_and_zero(self):
        row = self.row("feedback", .001, b"\xa3"+struct.pack("<Hi", 100, 100), can_opcode=163,
                       encoder_count=100, encoder_single_turn_count=100,
                       position_unwrapped_rad=-100*RADIANS_PER_COUNT,
                       position_rad=-100*RADIANS_PER_COUNT)
        check_wire(row)
        row["position_unwrapped_rad"] *= -1
        with self.assertRaises(ValueError):
            check_wire(row)

    def test_a2_native_rpm_conversion(self):
        row = self.row("feedback", .001, b"\xa2"+struct.pack("<i", 1000), can_opcode=162,
                       velocity_rad_s=-10*2*math.pi/60)
        check_wire(row)

    def test_a1_current_torque_independent_sign(self):
        row = self.row("feedback", .001, b"\xa1"+struct.pack("<i", 500), can_opcode=161,
                       current_a=.5, torque_nm=-.39)
        check_wire(row)
        row["torque_nm"] *= -1
        with self.assertRaises(ValueError):
            check_wire(row)

    def test_f1_bits_and_quantized_native_units(self):
        payload = bytes.fromhex("F1800080080001")
        row = self.row("mit_feedback", .001, payload, mit_position_code=32768, mit_velocity_code=2048,
                       mit_torque_code=2048, mit_feedback_position_motor_rad=3/65535,
                       mit_feedback_velocity_motor_rad_s=45/4095,
                       mit_feedback_torque_motor_nm=18/4095,
                       mit_feedback_position_rad=-6/65535,
                       mit_feedback_velocity_rad_s=-45/4095,
                       mit_feedback_torque_nm=-18/4095,
                       mit_status_raw=1, mit_is_active="True", mit_has_fault="False")
        check_wire(row)
        row["mit_has_fault"] = "True"
        with self.assertRaises(ValueError):
            check_wire(row)

    def test_sparse_trace_uses_own_time_not_other_query(self):
        rows = [dict(elapsed_s=i*.005, position_unwrapped_rad=i*.01, velocity_rad_s="") for i in range(10)]
        result = trace(rows, "position_unwrapped_rad")
        np.testing.assert_array_equal(result[:, 0], np.arange(10)*.005)
        with self.assertRaises(ValueError):
            trace(rows, "velocity_rad_s")

    def test_reversed_or_duplicate_time_rejected(self):
        rows = [dict(elapsed_s=i*.005, velocity_rad_s=1) for i in range(10)]
        rows[5]["elapsed_s"] = rows[4]["elapsed_s"]
        with self.assertRaises(ValueError):
            trace(rows, "velocity_rad_s")

    def test_missing_f1_or_sparse_feedback_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"202610071628_mit_10_02_1Hz.csv"
            with path.open("w", newline="") as h:
                writer = csv.DictWriter(h, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(self.target())
            with self.assertRaisesRegex(ValueError, "Need MIT targets"):
                load_recording(path)

    def test_real_recordings_wire_metadata_and_cross_checks(self):
        directory = Path(__file__).resolve().parents[1]/"logs"
        names = ("202610071628_mit_10_02_1Hz", "202610071632_mit_10_02_2Hz", "202610071634_mit_10_02_5Hz",
                 "202610071637_mit_10_01_1Hz", "202610071639_mit_10_01_2Hz", "202610071641_mit_10_01_5Hz",
                 "202610071642_mit_10_01_10Hz", "202610071643_mit_10_02_10Hz")
        paths = [directory/(name+".csv") for name in names]
        if not all(p.exists() for p in paths):
            self.skipTest("Real-bike recordings are optional, uncommitted experimental inputs")
        for path in paths:
            with self.subTest(source=path.name):
                result, traces = analyze(path)
                self.assertTrue(result["filename_verified"])
                self.assertLess(abs(result["frequency_fit_hz"]-result["labels"]["frequency_hz"]), .0001)
                self.assertFalse(result["provenance"]["shutdown_confirmed"])
                self.assertEqual(result["provenance"]["status"][-1]["mode"], 5)
                self.assertEqual(result["skipped_command_ticks"], 0)
                self.assertEqual(result["provenance"]["f1_motion_fault_rows"], 0)
                self.assertTrue(all(len(t) > 100 for t in traces))
                self.assertLess(abs(result["f1_cross_check"]["gain"]-result["steady"]["gain"]), .004)
                self.assertLess(abs(result["correlation_check"]["lag_ms"]-result["steady"]["phase_equivalent_lag_ms"]), .5)


if __name__ == "__main__":
    unittest.main()
