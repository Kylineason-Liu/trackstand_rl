"""Integrated MIT experiment checks; virtual clocks/CAN, never real motor commands."""

import contextlib
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
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import steering_sine_mit as original
import steering_sine_mit_io as combined


class MemoryLog:
    def __init__(self, fail_on_target=False):
        self.rows = []
        self.fail_on_target = fail_on_target
        self.failed = False

    def check(self):
        if self.failed:
            raise RuntimeError("simulated CSV failure")

    def emit(self, row):
        if self.fail_on_target and row["record_type"] == "mit_target":
            self.failed = True
        self.check()
        self.rows.append(row.copy())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeCAN:
    """One nonblocking CAN endpoint with scheduled, externally originated replies."""

    def __init__(self, fault=None):
        self.now, self.sequence = 0.0, 0
        self.frames, self.replies, self.options = [], [], []
        self.fault = fault
        self.configs = {2: (120, 800, 300), 1: (955, 4500, 1800)}
        self.positions = {2: 32767, 1: 32767}
        self.f0_reads = 0
        self.enabled, self.closed = False, False
        self.drop_aux, self.drop_mit = set(), False
        self.reply_delay, self.mit_delay = .0002, .0004
        self.recv_cost = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def setsockopt(self, level, option, value):
        self.options.append((level, option, value))

    def setblocking(self, value):
        assert value is False

    def bind(self, address):
        self.address = address

    def enqueue(self, cid, data, delay=0, ancillary=None, flags=0):
        self.sequence += 1
        frame = combined.CAN_FRAME.pack(cid, len(data), data)
        heapq.heappush(self.replies, (self.now + delay, self.sequence, frame, ancillary or [], flags))

    def send(self, frame):
        cid, length, padded = combined.CAN_FRAME.unpack(frame)
        data, mid = padded[:length], cid & 0xFF
        self.frames.append((self.now, cid, data))
        if cid & 0x400:
            if self.fault == "send_failed":
                raise OSError("simulated MIT send failure")
            self.enabled = True
            self.positions[mid] = int.from_bytes(data[:2], "big")
            if not self.drop_mit:
                status = 3 if self.fault == "runtime_fault" else 0 if self.fault == "runtime_inactive" else 1
                self.enqueue(mid, b"\xf1" + data[:2] + bytes.fromhex("800800") + bytes([status]), self.mit_delay)
        elif data == b"\xcf":
            if self.fault == "cf_send_failed" and self.enabled:
                raise OSError("simulated shutdown CF failure")
            mode = 4 if self.fault == "cf_bad_mode" and not self.enabled else 0
            self.enabled = False
            self.enqueue(mid, b"\xcf" + struct.pack("<HH", 4800, 125) + bytes([35, mode, 0]), self.reply_delay)
        elif data == b"\xb1":
            self.positions[mid] = 32767
            if self.fault != "missing_b1":
                self.enqueue(mid, b"\xb1\x00\x00", self.reply_delay)
        elif data[0] == 0xF0:
            if len(data) == 7:
                if self.fault != "write_not_applied":
                    self.configs[mid] = struct.unpack("<HHH", data[1:])
                if self.fault == "missing_write":
                    return len(frame)
            else:
                self.f0_reads += 1
                if self.f0_reads == 2 and self.fault == "bad_readback":
                    self.configs[mid] = (120, 800, 300)
            self.enqueue(mid, b"\xf0" + struct.pack("<HHH", *self.configs[mid]), self.reply_delay)
        elif data == b"\xf1":
            status = 2 if self.fault == "startup_fault" else 1 if self.fault == "startup_active" else 0
            p = 40000 if self.fault == "bad_f1_zero" else self.positions[mid]
            v = 3000 if self.fault == "moving_startup" else 2048
            fields = bytes([v >> 4, ((v & 15) << 4) | 8, 0, status])
            self.enqueue(mid, b"\xf1" + p.to_bytes(2, "big") + fields, self.reply_delay)
        elif data == b"\xb0":
            constant = math.nan if self.fault == "bad_constant" else .78 if mid == 2 else .3
            self.enqueue(mid, b"\xb0\x00" + struct.pack("<f", constant) + b"\x00", self.reply_delay)
        elif data[0] not in self.drop_aux or not self.enabled:
            if data == b"\xa3":
                count = 100 if self.fault == "bad_a3_zero" else 0 if mid == 2 else 1000
                self.enqueue(mid, b"\xa3" + struct.pack("<Hi", count % 16384, count), self.reply_delay)
            elif data == b"\xa2":
                self.enqueue(mid, b"\xa2" + struct.pack("<i", 10000), self.reply_delay)
            elif data == b"\xa1":
                self.enqueue(mid, b"\xa1" + struct.pack("<i", 1500), self.reply_delay)
        return len(frame)

    def recvmsg(self, frame_size, ancillary_size):
        assert frame_size == 16
        if not self.replies or self.replies[0][0] > self.now:
            raise BlockingIOError
        _, _, frame, ancillary, flags = heapq.heappop(self.replies)
        self.now += self.recv_cost
        return frame, ancillary, flags, ("can0",)

    def select(self, readable, writable, exceptional, timeout):
        assert readable == [self]
        ready = self.replies[0][0] if self.replies else math.inf
        self.now = max(self.now, min(ready, self.now + timeout))
        return ([self] if ready <= self.now else []), [], []


class IntegratedMITTests(unittest.TestCase):
    def setUp(self):
        self.args = combined.parse_args(["--duration", ".16"])

    @contextlib.contextmanager
    def clock(self, bus):
        with patch.object(combined.time, "monotonic", side_effect=lambda: bus.now), \
                patch.object(combined.time, "time", side_effect=lambda: 1700000000 + bus.now), \
                patch.object(combined.select, "select", side_effect=bus.select), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield

    def run_fake(self, bus, log=None):
        log = MemoryLog() if log is None else log
        with self.clock(bus), patch.object(combined.socket, "socket", return_value=bus) as create_socket, \
                patch.object(combined, "CsvLog", return_value=log):
            combined.execute(self.args)
            create_socket.assert_called_once()
        return log

    def motor(self):
        return combined.Motor("steering", 2, 1, -1, -1)

    def test_defaults_fixed_50hz_and_extended_200hz(self):
        args = combined.parse_args([])
        self.assertEqual(combined.COMMAND_RATE_HZ, 50)
        self.assertEqual((args.motor_id, args.sign, args.torque_sign, args.gear_ratio), (2, -1, -1, 1))
        self.assertEqual((args.amplitude, args.period, args.feedback, args.feedback_rate), (1, 1, "extended", 200))
        self.assertFalse(args.include_drive)

    def test_payload_and_native_origin_match_proven_sender(self):
        ranges = (3, 45, 18)
        for p in (-3, -1, 0, 1, 3):
            self.assertEqual(combined.mit_payload(p, ranges), original.mit_payload(p, original.MITRanges(*ranges)))
        self.args.gear_ratio = 2
        self.args.period = 10  # Explicit quarter-cycle test, independent of CLI defaults.
        self.assertEqual(combined.sine_position(self.args, 2.5, .4), -1.6)

    def test_cli_invalid_values_rejected(self):
        invalid = (("--period", ".04"), ("--amplitude", "nan"), ("--feedback-rate", "0"),
                   ("--duration", "-1"), ("--motor-id", "0"), ("--drive-id", "255"),
                   ("--query-timeout", "inf"), ("--gear-ratio", "0"), ("--amplitude", "4"),
                   ("--include-drive", "--drive-id", "2"), ("--include-drive", "--feedback", "mit"))
        for arguments in invalid:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                combined.parse_args(list(arguments))

    def test_preview_50hz_no_can_no_csv(self):
        with patch.object(combined.socket, "socket") as create_socket, patch.object(combined, "CsvLog") as csv_log, \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(combined.main(["--preview", "--duration", ".1"]), 0)
        create_socket.assert_not_called()
        csv_log.assert_not_called()
        self.assertEqual(len([line for line in printed.getvalue().splitlines() if line.startswith("t=")]), 5)

    def test_one_socket_startup_order_preserves_ranges_and_csv_fields(self):
        bus = FakeCAN()
        log = self.run_fake(bus)
        first_target = next(i for i, (_, cid, _) in enumerate(bus.frames) if cid == 0x502)
        startup = [(cid, data) for _, cid, data in bus.frames[:first_target]]
        self.assertEqual([data[0] for _, data in startup], [0xCF, 0xB1, 0xF0, 0xF0, 0xF0, 0xA3, 0xF1, 0xB0])
        self.assertEqual(startup[3], (0x102, bytes.fromhex("F01E0020032C01")))
        self.assertEqual(bus.configs[2], (30, 800, 300))
        self.assertTrue(bus.closed)
        self.assertTrue(any(row["record_type"] == "mit_feedback" and row["phase"] == "motion" for row in log.rows))
        self.assertTrue(any(row["record_type"] == "event" and "acknowledged" in row["can_note"] for row in log.rows))
        self.assertTrue(all(set(row) <= set(combined.FIELDS) for row in log.rows))
        filters = next(value for level, option, value in bus.options if level == socket.SOL_CAN_RAW
                       and option == socket.CAN_RAW_FILTER)
        self.assertTrue(all(mask == 0xC00007FF for cid, mask in struct.iter_unpack("=II", filters)))

    def test_command_intervals_20ms_despite_missing_aux_replies(self):
        for suppress in (set(), {0xA1, 0xA2, 0xA3}):
            with self.subTest(suppress=suppress):
                bus = FakeCAN()
                bus.drop_aux = suppress
                log = self.run_fake(bus)
                targets = [row for row in log.rows if row["record_type"] == "mit_target"]
                self.assertEqual(len(targets), 8)
                for row in targets[1:]:
                    self.assertAlmostEqual(row["command_interval_s"], .02, places=9)
                self.assertTrue(all(row["skipped_command_ticks"] == 0 for row in targets))
                if suppress:
                    self.assertTrue(any(row.get("query_status") == "timed_out" for row in log.rows))

    def test_mit_only_no_motion_time_read_queries(self):
        self.args.feedback = "mit"
        bus = FakeCAN()
        log = self.run_fake(bus)
        self.assertFalse(any(row["phase"] == "motion" and row["record_type"] == "can_tx" for row in log.rows))
        self.assertTrue(any(row["record_type"] == "mit_feedback" and row["phase"] == "motion" for row in log.rows))

    def test_delayed_f1_does_not_set_command_frequency(self):
        bus = FakeCAN()
        bus.mit_delay = .03  # Longer than the 20-ms setpoint interval.
        log = self.run_fake(bus)
        targets = [row for row in log.rows if row["record_type"] == "mit_target"]
        self.assertEqual(len(targets),8)
        self.assertTrue(all(abs(row["command_interval_s"]-.02) < 1e-9 for row in targets[1:]))

    def test_pre_b1_unsolicited_samples_cannot_define_motion_reference(self):
        bus = FakeCAN()
        bus.enqueue(2,b"\xf0"+struct.pack("<HHH",120,800,300))
        bus.enqueue(2,bytes.fromhex("F1900080080000"))
        bus.enqueue(2,b"\xa3"+struct.pack("<Hi",1000,1000))
        log = self.run_fake(bus)
        targets = [row for row in log.rows if row["record_type"] == "mit_target"]
        self.assertEqual(targets[0]["mit_position_code"],32767)
        self.assertTrue(all(row["zero_count"]==0 and row["reference_epoch"]==1 for row in targets))

    def test_stale_f0_write_reply_is_logged_not_confirmed(self):
        class StaleConfig(FakeCAN):
            def send(self,frame):
                cid,length,payload = combined.CAN_FRAME.unpack(frame)
                if length==7 and payload[0]==0xF0:
                    self.enqueue(2,b"\xf0"+struct.pack("<HHH",120,800,300))
                return super().send(frame)
        log = self.run_fake(StaleConfig())
        self.assertTrue(any("differs from requested" in row.get("can_note","") for row in log.rows))
        self.assertTrue(all(row["mit_position_max_rad"]==3 for row in log.rows if row["record_type"]=="mit_target"))

    def test_request_rejects_wrong_motor_opcode_and_length(self):
        class NoisyCAN(FakeCAN):
            def send(self,frame):
                cid,length,payload = combined.CAN_FRAME.unpack(frame)
                if payload[:length]==b"\xa3":
                    self.enqueue(9,b"\xa3"+struct.pack("<Hi",0,9999))
                    self.enqueue(2,b"\xa2"+struct.pack("<i",6000))
                    self.enqueue(2,b"\xa3"+bytes(5))
                return super().send(frame)
        bus = NoisyCAN()
        motor = self.motor()
        obj = combined.CanIO(bus,MemoryLog(),[motor])
        with self.clock(bus):
            reply = obj.request(motor,0xA3)
        self.assertEqual(struct.unpack("<i",reply[3:])[0],0)

    def test_async_late_reply_has_explicit_status(self):
        bus = FakeCAN()
        motor = self.motor()
        log = MemoryLog()
        obj = combined.CanIO(bus,log,[motor])
        obj.pending = combined.PendingRead(motor,0xA1,0,.002)
        bus.now = .003
        bus.enqueue(2,b"\xa1"+struct.pack("<i",1500))
        with self.clock(bus):
            obj.drain()
        self.assertIsNone(obj.pending)
        self.assertEqual(log.rows[-1]["query_status"],"late_reply")

    def test_origin_or_config_change_during_motion_stops(self):
        for data in (b"\xb1\x00\x00",b"\xf0"+struct.pack("<HHH",120,800,300)):
            with self.subTest(data=data):
                bus = FakeCAN()
                bus.enqueue(2,data)
                motor = self.motor()
                motor.ranges = (3,8,3)
                obj = combined.CanIO(bus,MemoryLog(),[motor])
                obj.phase = "motion"
                with self.clock(bus),self.assertRaises(RuntimeError):
                    obj.receive()

    def test_shutdown_without_ack_is_explicit_failure(self):
        class NoShutdownAck(FakeCAN):
            def send(self,frame):
                cid,length,data = combined.CAN_FRAME.unpack(frame)
                if data[:length]==b"\xcf" and self.enabled:
                    self.frames.append((self.now,cid,b"\xcf"))
                    return len(frame)
                return super().send(frame)
        with self.assertRaisesRegex(RuntimeError,"disabled mode not confirmed"):
            self.run_fake(NoShutdownAck())

    def test_busy_receive_work_is_bounded_by_command_deadline(self):
        bus = FakeCAN()
        bus.recv_cost = .0005
        for _ in range(100):
            bus.enqueue(2,b"\xa1"+struct.pack("<i",1500))
        obj = combined.CanIO(bus,MemoryLog(),[self.motor()])
        obj.phase,obj.next_command = "motion",.02
        with self.clock(bus):
            self.assertFalse(obj.drain())
        self.assertLessEqual(bus.now,.0205)
        self.assertTrue(bus.replies)

    def test_drive_collection_never_controls_drive(self):
        self.args.include_drive = True
        bus = FakeCAN()
        log = self.run_fake(bus)
        drive_tx = [data for _, cid, data in bus.frames if cid & 0xFF == 1]
        self.assertTrue(drive_tx)
        self.assertTrue(all(len(data) == 1 and data[0] in combined.READ_LENGTHS for data in drive_tx))
        self.assertTrue(any(row["motor"] == "drive" and row["record_type"] == "feedback" for row in log.rows))

    def test_startup_failures_no_motion_final_cf_attempted(self):
        faults = ("cf_bad_mode", "missing_b1", "missing_write", "write_not_applied", "bad_readback",
                  "bad_a3_zero", "bad_f1_zero", "startup_fault", "startup_active", "moving_startup")
        for fault in faults:
            with self.subTest(fault=fault):
                bus = FakeCAN(fault)
                with self.assertRaises((RuntimeError, TimeoutError)):
                    self.run_fake(bus)
                self.assertFalse(any(cid == 0x502 for _, cid, _ in bus.frames))
                self.assertEqual(bus.frames[-1][2], b"\xcf")

    def test_logging_failure_does_not_prevent_physical_cf_send(self):
        bus = FakeCAN()
        with self.assertRaisesRegex(RuntimeError, "CSV failure"):
            self.run_fake(bus, MemoryLog(fail_on_target=True))
        self.assertEqual(bus.frames[-1][1:], (0x102, b"\xcf"))

    def test_missing_f1_watchdog_stops_and_disables(self):
        bus = FakeCAN()
        bus.drop_mit = True
        self.args.duration, self.args.feedback_timeout = 1, .08
        with self.assertRaisesRegex(RuntimeError, "F1 feedback timeout"):
            self.run_fake(bus)
        self.assertEqual(bus.frames[-1][2], b"\xcf")
        self.assertLessEqual(len([cid for _, cid, _ in bus.frames if cid == 0x502]), 4)

    def test_runtime_fault_and_failed_transmission_attempt_cf(self):
        for fault in ("runtime_fault", "runtime_inactive", "send_failed"):
            with self.subTest(fault=fault):
                bus = FakeCAN(fault)
                with self.assertRaises((OSError, RuntimeError)):
                    self.run_fake(bus)
                self.assertEqual(bus.frames[-1][2], b"\xcf")

    def test_steering_references_require_explicit_verified_startup(self):
        motor = self.motor()
        motor.ranges = (3,8,3)
        obj = combined.CanIO(None,MemoryLog(),[motor])
        a3 = obj.row(motor,"can_rx")
        f1 = obj.row(motor,"can_rx")
        obj.decode_reply(motor,b"\xa3"+struct.pack("<Hi",1000,1000),0,a3)
        obj.decode_reply(motor,bytes.fromhex("F17FFF80080000"),0,f1)
        self.assertIsNone(motor.zero_count)
        self.assertTrue(math.isnan(motor.zero_rad))
        self.assertTrue(math.isnan(a3["position_unwrapped_rad"]))
        self.assertTrue(math.isnan(f1["mit_feedback_position_rad"]))

    def test_shutdown_send_failure_is_not_reported_success(self):
        with self.assertRaisesRegex(OSError, "shutdown CF failure"):
            self.run_fake(FakeCAN("cf_send_failed"))

    def test_unknown_torque_constant_stays_nan(self):
        log = self.run_fake(FakeCAN("bad_constant"))
        rows = [row for row in log.rows if "current_a" in row]
        self.assertTrue(rows)
        self.assertTrue(all(math.isnan(row["torque_nm"]) for row in rows))

    def test_feedback_units_and_signs(self):
        motor = self.motor()
        motor.ranges, motor.zero_rad, motor.zero_count, motor.torque_constant = (3,45,18), .2, 1000, .78
        obj = combined.CanIO(None, MemoryLog(), [motor])
        for data, expected in ((b"\xa3" + struct.pack("<Hi", 123, 17384), {"position_unwrapped_rad":-2*math.pi}),
                               (b"\xa2" + struct.pack("<i", 6000), {"velocity_rad_s":-2*math.pi}),
                               (b"\xa1" + struct.pack("<i", 1500), {"current_a":1.5,"torque_nm":-1.17}),
                               (bytes.fromhex("F1800080080001"), {"mit_status_raw":1,"mit_has_fault":False})):
            row = obj.row(motor,"can_rx")
            obj.decode_reply(motor,data,1,row)
            for key,value in expected.items():
                self.assertAlmostEqual(row[key],value)
        payload = combined.mit_payload(1,motor.ranges)
        fields = combined.target_fields(motor,payload)
        self.assertAlmostEqual(fields["mit_kp_nm_per_motor_rad"],82*500/4095)
        self.assertAlmostEqual(fields["mit_kd_nm_s_per_motor_rad"],164*5/4095)
        self.assertEqual(fields["mit_velocity_code"],2048)
        self.assertAlmostEqual(fields["mit_feedforward_torque_motor_nm"],18/4095)
        self.assertAlmostEqual(fields["mit_target_position_rad"],-(fields["mit_target_position_motor_rad"]-.2))

    def test_receive_interference_drop_and_truncation_fail(self):
        for cid, data, ancillary, flags in (
            (0x502, bytes(8), [], 0), (0x102, b"\xa3", [], socket.MSG_DONTROUTE),
            (2,b"\xcf",[],0), (2,bytes.fromhex("00FF00FF00FF00FF"),[],0),
            (2,b"\xa1"+bytes(4),[(socket.SOL_SOCKET,combined.SO_RXQ_OVFL,struct.pack("=I",1))],0),
            (2,b"\xa1"+bytes(4),[],socket.MSG_CTRUNC),
        ):
            with self.subTest(cid=cid,data=data,flags=flags):
                bus = FakeCAN()
                bus.enqueue(cid,data,ancillary=ancillary,flags=flags)
                obj = combined.CanIO(bus,MemoryLog(),[self.motor()])
                with self.clock(bus), self.assertRaises(RuntimeError):
                    obj.receive()

    def test_overwrite_refused_before_can_open(self):
        with tempfile.TemporaryDirectory() as directory:
            self.args.output = str(Path(directory)/"existing.csv")
            Path(self.args.output).touch()
            with patch.object(combined.socket,"socket") as create_socket, self.assertRaises(FileExistsError):
                combined.execute(self.args)
            create_socket.assert_not_called()

    def test_real_csv_writer_full_fake_experiment_flushes_rows(self):
        bus = FakeCAN()
        with tempfile.TemporaryDirectory() as directory:
            self.args.output = str(Path(directory)/"experiment.csv")
            with self.clock(bus), patch.object(combined.socket,"socket",return_value=bus):
                combined.execute(self.args)
            with open(self.args.output,newline="") as saved:
                rows = list(csv.DictReader(saved))
            self.assertEqual(len([row for row in rows if row["record_type"]=="mit_target"]),8)
            self.assertTrue(any(row["record_type"]=="mit_feedback" for row in rows))
            self.assertIn("acknowledged",rows[-1]["can_note"])

    def test_csv_backlog_and_writer_errors_visible(self):
        log = combined.CsvLog("unused")
        log.pending = combined.queue.Queue(maxsize=1)
        log.emit({})
        with self.assertRaisesRegex(RuntimeError,"backlog"):
            log.emit({})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,"CSV writer failed"):
                with combined.CsvLog(str(Path(directory)/"invalid.csv")) as writer:
                    writer.emit({"not_a_csv_field":True})

    def test_sigterm_and_keyboard_interrupt_cleanup(self):
        previous = combined.signal.getsignal(combined.signal.SIGTERM)
        with patch.object(combined,"execute",side_effect=KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(combined.main([]),130)
        self.assertIs(combined.signal.getsignal(combined.signal.SIGTERM),previous)
        bus = FakeCAN()
        with self.clock(bus), patch.object(combined.socket,"socket",return_value=bus), \
                patch.object(combined,"CsvLog",return_value=MemoryLog()), \
                patch.object(combined,"motion",side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            combined.execute(self.args)
        self.assertEqual(bus.frames[-1][2],b"\xcf")


if __name__ == "__main__":
    unittest.main()
