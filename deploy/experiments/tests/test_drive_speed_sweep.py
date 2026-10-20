"""Rear speed-sweep tests with simulated CAN/time only; never open a CAN device."""

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
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import drive_speed_sweep as sweep


class Clock:
    now = 10.0

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
    def __init__(self, clock):
        self.clock, self.frames, self.rx = clock, [], []
        self.sequence, self.mode, self.speed_code = 0, 0, 0
        self.config = {0xB3: b"\xb3"+struct.pack("<I", 3000),
                       0xCD: b"\xcd\x00"+struct.pack("<H", 5000)+b"\x00"}
        self.drop, self.temperature, self.fault, self.bad_cf = set(), 25, 0, False
        self.reverse_speed, self.ignore_writes = False, False
        self.query_delay = 0.0002
        self.closed = False

    def close(self):
        self.closed = True

    def enqueue(self, data, cid=1, delay=0, flags=0, ancillary=None):
        self.sequence += 1
        heapq.heappush(self.rx, (self.clock.now+delay, self.sequence,
                                sweep.CAN_FRAME.pack(cid,len(data),data), ancillary or [], flags))

    def status(self, opcode):
        return bytes([opcode])+struct.pack("<HH",2400,20)+bytes([self.temperature,self.mode,self.fault])

    def send(self, frame):
        cid, length, data = sweep.CAN_FRAME.unpack(frame)
        assert cid == 0x101, "No other motor may be commanded"
        data, op = data[:length], data[0]
        self.frames.append((self.clock.now,data))
        if op == 0xCF:
            self.mode, self.speed_code = (3 if self.bad_cf else 0), 0
            reply = self.status(op)
        elif op == 0xC1:
            self.mode = 3
            self.speed_code = struct.unpack("<i",data[1:])[0]*(-1 if self.reverse_speed else 1)
            reply = b"\xc1"+struct.pack("<i",self.speed_code)
        elif op in self.config:
            if len(data)>1 and not self.ignore_writes:
                self.config[op] = data
            reply = self.config[op]
        elif op == 0xB0:
            reply = b"\xb0\x00"+struct.pack("<f",0.3)+b"\x00"
        elif op == 0xA2:
            reply = b"\xa2"+struct.pack("<i",self.speed_code)
        elif op == 0xA1:
            current = 0 if self.mode == 0 or self.speed_code == 0 else (100 if self.speed_code>0 else -100)
            reply = b"\xa1"+struct.pack("<i",current)
        elif op == 0xAE:
            reply = self.status(op)
        else:
            raise AssertionError(f"Unexpected operation {op:02X}")
        if op not in self.drop:
            self.enqueue(reply,delay=self.query_delay)
        return len(frame)

    def recvmsg(self, size, ancillary_size):
        if not self.rx or self.rx[0][0]>self.clock.now:
            raise BlockingIOError
        _,_,frame,ancillary,flags=heapq.heappop(self.rx)
        return frame,ancillary,flags,("fake",)

    def wait(self, duration):
        ready=self.rx[0][0] if self.rx else math.inf
        self.clock.now=min(self.clock.now+max(0,duration),ready)


class SpeedSweepTests(unittest.TestCase):
    def setUp(self):
        self.args=sweep.parse_args(["--speeds","1","-1","--hold",".04","--settle",".02",
                                   "--rest",".04","--acceleration","20"])
        self.clock,self.log=Clock(),MemoryLog()
        self.bus=FakeCAN(self.clock)
        self.runner=sweep.Sweep(self.args,self.bus,self.log,clock=self.clock)
        self.runner.wait=self.bus.wait
        self.quiet_out,self.quiet_err=redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO())
        self.quiet_out.__enter__()
        self.quiet_err.__enter__()

    def tearDown(self):
        self.quiet_err.__exit__(None,None,None)
        self.quiet_out.__exit__(None,None,None)

    def test_preview_has_no_can_or_file_side_effects(self):
        with patch.object(sweep,"open_bus") as can,patch.object(sweep,"CsvLog") as log:
            self.assertEqual(sweep.main(["--speeds","1"]),0)
        can.assert_not_called()
        log.assert_not_called()

    def test_socket_setup_with_and_without_python_error_filter_constant(self):
        names=("AF_CAN","SOCK_RAW","CAN_RAW","SOL_CAN_RAW","CAN_RAW_FILTER",
               "CAN_RAW_RECV_OWN_MSGS","SOL_SOCKET")
        for exports_constant in (False,True):
            with self.subTest(exports_constant=exports_constant):
                bus=Mock()
                portable=SimpleNamespace(**{name:getattr(socket,name) for name in names},
                                         socket=Mock(return_value=bus))
                if exports_constant:
                    portable.CAN_RAW_ERR_FILTER=2
                with patch.object(sweep,"socket",portable):
                    self.assertIs(sweep.open_bus(self.args),bus)
                bus.setsockopt.assert_any_call(socket.SOL_CAN_RAW,2,struct.pack("=I",0x1FFFFFFF))
                bus.bind.assert_called_once_with(("can0",))
                bus.send.assert_not_called()
                bus.close.assert_not_called()

    def test_socket_option_failure_closes_socket_without_sending(self):
        bus=Mock()
        bus.setsockopt.side_effect=OSError("unsupported kernel option")
        with patch.object(sweep.socket,"socket",return_value=bus),self.assertRaises(OSError):
            sweep.open_bus(self.args)
        bus.close.assert_called_once()
        bus.send.assert_not_called()

    def test_cli_rejects_invalid_numbers_and_limits(self):
        for flags in (["--speeds","nan"],["--speeds","11"],["--speeds","0"],
                      ["--hold","0"],["--command-rate","1"],["--query-timeout",".3"],
                      ["--motor-id","0"],["--current-limit-a","1e30"]):
            with self.subTest(flags=flags),self.assertRaises(SystemExit):
                sweep.parse_args(flags)

    def test_c1_units_sign_and_quantization(self):
        payload,encoded,rpm=sweep.speed_payload(self.args,2*math.pi)
        self.assertEqual(payload,b"\xc1"+struct.pack("<i",-6000))
        self.assertAlmostEqual(encoded,2*math.pi)
        self.assertEqual(rpm,-60)

    def test_plan_ramps_to_zero_between_both_directions(self):
        plan=sweep.make_plan(self.args)
        for left,right in zip(plan,plan[1:]):
            self.assertAlmostEqual(left.start+left.duration,right.start)
            self.assertEqual(left.final,right.initial)
        self.assertEqual([p.final for p in plan if p.name=="hold"],[1,-1])
        ramp=next(p for p in plan if p.name=="ramp_up")
        self.assertAlmostEqual(ramp.target(ramp.start+ramp.duration/2),.5)
        self.assertEqual(plan[-1].final,0)

    def test_complete_sweep_queries_logs_and_restores(self):
        original=self.bus.config.copy()
        self.runner.startup()
        self.assertAlmostEqual(self.runner.current_limit,1)
        self.assertEqual(self.bus.config[0xCD],b"\xcd\x01"+struct.pack("<H",500)+b"\x00")
        self.runner.run()
        self.assertTrue(self.runner.shutdown())
        self.assertEqual(self.bus.config,original)
        self.assertEqual(self.bus.mode,0)
        codes=[struct.unpack("<i",d[1:])[0] for _,d in self.bus.frames if d[0]==0xC1]
        self.assertTrue(any(n>0 for n in codes) and any(n<0 for n in codes))
        self.assertTrue({"A2","A1","B0","AE"}.issubset({r.get("opcode_hex") for r in self.log.rows}))
        self.assertTrue(any(r.get("torque_estimate_nm",0)>0 for r in self.log.rows))
        self.assertTrue(all(set(r).issubset(sweep.FIELDS) for r in self.log.rows))

    def test_never_raises_a_lower_existing_current_limit(self):
        self.bus.config[0xB3]=b"\xb3"+struct.pack("<I",400)
        self.runner.startup()
        self.assertAlmostEqual(self.runner.current_limit,.4)
        self.assertTrue(self.runner.shutdown())
        self.assertEqual(self.bus.config[0xB3],b"\xb3"+struct.pack("<I",400))

    def test_keeps_shorter_existing_watchdog_and_actions(self):
        original=b"\xcd\x01"+struct.pack("<H",200)+b"\x03"
        self.bus.config[0xCD]=original
        self.runner.startup()
        self.assertEqual(self.bus.config[0xCD],original)
        self.assertTrue(self.runner.shutdown())

    def test_readback_mismatch_prevents_any_speed_command(self):
        self.bus.ignore_writes=True
        with self.assertRaisesRegex(RuntimeError,"acknowledgement"):
            self.runner.startup()
        self.assertFalse(any(data[0]==0xC1 for _,data in self.bus.frames))
        self.assertTrue(self.runner.shutdown())

    def test_missing_query_stops_without_using_cached_reply(self):
        self.runner.startup()
        self.bus.drop.add(0xA1)
        with self.assertRaises(TimeoutError):
            self.runner.run()
        self.assertTrue(self.runner.shutdown())
        self.assertTrue(any(r.get("query_status")=="timeout" for r in self.log.rows))

    def test_disk_failure_cannot_prevent_disable_and_restore(self):
        original=self.bus.config.copy()
        self.runner.startup()
        self.log.fail=True
        with self.assertRaises(OSError):
            self.runner.run()
        self.assertFalse(self.runner.shutdown())  # CSV error is reported, even though shutdown succeeded.
        self.assertEqual(self.bus.mode,0)
        self.assertEqual(self.bus.config,original)

    def test_cf_is_sent_before_processing_unsafe_backlog(self):
        self.runner.startup()
        self.bus.enqueue(b"\xa2"+struct.pack("<i",10000000))
        prior=len(self.bus.frames)
        self.assertTrue(self.runner.shutdown())
        self.assertEqual(self.bus.frames[prior][1],b"\xcf")

    def test_no_restore_without_confirmed_disable(self):
        self.runner.startup()
        self.bus.drop.add(0xCF)
        prior=len(self.bus.frames)
        self.assertFalse(self.runner.shutdown())
        self.assertTrue(all(data==b"\xcf" for _,data in self.bus.frames[prior:]))
        self.assertEqual(self.bus.config[0xB3],b"\xb3"+struct.pack("<I",1000))

    def test_bad_cf_mode_never_restores_settings(self):
        self.runner.startup()
        self.bus.bad_cf=True
        self.assertFalse(self.runner.shutdown())
        self.assertEqual(self.bus.config[0xB3],b"\xb3"+struct.pack("<I",1000))

    def test_stale_disabled_cf_cannot_authorize_restore_when_ae_is_active(self):
        self.runner.startup()
        self.bus.enqueue(self.bus.status(0xCF))  # Queued stale mode=0 acknowledgement.
        self.bus.bad_cf=True
        self.assertFalse(self.runner.shutdown())
        self.assertEqual(self.bus.config[0xB3],b"\xb3"+struct.pack("<I",1000))

    def test_opposite_feedback_direction_aborts(self):
        self.args.hold=.5
        self.runner.startup()
        self.bus.reverse_speed=True
        with self.assertRaisesRegex(RuntimeError,"direction mismatch"):
            self.runner.run()
        self.assertTrue(self.runner.shutdown())

    def test_motion_fault_and_mode_exit_abort(self):
        self.runner.startup()
        self.runner.motion=True
        for mode,fault in ((3,4),(0,0),(2,0)):
            data=b"\xae"+struct.pack("<HH",2400,0)+bytes([25,mode,fault])
            with self.subTest(mode=mode,fault=fault),self.assertRaises(RuntimeError):
                self.runner.decode(data,self.clock.now)
        self.assertTrue(self.runner.shutdown())

    def test_old_feedback_and_command_scheduling_gap_abort(self):
        self.runner.startup()
        self.runner.motion=True
        self.runner.last_command=self.clock.now
        self.clock.now+=.3
        with self.assertRaisesRegex(RuntimeError,"stale"):
            self.runner.service()
        self.runner.times["velocity"]=self.runner.times["current"]=self.clock.now
        with self.assertRaisesRegex(RuntimeError,"scheduling gap"):
            self.runner.service()
        self.assertTrue(self.runner.shutdown())

    def test_protocol_decoding_does_not_use_bus_current_as_torque(self):
        self.runner.decode(b"\xb0\x00"+struct.pack("<f",.3)+b"\x00",10)
        self.runner.decode(b"\xa2"+struct.pack("<i",-6000),10)
        self.runner.decode(b"\xa1"+struct.pack("<i",-200),10)
        self.runner.decode(b"\xae"+struct.pack("<HH",2400,65000)+bytes([25,0,0]),10)
        self.runner.emit("test")
        row=self.log.rows[-1]
        self.assertAlmostEqual(row["velocity_rad_s"],2*math.pi)
        self.assertAlmostEqual(row["torque_estimate_nm"],.06,places=6)
        self.assertEqual(row["bus_current_raw"],65000)

    def test_fault_temperature_overspeed_overcurrent_checks(self):
        self.runner.current_limit=1
        for payload in (b"\xa2"+struct.pack("<i",-100000),b"\xa1"+struct.pack("<i",2000),
                        b"\xae"+struct.pack("<HH",2400,0)+bytes([60,0,0]),
                        b"\xae"+struct.pack("<HH",2400,0)+bytes([25,0,2])):
            with self.subTest(payload=payload),self.assertRaises(RuntimeError):
                self.runner.decode(payload,10)

    def test_conflicting_writer_malformed_reply_and_overflow(self):
        cases=((b"\xc1\0\0\0\0",0x101,[]),
               (b"\xa2\0",1,[]),
               (b"\xa1\0\0\0\0",1,[(socket.SOL_SOCKET,sweep.SO_RXQ_OVFL,struct.pack("=I",1))]))
        for data,cid,ancillary in cases:
            with self.subTest(data=data),self.assertRaises(RuntimeError):
                self.bus.enqueue(data,cid=cid,ancillary=ancillary)
                self.runner.receive()

    def test_csv_is_exclusive_and_has_documented_header(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"test.csv"
            log=sweep.CsvLog(path)
            log.emit({"record_type":"test"})
            log.close()
            with path.open(newline="") as stream:
                self.assertEqual(next(csv.reader(stream)),list(sweep.FIELDS))
            with self.assertRaises(FileExistsError):
                sweep.CsvLog(path)

    def test_execute_entrypoint_writes_real_csv_with_fake_can(self):
        def wait(readable,writable,exceptional,timeout):
            self.bus.wait(timeout)
            return readable,[],[]
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/"record.csv"
            with patch.object(sweep,"open_bus",return_value=self.bus), \
                    patch.object(sweep.time,"monotonic",side_effect=self.clock.monotonic), \
                    patch.object(sweep.time,"time",side_effect=self.clock.time), \
                    patch.object(sweep.select,"select",side_effect=wait):
                result=sweep.main(["--execute","--speeds","1","-1","--hold",".04",
                                   "--settle",".02","--rest",".04","--acceleration","20",
                                   "--output",str(output)])
            self.assertEqual(result,0)
            self.assertTrue(self.bus.closed)
            with output.open(newline="") as stream:
                rows=list(csv.DictReader(stream))
            self.assertTrue(any(row["phase"]=="hold" for row in rows))
            self.assertTrue(any(row["record_type"]=="command" for row in rows))
            self.assertTrue(any(row["opcode_hex"]=="A1" and row["record_type"]=="rx" for row in rows))
            self.assertEqual(rows[-1]["motor_mode"],"0")
            self.assertEqual(self.bus.config[0xB3],b"\xb3"+struct.pack("<I",3000))


if __name__=="__main__":
    unittest.main()
