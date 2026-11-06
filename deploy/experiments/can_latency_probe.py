#!/usr/bin/env python3
"""Disabled deployment-load CAN probe, or transmit-free capture during real replay.

No ROS dependency unless --ros is used. Default is preview; --execute opens CAN.
Never sends motion/zeroing/configuration commands. Disabled mode sends CF then
read-only queries. Passive mode sends nothing, including on exit.
"""
import argparse
from collections import defaultdict, deque
import json
import math
from pathlib import Path
import select
import signal
import socket
import struct
import subprocess
import threading
import time
from datetime import datetime

from latency_common import ClockMap, CsvLog, dump

FRAME = struct.Struct('=IB3x8s')
SO_TIMESTAMPNS_NEW, SO_RXQ_OVFL = 64, 40
CAN_ERR_FLAG = 0x20000000
LENGTHS = {0xA3: 7, 0xA2: 5, 0xA1: 5, 0xAE: 8, 0xB0: 7, 0xF0: 7, 0xF1: 7, 0xCD: 5, 0xCF: 8}
BATCH = ((2, 0xA3), (2, 0xA2), (1, 0xA3), (1, 0xA2),
         (2, 0xA1), (1, 0xA1), (2, 0xAE), (1, 0xAE))


def frame_bits(dlc):
    # Standard Classical CAN: SOF..EOF plus 3-bit intermission, excluding stuffing/retries.
    return 47 + 8 * dlc


def load_plan(scale=1.):
    base_bits = 100 * sum(frame_bits(1) + frame_bits(LENGTHS[op]) for _, op in BATCH)
    mit_bits = 2 * 200 * (frame_bits(8) + frame_bits(7))
    f1_rate = 200 * (frame_bits(8) + frame_bits(7)) / (frame_bits(1) + frame_bits(7))
    return dict(query_batch_hz=100 * scale, f1_queries_per_motor_hz=f1_rate * scale,
                reference_bits_per_s=base_bits + mit_bits,
                probe_expected_bits_per_s=(base_bits + mit_bits) * scale,
                reference_frames_per_s=2400, probe_expected_frames_per_s=(1600 + 4*f1_rate)*scale,
                excludes='bit stuffing, retries, startup and low-rate configuration reads',
                substitute='F1 reads replace two 200-Hz 8-byte MIT streams; no motor activation')


class Tracker:
    """CAN protocol has no transaction number: report ambiguity, never invent correlation."""
    def __init__(self):
        self.pending = defaultdict(deque)
        self.tainted = set()

    def sent(self, motor, opcode, seq, stamp):
        key = (motor, opcode)
        if self.pending[key]:
            self.tainted.add(key)
        self.pending[key].append((seq, stamp))

    def expire(self, now, timeout=.05):
        expired = []
        for key, pending in self.pending.items():
            while pending and now - pending[0][1] > timeout:
                self.tainted.add(key)
                expired.append((key, *pending.popleft()))
        return expired

    def reply(self, motor, opcode, stamp, valid_clock):
        key = (motor, opcode)
        pending = self.pending[key]
        if not pending:
            return None, None, False
        seq, sent = pending.popleft()
        valid = valid_clock and key not in self.tainted and stamp is not None and stamp >= sent
        return seq, (stamp-sent)*1000 if valid else None, valid


class Bus:
    def __init__(self, interface):
        self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS_NEW, 1)
            self.sock.setsockopt(socket.SOL_SOCKET, SO_RXQ_OVFL, 1)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
            # Include error frames; CPython on Pi may omit this Linux constant.
            self.sock.setsockopt(socket.SOL_CAN_RAW, getattr(socket, 'CAN_RAW_ERR_FILTER', 2), struct.pack('=I', 0x1FFFFFFF))
            self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_RECV_OWN_MSGS, 0)
            self.sock.bind((interface,))
            self.sock.setblocking(False)
        except BaseException:
            self.sock.close()
            raise
        self.drops = 0

    def send(self, motor, opcode):
        # Structural allowlist: only one-byte read commands and disable are possible.
        if motor not in (1, 2) or opcode not in LENGTHS:
            raise ValueError('not an allowed diagnostic request')
        if self.sock.send(FRAME.pack(0x100 | motor, 1, bytes([opcode]))) != FRAME.size:
            raise OSError('short CAN send')

    def receive(self):
        try:
            raw, anc, flags, _ = self.sock.recvmsg(FRAME.size, socket.CMSG_SPACE(16)+socket.CMSG_SPACE(4))
        except BlockingIOError:
            return None
        dequeue = time.monotonic()
        if len(raw) != FRAME.size or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise RuntimeError('truncated CAN frame/timestamp')
        can_id, dlc, data = FRAME.unpack(raw)
        if dlc > 8:
            raise RuntimeError('invalid Classical CAN DLC')
        wall = None
        for level, kind, value in anc:
            if level == socket.SOL_SOCKET and kind == SO_TIMESTAMPNS_NEW and len(value) == 16:
                sec, ns = struct.unpack('=qq', value)
                wall = sec + ns*1e-9
            if level == socket.SOL_SOCKET and kind == SO_RXQ_OVFL and len(value) == 4:
                self.drops = struct.unpack('=I', value)[0]
        return can_id, data[:dlc], wall, dequeue, bool(flags & socket.MSG_DONTROUTE), self.drops

    def close(self):
        self.sock.close()


def system_snapshot(interface):
    result = {'wall_s': time.time(), 'monotonic_s': time.monotonic()}
    try:
        p = subprocess.run(['ip', '-details', '-statistics', 'link', 'show', interface], capture_output=True, text=True, timeout=3)
        result['ip_link'] = p.stdout + p.stderr
    except (OSError, subprocess.TimeoutExpired) as error:
        result['ip_link_error'] = str(error)
    for path in ('/proc/stat', '/proc/loadavg', '/proc/meminfo', '/proc/diskstats',
                 '/sys/class/thermal/thermal_zone0/temp'):
        try:
            result[path] = Path(path).read_text()
        except OSError:
            pass
    return result


class Probe:
    def __init__(self, args, log, stop):
        self.args, self.log, self.stop = args, log, stop
        self.bus = None
        self.clock = ClockMap()
        self.tracker = Tracker()
        self.seq = 0
        self.latest = {}  # Kernel timestamp/dequeue/process per motor/opcode.
        self.lock = threading.Lock()
        self.verified = set()
        self.base_wait = None
        self.tx = self.rx = self.bits = 0
        self.error = None
        self.started = self.finished = None

    def send(self, motor, opcode, scheduled):
        self.seq += 1
        start = time.monotonic()
        self.bus.send(motor, opcode)
        end = time.monotonic()
        self.tx += 1; self.bits += frame_bits(1)
        self.tracker.sent(motor, opcode, self.seq, start)
        self.log.emit('send', seq=self.seq, motor=motor, opcode=hex(opcode), scheduled_s=scheduled,
                      send_start_s=start, send_end_s=end, can_id=hex(0x100|motor), dlc=1)
        return start

    def receive(self):
        frame = self.bus.receive()
        if frame is None:
            return False
        can_id, data, wall, dequeue, local, dropped = frame
        offset, width, jump, valid = self.clock.sample()
        kernel = None if wall is None else wall-offset
        valid = valid and kernel is not None and kernel <= time.monotonic() + width/2000.
        if not valid:
            kernel = None
        motor = can_id if can_id in (1, 2) else None
        opcode = data[0] if data else None
        seq, rtt, paired = (None, None, False)
        unsafe_reason = None
        if motor and opcode in LENGTHS and len(data) == LENGTHS[opcode] and not local:
            seq, rtt, paired = self.tracker.reply(motor, opcode, kernel, valid)
            if self.base_wait and self.base_wait[:2] == (motor, opcode):
                # A kernel timestamp preceding the query cannot complete that query.
                if kernel is not None and kernel >= self.base_wait[2]:
                    self.base_wait = None
            if self.args.mode == 'disabled' and opcode == 0xF1:
                if data[6] != 0:
                    unsafe_reason = f'motor {motor} F1 active/fault/unknown status {data[6]}; stop other writers'
                if data[6] == 0 and kernel is not None and kernel >= self.disable_sent[motor]:
                    self.verified.add(motor)
            processed = time.monotonic()
            with self.lock:
                self.latest[(motor, opcode)] = (kernel, dequeue, processed)
        else:
            processed = time.monotonic()
        self.rx += 1; self.bits += frame_bits(len(data))
        self.log.emit('receive', seq=seq, motor=motor, opcode='' if opcode is None else hex(opcode),
                      can_id=hex(can_id), dlc=len(data), payload=data.hex(), kernel_wall_s=wall,
                      kernel_mono_s=kernel, dequeue_s=dequeue, processed_s=processed,
                      clock_uncertainty_ms=width, clock_offset_jump_ms=jump, clock_valid=int(valid),
                      local_tx=int(local), rx_dropped=dropped, rtt_ms=rtt, pair_valid=int(paired),
                      queue_ms=None if kernel is None else (dequeue-kernel)*1000)
        if unsafe_reason is not None:
            raise RuntimeError(unsafe_reason)
        if self.args.mode == 'disabled':
            if local or can_id in (0, 0xFF, 0x101, 0x102, 0x400, 0x4FF, 0x401, 0x402, 0x501, 0x502):
                raise RuntimeError(f'another CAN writer detected: {can_id:#x}; disabled test invalid')
        return True

    def run(self):
        jobs = deque()
        try:
            self.bus = Bus(self.args.interface)
            if self.args.mode == 'disabled':
                self.disable_sent = {m: self.send(m, 0xCF, time.monotonic()) for m in (1, 2)}
                deadline = time.monotonic()+3.
                next_check = 0.
                while not self.stop.is_set() and self.verified != {1, 2}:
                    now = time.monotonic()
                    if now > deadline:
                        raise RuntimeError('cannot confirm both motors disabled via F1')
                    if now >= next_check:
                        for m in (1, 2):self.send(m, 0xF1, now)
                        next_check = now+.1
                    select.select([self.bus.sock], [], [], .002)
                    for _ in range(256):
                        if not self.receive():break
                if self.stop.is_set():return
            self.started = time.monotonic()
            self.log.emit('measurement_start', detail=json.dumps({'mode': self.args.mode, 'tx_before':self.tx,'rx_before':self.rx,'bits_before':self.bits}))
            plan = load_plan(self.args.load_scale)
            next_batch = next_f1 = next_config = self.started
            while not self.stop.is_set() and time.monotonic()-self.started < self.args.duration:
                # Bound each drain so an overloaded receive queue cannot starve scheduling.
                for _ in range(256):
                    if not self.receive():break
                now = time.monotonic()
                for key, seq, sent in self.tracker.expire(now):
                    self.log.emit('reply_timeout', seq=seq, motor=key[0], opcode=hex(key[1]), send_start_s=sent)
                if self.args.mode == 'disabled':
                    if self.base_wait and now-self.base_wait[2] > .002:
                        self.log.emit('query_2ms_budget_exceeded', motor=self.base_wait[0], opcode=hex(self.base_wait[1]))
                        self.base_wait = None
                    if now >= next_batch and not jobs:
                        scheduled = next_batch
                        jobs.extend((m, op, scheduled) for m, op in BATCH)
                        period = 1/plan['query_batch_hz']
                        missed = max(0, math.floor((now-next_batch)/period))
                        next_batch += (missed+1)*period
                        self.log.emit('query_batch', scheduled_s=scheduled, detail=str(missed))
                        if now >= next_config:
                            jobs.extend((m, op, now) for m, op in ((2,0xF0),(1,0xF0),(2,0xCD),(1,0xCD)))
                            next_config = now+1.
                    if now >= next_f1:
                        scheduled = next_f1
                        period = 1/plan['f1_queries_per_motor_hz']
                        next_f1 += (max(0,math.floor((now-next_f1)/period))+1)*period
                        for m in (2,1):self.send(m,0xF1,scheduled)
                    if self.base_wait is None and jobs:
                        m,op,scheduled = jobs.popleft()
                        self.base_wait=(m,op,self.send(m,op,scheduled))
                select.select([self.bus.sock], [], [], .0005)
        except BaseException as error:
            self.error = f'{type(error).__name__}: {error}'
            try:self.log.emit('fatal',detail=self.error)
            except Exception:pass
        finally:
            self.finished = time.monotonic()
            if self.bus is not None:
                if self.args.mode == 'disabled':
                    for m in (1,2):
                        try:self.send(m,0xCF,time.monotonic())
                        except Exception:pass
                self.bus.close()
            self.stop.set()


def run_ros(probe, log, stop, run_id):
    """Synthetic deployment-like publication load; never publishes motor authority."""
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray, String, Bool
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node('trackstand_latency_probe')
    prefix='/trackstand/latency_probe/'
    pubs={name:node.create_publisher(kind,prefix+name,10) for name,kind in
          (('state',Float64MultiArray),('control',Float64MultiArray),('status',String),('fault',Bool))}
    seq=0
    def tick():
        nonlocal seq
        seq+=1;now=time.monotonic()
        with probe.lock:
            samples=[probe.latest.get(key,(None,None,None)) for key in ((2,0xA3),(2,0xA2),(1,0xA3),(1,0xA2))]
        stamps=[v if v is not None else -1. for row in samples for v in row]
        # Same nominal 200 control + 100 state/status/fault publication rates.
        pubs['control'].publish(Float64MultiArray(data=[float(seq),now]+[0.]*126))
        if seq%2==0:
            pub=time.monotonic()
            pubs['state'].publish(Float64MultiArray(data=[float(seq),pub,*stamps]))
            log.emit('ros_publish',seq=seq,publish_s=pub,detail=json.dumps({'run_id':run_id,'sample_kernel_dequeue_processed':samples}))
            pubs['status'].publish(String(data=json.dumps({'run_id':run_id,'tx':probe.tx,'rx':probe.rx,'diagnostic_padding':'x'*1000})))
            pubs['fault'].publish(Bool(data=probe.error is not None))
    node.create_timer(.005,tick)
    try:
        while not stop.is_set() and rclpy.ok():rclpy.spin_once(node,timeout_sec=.01)
    finally:
        node.destroy_node();rclpy.try_shutdown()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--mode',choices=('disabled','passive'),default='disabled')
    p.add_argument('--interface',default='can0')
    p.add_argument('--duration',type=float,default=120.)
    p.add_argument('--load-scale',type=float,default=1.,help='Disabled mode only; 1 matches estimated deployment bus airtime')
    p.add_argument('--ros',action='store_true',help='Add deployment-like ROS publisher load; no policy inference')
    p.add_argument('--label',default='minimal')
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'logs')
    args=p.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration<=0 or not math.isfinite(args.load_scale) or not 0<args.load_scale<=2:
        p.error('duration must be positive and finite; load-scale must be in (0,2]')
    print(json.dumps(load_plan(args.load_scale),indent=2))
    print('Disabled: independent support, motors unpowered by control; stop ALL other CAN writers. Passive: no transmissions.')
    if not args.execute:
        print('Preview only; add --execute to record.');return 0
    if args.ros:
        import rclpy  # Fail before touching CAN if ROS environment is not sourced.
    args.output_dir.mkdir(parents=True,exist_ok=True)
    name=datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_can_latency'
    out=args.output_dir/name;out.mkdir()
    stop=threading.Event()
    handlers={s:signal.signal(s,lambda *_:stop.set()) for s in (signal.SIGINT,signal.SIGTERM)}
    dump(out/'metadata.json',dict(mode=args.mode,label=args.label,interface=args.interface,duration_s=args.duration,
         ros=args.ros,load_plan=load_plan(args.load_scale),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         note='read-only substitute has similar estimated airtime, not identical firmware/actuation load'))
    dump(out/'system_before.json',system_snapshot(args.interface))
    log=CsvLog(out/'can.csv');probe=Probe(args,log,stop)
    thread=threading.Thread(target=probe.run,daemon=True)
    print(f'Logs: {out}',flush=True)
    thread.start();samples=[]
    def monitor():
        while not stop.wait(1.):samples.append(system_snapshot(args.interface))
    mon=threading.Thread(target=monitor,daemon=True);mon.start()
    error=None
    try:
        if args.ros:run_ros(probe,log,stop,name)
        else:
            while not stop.wait(.2):pass
    except BaseException as e:
        error=f'{type(e).__name__}: {e}'
    finally:
        stop.set();thread.join(timeout=5);mon.join(timeout=5)
        try:
            dump(out/'system_samples.json',samples)
            dump(out/'system_after.json',system_snapshot(args.interface))
            dump(out/'result.json',{'error':probe.error or error,'worker_stopped':not thread.is_alive(),
                  'tx':probe.tx,'rx':probe.rx,'estimated_bits':probe.bits,'started_s':probe.started,'finished_s':probe.finished,
                  'tainted_pair_keys':[list(k) for k in sorted(probe.tracker.tainted)]})
        finally:
            log.close()
            for s,h in handlers.items():signal.signal(s,h)
    print('Finished:',probe.error or error or 'complete; inspect achieved traffic and timing, not just exit status')
    return 1 if probe.error or error or thread.is_alive() else 0


if __name__=='__main__':
    raise SystemExit(main())
