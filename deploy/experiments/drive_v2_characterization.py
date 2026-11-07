#!/usr/bin/env python3
"""Rear-only v2 MIT PI characterization; preview unless --execute.

300 positive holds (0.1..30 rad/s, 5 s each), each followed by CF-disabled
2 s coasting, then +/-4 rad/s sines at .5,1,2,5,10 Hz (10 s each).
No steering traffic. Uses production v2 codec/integrator, requires numpy.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import select
import signal
import socket
import struct
import sys
import threading

import drive_mit_velocity_test as legacy
import drive_velocity_sine as shared

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'source' / 'trackstand_deploy'))
from trackstand_deploy import contracts_v2 as contract
from trackstand_deploy.mit_protocol import parse_f0

EXTRA_FIELDS = ('trial', 'phase', 'trial_position_rad', 'effective_target_velocity_rad_s',
                'integral_dt_s', 'integral_limited_updates', 'sim_reference_clipped_torque_nm')


@dataclass(frozen=True)
class Trial:
    trial: int
    kind: str
    target_rad_s: float
    frequency_hz: float
    duration_s: float
    rest_s: float = 2.

    def target(self, elapsed):
        return self.target_rad_s if self.kind == 'hold' else self.target_rad_s * math.sin(
            2 * math.pi * self.frequency_hz * elapsed)


def make_plan(args):
    holds = [Trial(i, 'hold', i / 10, 0., 5.) for i in range(1, 301)]
    sines = [Trial(301+i, 'sine', args.amplitude, f, args.sine_duration)
             for i, f in enumerate(args.frequencies)]
    return holds + sines


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--interface', default='can0')
    p.add_argument('--recover-settings', type=Path, help='Restore pending F0/CD from this script JSON; no motion')
    p.add_argument('--amplitude', type=shared.finite, default=4.)
    p.add_argument('--frequencies', nargs='+', type=shared.finite, default=[.5, 1., 2., 5., 10.])
    p.add_argument('--sine-duration', type=shared.finite, default=10.)
    p.add_argument('--feedback-timeout', type=shared.finite, default=1., help='A2 validity deadline in seconds')
    p.add_argument('--output', type=Path, default=None)
    a = p.parse_args(argv)
    if not 0 < a.amplitude <= 61.6 or not 0 < a.sine_duration or not 0 < a.feedback_timeout:
        p.error('amplitude must be in (0,61.6], duration and feedback timeout must be positive')
    if any(not 0 < f < 100 for f in a.frequencies):
        p.error('sine frequencies must be in (0,100) Hz for 200 Hz packets')
    a.output = a.output or HERE / 'logs' / f'{datetime.now():%Y%m%d_%H%M%S_%f}_drive_v2_characterization.csv'
    if a.output.suffix.lower() != '.csv':
        p.error('--output requires .csv extension')
    return a


class CsvLog(shared.CsvLog):
    def __enter__(self):
        self.file = self.path.open('x', newline='')
        try:
            self.writer = csv.DictWriter(self.file, fieldnames=shared.FIELDS + EXTRA_FIELDS)
            self.writer.writeheader()
            self.file.flush()
            self.thread = threading.Thread(target=self._write, daemon=True)
            self.thread.start()
        except BaseException:
            self.file.close()
            raise
        return self


class CharacterizationIO(legacy.MitVelocityTestIO):
    """Reuse rear-only transactions/CF restoration, not legacy experiment guards."""
    def __init__(self, args, *values, persist=lambda: None, **kwargs):
        shared.VelocityTest.__init__(self, args, *values, **kwargs)
        self.persist = persist
        self.disable_confirmed = False
        self.disable_confirmation_method = None
        self.command_ranges = None
        self.encoder_zero_at = None
        self.trial = None
        self.trial_origin = math.nan
        self.phase = 'startup'
        self.reset_integral()

    def reset_integral(self):
        self.integral = contract.VelocityIntegral()
        self.integral_nm = 0.

    def emit(self, kind, stamp=None, **fields):
        fields.update(trial=self.trial.trial if self.trial else '', phase=self.phase,
                      trial_position_rad=self.position_rad-self.trial_origin)
        shared.VelocityTest.emit(self, kind, stamp, **fields)

    def send(self, payload, can_id=0x101, **fields):
        if can_id == 0x501:
            if len(payload) != 8 or self.command_ranges is None:
                raise ValueError('invalid/unconfigured rear MIT packet')
            self.disable_confirmed = False
        elif can_id == 0x101:
            if not payload or payload[0] not in shared.LENGTHS or payload[0] == 0xC1:
                raise ValueError('only rear MIT and read/settings/disable transactions allowed')
            if len(payload) > 1 and (payload[0] not in (0xF0, 0xCD) or not self.disable_confirmed):
                raise ValueError('only confirmed-disabled F0/CD writes allowed')
        else:
            raise ValueError('rear-only: no steering or broadcast transmissions')
        return shared.VelocityTest.send(self, payload, can_id, **fields)

    def startup(self):
        self.disable()
        for op in (0xB0, 0xB3, 0xB5, 0xB8, 0xB9, 0xCD, 0xF0):
            self.settings[f'{op:02X}'] = self.request(op).hex()
        f0, cd = (bytes.fromhex(self.settings[k]) for k in ('F0', 'CD'))
        if cd[1] not in (0, 1):
            raise ValueError('invalid CD watchdog')
        wanted = (contract.rear_f0_payload(f0), b'\xcd\x01' + struct.pack('<H', 500) + cd[4:])
        # One durable pre-write backup, including both originals, before changing either setting.
        self.saved = [original for original, new in zip((f0, cd), wanted) if original != new]
        self.persist()
        for original, new in zip((f0, cd), wanted):
            if original != new:
                self.configure_setting(new)
            self.applied_settings[f'{new[0]:02X}'] = new.hex()
        self.command_ranges = parse_f0(wanted[0])
        self.last_count = self.zero_count = None
        self.travel_counts = 0
        self.request(0xA3)
        self.encoder_zero_at = self.clock.monotonic()
        for op in (0xA2, 0xA1, 0xAE):
            self.request(op)

    def check_feedback(self, now):
        if not math.isfinite(self.velocity) or now-self.velocity_at > self.args.feedback_timeout:
            raise RuntimeError('A2 feedback invalid/stale: cannot integrate velocity error')

    def command(self, target, now, **fields):
        self.check_feedback(now)
        self.integral.update(target-self.velocity, now, contract.encoded_drive_kd(),
                             contract.integral_limit(self.command_ranges))
        self.integral_nm = self.integral.torque_nm
        packet = contract.velocity_packet(target, self.velocity, self.integral_nm, self.command_ranges, -1.)
        c = packet.command
        return self.send(packet.payload, 0x501, target_velocity_rad_s=target,
                         encoded_target_velocity_rad_s=-c.velocity_rad_s,
                         effective_target_velocity_rad_s=packet.effective_target_rad_s,
                         mit_kp=c.kp_nm_per_rad, mit_kd=c.kd_nm_s_per_rad, mit_feedforward_native_nm=c.feedforward_torque_nm,
                         integral_error_rad=self.integral.error_rad, integral_torque_nm=self.integral_nm,
                         integral_dt_s=self.integral.last_dt_s,
                         integral_limited_updates=self.integral.limited_updates,
                         estimated_mit_total_torque_nm=packet.estimated_total_torque_nm,
                         sim_reference_clipped_torque_nm=max(-6.28, min(6.28, packet.estimated_total_torque_nm)),
                         **fields)

    def run_window(self, duration, enabled):
        self.phase_start = self.next_command = self.clock.monotonic()
        end = self.phase_start + duration
        # Same rear query cadence as v2's state/query timer; MIT replies also carry F1.
        due = self.phase_start
        self.active = enabled
        count = skipped = 0
        while self.clock.monotonic() < end:
            if self.stop_requested():
                raise KeyboardInterrupt
            self.log.check()
            now = self.clock.monotonic()
            if enabled and now >= self.next_command:
                omitted = max(0, int((now-self.next_command)*200))
                skipped += omitted
                self.command(self.trial.target(now-self.phase_start), now,
                             scheduled_at=self.next_command, skipped_slots=omitted,
                             command_index=count, target_sample_time_s=now-self.phase_start)
                count += 1
                self.next_command = self.phase_start + (int((self.clock.monotonic()-self.phase_start)*200+1e-7)+1)/200
            if now >= due:
                for op in (0xA1, 0xA2, 0xA3, 0xAE):
                    self.send(bytes([op]))
                due = self.clock.monotonic() + .01
            self.drain()
            deadline = min(end, due, self.next_command if enabled else end)
            select.select([self.bus], [], [], max(0., min(.01, deadline-self.clock.monotonic())))
        self.active = False
        self.phases.append(dict(trial=self.trial.trial, phase=self.phase, commands=count,
                                skipped_slots=skipped, start_monotonic_s=self.phase_start,
                                actual_duration_s=self.clock.monotonic()-self.phase_start,
                                average_packet_rate_hz=count/(self.clock.monotonic()-self.phase_start)))

    def run_trial(self, trial):
        self.trial, self.phase = trial, trial.kind
        self.request(0xA3)
        self.request(0xA2)
        self.trial_origin = self.position_rad
        self.reset_integral()
        self.emit('event', note='Trial start; software displacement reference refreshed; integral reset')
        self.run_window(trial.duration_s, True)
        self.disable()
        self.phase = 'disabled_rest'
        self.emit('event', note='CF confirmed; exactly 2 seconds nominal coast; no rest-speed criterion')
        self.run_window(trial.rest_s, False)
        print(f'Trial {trial.trial}: {trial.kind} {trial.target_rad_s:g} rad/s complete', flush=True)

    def receive(self):
        frame, ancillary, flags, _ = self.bus.recvmsg(shared.CAN_FRAME.size, socket.CMSG_SPACE(4))
        now = self.clock.monotonic()
        if len(frame) != shared.CAN_FRAME.size or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise RuntimeError("truncated CAN frame")
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == shared.RX_OVERFLOW and struct.unpack("=I", data)[0]:
                raise RuntimeError("CAN receive queue overflow; log is incomplete")
        cid, length, padded = shared.CAN_FRAME.unpack(frame)
        data = padded[:length]
        if cid & ~0x7ff or length > 8:
            raise RuntimeError(f"CAN error/nonstandard frame {cid:08X}")
        if cid != 1 or flags & socket.MSG_DONTROUTE:
            raise RuntimeError("another rear/broadcast CAN writer observed; stop other writers")
        if not data or shared.LENGTHS.get(data[0]) != length:
            raise RuntimeError(f"unexpected motor response: {data.hex()}")
        extra, fault = {}, None
        op = data[0]
        if op in (0xA2, 0xC1):
            self.velocity = shared.SIGN * struct.unpack("<i", data[1:])[0] * .01 * 2 * math.pi / 60
            self.velocity_at = now
        elif op == 0xA1:
            self.current, self.current_at = struct.unpack("<i", data[1:])[0] * .001, now
        elif op == 0xA3:
            count = struct.unpack("<i", data[3:7])[0]
            if self.last_count is None:
                self.zero_count = count
            else:
                self.travel_counts += (count - self.last_count + 2**31) % 2**32 - 2**31
            self.last_count, self.position_rad = count, shared.SIGN * self.travel_counts * 2 * math.pi / 16384
            extra["multi_turn_count"] = count
        elif op == 0xB0:
            self.kt = struct.unpack("<f", data[2:6])[0]
            if not math.isfinite(self.kt) or self.kt <= 0:
                fault = "invalid B0 torque constant"
        elif op in (0xB8, 0xB9):
            self.raw_gains[f"{op:02X}"] = struct.unpack("<f", data[1:])[0]
        elif op == 0xF0:
            p, v, t = struct.unpack("<HHH", data[1:])
            self.ranges = (p * .1, v * .01, t * .01)
            if min(self.ranges) <= 0:
                fault = "invalid F0 encoding ranges"
        elif op == 0xF1 and self.ranges is not None:
            p = (data[1] << 8) | data[2]
            v = (data[3] << 4) | (data[4] >> 4)
            t = ((data[4] & 15) << 8) | data[5]
            self.f1_at, self.f1_status = now, data[6]
            extra = dict(mit_position_native_rad=shared.decode(p, self.ranges[0], 16),
                         mit_velocity_rad_s=shared.SIGN * shared.decode(v, self.ranges[1], 12),
                         mit_torque_nm=shared.SIGN * shared.decode(t, self.ranges[2], 12), mit_status=self.f1_status)
            if self.f1_status & 2:
                fault = "MIT driver system fault"
        elif op in (0xAE, 0xCF):
            volts, amps = struct.unpack("<HH", data[1:5])
            self.status_at = now
            self.driver_status = dict(bus_voltage_v=volts * .01, bus_current_a=amps * .01,
                                      temperature_c=data[5], driver_mode=data[6], fault_code=data[7])
            extra.update(self.driver_status)
            if data[7]:
                fault = f"motor driver fault code {data[7]}"
        self.emit("rx", now, can_id=f"{cid:03X}", payload_hex=data.hex().upper(), opcode=f"{op:02X}",
                  note=fault or "", **extra)
        if fault and not self.cleaning:
            raise RuntimeError(fault)
        return data


def main(argv=None):
    args = parse_args(argv)
    plan = [] if args.recover_settings else make_plan(args)
    recovery = None
    if args.recover_settings:
        recovery = json.loads(args.recover_settings.read_text())
        if recovery.get('schema') != 'drive_v2_characterization_v1' or recovery.get('motor_id') != 1:
            raise ValueError('recovery requires this rear-only characterization JSON')
        for raw in recovery.get('pending_restore', []):
            payload = bytes.fromhex(raw)
            if not payload or payload[0] not in (0xF0, 0xCD) or len(payload) != shared.LENGTHS[payload[0]]:
                raise ValueError('invalid pending restore payload')
    print(f'Rear only: {len(plan)} trials (300 holds + 5 sines by default; recovery runs none); '
          f'{sum(t.duration_s+t.rest_s for t in plan):g} s nominal, excluding setup/CF acknowledgement.')
    print('MIT Kp=0, Kd=.3, host Ki=3.3; packets/integral 200 Hz, queries 100 Hz.')
    if not args.execute:
        print('PREVIEW: no CAN or files. Add --execute; rear wheel airborne/free, other writers stopped.')
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = args.output.with_suffix('.json')
    plan_path = args.output.with_name(args.output.stem+'_plan.csv')
    if any(p.exists() for p in (args.output, meta, plan_path, meta.with_suffix('.json.tmp'))):
        raise FileExistsError('refusing to overwrite experiment files')
    with plan_path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=Trial.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(asdict(t) for t in plan)
    report = dict(schema='drive_v2_characterization_v1', completed=False, disabled_and_restored=False,
                  motor_id=1, steering_traffic=False, total_internal_torque_clamp_verified=False,
                  arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  helper_sha256={Path(mod.__file__).name: hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
                                 for mod in (legacy, legacy.c1, shared)},
                  contract_sha256=hashlib.sha256(Path(contract.__file__).read_bytes()).hexdigest(),
                  plan=[asdict(t) for t in plan])
    bus = motor = None
    def persist():
        if motor is not None:
            report.update(original_settings=motor.settings, applied_settings=motor.applied_settings,
                          pending_restore=[p.hex() for p in motor.saved], phases=motor.phases,
                          initial_a3_count=motor.zero_count, torque_constant_nm_per_a=motor.kt,
                          current_limit_b3_a=(struct.unpack('<I',bytes.fromhex(motor.settings['B3'])[1:])[0]*.001
                                              if 'B3' in motor.settings else None))
            if report['current_limit_b3_a'] is not None:
                report['current_limit_b3_times_kt_nm'] = report['current_limit_b3_a']*motor.kt
        legacy.c1.save_report(meta,report)
    persist()
    print(f'CSV: {args.output}', flush=True)
    print('Total MIT PD torque clamp unverified: a 30 rad/s step from rest initially requests about 9 Nm.')
    stopped = threading.Event()
    handlers={sig:signal.signal(sig,lambda *_:stopped.set()) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        with CsvLog(args.output) as log:
            try:
                bus=legacy.open_bus(args.interface)
                motor=CharacterizationIO(args,bus,log,stopped.is_set,persist=persist)
                if recovery is not None:
                    motor.saved = [bytes.fromhex(raw) for raw in recovery.get('pending_restore', [])]
                    report['recovery_source'] = str(args.recover_settings)
                    persist()
                else:
                    motor.startup()
                for trial in plan:
                    motor.run_trial(trial)
                report['completed']=True
            except BaseException as error:
                report['error']=f'{type(error).__name__}: {error}'
                print('STOPPED: '+report['error'],file=sys.stderr)
            finally:
                if motor is not None:
                    try:
                        motor.cleanup()
                        report['disabled_and_restored']=True
                    except BaseException as error:
                        report['cleanup_error']=f'{type(error).__name__}: {error}'
                if bus is not None:
                    bus.close()
                persist()
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
    return 0 if report['completed'] and report['disabled_and_restored'] else 1


if __name__ == '__main__':
    sys.exit(main())
